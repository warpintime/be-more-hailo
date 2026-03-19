import base64
import os
import requests
import logging
import re
import json
import urllib.parse
import numpy as np
from .config import (
    LLM_PROVIDER,
    LLM_URL,
    LLM_MODEL,
    FAST_LLM_MODEL,
    VISION_MODEL,
    VLM_HEF_PATH,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    get_system_prompt,
)
from .tts import add_pronunciation
from .search import search_web

try:
    import google.generativeai as genai
except Exception:
    genai = None

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Hailo VLM (Vision Language Model) singleton
# --------------------------------------------------------------------------- #
# Lazy-loaded on first image analysis request.  Kept alive so subsequent
# requests don't pay the ~3 s init cost again.
_vlm_instance = None
_vlm_vdevice = None


def _get_vlm():
    """Return a (vlm, frame_shape, frame_dtype) tuple, initialising on first call."""
    global _vlm_instance, _vlm_vdevice

    if _vlm_instance is not None:
        shape = _vlm_instance.input_frame_shape()
        dtype = _vlm_instance.input_frame_format_type()
        return _vlm_instance, shape, dtype

    hef = VLM_HEF_PATH
    if not os.path.isabs(hef):
        # Resolve relative to project root (where the scripts live)
        hef = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), hef)

    if not os.path.exists(hef):
        raise FileNotFoundError(f"VLM HEF not found at {hef}. Run setup.sh to download it.")

    from hailo_platform import VDevice
    from hailo_platform.genai import VLM

    logger.info(f"Initialising Hailo VLM from {hef} ...")
    _vlm_vdevice = VDevice()
    _vlm_instance = VLM(_vlm_vdevice, hef)
    shape = _vlm_instance.input_frame_shape()
    dtype = _vlm_instance.input_frame_format_type()
    logger.info(f"VLM ready — frame shape {shape}, dtype {dtype}")
    return _vlm_instance, shape, dtype


def _decode_image_to_frame(image_b64: str, target_shape, target_dtype=np.uint8):
    """Decode a base64 JPEG/PNG into a numpy array matching VLM input requirements."""
    import cv2

    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from base64 data")

    # OpenCV loads BGR; VLM expects RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = target_shape
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    return img.astype(target_dtype)

# Keep at most this many messages (plus the system prompt) to avoid
# unbounded memory growth on memory-constrained devices like a Pi.
MAX_HISTORY_MESSAGES = 20

_DISPLAY_IMAGE_KEYWORDS = [
    "show me a picture", "show me an image", "show me a photo",
    "show a picture", "show an image", "show a photo",
    "display a picture", "display an image", "display a photo",
    "picture of", "image of", "photo of",
    "generate an image", "generate a picture",
    "draw me", "draw a",
]

# Phrases that indicate the model is regurgitating system prompt instructions
_PROMPT_LEAK_PATTERNS = [
    re.compile(r'CRIT(?:IC)?AL:.*?JSON', re.IGNORECASE | re.DOTALL),
    re.compile(r'you MUST output.*?JSON', re.IGNORECASE | re.DOTALL),
    re.compile(r'output (?:exactly )?this JSON', re.IGNORECASE),
    re.compile(r'If the user asks.*?(?:output|JSON)', re.IGNORECASE | re.DOTALL),
    re.compile(r'Replace YOUR_PROMPT_HERE', re.IGNORECASE),
    re.compile(r'Valid emotions are:', re.IGNORECASE),
]


def _build_display_image_action(user_text: str) -> str:
    """Extract the subject from user text and return a display_image JSON action."""
    # Strip common prefixes to get the image subject
    subject = user_text
    for prefix in ["show me a picture of", "show me an image of", "show me a photo of",
                    "show a picture of", "show an image of", "show a photo of",
                    "display a picture of", "display an image of", "display a photo of",
                    "generate an image of", "generate a picture of",
                    "draw me a", "draw me an", "draw me", "draw a", "draw an",
                    "picture of", "image of", "photo of"]:
        lower = subject.lower()
        if lower.startswith(prefix):
            subject = subject[len(prefix):].strip()
            break
    # Remove trailing punctuation
    subject = subject.rstrip("?.!")
    prompt = urllib.parse.quote(subject.strip() or "something fun")
    return json.dumps({"action": "display_image", "image_url": f"https://image.pollinations.ai/prompt/{prompt}"})


def _strip_prompt_leakage(content: str) -> str:
    """Remove text that looks like the model echoing system prompt instructions."""
    for pat in _PROMPT_LEAK_PATTERNS:
        # Remove the matched pattern and everything after it (it's usually trailing noise)
        match = pat.search(content)
        if match:
            content = content[:match.start()].strip()
    return content


def _extract_gemini_text(response) -> str:
    """Best-effort extraction of plain text from Gemini SDK responses."""
    text = getattr(response, "text", "")
    if text:
        return text

    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", "")
            if part_text:
                parts.append(part_text)
    return "".join(parts).strip()


def _split_into_speak_chunks(text: str):
    """Split generated text into sentence-ish chunks for TTS streaming compatibility."""
    if not text:
        return []
    chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+|\n+", text) if c and c.strip()]
    return chunks or [text.strip()]


class Brain:
    def __init__(self):
        self.history = [{"role": "system", "content": get_system_prompt()}]

    def _trim_history(self):
        """Keep the system prompt + the most recent MAX_HISTORY_MESSAGES messages."""
        # history[0] is always the system prompt
        non_system = self.history[1:]
        if len(non_system) > MAX_HISTORY_MESSAGES:
            self.history = [self.history[0]] + non_system[-MAX_HISTORY_MESSAGES:]

    def _choose_model(self, user_text: str) -> str:
        complex_keywords = [
            "explain", "story", "how", "why", "code", "write", "create",
            "analyze", "compare", "difference", "history", "long"
        ]
        words = user_text.lower().split()
        if len(words) > 15 or any(kw in words for kw in complex_keywords):
            return LLM_MODEL
        return FAST_LLM_MODEL

    def _provider(self) -> str:
        provider = (LLM_PROVIDER or "ollama").strip().lower()
        if provider not in ("ollama", "gemini"):
            provider = "ollama"
        return provider

    def _call_gemini(self, messages, temperature=0.4, max_tokens=120) -> str:
        if genai is None:
            raise RuntimeError("Gemini SDK not available. Install google-generativeai.")
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        genai.configure(api_key=GEMINI_API_KEY)

        system_parts = []
        contents = []
        for msg in messages:
            role = msg.get("role")
            text = str(msg.get("content", ""))
            if not text:
                continue
            if role == "system":
                system_parts.append(text)
            elif role == "assistant":
                contents.append({"role": "model", "parts": [text]})
            else:
                contents.append({"role": "user", "parts": [text]})

        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction="\n".join(system_parts).strip() or None,
        )
        response = model.generate_content(
            contents or [{"role": "user", "parts": [""]}],
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        return _extract_gemini_text(response)

    def _call_ollama_once(self, messages, model_name, temperature=0.4, max_tokens=120) -> str:
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        response = requests.post(LLM_URL, json=payload, timeout=180)
        if response.status_code != 200:
            raise RuntimeError(f"LLM Error: {response.status_code} - {response.text}")
        data = response.json()
        return data.get("message", {}).get("content", "")

    def _call_llm_once(self, messages, model_name, temperature=0.4, max_tokens=120) -> str:
        if self._provider() == "gemini":
            return self._call_gemini(messages, temperature=temperature, max_tokens=max_tokens)
        return self._call_ollama_once(messages, model_name, temperature=temperature, max_tokens=max_tokens)

    def think(self, user_text: str) -> str:
        """
        Send text to local LLM (Hailo/Ollama) and get response.
        """
        self.history.append({"role": "user", "content": user_text})

        lower_text = user_text.lower()

        # Pre-LLM camera check — same logic as stream_think
        camera_keywords = [
            "take a photo", "take a picture", "take photo", "take picture",
            "look at", "what do you see", "what can you see", "use your camera",
            "photograph", "snap a photo",
        ]
        if any(kw in lower_text for kw in camera_keywords):
            action = '{"action": "take_photo"}'
            self.history.append({"role": "assistant", "content": action})
            return action

        # Pre-LLM display_image check — handle image generation requests
        # directly instead of relying on the small model to emit correct JSON
        if any(kw in lower_text for kw in _DISPLAY_IMAGE_KEYWORDS):
            action = _build_display_image_action(user_text)
            self.history.append({"role": "assistant", "content": action})
            return action

        # Pre-LLM web search — same logic as stream_think
        realtime_keywords = [
            "weather", "forecast", "temperature", "tonight", "tomorrow",
            "news", "latest", "right now", "score", "stocks", "bitcoin",
            "crypto", "price of", "happening", "recently", "live",
        ]
        question_markers = [
            "what", "who", "when", "where", "find", "search", "tell me",
            "look up", "check", "is there", "did", "?",
        ]
        has_realtime_kw = any(kw in lower_text for kw in realtime_keywords)
        has_question = any(q in lower_text for q in question_markers)
        search_injected = False
        if has_realtime_kw and has_question:
            try:
                search_result = search_web(user_text)
                if search_result and search_result not in ("SEARCH_EMPTY", "SEARCH_ERROR") and len(search_result) > 50:
                    clean_result = re.sub(r"^SEARCH RESULTS for '.*?':\n?", "", search_result).strip()
                    self.history[-1]["content"] = (
                        f"[LIVE DATA: {clean_result}] "
                        f"Using only the above live data, answer in one or two sentences as BMO: {user_text}"
                    )
                    search_injected = True
            except Exception as e:
                logger.warning(f"Pre-LLM web search failed: {e}")

        chosen_model = self._choose_model(user_text)

        try:
            provider = self._provider()
            logger.info(f"Sending request to LLM ({provider}/{chosen_model})")
            content = self._call_llm_once(self.history, chosen_model, temperature=0.4, max_tokens=120)

            try:
                clean_content = content.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
                json_match = re.search(r'\{.*?\}', clean_content, re.DOTALL)
                if json_match:
                    action_data = json.loads(json_match.group(0))

                    if action_data.get("action") == "take_photo":
                        logger.info("LLM requested to take a photo.")
                        return json.dumps({"action": "take_photo"})

                    elif action_data.get("action") == "search_web":
                        query = action_data.get("query", "")
                        logger.info(f"LLM requested web search for: {query}")
                        search_result = search_web(query)

                        summary_prompt = [
                            {"role": "system", "content": "Summarize this search result in one short, conversational sentence as BMO. Do not use markdown."},
                            {"role": "user", "content": f"RESULT: {search_result}\\nUser Question: {user_text}"}
                        ]
                        content = self._call_llm_once(summary_prompt, FAST_LLM_MODEL, temperature=0.4, max_tokens=120)
            except json.JSONDecodeError:
                pass

            pronounce_match = re.search(r'!PRONOUNCE:\s*([a-zA-Z0-9_-]+)\s*=\s*([a-zA-Z0-9_-]+)', content, re.IGNORECASE)
            if pronounce_match:
                word = pronounce_match.group(1).strip()
                phonetic = pronounce_match.group(2).strip()
                logger.info(f"Learned new pronunciation from LLM: {word} -> {phonetic}")
                add_pronunciation(word, phonetic)
                content = re.sub(r'!PRONOUNCE:.*', '', content, flags=re.IGNORECASE).strip()

            content = _strip_prompt_leakage(content)
            content = re.sub(r'\bBeemo\b', 'BMO', content, flags=re.IGNORECASE)

            if not content.strip():
                content = "BMO is here! How can I help?"

            self.history.append({"role": "assistant", "content": content})

            if search_injected:
                for msg in reversed(self.history):
                    if msg.get("role") == "user" and msg.get("content", "").startswith("[LIVE DATA:"):
                        msg["content"] = user_text
                        break

            self._trim_history()
            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"Connection Error: {e}")
            return "Could not connect to my brain. Is the LLM server running?"
        except Exception as e:
            logger.error(f"Brain Exception: {e}")
            return "I'm having trouble thinking right now."

    def get_history(self):
        return self.history

    def stream_think(self, user_text: str):
        """
        Send text to local LLM and yield full sentences as they are generated.
        Useful for TTS chunking (speaking while generating).
        """
        self.history.append({"role": "user", "content": user_text})

        lower_text = user_text.lower()

        # Pre-LLM camera check: if user asks to take a photo / look at something,
        # emit the action JSON directly without calling the LLM.
        # This is more reliable than hoping the small model emits the right JSON.
        camera_keywords = [
            "take a photo", "take a picture", "take photo", "take picture",
            "look at", "what do you see", "what can you see", "use your camera",
            "photograph", "snap a photo",
        ]
        if any(kw in lower_text for kw in camera_keywords):
            action = '{"action": "take_photo"}'
            self.history.append({"role": "assistant", "content": action})
            yield action
            return

        # Pre-LLM display_image check
        if any(kw in lower_text for kw in _DISPLAY_IMAGE_KEYWORDS):
            action = _build_display_image_action(user_text)
            self.history.append({"role": "assistant", "content": action})
            yield action
            return

        # Pre-LLM keyword check: if the question likely needs real-time info,
        # do the web search now rather than relying on the model to emit JSON.
        # Require at least one realtime keyword AND the text to look like a question
        # (contains 'what', 'who', 'when', 'find', 'search', '?', etc.) to avoid
        # false triggers on casual phrases like 'how are you doing today'.
        realtime_keywords = [
            "weather", "forecast", "temperature", "tonight", "tomorrow",
            "news", "latest", "right now", "score", "stocks", "bitcoin",
            "crypto", "price of", "happening", "recently", "live",
        ]
        question_markers = [
            "what", "who", "when", "where", "find", "search", "tell me",
            "look up", "check", "is there", "did", "?",
        ]
        has_realtime_kw = any(kw in lower_text for kw in realtime_keywords)
        has_question = any(q in lower_text for q in question_markers)
        needs_search = has_realtime_kw and has_question
        search_injected = False
        if needs_search:
            try:
                search_result = search_web(user_text)
                # Only inject if we got a real result (not empty/error sentinel)
                if search_result and search_result not in ("SEARCH_EMPTY", "SEARCH_ERROR") and len(search_result) > 50:
                    # Strip verbose "SEARCH RESULTS for '...':" prefix from search.py
                    clean_result = re.sub(r"^SEARCH RESULTS for '.*?':\n?", "", search_result).strip()
                    self.history[-1]["content"] = (
                        f"[LIVE DATA: {clean_result}] "
                        f"Using only the above live data, answer in one or two sentences as BMO: {user_text}"
                    )
                    search_injected = True
            except Exception as e:
                logger.warning(f"Pre-LLM web search failed: {e}")

        # Simple heuristic to route to a faster model for simple chat
        chosen_model = self._choose_model(user_text)

        full_content = ""
        buffer = ""

        try:
            provider = self._provider()
            logger.info(f"Stream request to LLM ({provider}/{chosen_model})")

            if provider == "gemini":
                full_content = self._call_llm_once(self.history, chosen_model, temperature=0.4, max_tokens=120)
                for chunk in _split_into_speak_chunks(full_content):
                    cleaned = _strip_prompt_leakage(chunk)
                    out_chunk = re.sub(r'\bBeemo\b', 'BMO', cleaned, flags=re.IGNORECASE)
                    if out_chunk.strip():
                        yield out_chunk
            else:
                payload = {
                    "model": chosen_model,
                    "messages": self.history,
                    "stream": True,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 120,
                    }
                }

                with requests.post(LLM_URL, json=payload, stream=True, timeout=180) as response:
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    chunk = data.get("message", {}).get("content", "")
                                    if not chunk:
                                        continue

                                    chunk = chunk.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
                                    buffer += chunk
                                    full_content += chunk

                                    if any(buffer.endswith(punc) for punc in ['.', '!', '?', '\n']) or "\n\n" in buffer:
                                        cleaned = _strip_prompt_leakage(buffer)
                                        out_chunk = re.sub(r'\bBeemo\b', 'BMO', cleaned, flags=re.IGNORECASE)
                                        if out_chunk.strip():
                                            yield out_chunk
                                        buffer = ""
                                except json.JSONDecodeError:
                                    pass

                        if buffer.strip():
                            cleaned = _strip_prompt_leakage(buffer)
                            out_chunk = re.sub(r'\bBeemo\b', 'BMO', cleaned, flags=re.IGNORECASE)
                            if out_chunk.strip():
                                yield out_chunk

                        json_match = re.search(r'\{.*?\}', full_content, re.DOTALL)
                        if json_match and "action" in json_match.group(0):
                            pass
                    else:
                        logger.error(f"LLM Stream Error: {response.status_code} - {response.text}")
                        yield "I'm having trouble thinking."
                        return

            self.history.append({"role": "assistant", "content": full_content})

            if search_injected:
                for msg in reversed(self.history):
                    if msg.get("role") == "user" and msg.get("content", "").startswith("[LIVE DATA:"):
                        msg["content"] = user_text
                        break

            self._trim_history()

        except requests.exceptions.RequestException as e:
            logger.error(f"Connection Error: {e}")
            yield "Could not connect to my brain."
        except Exception as e:
            logger.error(f"Brain Exception: {e}")
            yield "I'm having trouble right now."

    def set_history(self, new_history):
        # Ensure system prompt is always present and up to date
        if not new_history or new_history[0].get("role") != "system":
            new_history.insert(0, {"role": "system", "content": get_system_prompt()})
        else:
            new_history[0]["content"] = get_system_prompt()
        self.history = new_history

    def analyze_image(self, image_base64: str, user_text: str) -> str:
        """
        Analyse an image using the Hailo VLM (Qwen2-VL-2B) running directly
        on the NPU via the HailoRT Python API.  Falls back to a polite error
        message if the HEF isn't available or the hardware can't be reached.
        """
        # Strip data URI prefix if present (browser sends "data:image/jpeg;base64,...")
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # We don't append the image to the main history to save context window,
        # but we do append the user's question and the assistant's answer.
        self.history.append({"role": "user", "content": user_text})

        try:
            vlm, frame_shape, frame_dtype = _get_vlm()

            # Decode the base64 image into a numpy frame the VLM expects
            frame = _decode_image_to_frame(image_base64, frame_shape, frame_dtype)

            # Build the structured prompt expected by the Qwen2-VL model
            prompt = [
                {"role": "system", "content": [
                    {"type": "text", "text": "You are BMO, a helpful robot assistant. Describe what you see concisely and conversationally in English."}
                ]},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text or "What do you see in this image?"}
                ]}
            ]

            logger.info("Running VLM inference on Hailo NPU ...")
            vlm.clear_context()
            content = vlm.generate_all(
                prompt=prompt,
                frames=[frame],
                max_generated_tokens=150,
                temperature=0.4,
            )

            # Clean up any smart quotes, stop tokens, or stray formatting
            content = content.replace('\u201c', '"').replace('\u201d', '"')
            content = content.replace('\u2018', "'").replace('\u2019', "'")
            for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
                content = content.replace(tok, "")
            content = content.strip()

            logger.info(f"VLM response ({len(content)} chars): {content[:120]}...")

            self.history.append({"role": "assistant", "content": content})
            return content

        except FileNotFoundError as e:
            logger.warning(f"VLM HEF not found: {e}")
            return "BMO's vision model isn't installed yet. Run setup.sh to download it!"
        except Exception as e:
            logger.error(f"VLM Exception: {e}", exc_info=True)
            return "I tried to look, but my eyes aren't working right now."
