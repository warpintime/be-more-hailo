# Be More Agent — Hailo-10H Edition

<p align="center">
  <img src="bmo_irl.jpg" height="300" alt="BMO On-Device" />
  <img src="bmo-web.png" height="300" alt="BMO Web Interface" />
</p>

A fork of [@brenpoly's be-more-agent](https://github.com/brenpoly/be-more-agent) project, built to run fully on-device on a **Raspberry Pi 5** with the **Raspberry Pi AI HAT 2+** (Hailo-10H). BMO listens for its wake word, understands what you say, thinks about it locally, and talks back — no cloud, no subscriptions, no data leaving your house.

This fork adds a browser-based **web interface**, a shared `core/` module layer used by both interfaces, and updated support for the Hailo NPU hardware.

---

## What runs where

| Component | Where it runs | Notes |
|-----------|--------------|-------|
| LLM (`qwen2.5-instruct:1.5b`) | Hailo-10H NPU | via `hailo-ollama` |
| Vision (`qwen2-vl-instruct:2b`) | Hailo-10H NPU | optional; requires camera |
| STT (Whisper base.en) | CPU | via `whisper.cpp`; NPU path causes PCIe timeouts |
| TTS (Piper) | CPU | streams sentence-by-sentence while LLM generates |
| Wake word (openWakeWord) | CPU | "Hey BMO" custom model |

STT runs on the CPU by design. Pushing 16kHz audio arrays through the Hailo PCIe bus caused consistent 15+ second timeouts during development. `whisper.cpp` on the quad-core ARM is fast enough and keeps the NPU free for inference.

---

## Interfaces

### On-Device (`agent_hailo.py`)
BMO in its natural habitat. Plug in a screen, a USB mic, and a USB speaker and you get the full experience: animated faces, wake word detection, and the whole listen → think → speak loop running locally. After each response, BMO stays in "Still listening..." mode for 8 seconds so you can keep a conversation going without re-saying the wake word every time.

### Web (`web_app.py`)
A FastAPI server with a browser-based UI — useful if you want to talk to BMO from another room, or you'd rather not have a screen hanging off your Pi. Hold a button to record, and BMO responds with audio in your browser.

The web interface includes:
- **Debug panel** — conversation history and live server logs
- **Pronunciation override** — corrects how Piper pronounces specific words
- **LLM status indicator** — shows whether the NPU model is ready
- **Hands-free mode** — enables wake word detection so you don't need to hold the button
- **Pi Audio toggle** — routes audio to the Pi's physical speaker instead of browser playback

---

## Secure remote access

Modern browsers require HTTPS for microphone access, which makes things awkward when your Pi is just sitting on your local network. [Tailscale](https://tailscale.com/) solves this elegantly — install it on your Pi and your other devices, enable HTTPS certificates, and you get a proper `*.ts.net` address with a real cert, reachable from anywhere on your Tailnet. No port forwarding, no dynamic DNS nonsense.

> **Disclosure:** I work at Tailscale. That said, I genuinely use it for this project and it's the best solution I've found for exactly this problem.

1. Install Tailscale on the Pi and your client device
2. Enable [HTTPS certificates](https://tailscale.com/kb/1153/enabling-https/) in the Tailscale admin console
3. On the Pi, run:
   ```bash
   tailscale serve --bg --https=443 localhost:8080
   ```
4. Access the web UI at `https://<your-pi-hostname>.ts.net`

Your BMO is then reachable from your phone, laptop, or any device on your Tailnet — mic access works, and it's not exposed to the open internet.

---

## Hardware

- Raspberry Pi 5 (4GB or 8GB recommended)
- Raspberry Pi AI HAT 2+ (Hailo-10H, required for NPU features)
- USB microphone and speaker (for on-device mode)
- HDMI or DSI display (for on-device GUI)
- Raspberry Pi Camera Module (optional, for vision/photo features)

---

## Project structure

```
be-more-agent/
├── agent_hailo.py          # On-device GUI application
├── web_app.py              # FastAPI web server
├── core/
│   ├── config.py           # All configuration (models, devices, paths, system prompt)
│   ├── llm.py              # LLM inference, web search, conversation history
│   ├── tts.py              # Text-to-speech via Piper
│   └── stt.py              # Speech-to-text via whisper.cpp
├── templates/              # Jinja2 HTML templates for the web UI
├── static/                 # CSS, JS, favicon
├── setup.sh                # Automated installation script
├── setup_services.sh       # Installs systemd background services
├── start_web.sh            # Starts the web server
├── start_agent.sh          # Starts the on-device GUI
├── requirements.txt        # Python dependencies
├── wakeword.onnx           # OpenWakeWord model
├── piper/                  # Piper TTS engine and voice model
├── models/                 # Whisper model weights
├── whisper.cpp/            # Compiled whisper.cpp STT binary
└── faces/ sounds/          # GUI assets (swap these to customise BMO's look and feel)
```

---

## Installation

### Prerequisites

- Raspberry Pi OS (64-bit, current stable)
- `hailo-ollama` installed and running — follow [Hailo's documentation](https://github.com/hailo-ai/hailo-ollama) for setup

### Automated install

```bash
curl -sSL https://raw.githubusercontent.com/moorew/be-more-hailo/main/setup.sh | bash
cd be-more-agent
```

The script handles everything:
- Installs system packages including `libcamera-apps` for camera support
- Downloads and extracts the Piper TTS engine
- Clones and compiles `whisper.cpp`
- Downloads the `ggml-base.en` Whisper model
- Creates a Python virtual environment and installs dependencies
- Pulls `qwen2.5-instruct:1.5b` (LLM) and `qwen2-vl-instruct:2b` (vision) via `hailo-ollama`
- Checks camera availability and lets you know if anything's missing

### Manual install

```bash
git clone https://github.com/moorew/be-more-hailo.git be-more-agent
cd be-more-agent
chmod +x *.sh
./setup.sh
```

---

## Running

**Web server:**
```bash
source venv/bin/activate
./start_web.sh
```
Open `http://<YOUR_PI_IP>:8080` in a browser (or your Tailscale HTTPS address for microphone access).

**On-device GUI:**
```bash
source venv/bin/activate
./start_agent.sh
```

**Auto-start on boot:**
```bash
./setup_services.sh
```
Then manage with `sudo systemctl start|stop|restart bmo-web` or `bmo-ollama`.

---

## Configuration

All settings live in `core/config.py`. The most commonly changed values:

```python
# LLM models (must be pulled via hailo-ollama)
LLM_MODEL       = "qwen2.5-instruct:1.5b"
FAST_LLM_MODEL  = "qwen2.5-instruct:1.5b"
VISION_MODEL    = "qwen2-vl-instruct:2b"

# Audio device for local hardware playback (run `aplay -l` to find yours)
# The USB speaker is typically on a different ALSA card from the mic — check both.
ALSA_DEVICE = "plughw:UACDemoV10,0"

# Microphone device index (run `python3 -c "import sounddevice as sd; print(sd.query_devices())"`)
MIC_DEVICE_INDEX = 1
MIC_SAMPLE_RATE  = 48000

# STT binary and model
WHISPER_CMD   = "./whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "./models/ggml-base.en.bin"
```

Environment variables override any of these at runtime:
```bash
export ALSA_DEVICE="plughw:2,0"
```

---

## Dual-model routing

By default, all queries go to a single model (`qwen2.5-instruct:1.5b`). If you want to route longer or more complex queries to a larger model:

1. Pull the larger model via `hailo-ollama`
2. Set `LLM_MODEL` to the larger model name in `core/config.py`
3. Keep `FAST_LLM_MODEL` pointing to `qwen2.5-instruct:1.5b`

Short, simple prompts (under 15 words, no complex keywords) stay on the fast model. Longer or more complex ones go to `LLM_MODEL`. Note that swapping models on the Hailo-10H takes a few seconds on the first query after a switch.

---

## Camera and vision

If you have a Raspberry Pi Camera Module connected:

1. Enable the camera interface in `raspi-config`
2. Install camera tools if not already present:
   ```bash
   sudo apt install -y libcamera-apps
   ```
3. Say something like "Hey BMO, take a photo and tell me what you see" — the agent captures a frame with `rpicam-still` and sends it to the vision model (`qwen2-vl-instruct:2b`) on the NPU

If no camera is found, BMO will say so rather than crashing.

---

## Customisation

BMO is pretty easy to make your own:

**Personality:** Edit `get_system_prompt()` in `core/config.py`. This is where BMO's voice, tone, and quirks are defined.

**Faces:** Drop PNG sequences into `faces/<state>/`. The GUI loops through all images in each folder for each state (idle, listening, thinking, speaking).

**Sounds:** Put `.wav` files in `sounds/<category>/`. BMO picks one at random per event.

**Wake word:** Replace `wakeword.onnx` with any [OpenWakeWord](https://github.com/dscripka/openWakeWord)-compatible model.

---

## Credits

The original project is entirely the work of [@brenpoly](https://github.com/brenpoly/be-more-agent) — the concept, the character, and the original implementation. This fork adds Hailo NPU support, the web interface, dual-interface `core/` modules, and various fixes and improvements.

**"BMO"** and **"Adventure Time"** are trademarks of Cartoon Network (Warner Bros. Discovery). This is a fan project for personal and educational use only, not affiliated with or endorsed by Cartoon Network.

---

## License

MIT — see [LICENSE](LICENSE).
