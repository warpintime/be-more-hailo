const SHELL_CACHE = 'bmo-shell-v1';
const RUNTIME_CACHE = 'bmo-runtime-v1';
const SHELL_URLS = [
  '/',
  '/favicon.png',
  '/apple-touch-icon.png',
  '/manifest.webmanifest'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL_URLS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== SHELL_CACHE && key !== RUNTIME_CACHE)
          .map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') {
    return;
  }

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) {
    return;
  }

  if (url.pathname.startsWith('/api/')) {
    event.respondWith(fetch(request));
    return;
  }

  const cacheFirst =
    url.pathname === '/' ||
    url.pathname.startsWith('/static/') ||
    url.pathname.startsWith('/faces/') ||
    url.pathname.startsWith('/sounds/') ||
    url.pathname === '/favicon.png' ||
    url.pathname === '/apple-touch-icon.png' ||
    url.pathname === '/manifest.webmanifest';

  if (cacheFirst) {
    event.respondWith(
      caches.match(request).then((cached) => {
        const networkFetch = fetch(request)
          .then((response) => {
            if (response.ok) {
              const responseClone = response.clone();
              caches.open(RUNTIME_CACHE).then((cache) => cache.put(request, responseClone));
            }
            return response;
          })
          .catch(() => cached);

        return cached || networkFetch;
      })
    );
    return;
  }

  event.respondWith(
    fetch(request).catch(() => caches.match(request).then((cached) => cached || caches.match('/')))
  );
});
