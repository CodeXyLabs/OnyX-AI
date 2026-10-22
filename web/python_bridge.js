/* ============================================================================
 * OnyX · Python brain bridge
 * ============================================================================
 * Связывает интерфейс с локальным Python-сервером (python_core/server.py).
 *
 * Если сервер недоступен (страница открыта с file:// и сервер не запущен),
 * весь интерфейс продолжает работать на встроенном JS-движке. Если сервер
 * доступен — ответы, изображения и видео генерирует Python-мозг, что
 * заметно умнее и быстрее обучается.
 *
 * API:
 *   OnyXPython.available      — bool (стало известно после проверки)
 *   OnyXPython.chat(q, opts)  — Promise<ответ> с fallback на JS-движок
 *   OnyXPython.image(prompt)  — Promise<{url, watermarked}>
 *   OnyXPython.video(prompt)  — Promise<{url, ext, watermarked}>
 *   OnyXPython.learn(q, a)    — асинхронное дообучение
 *   OnyXPython.status()       — Promise<статус>
 * ==========================================================================*/

function OnyXPythonBridgeFactory(global) {
'use strict';

var BASE = 'http://127.0.0.1:8787';
var TIMEOUT = 3000;
var available = null;   // true — ядро ответило; null — ещё не готово (пробуем снова)

function fetchJson(url, opts, timeoutMs) {
  if (typeof fetch !== 'function') return Promise.reject(new Error('fetch unavailable'));
  var controller = null, timer = null;
  try {
    if (typeof AbortController !== 'undefined') {
      controller = new AbortController();
      timer = setTimeout(function(){ controller.abort(); }, timeoutMs || TIMEOUT);
    }
    return fetch(url, Object.assign({}, opts, controller ? { signal: controller.signal } : {}))
      .then(function(res){
        if (timer) clearTimeout(timer);
        if (!res.ok) {
          var err = new Error('HTTP ' + res.status);
          err.status = res.status;
          throw err;
        }
        return res.json();
      })
      .catch(function(e){ if (timer) clearTimeout(timer); throw e; });
  } catch (e) {
    return Promise.reject(e);
  }
}

/* Один запрос к /api/status. Возвращает:
 *   {reachable, ready, loading, error, st}
 * Недоступность НЕ кэшируется: сервер может подняться позже. */
function probe() {
  return fetchJson(BASE + '/api/status', { method: 'GET' }, 3000)
    .then(function(st){
      if (st && st.ready) { available = true; return { reachable: true, ready: true, loading: false, error: null, st: st }; }
      available = null;
      return { reachable: true, ready: false, loading: !!(st && st.loading), error: st && st.error ? st.error : null, st: st || null };
    })
    .catch(function(){
      available = null;
      return { reachable: false, ready: false, loading: false, error: null, st: null };
    });
}

function check(force) {
  if (!force && available === true) return Promise.resolve(true);
  return probe().then(function(p){ return p.ready; });
}

/* Ждём готовности ядра. Модель грузится ~40 секунд, поэтому проверяем
 * статус повторно, вместо того чтобы навсегда запомнить «недоступно». */
function waitReady(timeoutMs) {
  var deadline = Date.now() + (timeoutMs || 120000);
  return new Promise(function(resolve){
    (function attempt(){
      probe().then(function(p){
        if (p.ready) return resolve(true);
        if (p.error) return resolve(false);          // ядро не поднимется — не ждём зря
        if (Date.now() >= deadline) return resolve(false);
        setTimeout(attempt, 1500);
      });
    })();
  });
}

/* POST с повтором: пока сервер отвечает 503 «модель грузится» или ещё не
 * принимает соединения — пробуем снова до дедлайна. */
function postJson(path, payload, timeoutMs, waitMs) {
  var deadline = Date.now() + (waitMs || 120000);
  return new Promise(function(resolve){
    (function attempt(){
      fetchJson(BASE + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }, timeoutMs).then(function(r){
        available = true;
        resolve(r);
      }).catch(function(e){
        var loading = e && e.status === 503;
        var offline = !e || !e.status;
        if ((loading || offline) && Date.now() < deadline) {
          setTimeout(attempt, loading ? 1500 : 2000);
          return;
        }
        resolve(null);
      });
    })();
  });
}

/* ------------------------------------------------------------------ chat -- */

/* Отвечает через Python-сервер. Если сервера нет вовсе — null (интерфейс
 * переходит на JS-движок). Если сервер запущен и грузит модель — ждём. */
function chat(question, opts) {
  opts = opts || {};
  return probe().then(function(p){
    if (!p.reachable || p.error) return null;      // ядро не запущено — сразу JS
    var ready = p.ready ? Promise.resolve(true) : waitReady(opts.waitMs || 120000);
    return ready.then(function(ok){
      if (!ok) return null;
      return postJson('/api/chat', {
        question: question,
        temperature: opts.temperature != null ? opts.temperature : 0.8,
        max_tokens: opts.max_tokens || 220,
        intent: opts.intent || 'default',
        use_web: opts.use_web !== false
      }, 90000).then(function(r){
        if (!r) return null;
        r.__python = true;
        return r;
      });
    });
  });
}

/* ----------------------------------------------------------------- image -- */

function image(prompt, opts) {
  opts = opts || {};
  return probe().then(function(p){
    if (!p.reachable || p.error) return null;
    var ready = p.ready ? Promise.resolve(true) : waitReady(opts.waitMs || 60000);
    return ready.then(function(ok){
      if (!ok) return null;
      return postJson('/api/image', {
        prompt: prompt,
        size: opts.size || 768,
        online: !!opts.online,
        style: opts.style || null
      }, 90000);
    });
  });
}

/* ----------------------------------------------------------------- video -- */

function video(prompt, opts) {
  opts = opts || {};
  return probe().then(function(p){
    if (!p.reachable || p.error) return null;
    var ready = p.ready ? Promise.resolve(true) : waitReady(opts.waitMs || 90000);
    return ready.then(function(ok){
      if (!ok) return null;
      return postJson('/api/video', {
        prompt: prompt,
        seconds: opts.seconds || 5,
        width: opts.width || 800,
        height: opts.height || 450,
        online: !!opts.online
      }, 120000);
    });
  });
}

/* ----------------------------------------------------------------- learn -- */

function learn(question, answer) {
  check().then(function(ok){
    if (!ok) return;
    fetchJson(BASE + '/api/learn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question, answer: answer })
    }, 30000).catch(function(){});
  });
}

function status() {
  return probe().then(function(p){
    if (p.st) return p.st;                       // включает ready/loading
    if (!p.reachable) return null;
    return fetchJson(BASE + '/api/status', { method: 'GET' }, 10000).catch(function(){ return null; });
  });
}

var OnyXPython = {
  BASE: BASE,
  check: check,
  probe: probe,
  waitReady: waitReady,
  chat: chat,
  image: image,
  video: video,
  learn: learn,
  status: status,
  get available(){ return available === true; }
};

global.OnyXPython = OnyXPython;
try { if (typeof module !== 'undefined' && module.exports) module.exports = OnyXPython; } catch (e) { }
return OnyXPython;
}

if (typeof window !== 'undefined') OnyXPythonBridgeFactory(window);