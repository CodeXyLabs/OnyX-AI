/* ============================================================================
 * OnyX · assembled-page verification
 * ============================================================================
 * Boots the built index.html in a simulated browser: a minimal DOM, a Worker
 * stand-in that really executes the engine source, and stub media APIs. Then it
 * exercises everything the page promises — training, the confidence gate, quiet
 * retrieval, watermarking, and the startup splash.
 * ==========================================================================*/
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');

let failures = 0;
function check(name, ok, detail) {
  console.log((ok ? '  PASS  ' : '  FAIL  ') + name + (detail ? '   ' + detail : ''));
  if (!ok) failures++;
}

/* ---------------------------------------------------- pull the inline blocks */

function extractScripts(source) {
  const out = [];
  const re = /<script([^>]*)>([\s\S]*?)<\/script>/g;
  let m;
  while ((m = re.exec(source)) !== null) out.push({ attrs: m[1], body: m[2] });
  return out;
}

const scripts = extractScripts(html);
const engineBlock = scripts.find(s => /id="onyx-llm-engine-source"/.test(s.attrs));
const plain = scripts.filter(s => !/id="onyx-llm-engine-source"/.test(s.attrs) && !/src=/.test(s.attrs));

console.log('page size      : ' + (html.length / 1024).toFixed(1) + ' KB');
console.log('script blocks  : ' + scripts.length);
check('engine source is inline', !!engineBlock, engineBlock ? (engineBlock.body.length / 1024).toFixed(1) + ' KB' : 'missing');
if (!engineBlock) process.exit(1);

/* The asset block is tiny, the six modules are short; the long one is the app. */
const allModules = plain.filter(b => b.body.length < 80000);
const appBlock = plain.find(b => b.body.length >= 80000);
console.log('module blocks  : ' + allModules.length + ' (' + allModules.map(b => (b.body.length / 1024).toFixed(1)).join(', ') + ' KB)');
check('the site application script is intact', !!appBlock, appBlock ? (appBlock.body.length / 1024).toFixed(1) + ' KB' : 'missing');

/* --------------------------------------------------------------- fake DOM */

function canvasContext() {
  const noop = () => { };
  return {
    save: noop, restore: noop, beginPath: noop, closePath: noop, moveTo: noop, lineTo: noop,
    quadraticCurveTo: noop, arc: noop, fill: noop, stroke: noop, fillRect: noop, strokeRect: noop,
    clearRect: noop, drawImage: noop, fillText: noop, setTransform: noop, scale: noop,
    createLinearGradient: () => ({ addColorStop: noop }),
    createRadialGradient: () => ({ addColorStop: noop }),
    globalAlpha: 1, globalCompositeOperation: 'source-over', fillStyle: '', strokeStyle: '',
    lineWidth: 1, shadowColor: '', shadowBlur: 0, font: ''
  };
}

class El {
  constructor(tag) {
    this.tagName = (tag || 'div').toUpperCase();
    this.children = []; this.parentNode = null;
    this.style = {}; this.dataset = {}; this._text = ''; this._html = '';
    this.width = 300; this.height = 150;
    this.classList = {
      _s: new Set(),
      add: (...c) => c.forEach(x => this.classList._s.add(x)),
      remove: (...c) => c.forEach(x => this.classList._s.delete(x)),
      toggle: (c, on) => {
        if (on === undefined) { this.classList._s.has(c) ? this.classList._s.delete(c) : this.classList._s.add(c); }
        else if (on) this.classList._s.add(c); else this.classList._s.delete(c);
      },
      contains: (c) => this.classList._s.has(c)
    };
    this.attributes = {}; this.listeners = {};
  }
  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v); }
  get innerHTML() { return this._html; }
  set innerHTML(v) { this._html = String(v); }
  get innerText() { return this._text; }
  setAttribute(k, v) { this.attributes[k] = v; }
  getAttribute(k) { return this.attributes[k] === undefined ? null : this.attributes[k]; }
  removeAttribute(k) { delete this.attributes[k]; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  insertBefore(c) { c.parentNode = this.parentNode; this.children.push(c); return c; }
  removeChild(c) { this.children = this.children.filter(x => x !== c); return c; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); }
  removeEventListener() { }
  focus() { }
  click() { (this.listeners.click || []).forEach(f => f({ target: this })); }
  closest() { return null; }
  contains() { return false; }
  get firstChild() { return this.children[0] || null; }
  get nextSibling() { return null; }
  getContext() { return canvasContext(); }
  toDataURL() { return 'data:image/png;base64,AAAA'; }
  captureStream() { return { getTracks: () => [{ stop() { } }] }; }
  play() { return Promise.resolve(); }
  pause() { }
  load() { }
}

const registry = new Map();
const IDS = ['brainPanel', 'llmPanel', 'llmTileStats', 'llmBody', 'llmStatus', 'llmModal',
  'bNodes', 'bEdges', 'bRules', 'lmState', 'lmParams', 'lmSteps', 'lmLoss', 'lmCorpus', 'lmVocab',
  'settingsModal', 'settingsClose', 'llmClose', 'llmTrain30', 'llmTrain120', 'llmSave',
  'llmExportW', 'llmExportC', 'llmReset', 'stileLlm', 'input', 'send', 'chat', 'history',
  'toastBox', 'welcome', 'paramsModal', 'versionModal', 'accountModal', 'aboutModal',
  'lightbox', 'sidebar', 'sidebarBackdrop'];
IDS.forEach(id => registry.set(id, new El('div')));
registry.get('brainPanel').parentNode = new El('aside');

const document = {
  readyState: 'complete',
  hidden: false,
  body: new El('body'),
  head: new El('head'),
  createElement: (t) => new El(t),
  createDocumentFragment: () => new El('fragment'),
  getElementById: (id) => {
    const direct = registry.get(id);
    if (direct) return direct;
    for (const el of registry.values()) {
      if (el._html && el._html.indexOf('id="' + id + '"') !== -1) {
        const made = new El('b'); registry.set(id, made); return made;
      }
    }
    return null;
  },
  querySelector: (sel) => (sel && sel[0] === '#' ? document.getElementById(sel.slice(1)) : null),
  querySelectorAll: () => [],
  addEventListener: () => { },
  removeEventListener: () => { }
};

/* ---------------------------------------------------------- mock Worker --- */

class MockWorker {
  constructor() { this.onmessage = null; this.onerror = null; this._ctx = null; }
  _ensure() {
    if (this._ctx) return this._ctx;
    const sandbox = {
      console,
      setTimeout, clearTimeout, setInterval, clearInterval,
      performance: { now: () => Date.now() },
      indexedDB: undefined,
      Float32Array, Int8Array, Int32Array, Uint8Array, Float64Array,
      Map, Set, Promise, Math, Date, JSON, Object, Array, Number, String, Error, RegExp,
      isFinite, isNaN, parseInt, parseFloat, encodeURIComponent, decodeURIComponent,
      WorkerGlobalScope: function WorkerGlobalScope() { }
    };
    sandbox.globalThis = sandbox;
    sandbox.self = sandbox;
    vm.createContext(sandbox);
    vm.runInContext('self.__proto__ = WorkerGlobalScope.prototype;', sandbox);
    vm.runInContext(engineBlock.body, sandbox, { filename: 'engine.js' });
    this._ctx = sandbox;
    return sandbox;
  }
  postMessage(msg) {
    const sb = this._ensure();
    setTimeout(() => {
      try {
        vm.runInContext('(function(m){ if (self.onmessage) self.onmessage({data:m}); })(' +
          JSON.stringify(msg) + ')', sb);
      } catch (e) { if (this.onerror) this.onerror({ message: String(e && e.message || e) }); }
    }, 0);
  }
  terminate() { }
}

class MockBlob {
  constructor(parts, opts) {
    this.parts = parts;
    this.type = (opts && opts.type) || '';
    this.size = parts.join('').length;
  }
}
class MockMediaRecorder {
  constructor(stream, opts) {
    this.stream = stream;
    this.mimeType = (opts && opts.mimeType) || 'video/webm';
    this.state = 'inactive';
  }
  static isTypeSupported(t) { return /webm|mp4/.test(t); }
  start() { this.state = 'recording'; }
  stop() { this.state = 'inactive'; if (this.onstop) setTimeout(() => this.onstop(), 0); }
}

/* -------------------------------------------------------- sandbox the page */

const sandbox = {
  document, console,
  setTimeout, clearTimeout, setInterval, clearInterval,
  Blob: MockBlob,
  URL: { createObjectURL: () => 'blob:mock', revokeObjectURL: () => { } },
  Worker: MockWorker,
  MediaRecorder: MockMediaRecorder,
  HTMLCanvasElement: { prototype: { captureStream: () => ({ getTracks: () => [{ stop() { } }] }) } },
  Image: function () { const self = this; self.width = 256; self.height = 128; setTimeout(() => { if (self.onload) self.onload(); }, 0); },
  FileReader: function () {
    this.readAsDataURL = () => {
      const s = this;
      setTimeout(() => { s.result = 'data:image/png;base64,AAAA'; if (s.onload) s.onload(); }, 0);
    };
  },
  AbortController: function () { this.signal = {}; this.abort = () => { }; },
  localStorage: {
    _d: {},
    getItem(k) { return this._d[k] === undefined ? null : this._d[k]; },
    setItem(k, v) { this._d[k] = String(v); },
    removeItem(k) { delete this._d[k]; }
  },
  performance: { now: () => Date.now() },
  navigator: { userAgent: 'node' },
  requestAnimationFrame: (fn) => setTimeout(() => fn(Date.now()), 16),
  cancelAnimationFrame: (id) => clearTimeout(id),
  matchMedia: () => ({ matches: false }),
  location: { href: 'file:///index.html', protocol: 'file:' },
  fetch: () => Promise.resolve({
    ok: false, status: 404,
    json: () => Promise.resolve(null), text: () => Promise.resolve('')
  }),
  indexedDB: undefined,
  confirm: () => true,
  alert: () => { },
  open: () => null,
  Float32Array, Int8Array, Int32Array, Uint8Array, Float64Array,
  Map, Set, Promise, Math, Date, JSON, Object, Array, Number, String, Error, RegExp,
  isFinite, isNaN, parseInt, parseFloat, encodeURIComponent, decodeURIComponent
};
sandbox.globalThis = sandbox;
sandbox.self = sandbox;
sandbox.window = sandbox;
sandbox.addEventListener = function () { };
sandbox.removeEventListener = function () { };
vm.createContext(sandbox);

/* --------------------------------------------------------------- run them */

async function main() {
  for (const mod of allModules) {
    const name = (mod.body.match(/function (OnyX\w+Factory)/) || [])[1] ||
      (/ONYX_WATERMARK_DATA_URI/.test(mod.body) ? 'assets' : 'module');
    try {
      vm.runInContext(mod.body, sandbox, { filename: name + '.js' });
      check('module ' + name + ' loads', true, (mod.body.length / 1024).toFixed(1) + ' KB');
    } catch (e) {
      check('module ' + name + ' loads', false, e.message);
    }
  }

  check('engine namespace is published', typeof sandbox.OnyXLLM === 'object');
  check('corpus generates text', !!sandbox.OnyXCorpus && sandbox.OnyXCorpus.build().length > 100000,
    sandbox.OnyXCorpus ? (sandbox.OnyXCorpus.build().length / 1024).toFixed(0) + ' KB' : 'missing');
  check('search helpers are exposed', !!sandbox.OnyXWeb && typeof sandbox.OnyXWeb.shouldSearch === 'function');
  check('image service is exposed', !!sandbox.OnyXImage && typeof sandbox.OnyXImage.generate === 'function');
  check('video service is exposed', !!sandbox.OnyXVideo && typeof sandbox.OnyXVideo.generate === 'function');
  check('splash is exposed', !!sandbox.OnyXSplash && typeof sandbox.OnyXSplash.show === 'function');
  check('watermark data URI is embedded',
    typeof sandbox.ONYX_WATERMARK_DATA_URI === 'string' &&
    sandbox.ONYX_WATERMARK_DATA_URI.indexOf('data:image/png;base64,') === 0,
    (String(sandbox.ONYX_WATERMARK_DATA_URI).length / 1024).toFixed(0) + ' KB');
  check('video mark data URI is embedded',
    typeof sandbox.ONYX_MARK_DATA_URI === 'string' &&
    sandbox.ONYX_MARK_DATA_URI.indexOf('data:image/png;base64,') === 0,
    (String(sandbox.ONYX_MARK_DATA_URI).length / 1024).toFixed(0) + ' KB');

  const caps = sandbox.OnyXLLM.MODEL_CAPABILITIES || {};
  check('V0.5 is registered with video', !!caps['0.5'] && caps['0.5'].video === true,
    Object.keys(caps).join(', '));
  check('earlier models do not offer video', !caps['4.3'].video);

  /* Boot the bridge. */
  const B = sandbox.OnyXLLM.bridge;
  B.seedText = sandbox.OnyXCorpus.build();
  B.start();
  check('worker was created', !!B.worker);

  const t0 = Date.now();
  const booted = await new Promise((resolve) => {
    const iv = setInterval(() => {
      if (B.ready || B.failed) { clearInterval(iv); resolve(!B.failed); }
      else if (Date.now() - t0 > 240000) { clearInterval(iv); resolve(false); }
    }, 100);
  });
  check('bridge boots and reports status', booted, (Date.now() - t0) + ' ms');
  if (!booted) { console.log('\nFAILED: the bridge did not boot'); process.exit(1); }

  const st = B.stats || {};
  console.log('     status: ' + JSON.stringify({
    params: st.params, vocab: st.vocab, steps: st.steps,
    loss: st.loss != null ? +st.loss.toFixed(3) : null, corpusChars: st.corpusChars, docs: st.corpusDocs
  }));
  check('status carries real model numbers',
    st.params > 100000 && st.vocab > 50 && st.corpusChars > 100000,
    'params ' + st.params + ', vocab ' + st.vocab + ', corpus ' + st.corpusChars);
  check('bootstrap training produced steps', st.steps > 0, st.steps + ' steps');

  const panel = registry.get('llmPanel');
  check('sidebar statistics panel was added', !!panel && /lmParams/.test(panel.innerHTML));
  const tile = document.getElementById('llmTileStats');
  check('settings tile shows statistics', !!tile && tile.textContent.length > 10, tile ? tile.textContent : '');

  /* Generation, the gate, and retrieval. */
  const res = await B.ask('что такое вода');
  check('ask() returns an answer envelope', !!res && !!res.answer,
    res && res.answer ? (res.answer.genTokens + ' tokens, low=' + res.answer.lowConfidence) : 'null');
  if (res && res.answer) console.log('     sample: ' + JSON.stringify(res.answer.text.slice(0, 110)));

  const woven = sandbox.OnyXWeb.blend([
    { title: 'Вода', text: 'Вода — это прозрачная жидкость без цвета и запаха. Она закипает при ста градусах Цельсия.' }
  ], 'что такое вода', 'definition');
  check('blended retrieval reads as prose without links',
    woven.length > 60 && woven.indexOf('http') === -1, JSON.stringify(woven.slice(0, 90)));

  /* Learning then training. */
  const before = (B.stats && B.stats.steps) || 0;
  B.learn('что такое вода', 'вода — это соединение водорода и кислорода.',
    [{ title: 'Вода', text: 'Вода закипает при ста градусах Цельсия.' }]);
  await new Promise(r => setTimeout(r, 3000));
  const after = await B.call('status', {}, 20000);
  check('learning advanced the training', after.steps > before, before + ' -> ' + after.steps);

  /* Media. */
  const img = await sandbox.OnyXImage.placeholder('киберпанк город ночью', { size: 256 });
  check('image service produces a watermarked picture', !!img && !!img.url && img.watermarked,
    img ? img.width + 'x' + img.height : 'null');

  check('video service reports encoder support', sandbox.OnyXVideo.supported() === true);
  const clip = await sandbox.OnyXVideo.generate('космос и звёзды',
    { seconds: 2, width: 320, height: 180, fps: 12 });
  check('video service produces a clip with a baked watermark',
    !!clip && !!clip.blob && clip.watermarked === true && clip.seconds === 2,
    clip ? (clip.width + 'x' + clip.height + ' ' + clip.ext + ' ' + clip.bytes + ' bytes') : 'null');

  /* Splash. */
  sandbox.OnyXSplash.show();
  check('splash overlay mounts', !!registry.get('onyxSplash') || sandbox.OnyXSplash.done);
  sandbox.OnyXSplash.finish(true);
  check('splash dismisses cleanly', sandbox.OnyXSplash.done === true);

  const desc = B.describe();
  check('describe() renders a readable summary', typeof desc === 'string' && desc.length > 100,
    desc.split('<br>').length + ' lines');

  console.log('\n' + (failures === 0 ? 'PAGE VERIFICATION PASSED' : failures + ' CHECK(S) FAILED'));
  process.exit(failures === 0 ? 0 : 1);
}

main().catch(e => { console.error('verification crashed:', e && e.stack || e); process.exit(1); });
