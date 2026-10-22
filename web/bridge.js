/* ============================================================================
 * OnyX LLM · host bridge
 * ============================================================================
 * Runs on the page and connects four things:
 *
 *   · the network, inside a Web Worker built from the inline engine source;
 *   · open sources, consulted quietly and folded into the answer as prose;
 *   · image and video generation, with the OnyX watermark applied to both;
 *   · the chat harness, which is only told whether the reply is ready.
 *
 * The page's layout, colours and markup are not restructured: statistics go
 * into the block that already exists, plus one additive settings screen.
 * ==========================================================================*/

function OnyXLLMBridgeFactory(global) {
'use strict';

var CALL_TIMEOUT = 60000;
var TRAIN_SLICE = 900;
var STATS_INTERVAL = 1200;

/* Every model the picker offers, with the capabilities each one unlocks. */
var MODEL_CAPABILITIES = {
  '2.5': { video: false, images: false, web: false, vision: 'граф знаний' },
  '3.1': { video: false, images: true, web: true, vision: 'граф + источники' },
  '4.0': { video: false, images: true, web: true, vision: 'глубокий анализ' },
  '4.2': { video: false, images: true, web: true, vision: 'все источники' },
  '4.3': { video: false, images: true, web: true, vision: 'максимум · nova' },
  '0.5': { video: true, images: true, web: true, vision: 'видео · v0.5' }
};

var Bridge = {
  worker: null,
  ready: false,
  failed: false,
  pending: Object.create(null),
  seq: 1,
  stats: null,
  training: false,
  wantMore: 0,
  budgetScale: 1,
  lastStatsAt: 0,
  seedText: '',
  lastWoven: '',
  lastResults: [],
  lastIntent: '',
  caps: MODEL_CAPABILITIES['4.3'],

  /* ------------------------------------------------------------- bootstrap */

  start: function () {
    var self = this;
    if (this.worker || this.failed) return;
    var src = this.engineSource();
    if (!src) { this.failed = true; this.render(); return; }
    try {
      var blob = new Blob([src], { type: 'application/javascript' });
      var url = URL.createObjectURL(blob);
      this.worker = new Worker(url);
      this.worker.onmessage = function (e) { self.onMessage(e.data); };
      this.worker.onerror = function (e) {
        self.failed = true;
        self.worker = null;
        try { console.warn('[OnyX LLM] worker error', e.message); } catch (x) { }
        self.render();
      };
      this.call('init', { seed: this.seedText || (global.OnyXCorpus ? global.OnyXCorpus.build() : '') }, 180000)
        .then(function () { return self.call('status', {}, 30000); })
        .then(function (st) {
          self.ready = true;
          self.stats = st || self.stats;
          self.render();
          self.scheduleTraining(6);
          self.warmMedia();
        })
        .catch(function () { self.failed = true; self.render(); });
    } catch (e) {
      this.failed = true;
      this.render();
    }
  },

  /* Prepare the watermarks once, so the first generation is not slow. */
  warmMedia: function () {
    try { if (global.OnyXImage && OnyXImage.warmup) OnyXImage.warmup(); } catch (e) { }
    try { if (global.OnyXVideo && OnyXVideo.warmup) OnyXVideo.warmup(); } catch (e) { }
  },

  /* The engine source lives in an inline <script id="onyx-llm-engine-source">.
   * Reading its textContent gives the worker the exact source, which keeps
   * everything working from file:// where fetching a separate .js is blocked. */
  engineSource: function () {
    var el = document.getElementById('onyx-llm-engine-source');
    if (!el) return null;
    var src = el.textContent || el.innerText || '';
    if (src.length < 1000) return null;
    if (src.indexOf('</scr' + 'ipt>') !== -1) src = src.split('</scr' + 'ipt>').join('<\\/script>');
    return src;
  },

  /* ------------------------------------------------------------------- rpc */

  call: function (type, payload, timeout) {
    var self = this;
    if (!this.worker) return Promise.reject(new Error('no worker'));
    var id = this.seq++;
    return new Promise(function (resolve, reject) {
      var timer = setTimeout(function () {
        delete self.pending[id];
        reject(new Error('timeout'));
      }, timeout || CALL_TIMEOUT);
      self.pending[id] = {
        resolve: function (v) { clearTimeout(timer); resolve(v); },
        reject: function (e) { clearTimeout(timer); reject(e); }
      };
      try { self.worker.postMessage(Object.assign({ id: id, type: type }, payload || {})); }
      catch (e) { clearTimeout(timer); delete self.pending[id]; reject(e); }
    });
  },

  onMessage: function (msg) {
    if (!msg || !msg.id) return;
    var p = this.pending[msg.id];
    if (!p) return;
    delete this.pending[msg.id];
    if (msg.ok) p.resolve(msg.result);
    else p.reject(new Error(msg.error || 'worker error'));
  },

  /* -------------------------------------------------------------- training */

  scheduleTraining: function (units) {
    this.wantMore += (units || 1);
    if (this.training) return;
    var self = this;
    this.training = true;
    this.render();
    var loop = function () {
      if (!self.worker || self.failed) { self.training = false; self.render(); return; }
      if (self.wantMore <= 0) { self.training = false; self.flushStats(true); self.render(); return; }
      self.wantMore -= 1;
      var steps = Math.max(1, Math.min(24, Math.round(6 * self.budgetScale)));
      self.call('train', { steps: steps, opts: { timeBudgetMs: TRAIN_SLICE } }, 180000)
        .then(function (r) {
          if (r && r.ema != null && self.stats) self.stats.loss = r.ema;
          self.flushStats(false);
          setTimeout(loop, 60);
        })
        .catch(function () { self.training = false; self.render(); });
    };
    setTimeout(loop, 40);
  },

  /* --------------------------------------------------- answering a question */

  /* 1. Ask the network. 2. If it is unsure, quietly consult open sources and let
   *    it try again with that material. 3. Never show links or citations — the
   *    retrieved facts are simply spoken inside the answer. */
  ask: function (question, intent) {
    var self = this;
    if (!this.worker) return Promise.resolve(null);
    var t0 = Date.now();
    if (intent) this.lastIntent = intent;

    var finish = function (res, sources, woven) {
      if (res) res.ms = Date.now() - t0;
      self.lastResults = sources || [];
      self.lastWoven = woven || '';
      if (self.stats) self.stats.lastGen = res;
      return { answer: res, sources: sources || [], woven: woven || '' };
    };

    var askModel = function (context) {
      var type = context ? 'generateWithContext' : 'generate';
      return self.call(type, { question: question, context: context, opts: {} }, 45000)
        .catch(function () { return null; });
    };

    return askModel('').then(function (first) {
      var needsHelp = !first || first.lowConfidence;
      var wantWeb = global.OnyXWeb && OnyXWeb.shouldSearch ? OnyXWeb.shouldSearch(question) : true;
      if (!needsHelp || !wantWeb || !global.OnyXWeb) return finish(first, [], '');

      var techy = /(код|программ|python|javascript|ошибк|функци|массив|api|sql|сервер|компил|отлад|алгоритм|библиотек)/i.test(question);
      return OnyXWeb.search(question, { tech: techy, max: 4 })
        .then(function (results) {
          if (!results || !results.length) return finish(first, [], '');
          var woven = OnyXWeb.blend(results, question, self.lastIntent || 'default');
          var context = OnyXWeb.contextFor(results, 380);
          return askModel(context).then(function (second) {
            var best = (second && !second.lowConfidence) ? second
              : (first && !first.lowConfidence) ? first
                : (second || first);
            return finish(best, results, woven);
          });
        })
        .catch(function () { return finish(first, [], ''); });
    });
  },

  learn: function (question, answer, sources) {
    if (!this.worker) return;
    var src = [];
    var list = (sources && sources.length) ? sources : this.lastResults;
    if (list && list.length) {
      for (var i = 0; i < Math.min(list.length, 3); i++) {
        var s = list[i];
        if (s && s.text) src.push({ title: s.title || '', text: s.text });
      }
    }
    this.call('learn', { question: question, answer: answer, sources: src }, 40000)
      .then(function (r) { if (r && r.added) Bridge.scheduleTraining(2); })
      .catch(function () { });
  },

  feedback: function (question, answer, score) {
    if (!this.worker) return Promise.resolve(false);
    return this.call('feedback', { question: question, answer: answer, score: score }, 30000)
      .then(function (r) { Bridge.scheduleTraining(3); return !!(r && r.ok); })
      .catch(function () { return false; });
  },

  trainNow: function (seconds) {
    this.scheduleTraining(Math.max(20, Math.round((seconds || 20) * 10)));
  },

  save: function (force) {
    if (!this.worker) return Promise.resolve(false);
    return this.call('save', { force: !!force }, 40000).catch(function () { return false; });
  },

  exportWeights: function () {
    if (!this.worker) return Promise.resolve(null);
    return this.call('export', {}, 120000).catch(function () { return null; });
  },

  importWeights: function (data) {
    if (!this.worker) return Promise.resolve(false);
    return this.call('import', { data: data }, 120000)
      .then(function (r) { return !!(r && r.ok); })
      .catch(function () { return false; });
  },

  exportCorpus: function () {
    if (!this.worker) return Promise.resolve(null);
    return this.call('exportCorpus', {}, 90000).catch(function () { return null; });
  },

  reset: function () {
    if (!this.worker) return Promise.resolve(false);
    var self = this;
    var text = this.seedText || (global.OnyXCorpus ? global.OnyXCorpus.build() : '');
    return this.call('reset', { seed: text }, 240000)
      .then(function (r) { self.scheduleTraining(4); return r; })
      .catch(function () { return false; });
  },

  /* ------------------------------------------------------------ reporting */

  flushStats: function (force) {
    if (!this.worker) return;
    var now = Date.now();
    if (!force && now - this.lastStatsAt < STATS_INTERVAL) return;
    this.lastStatsAt = now;
    var self = this;
    this.call('status', {}, 20000).then(function (st) {
      self.stats = Object.assign(self.stats || {}, st);
      self.render();
    }).catch(function () { });
  },

  render: function () {
    var el = document.getElementById('brainPanel');
    if (!el) return;
    var box = document.getElementById('llmPanel');
    if (!box) {
      box = document.createElement('div');
      box.id = 'llmPanel';
      box.className = 'brain-panel';
      box.style.borderTop = '1px solid var(--line2)';
      var host = el.parentNode || document.body;
      if (host && host.insertBefore) host.insertBefore(box, el.nextSibling);
      else if (host && host.appendChild) host.appendChild(box);
    }
    if (!box.innerHTML) {
      box.innerHTML =
        '<div class="row"><span style="color:var(--red)">нейросеть onyx llm</span><b id="lmState">—</b></div>' +
        '<div class="row"><span>параметров</span><b class="blue" id="lmParams">0</b></div>' +
        '<div class="row"><span>шагов обучения</span><b class="green" id="lmSteps">0</b></div>' +
        '<div class="row"><span>loss</span><b class="yellow" id="lmLoss">—</b></div>' +
        '<div class="row"><span>корпус</span><b id="lmCorpus">0</b></div>' +
        '<div class="row"><span>словарь</span><b id="lmVocab">0</b></div>';
    }
    var st = this.stats || {};
    var put = function (id, v) {
      var e = null;
      try { e = document.getElementById(id); } catch (x) { }
      if (e) e.textContent = String(v);
    };
    var state;
    if (this.failed) state = 'недоступна';
    else if (!this.ready) state = 'запуск…';
    else if (this.training) state = 'обучается';
    else state = 'готова';
    put('lmState', state);
    put('lmParams', st.params ? Number(st.params).toLocaleString('ru-RU') : '0');
    put('lmSteps', st.steps ? Number(st.steps).toLocaleString('ru-RU') : '0');
    put('lmLoss', st.loss != null ? Number(st.loss).toFixed(3) : '—');
    put('lmCorpus', st.corpusChars ? Math.round(st.corpusChars / 1000) + ' тыс. симв.' : '0');
    put('lmVocab', st.vocab ? Number(st.vocab).toLocaleString('ru-RU') : '0');
    this.renderTile();
  },

  renderTile: function () {
    var el = null;
    try { el = document.getElementById('llmTileStats'); } catch (e) { }
    if (!el) return;
    var st = this.stats || {};
    var bits = [];
    if (this.failed) bits.push('недоступна');
    else if (!this.ready) bits.push('запуск…');
    else {
      bits.push('параметров ' + (st.params ? Number(st.params).toLocaleString('ru-RU') : '0'));
      bits.push('шагов ' + (st.steps ? Number(st.steps).toLocaleString('ru-RU') : '0'));
      if (st.loss != null) bits.push('loss ' + Number(st.loss).toFixed(3));
      bits.push('корпус ' + Math.round((st.corpusChars || 0) / 1000) + ' тыс. симв.');
      bits.push('словарь ' + (st.vocab || 0));
    }
    el.textContent = bits.join(' · ');
  },

  describe: function () {
    var st = this.stats || {};
    var lines = [];
    lines.push('движок: ' + (this.failed ? 'недоступен' : (this.ready ? 'работает' : 'запускается')));
    if (st.params) lines.push('параметров: ' + Number(st.params).toLocaleString('ru-RU'));
    if (st.layers) lines.push('архитектура: ' + st.layers + ' слоёв · ' + st.dModel + ' каналов · ' + st.heads + ' голов · контекст ' + st.ctx + ' · ' + (st.preset || ''));
    if (st.vocab) lines.push('словарь: ' + st.vocab + ' символов (посимвольная токенизация)');
    if (st.steps) lines.push('шагов обучения: ' + Number(st.steps).toLocaleString('ru-RU') + ' · увидено токенов: ' + Number(st.tokens || 0).toLocaleString('ru-RU'));
    if (st.loss != null) lines.push('loss: ' + Number(st.loss).toFixed(4) + ' · perplexity: ' + (Math.exp(st.loss) < 1e6 ? Math.exp(st.loss).toFixed(2) : 'очень высокая'));
    if (st.corpusDocs != null) lines.push('корпус: ' + st.corpusDocs + ' документов · ' + Math.round((st.corpusChars || 0) / 1000) + ' тыс. символов · ' + Number(st.streamTokens || 0).toLocaleString('ru-RU') + ' токенов');
    if (st.generations) lines.push('генераций: ' + st.generations + ' · ответов отклонено гейтом: ' + (st.fallbacks || 0));
    if (st.storage === 'error') lines.push('внимание: сохранение в браузере недоступно, прогресс не переживёт перезагрузку');
    if (this.training) lines.push('статус: идёт обучение в фоне');
    lines.push('возможности: изображения — ' + (this.caps.images ? 'да' : 'нет') +
      ', видео — ' + (this.caps.video ? 'да' : 'нет') +
      ', открытые источники — ' + (this.caps.web ? 'да' : 'нет'));
    return lines.join('<br>');
  },

  boot: function () {
    var self = this;
    var go = function () {
      try { self.start(); } catch (e) { self.failed = true; self.render(); }
      global.addEventListener('beforeunload', function () { try { self.save(true); } catch (e) { } });
      global.addEventListener('load', function () { setTimeout(function () { self.save(false); }, 20000); });
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go);
    else go();
  }
};

global.OnyXLLM = global.OnyXLLM || {};
global.OnyXLLM.bridge = Bridge;
global.OnyXLLM.MODEL_CAPABILITIES = MODEL_CAPABILITIES;
global.OnyXLLM.capabilities = function (version) { return MODEL_CAPABILITIES[version] || MODEL_CAPABILITIES['4.3']; };
global.OnyXLLM.ask = function (q) { return Bridge.ask(q); };
global.OnyXLLM.train = function (seconds) { return Bridge.trainNow(seconds || 30); };
global.OnyXLLM.status = function () { return Bridge.stats; };
global.OnyXLLM.save = function () { return Bridge.save(true); };
global.OnyXLLM.reset = function () { return Bridge.reset(); };
global.OnyXLLM.exportWeights = function () { return Bridge.exportWeights(); };
global.OnyXLLM.importWeights = function (d) { return Bridge.importWeights(d); };
global.OnyXLLM.exportCorpus = function () { return Bridge.exportCorpus(); };
global.OnyXLLM.learn = Bridge.learn.bind(Bridge);
global.OnyXLLM.search = function (q, o) { return global.OnyXWeb ? OnyXWeb.search(q, o) : Promise.resolve([]); };

return Bridge;
}

if (typeof window !== 'undefined') OnyXLLMBridgeFactory(window);
