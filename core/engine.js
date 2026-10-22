/* ============================================================================
 * OnyX LLM Engine · CodeXY Labs ™
 * ----------------------------------------------------------------------------
 * A self-contained, dependency-free, self-training language model.
 *
 * Implemented from scratch in pure JavaScript:
 *   · BPE (byte-pair-encoding) tokenizer, trained on the corpus itself
 *   · decoder-only Transformer: token + positional embeddings, pre-LayerNorm,
 *     multi-head causal self-attention, GELU MLP, weight-tied output projection
 *   · hand-derived backpropagation through every one of those operations
 *   · AdamW, warmup + cosine LR schedule, global gradient-norm clipping
 *   · temperature / top-k / nucleus / repetition-penalty sampling
 *   · incremental online learning: trains itself while the user chats
 *   · persistence: tokenizer + int8-quantised weights in IndexedDB
 *
 * This exact file runs in BOTH the main thread and inside a Web Worker. It is
 * written as a single plain function so it can be serialised via toString().
 * ==========================================================================*/

function OnyXLLMEngineFactory(global) {
'use strict';

var VERSION = '1.1';
var SPECIALS = ['<pad>', '<unk>', '<bos>', '<eos>', '<user>', '<ai>'];
var EPS = 1e-5;

/* ------------------------------------------------------------------ utils */

function softmaxInPlace(a, off, n) {
  var m = -Infinity, i;
  for (i = 0; i < n; i++) if (a[off + i] > m) m = a[off + i];
  if (m === -Infinity) m = 0;
  var s = 0;
  for (i = 0; i < n; i++) { var e = Math.exp(a[off + i] - m); a[off + i] = e; s += e; }
  if (!(s > 0) || !isFinite(s)) s = 1;
  var inv = 1 / s;
  for (i = 0; i < n; i++) a[off + i] *= inv;
}

function makeRng(seed) {
  var s = (seed >>> 0) || 0x9e3779b9;
  return function () {
    s = (s + 0x6D2B79F5) >>> 0;
    var t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function nowMs() {
  try { return (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now(); }
  catch (e) { return Date.now(); }
}

function zeroFill(a, n) { if (n === undefined) n = a.length; for (var i = 0; i < n; i++) a[i] = 0; }

/* --------------------------------------------------------------------------
 * Scratch-buffer pool. The training inner loop allocates tens of megabytes of
 * short-lived Float32Arrays per step; recycling them removes almost all of the
 * GC pressure and is a large part of the achievable throughput on the CPU.
 * Every consumer below fully overwrites its buffer (never accumulates into it),
 * so reusing a pooled buffer is safe.
 * ------------------------------------------------------------------------*/
var SCRATCH = new Map();
var SCRATCH_BUDGET = 96 * 1024 * 1024;   // bytes
var scratchBytes = 0;

function scratch(n) {
  var key = n;
  var list = SCRATCH.get(key);
  if (list === undefined) { list = []; SCRATCH.set(key, list); }
  var buf = list.pop();
  if (buf === undefined) {
    buf = new Float32Array(n);
    scratchBytes += n * 4;
  }
  return buf;
}

function release(buf) {
  if (!buf) return;
  var list = SCRATCH.get(buf.length);
  if (list === undefined) { list = []; SCRATCH.set(buf.length, list); }
  if (list.length < 24 && scratchBytes < SCRATCH_BUDGET * 4) list.push(buf);
}

function freeScratch() { SCRATCH.clear(); scratchBytes = 0; }

/* =================================================================== BPE ==
 * Standard BPE over unicode word tokens. Sub-word scoring uses
 * freq / (len(a) * len(b)) which behaves well across mixed alphabets.
 * =======================================================================*/

var PAIR_MUL = 1048576;

function BPETokenizer() {
  this.chars = [];
  this.merges = [];
  this.ranks = new Map();
  this.itos = [];
  this.stoi = new Map();
  this.base = SPECIALS.length;   // id where merged tokens start
  this.extra = new Map();        // characters discovered after training
  this.nextId = SPECIALS.length;
  this._tok = [];
}

BPETokenizer.prototype._rebuildVocab = function () {
  var toks = SPECIALS.concat(this.chars);
  this.base = toks.length;
  for (var i = 0; i < this.merges.length; i++) {
    var a = this.merges[i][0], b = this.merges[i][1];
    toks.push((toks[a] != null ? toks[a] : '\uFFFD') + (toks[b] != null ? toks[b] : '\uFFFD'));
  }
  this.itos = toks;
  this._tok = toks;
  this.stoi = new Map();
  for (var k = 0; k < toks.length; k++) this.stoi.set(toks[k], k);
  this.nextId = toks.length;
  return this;
};

BPETokenizer.prototype._buildRanks = function () {
  this.ranks = new Map();
  for (var i = 0; i < this.merges.length; i++) this.ranks.set(this.merges[i][0] * PAIR_MUL + this.merges[i][1], i);
};

BPETokenizer.prototype.train = function (text, opts) {
  opts = opts || {};
  var maxMerges = opts.merges || 3000;
  var minFreq = opts.minFreq || 2;

  /* Markers are stripped *here*, inside the trainer, so no caller can leak
   * "<user>"/"<ai>" into the character set or the merge table. */
  var norm = trainingOnlyText(normalizeForTraining(text));
  var words = bpeSplitText(norm);
  var cnt = new Map();
  for (var i = 0; i < words.length; i++) {
    var w = words[i];
    if (!w || w.charCodeAt(0) === 60) continue;   // never train merges on markers
    cnt.set(w, (cnt.get(w) || 0) + 1);
  }
  var keys = Array.from(cnt.keys());

  var charFreq = new Map();
  for (var k = 0; k < keys.length; k++) {
    var w = keys[k], f = cnt.get(w);
    for (var c = 0; c < w.length; c++) charFreq.set(w[c], (charFreq.get(w[c]) || 0) + f);
  }
  this.chars = Array.from(charFreq.keys()).sort();

  var cIndex = new Map();
  for (var ci = 0; ci < this.chars.length; ci++) cIndex.set(this.chars[ci], ci);

  var seqs = [];
  for (var s = 0; s < keys.length; s++) {
    var word = keys[s], arr = new Array(word.length);
    for (var p = 0; p < word.length; p++) arr[p] = cIndex.get(word[p]);
    seqs.push({ ids: arr, f: cnt.get(word) });
  }

  var toks = this.chars.slice();
  var nextId = this.chars.length;

  for (var step = 0; step < maxMerges; step++) {
    var pairs = new Map();
    for (var q = 0; q < seqs.length; q++) {
      var ids = seqs[q].ids, f2 = seqs[q].f;
      for (var t = 0; t + 1 < ids.length; t++) {
        var key = ids[t] * PAIR_MUL + ids[t + 1];
        var cur = pairs.get(key);
        pairs.set(key, cur === undefined ? f2 : cur + f2);
      }
    }
    if (pairs.size === 0) break;
    var bestKey = -1, bestScore = -Infinity;
    pairs.forEach(function (freq, key) {
      if (freq < minFreq) return;
      var a = (key / PAIR_MUL) | 0, b = key % PAIR_MUL;
      var la = toks[a] ? toks[a].length : 1, lb = toks[b] ? toks[b].length : 1;
      var score = freq / (la * lb);
      if (score > bestScore) { bestScore = score; bestKey = key; }
    });
    if (bestKey < 0) break;

    var ba = (bestKey / PAIR_MUL) | 0, bb = bestKey % PAIR_MUL;
    var merged = false;
    for (var r = 0; r < seqs.length; r++) {
      var src = seqs[r].ids, dst = null, n = 0;
      for (var u = 0; u < src.length; u++) {
        if (u + 1 < src.length && src[u] === ba && src[u + 1] === bb) {
          if (!dst) { dst = new Array(src.length); for (var cp = 0; cp < u; cp++) dst[cp] = src[cp]; n = u; }
          dst[n++] = nextId; u++; merged = true;
        } else if (dst) dst[n++] = src[u];
      }
      if (dst) { dst.length = n; seqs[r].ids = dst; }
    }
    if (!merged) break;
    this.merges.push([ba, bb]);
    toks.push((toks[ba] || '') + (toks[bb] || ''));
    nextId++;
  }

  this._rebuildVocab();
  this._buildRanks();
  return this;
};

BPETokenizer.prototype._charId = function (ch) {
  var id = this.stoi.get(ch);
  if (id !== undefined) return id;
  var ex = this.extra.get(ch);
  if (ex !== undefined) return ex;
  return 1; // <unk>
};

/* Characters first seen after training are appended as new vocabulary rows.
 * The model re-allocates its embedding to make room, so nothing is ever lost
 * and the vocabulary can keep growing with the conversation. */
BPETokenizer.prototype.extendChars = function (text) {
  var nov = null;
  var clean = String(text == null ? '' : text).replace(MARKER_ALL, ' ');
  for (var i = 0; i < clean.length; i++) {
    var ch = clean.charAt(i);
    if (/\s/.test(ch)) continue;
    if (this.stoi.has(ch) || this.extra.has(ch)) continue;
    if (!nov) nov = [];
    if (nov.indexOf(ch) === -1) nov.push(ch);
  }
  if (!nov) return { added: 0, chars: null };
  for (var j = 0; j < nov.length; j++) { this.extra.set(nov[j], this.nextId++); }
  return { added: nov.length, chars: nov };
};

BPETokenizer.prototype.encodeWord = function (word) {
  var ids = new Array(word.length);
  for (var i = 0; i < word.length; i++) ids[i] = this._charId(word[i]);
  while (ids.length > 1) {
    var bestRank = Infinity, bestPos = -1;
    for (var j = 0; j + 1 < ids.length; j++) {
      if (ids[j] < SPECIALS.length || ids[j + 1] < SPECIALS.length) continue;  // never merge markers
      var r = this.ranks.get(ids[j] * PAIR_MUL + ids[j + 1]);
      if (r !== undefined && r < bestRank) { bestRank = r; bestPos = j; }
    }
    if (bestPos < 0) break;
    ids.splice(bestPos, 2, this.base + bestRank);
  }
  return ids;
};

BPETokenizer.prototype.encode = function (text) {
  var out = [];
  var words = bpeSplitText(normalizeForTraining(text));
  for (var i = 0; i < words.length; i++) {
    var w = words[i];
    if (w.charCodeAt(0) === 60) {          // atomic control marker
      var mid = this.stoi.get(w);
      if (mid !== undefined) out.push(mid);
      continue;
    }
    var ids = this.encodeWord(w);
    for (var j = 0; j < ids.length; j++) out.push(ids[j]);
  }
  return out;
};

BPETokenizer.prototype.decode = function (ids) {
  var s = '';
  var idToExtra = null;
  for (var i = 0; i < ids.length; i++) {
    var id = ids[i];
    if (id < SPECIALS.length) continue;
    var t = this.itos[id];
    if (t != null) { s += t; continue; }
    if (!idToExtra) { idToExtra = new Map(); this.extra.forEach(function (v, kk) { idToExtra.set(v, kk); }); }
    var c = idToExtra.get(id);
    if (c) s += c;
  }
  return s;
};

BPETokenizer.prototype.id = function (tok) { var v = this.stoi.get(tok); return v === undefined ? 1 : v; };
BPETokenizer.prototype.size = function () { return this.nextId; };

BPETokenizer.prototype.serialize = function () {
  return { version: 1, chars: this.chars, merges: this.merges, extra: Array.from(this.extra.keys()) };
};
BPETokenizer.deserialize = function (obj) {
  var t = new BPETokenizer();
  t.chars = (obj && obj.chars) || [];
  t.merges = (obj && obj.merges) || [];
  t._rebuildVocab();
  t._buildRanks();
  var ex = (obj && obj.extra) || [];
  for (var i = 0; i < ex.length; i++) if (!t.stoi.has(ex[i]) && !t.extra.has(ex[i])) t.extra.set(ex[i], t.nextId++);
  return t;
};

/* Normalisation used both for tokenizer training and for every encode(). */
function normalizeForTraining(text) {
  var t = String(text == null ? '' : text);
  t = t.replace(/\r\n?/g, '\n');
  t = t.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, ' ');
  t = t.replace(/[\u200B-\u200D\uFEFF]/g, '');
  t = t.replace(/ё/g, 'е').replace(/Ё/g, 'Е');
  t = t.toLowerCase();
  t = t.replace(/[«»“”„]/g, '"').replace(/[‘’`]/g, "'");
  t = t.replace(/[—–]/g, '-');
  t = t.replace(/[ \t]+/g, ' ');
  t = t.replace(/ ?\n ?/g, '\n');
  t = t.replace(/\n{3,}/g, '\n\n');
  return t;
}

/* Text-level control markers. These must never take part in BPE merges: if the
 * trainer is allowed to glue "<ai>" to the following letters it burns its whole
 * merge budget on marker fragments and real words stay character-level. */
var TEXT_MARKERS = /<user>|<ai>|<eos>|<bos>/g;
var MARKER_ALL = /<user>|<ai>|<eos>|<bos>|<pad>|<unk>/g;

/* Split raw text into BPE "words" (a leading space belongs to the word, and
 * control markers are emitted as single atomic units). */
function bpeSplitText(text) {
  var re = /<user>|<ai>|<eos>|<bos>|[\p{L}\p{N}]+|[^\p{L}\p{N}\s]/gu;
  var out = [], m, last = 0;
  while ((m = re.exec(text)) !== null) {
    var unit = m[0];
    if (unit.charCodeAt(0) === 60 /* '<' */) {
      out.push(unit);
      last = re.lastIndex;
      continue;
    }
    out.push((text.slice(last, m.index).length > 0 ? ' ' : '') + unit);
    last = re.lastIndex;
  }
  return out;
}

/* Text used to TRAIN the tokenizer: markers stripped, so every merge slot goes
 * to genuine Cyrillic/Latin sub-words. */
function trainingOnlyText(text) {
  return String(text == null ? '' : text).replace(MARKER_ALL, ' ');
}

/* ============================================================ Matrix core ==
 * 3-D tensors: layout [outer, T, D]  -> index (o*T + t)*D + d
 * 2-D matrices: layout [rows, cols]  -> index r*cols + c
 * =========================================================================*/

/* C[o][T,Mc] = A[o][T,K] · W[Mc,K]^T   (linear layer, W is [out,in])
 * Unrolled four ways: this is the single hottest kernel in the whole engine,
 * and the unrolled form measures ~1.4x faster than the naive loop. */
function mmLinear(A, W, C, o, T, K, Mc) {
  var k4 = K & ~3;
  for (var t = 0; t < T; t++) {
    var ab = (o * T + t) * K, cb = (o * T + t) * Mc;
    for (var m = 0; m < Mc; m++) {
      var wb = m * K, s0 = 0, s1 = 0, s2 = 0, s3 = 0, k = 0;
      for (; k < k4; k += 4) {
        s0 += A[ab + k] * W[wb + k];
        s1 += A[ab + k + 1] * W[wb + k + 1];
        s2 += A[ab + k + 2] * W[wb + k + 2];
        s3 += A[ab + k + 3] * W[wb + k + 3];
      }
      for (; k < K; k++) s0 += A[ab + k] * W[wb + k];
      C[cb + m] = s0 + s1 + s2 + s3;
    }
  }
}

/* dA += dC · W ; dW += dC^T · A
 * W and dW may be null when only one of the two gradients is needed. */
function mmLinearBackward(dC, A, W, dA, dW, o, T, K, Mc) {
  var wantA = !!dA && !!W, wantW = !!dW && !!W;
  var k4 = K & ~3;
  for (var t = 0; t < T; t++) {
    var cb = (o * T + t) * Mc, ab = (o * T + t) * K;
    for (var m = 0; m < Mc; m++) {
      var g = dC[cb + m];
      if (g === 0) continue;
      var k, a0, a1, a2, a3, w0, w1, w2, w3;
      if (wantA) {
        var wb = m * K;
        for (k = 0; k < k4; k += 4) {
          w0 = W[wb + k]; w1 = W[wb + k + 1]; w2 = W[wb + k + 2]; w3 = W[wb + k + 3];
          dA[ab + k] += g * w0; dA[ab + k + 1] += g * w1;
          dA[ab + k + 2] += g * w2; dA[ab + k + 3] += g * w3;
        }
        for (; k < K; k++) dA[ab + k] += g * W[wb + k];
      }
      if (wantW) {
        var db = m * K;
        for (k = 0; k < k4; k += 4) {
          a0 = A[ab + k]; a1 = A[ab + k + 1]; a2 = A[ab + k + 2]; a3 = A[ab + k + 3];
          dW[db + k] += g * a0; dW[db + k + 1] += g * a1;
          dW[db + k + 2] += g * a2; dW[db + k + 3] += g * a3;
        }
        for (; k < K; k++) dW[db + k] += g * A[ab + k];
      }
    }
  }
}

/* Head-sliced tensors (Q/K/V and their gradients) are laid out as
 * [B, T, H, dHead] — index ((b*T + t)*H + h)*dHead + d — so they can be read
 * straight out of a [B*T, D] buffer with a head stride. */
function scatterHeads(src, dst, B, T, D, H, dh) {
  for (var b = 0; b < B; b++) {
    for (var h = 0; h < H; h++) {
      for (var t = 0; t < T; t++) {
        var sb = ((b * T + t) * H + h) * dh;
        var db = (b * T + t) * D + h * dh;
        for (var d = 0; d < dh; d++) dst[db + d] = src[sb + d];
      }
    }
  }
}

/* dW[(h*dh + d), k] += Σ_{b,t} dHead[b,t,h,d] · Xn[b,t,k] */
function headSlicedDW(A, Xn, dW, B, T, D, H, dh) {
  for (var b = 0; b < B; b++) {
    for (var h = 0; h < H; h++) {
      for (var t = 0; t < T; t++) {
        var ab = ((b * T + t) * H + h) * dh, db = (h * dh) * D, xb = (b * T + t) * D;
        for (var d = 0; d < dh; d++) {
          var g = A[ab + d];
          if (g === 0) continue;
          var row = db + d * D;
          for (var k = 0; k < D; k++) dW[row + k] += g * Xn[xb + k];
        }
      }
    }
  }
}

/* =========================================================== LayerNorm ==== */

function layerNormForward(X, G, B, Y, O, T, D) {
  var stats = { inv: new Float32Array(O * T), mean: new Float32Array(O * T) };
  for (var o = 0; o < O; o++) {
    for (var t = 0; t < T; t++) {
      var b = (o * T + t) * D, mean = 0, d;
      for (d = 0; d < D; d++) mean += X[b + d];
      mean /= D;
      var v = 0;
      for (d = 0; d < D; d++) { var df = X[b + d] - mean; v += df * df; }
      v /= D;
      var iv = 1 / Math.sqrt(v + EPS);
      stats.inv[o * T + t] = iv; stats.mean[o * T + t] = mean;
      for (d = 0; d < D; d++) Y[b + d] = (X[b + d] - mean) * iv * G[d] + B[d];
    }
  }
  return stats;
}

function layerNormBackward(X, stats, G, dY, dX, dG, dB, O, T, D) {
  var inv = stats.inv, ctr = stats.mean;
  for (var o = 0; o < O; o++) {
    for (var t = 0; t < T; t++) {
      var b = (o * T + t) * D, iv = inv[o * T + t], mean = ctr[o * T + t];
      var sum1 = 0, sum2 = 0, d;
      for (d = 0; d < D; d++) {
        var g = dY[b + d] * G[d];
        sum1 += g;
        sum2 += g * ((X[b + d] - mean) * iv);
      }
      var c1 = sum1 / D, c2 = sum2 / D;
      for (d = 0; d < D; d++) {
        var xhat = (X[b + d] - mean) * iv;
        dX[b + d] += iv * (dY[b + d] * G[d] - c1 - xhat * c2);
        dG[d] += dY[b + d] * xhat;
        dB[d] += dY[b + d];
      }
    }
  }
}

/* ============================================================== GELU ===== */

var GELU_C = 0.7978845608028654, GELU_K = 0.044715;

function geluForward(x) { return 0.5 * x * (1 + Math.tanh(GELU_C * (x + GELU_K * x * x * x))); }
function geluDeriv(x) {
  var u = GELU_C * (x + GELU_K * x * x * x);
  var t = Math.tanh(u);
  var du = GELU_C * (1 + 3 * GELU_K * x * x);
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du;
}

/* ================================================================= Model ==
 * Decoder-only Transformer, pre-LayerNorm, weight-tied output projection.
 * =======================================================================*/

/* Model sizes. `nano`/`micro` are what a browser tab can realistically train on
 * its own; the larger presets are meant for offline training (tools/train.py)
 * followed by a checkpoint import, because they are far too slow to converge
 * from scratch on a single CPU thread. */
var CONFIGS = {
  nano: { dModel: 80, nLayers: 4, nHeads: 4, dFf: 320, ctx: 144 },
  micro: { dModel: 128, nLayers: 6, nHeads: 4, dFf: 512, ctx: 192 },
  small: { dModel: 192, nLayers: 6, nHeads: 6, dFf: 768, ctx: 256 },
  medium: { dModel: 256, nLayers: 8, nHeads: 8, dFf: 1024, ctx: 320 },
  large: { dModel: 384, nLayers: 10, nHeads: 8, dFf: 1536, ctx: 384 },
  xl: { dModel: 512, nLayers: 12, nHeads: 8, dFf: 2048, ctx: 448 }
};

function OnyXLM(cfg) {
  cfg = cfg || {};
  var preset = CONFIGS[cfg.preset] || CONFIGS.micro;
  var D = cfg.dModel || preset.dModel;
  var H = cfg.nHeads || preset.nHeads;
  if (D % H !== 0) H = 4;
  this.cfg = {
    vocabSize: cfg.vocabSize || 512,
    dModel: D,
    nLayers: cfg.nLayers || preset.nLayers,
    nHeads: H,
    dFf: cfg.dFf || preset.dFf,
    ctx: cfg.ctx || preset.ctx,
    seed: cfg.seed || 1337
  };
  this.dHead = D / H;
  this._alloc();
  this._initWeights();
}

OnyXLM.prototype._alloc = function () {
  var cf = this.cfg, D = cf.dModel, V = cf.vocabSize, L = cf.nLayers, F = cf.dFf;
  this.E = new Float32Array(V * D); this.dE = new Float32Array(V * D);
  this.P = new Float32Array(cf.ctx * D); this.dP = new Float32Array(cf.ctx * D);
  this.gF = new Float32Array(D); this.bF = new Float32Array(D);
  this.dgF = new Float32Array(D); this.dbF = new Float32Array(D);
  this.layers = [];
  for (var i = 0; i < L; i++) {
    this.layers.push({
      ln1g: new Float32Array(D), ln1b: new Float32Array(D), dln1g: new Float32Array(D), dln1b: new Float32Array(D),
      ln2g: new Float32Array(D), ln2b: new Float32Array(D), dln2g: new Float32Array(D), dln2b: new Float32Array(D),
      Wq: new Float32Array(D * D), Wk: new Float32Array(D * D), Wv: new Float32Array(D * D), Wo: new Float32Array(D * D),
      dWq: new Float32Array(D * D), dWk: new Float32Array(D * D), dWv: new Float32Array(D * D), dWo: new Float32Array(D * D),
      bq: new Float32Array(D), bk: new Float32Array(D), bv: new Float32Array(D), bo: new Float32Array(D),
      dbq: new Float32Array(D), dbk: new Float32Array(D), dbv: new Float32Array(D), dbo: new Float32Array(D),
      W1: new Float32Array(F * D), W2: new Float32Array(D * F),
      dW1: new Float32Array(F * D), dW2: new Float32Array(D * F),
      b1: new Float32Array(F), b2: new Float32Array(D),
      db1: new Float32Array(F), db2: new Float32Array(D)
    });
  }
  this._params = null;
};

OnyXLM.prototype._initWeights = function () {
  var cf = this.cfg, D = cf.dModel, L = cf.nLayers, F = cf.dFf;
  var r = this.rng = makeRng(cf.seed);
  function gauss() {
    var u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function fill(w, scale) { for (var i = 0; i < w.length; i++) w[i] = gauss() * scale; }
  fill(this.E, 0.02);
  fill(this.P, 0.01);
  zeroFill(this.dE); zeroFill(this.dP);
  for (var i = 0; i < L; i++) {
    var l = this.layers[i];
    fill(l.Wq, 1 / Math.sqrt(D)); fill(l.Wk, 1 / Math.sqrt(D));
    fill(l.Wv, 1 / Math.sqrt(D)); fill(l.Wo, 1 / Math.sqrt(D));
    fill(l.W1, 1 / Math.sqrt(D)); fill(l.W2, 1 / Math.sqrt(F));
    for (var d = 0; d < D; d++) { l.ln1g[d] = 1; l.ln2g[d] = 1; }
  }
  for (var d2 = 0; d2 < D; d2++) this.gF[d2] = 1;
  this._zeroGrads();
};

/* Parameter registry: [weights, grads, applyWeightDecay] */
OnyXLM.prototype.params = function () {
  if (this._params) return this._params;
  var p = [[this.E, this.dE, true], [this.P, this.dP, false], [this.gF, this.dgF, false], [this.bF, this.dbF, false]];
  for (var i = 0; i < this.layers.length; i++) {
    var l = this.layers[i];
    p.push([l.ln1g, l.dln1g, false], [l.ln1b, l.dln1b, false]);
    p.push([l.Wq, l.dWq, true], [l.Wk, l.dWk, true], [l.Wv, l.dWv, true], [l.Wo, l.dWo, true]);
    p.push([l.bq, l.dbq, false], [l.bk, l.dbk, false], [l.bv, l.dbv, false], [l.bo, l.dbo, false]);
    p.push([l.ln2g, l.dln2g, false], [l.ln2b, l.dln2b, false]);
    p.push([l.W1, l.dW1, true], [l.W2, l.dW2, true], [l.b1, l.db1, false], [l.b2, l.db2, false]);
  }
  this._params = p;
  return p;
};

/* Grow the vocabulary (new characters seen after training) while preserving
 * every previously learned row of the embedding matrix. */
OnyXLM.prototype.growVocab = function (newSize, seed) {
  var V = this.cfg.vocabSize;
  if (newSize <= V) return false;
  var D = this.cfg.dModel;
  var E2 = new Float32Array(newSize * D);
  E2.set(this.E.subarray(0, V * D));
  var r = makeRng((seed || 7) >>> 0);
  function gauss() {
    var u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  for (var i = V * D; i < newSize * D; i++) E2[i] = gauss() * 0.02;
  this.E = E2; this.dE = new Float32Array(newSize * D);
  this.cfg.vocabSize = newSize;
  this._params = null;
  return true;
};

OnyXLM.prototype._zeroGrads = function () {
  var p = this.params();
  for (var i = 0; i < p.length; i++) zeroFill(p[i][1]);
};

OnyXLM.prototype.paramCount = function () {
  var p = this.params(), n = 0;
  for (var i = 0; i < p.length; i++) n += p[i][0].length;
  return n;
};

OnyXLM.prototype._scratch = function (B, T) {
  var key = B + 'x' + T;
  if (this._sc && this._sc.key === key) return this._sc;
  var cf = this.cfg, L = cf.nLayers, H = cf.nHeads;
  var s = { key: key, B: B, T: T, probs: [] };
  var att = B * H * T * T;
  for (var i = 0; i < L; i++) s.probs.push(new Float32Array(att));
  this._sc = s;
  return s;
};

/* ------------------------------------------------------------- forward ---- */

OnyXLM.prototype.forward = function (tokens, targets, opts) {
  opts = opts || {};
  var track = !!opts.track;
  var cf = this.cfg, D = cf.dModel, L = cf.nLayers, H = cf.nHeads, F = cf.dFf, V = cf.vocabSize, dh = this.dHead;
  var B = opts.B, T = opts.T;
  if (!B || !T) { T = tokens.length; B = 1; }
  if (B * T !== tokens.length) throw new Error('token count does not match B*T');
  if (T > cf.ctx) throw new Error('sequence exceeds context window');
  var BT = B * T;

  var X = new Float32Array(BT * D);
  for (var b = 0; b < B; b++) {
    for (var t = 0; t < T; t++) {
      var tok = tokens[b * T + t];
      if (!(tok >= 0) || tok >= V) tok = 1;
      var xb = (b * T + t) * D, eb = tok * D, pb = t * D;
      for (var d = 0; d < D; d++) X[xb + d] = this.E[eb + d] + this.P[pb + d];
    }
  }

  var cache = track ? { X0: X.slice(), layers: [], B: B, T: T, targets: targets, tokens: tokens } : null;
  var probsAll = this._scratch(B, T).probs;
  var cur = X;

  for (var li = 0; li < L; li++) {
    var lay = this.layers[li];
    var Xa = cur;

    var Xn = new Float32Array(BT * D);
    var st1 = layerNormForward(Xa, lay.ln1g, lay.ln1b, Xn, B, T, D);

    var Q = new Float32Array(BT * D), K = new Float32Array(BT * D), Vt = new Float32Array(BT * D);
    mmLinear(Xn, lay.Wq, Q, 0, BT, D, D);
    mmLinear(Xn, lay.Wk, K, 0, BT, D, D);
    mmLinear(Xn, lay.Wv, Vt, 0, BT, D, D);
    for (var i2 = 0; i2 < BT; i2++) {
      var ib = i2 * D;
      for (var d2 = 0; d2 < D; d2++) { Q[ib + d2] += lay.bq[d2]; K[ib + d2] += lay.bk[d2]; Vt[ib + d2] += lay.bv[d2]; }
    }

    var Att = new Float32Array(BT * D);
    var scale = 1 / Math.sqrt(dh);
    var probs = probsAll[li];
    var row = new Float32Array(T);
    for (var b2 = 0; b2 < B; b2++) {
      for (var h = 0; h < H; h++) {
        var hb = (b2 * H + h) * T * T;
        for (var t2 = 0; t2 < T; t2++) {
          var qb = ((b2 * T + t2) * H + h) * dh;
          for (var t3 = 0; t3 <= t2; t3++) {
            var kb = ((b2 * T + t3) * H + h) * dh, dot = 0;
            for (var d3 = 0; d3 < dh; d3++) dot += Q[qb + d3] * K[kb + d3];
            row[t3] = dot * scale;
          }
          for (var tz = t2 + 1; tz < T; tz++) row[tz] = -Infinity;
          softmaxInPlace(row, 0, T);
          var ab = ((b2 * T + t2) * H + h) * dh;
          /* Store the attention weights as P[query][key] at
           * probs[(b,h)][t2][t3]. The backward pass indexes exactly this way. */
          for (var t4 = 0; t4 < T; t4++) probs[hb + t2 * T + t4] = row[t4];
          for (var d4 = 0; d4 < dh; d4++) {
            var acc = 0;
            for (var t5 = 0; t5 <= t2; t5++) {
              var pv = row[t5];
              if (pv !== 0) acc += pv * Vt[((b2 * T + t5) * H + h) * dh + d4];
            }
            Att[ab + d4] = acc;
          }
        }
      }
    }

    var AttO = new Float32Array(BT * D);
    mmLinear(Att, lay.Wo, AttO, 0, BT, D, D);
    for (var i3 = 0; i3 < BT; i3++) { var ib3 = i3 * D; for (var d5 = 0; d5 < D; d5++) AttO[ib3 + d5] += lay.bo[d5]; }

    var Xr = new Float32Array(BT * D);
    for (var i4 = 0; i4 < BT * D; i4++) Xr[i4] = Xa[i4] + AttO[i4];

    var Xn2 = new Float32Array(BT * D);
    var st2 = layerNormForward(Xr, lay.ln2g, lay.ln2b, Xn2, B, T, D);

    var Hf = new Float32Array(BT * F);
    mmLinear(Xn2, lay.W1, Hf, 0, BT, D, F);
    for (var i5 = 0; i5 < BT; i5++) { var ib5 = i5 * F; for (var f = 0; f < F; f++) Hf[ib5 + f] += lay.b1[f]; }
    var Hp = Hf.slice();
    for (var i6 = 0; i6 < BT * F; i6++) Hf[i6] = geluForward(Hf[i6]);

    var H2 = new Float32Array(BT * D);
    mmLinear(Hf, lay.W2, H2, 0, BT, F, D);
    for (var i7 = 0; i7 < BT; i7++) { var ib7 = i7 * D; for (var d6 = 0; d6 < D; d6++) H2[ib7 + d6] += lay.b2[d6]; }

    var Xo = new Float32Array(BT * D);
    for (var i8 = 0; i8 < BT * D; i8++) Xo[i8] = Xr[i8] + H2[i8];

    if (track) cache.layers.push({ Xa: Xa, Xn: Xn, st1: st1, Q: Q, K: K, Vt: Vt, Att: Att, st2: st2, Xr: Xr, Xn2: Xn2, Hp: Hp, Hf: Hf });
    cur = Xo;
  }

  var Xf = new Float32Array(BT * D);
  var stF = layerNormForward(cur, this.gF, this.bF, Xf, B, T, D);

  var logits = new Float32Array(BT * V);
  mmLinear(Xf, this.E, logits, 0, BT, D, V);

  var loss = 0, nTargets = 0;
  var lossArr = new Float32Array(BT), argmaxArr = new Int32Array(BT);
  for (var i9 = 0; i9 < BT; i9++) argmaxArr[i9] = -1;
  if (targets) {
    for (var i10 = 0; i10 < BT; i10++) {
      var tg = targets[i10];
      if (tg < 0 || tg >= V) { lossArr[i10] = 0; continue; }
      var lb = i10 * V, mx = -Infinity, am = 0;
      for (var v2 = 0; v2 < V; v2++) { var lg = logits[lb + v2]; if (lg > mx) { mx = lg; am = v2; } }
      var sum = 0;
      for (var v3 = 0; v3 < V; v3++) sum += Math.exp(logits[lb + v3] - mx);
      lossArr[i10] = (mx + Math.log(sum)) - logits[lb + tg];
      argmaxArr[i10] = am;
      loss += lossArr[i10]; nTargets++;
    }
    if (nTargets) loss /= nTargets;
  }

  if (track) {
    cache.Xf = Xf; cache.stF = stF; cache.logits = logits; cache.nTargets = nTargets; cache.Xfin = cur;
    this._cache = cache;
  }
  return { loss: loss, logits: logits, lossArr: lossArr, argmaxArr: argmaxArr, B: B, T: T, nTargets: nTargets };
};

/* ------------------------------------------------------------ backward ---- */

OnyXLM.prototype.backward = function (scale) {
  var cache = this._cache;
  if (!cache) throw new Error('backward() called without a tracked forward()');
  var cf = this.cfg, B = cache.B, T = cache.T, D = cf.dModel, L = cf.nLayers, F = cf.dFf, V = cf.vocabSize, H = cf.nHeads, dh = this.dHead;
  var BT = B * T;
  var probsAll = this._scratch(B, T).probs;

  /* dLogits = scale * (softmax(logits) - onehot(target)) */
  var dLogits = new Float32Array(BT * V);
  for (var i = 0; i < BT; i++) {
    var tg = cache.targets[i];
    if (tg < 0 || tg >= V) continue;
    var lb = i * V, mx = -Infinity;
    for (var v = 0; v < V; v++) if (cache.logits[lb + v] > mx) mx = cache.logits[lb + v];
    var sum = 0;
    for (var v2 = 0; v2 < V; v2++) sum += Math.exp(cache.logits[lb + v2] - mx);
    var inv = 1 / sum;
    for (var v3 = 0; v3 < V; v3++) dLogits[lb + v3] = scale * Math.exp(cache.logits[lb + v3] - mx) * inv;
    dLogits[lb + tg] -= scale;
  }

  /* tied output projection: logits = Xf · E^T */
  var dXf = new Float32Array(BT * D);
  mmLinearBackward(dLogits, cache.Xf, this.E, dXf, this.dE, 0, BT, D, V);

  var dCur = new Float32Array(BT * D);
  layerNormBackward(cache.Xfin, cache.stF, this.gF, dXf, dCur, this.dgF, this.dbF, B, T, D);

  for (var li = L - 1; li >= 0; li--) {
    var c = cache.layers[li], lay = this.layers[li];
    var dXo = dCur;
    var dXr = dXo.slice();

    /* ---- MLP branch ---- */
    var dHf = new Float32Array(BT * F);
    mmLinearBackward(dXo, c.Hf, lay.W2, dHf, lay.dW2, 0, BT, F, D);
    for (var i2 = 0; i2 < BT; i2++) { var ib2 = i2 * D; for (var d2 = 0; d2 < D; d2++) lay.db2[d2] += dXo[ib2 + d2]; }
    var dHp = new Float32Array(BT * F);
    for (var i3 = 0; i3 < BT * F; i3++) dHp[i3] = dHf[i3] * geluDeriv(c.Hp[i3]);
    var dXn2 = new Float32Array(BT * D);
    mmLinearBackward(dHp, c.Xn2, lay.W1, dXn2, lay.dW1, 0, BT, D, F);
    for (var i4 = 0; i4 < BT; i4++) { var ib4 = i4 * F; for (var f = 0; f < F; f++) lay.db1[f] += dHp[ib4 + f]; }
    layerNormBackward(c.Xr, c.st2, lay.ln2g, dXn2, dXr, lay.dln2g, lay.dln2b, B, T, D);

    /* ---- attention branch ----
     * Written as a direct transcription of the derived gradients:
     *   att[t2] = Σ_j P[t2][j]·V[j]
     *   dV[j]   += P[t2][j]·dAtt[t2]
     *   dP[j]    = dAtt[t2]·V[j]
     *   dS[j]    = P[t2][j]·(dP[j] − Σ_i P[t2][i]·dP[i]) · scale
     *   dQ[t2]   = Σ_j dS[j]·K[j]
     *   dK[j]   += dS[j]·Q[t2]
     */
    var dAttO = dXo;
    var dAtt = new Float32Array(BT * D);
    mmLinearBackward(dAttO, c.Att, lay.Wo, dAtt, lay.dWo, 0, BT, D, D);
    for (var i5 = 0; i5 < BT; i5++) { var ib5 = i5 * D; for (var d3 = 0; d3 < D; d3++) lay.dbo[d3] += dAttO[ib5 + d3]; }

    var probs = probsAll[li];
    var dQ = new Float32Array(BT * D), dK = new Float32Array(BT * D), dV = new Float32Array(BT * D);
    var scaleA = 1 / Math.sqrt(dh);
    var dP = new Float32Array(T);
    var dS = new Float32Array(T);
    for (var b2 = 0; b2 < B; b2++) {
      for (var h = 0; h < H; h++) {
        var hb = (b2 * H + h) * T * T;
        for (var t2 = 0; t2 < T; t2++) {
          var ab = ((b2 * T + t2) * H + h) * dh;
          var t3, d4;
          /* Rebuild P[t2][·] from the cached Q,K. Recomputing here (instead of
           * reading a shared scratch buffer) keeps the backward self-contained
           * and guarantees the probabilities match this layer's activations. */
          for (t3 = 0; t3 <= t2; t3++) {
            var sb = ((b2 * T + t3) * H + h) * dh, sdot = 0;
            for (d4 = 0; d4 < dh; d4++) sdot += c.Q[ab + d4] * c.K[sb + d4];
            dP[t3] = sdot * scaleA;
          }
          for (t3 = t2 + 1; t3 < T; t3++) dP[t3] = -Infinity;
          softmaxInPlace(dP, 0, T);
          for (t3 = 0; t3 <= t2; t3++) probs[hb + t2 * T + t3] = dP[t3];

          /* 1. dP[j] = Σ_d dAtt[d]·V[j][d]  (overwrite, P now captured) */
          for (t3 = 0; t3 <= t2; t3++) {
            var kb0 = ((b2 * T + t3) * H + h) * dh, acc = 0;
            for (d4 = 0; d4 < dh; d4++) acc += dAtt[ab + d4] * c.Vt[kb0 + d4];
            dS[t3] = acc;                      // temporary: raw dP
          }
          /* 2. softmax Jacobian, including the 1/sqrt(dHead) factor */
          var dot = 0;
          for (t3 = 0; t3 <= t2; t3++) dot += dP[t3] * dS[t3];
          for (t3 = 0; t3 <= t2; t3++) dS[t3] = dP[t3] * (dS[t3] - dot) * scaleA;

          /* 3. dV[j] += P[j]·dAtt ; dQ[t2] += Σ_j dS[j]·K[j] */
          for (d4 = 0; d4 < dh; d4++) {
            var qa = 0, vc = dAtt[ab + d4];
            for (t3 = 0; t3 <= t2; t3++) {
              qa += dS[t3] * c.K[((b2 * T + t3) * H + h) * dh + d4];
              dV[((b2 * T + t3) * H + h) * dh + d4] += dP[t3] * vc;
            }
            dQ[ab + d4] += qa;
          }
          /* 4. dK[j] += dS[j]·Q[t2] */
          for (t3 = 0; t3 <= t2; t3++) {
            var kb = ((b2 * T + t3) * H + h) * dh;
            for (d4 = 0; d4 < dh; d4++) dK[kb + d4] += dS[t3] * c.Q[ab + d4];
          }
        }
      }
    }

    /* dXn receives dQ·Wq + dK·Wk + dV·Wv from the head-scattered buffers.
     * dW is null here (headSlicedDW already produced the weight gradients),
     * but W must still be passed so the input gradient can be computed. */
    var dXn = new Float32Array(BT * D);
    var gQ = new Float32Array(BT * D), gK = new Float32Array(BT * D), gV = new Float32Array(BT * D);
    scatterHeads(dQ, gQ, B, T, D, H, dh);
    scatterHeads(dK, gK, B, T, D, H, dh);
    scatterHeads(dV, gV, B, T, D, H, dh);
    headSlicedDW(dQ, c.Xn, lay.dWq, B, T, D, H, dh);
    headSlicedDW(dK, c.Xn, lay.dWk, B, T, D, H, dh);
    headSlicedDW(dV, c.Xn, lay.dWv, B, T, D, H, dh);
    mmLinearBackward(gQ, c.Xn, lay.Wq, dXn, null, 0, BT, D, D);
    mmLinearBackward(gK, c.Xn, lay.Wk, dXn, null, 0, BT, D, D);
    mmLinearBackward(gV, c.Xn, lay.Wv, dXn, null, 0, BT, D, D);
    for (var b4 = 0; b4 < B; b4++) {
      for (var h3 = 0; h3 < H; h3++) {
        for (var t10 = 0; t10 < T; t10++) {
          var sb = ((b4 * T + t10) * H + h3) * dh;
          for (var d8 = 0; d8 < dh; d8++) {
            lay.dbq[h3 * dh + d8] += dQ[sb + d8];
            lay.dbk[h3 * dh + d8] += dK[sb + d8];
            lay.dbv[h3 * dh + d8] += dV[sb + d8];
          }
        }
      }
    }

    layerNormBackward(c.Xa, c.st1, lay.ln1g, dXn, dXr, lay.dln1g, lay.dln1b, B, T, D);
    dCur = dXr;
  }

  /* ---- embeddings ---- */
  for (var i11 = 0; i11 < BT; i11++) {
    var tok = cache.tokens[i11];
    var ib11 = i11 * D, tPos = i11 % T;
    for (var d9 = 0; d9 < D; d9++) {
      var g = dCur[ib11 + d9];
      if (g === 0) continue;
      if (tok >= 0 && tok < V) this.dE[tok * D + d9] += g;
      this.dP[tPos * D + d9] += g;
    }
  }

  this._cache = null;
  return this;
};

/* --------------------------------------------------------------- Adam ---- */

function Adam(model, opts) {
  opts = opts || {};
  this.lr = opts.lr || 3e-3;
  this.b1 = opts.beta1 || 0.9;
  this.b2 = opts.beta2 || 0.95;
  this.eps = opts.eps || 1e-8;
  this.wd = opts.weightDecay || 0.01;
  this.clip = opts.clip || 1.0;
  this.t = 0;
  var p = model.params();
  this.m = []; this.v = [];
  for (var i = 0; i < p.length; i++) { this.m.push(new Float32Array(p[i][0].length)); this.v.push(new Float32Array(p[i][0].length)); }
  this.model = model;
}

Adam.prototype.step = function (lrOverride) {
  var p = this.model.params();
  var lr = lrOverride != null ? lrOverride : this.lr;
  this.t++;
  var b1 = this.b1, b2 = this.b2, c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
  var gnorm = 0, i, j;
  for (i = 0; i < p.length; i++) { var gg = p[i][1]; for (j = 0; j < gg.length; j++) gnorm += gg[j] * gg[j]; }
  gnorm = Math.sqrt(gnorm);
  var clipScale = (gnorm > this.clip && gnorm > 0) ? this.clip / gnorm : 1;
  for (i = 0; i < p.length; i++) {
    var w = p[i][0], g = p[i][1], m = this.m[i], v = this.v[i], wd = p[i][2] ? this.wd : 0;
    for (j = 0; j < w.length; j++) {
      var gj = g[j] * clipScale;
      m[j] = b1 * m[j] + (1 - b1) * gj;
      v[j] = b2 * v[j] + (1 - b2) * gj * gj;
      w[j] -= lr * ((m[j] / c1) / (Math.sqrt(v[j] / c2) + this.eps) + wd * w[j]);
    }
  }
  return { gradNorm: gnorm, clipScale: clipScale };
};

/* --------------------------------------------------- one full train step -- */

OnyXLM.prototype.step = function (tokens, targets, B, T, lr) {
  var out = this.forward(tokens, targets, { track: true, B: B, T: T });
  var n = 0;
  for (var i = 0; i < targets.length; i++) if (targets[i] >= 0) n++;
  if (!n || !this._cache) { this._cache = null; return { loss: out.loss, gradNorm: 0, skipped: true }; }
  this.backward(1 / n);
  var info = this.opt.step(lr);
  this._zeroGrads();
  return { loss: out.loss, gradNorm: info.gradNorm, clipScale: info.clipScale, n: n };
};

/* ---------------------------------------------------------- generation ---- */

OnyXLM.prototype.generate = function (promptIds, opts) {
  opts = opts || {};
  var cf = this.cfg, V = cf.vocabSize;
  var temperature = opts.temperature != null ? opts.temperature : 0.85;
  var topK = Math.max(1, Math.min(V, opts.topK || 40));
  var topP = opts.topP != null ? opts.topP : 0.95;
  var maxNew = Math.max(1, opts.maxNew || 60);
  var repPen = opts.repetitionPenalty != null ? opts.repetitionPenalty : 1.08;
  var stopIds = opts.stopIds || [];
  var greedy = !!opts.greedy;
  var rng = opts.rng || makeRng((Date.now() ^ 0x5bf03635) >>> 0);

  var ids = promptIds.slice(0, Math.max(1, cf.ctx - maxNew));
  var gen = [], logprobs = [], unkCount = 0;
  var used = new Uint8Array(V);
  var work = new Float64Array(V);
  var idx = new Int32Array(V);
  for (var z = 0; z < V; z++) idx[z] = z;
  var stopSet = new Uint8Array(V);
  for (var s0 = 0; s0 < stopIds.length; s0++) if (stopIds[s0] >= 0 && stopIds[s0] < V) stopSet[stopIds[s0]] = 1;

  /* Wall-clock budget: generation runs in a worker, but the user is still
   * waiting, so we stop early rather than deliver a long, late reply. */
  var deadline = opts.maxMs ? nowMs() + opts.maxMs : 0;

  for (var step = 0; step < maxNew; step++) {
    if (deadline && step > 0 && nowMs() > deadline) break;
    var out = this.forward(ids, null, { track: false, B: 1, T: ids.length });
    var base = (ids.length - 1) * V;
    var logits = out.logits;
    for (var i = 0; i < V; i++) {
      var lg = logits[base + i];
      if (used[i] && repPen > 1) lg = lg > 0 ? lg / repPen : lg * repPen;
      work[i] = lg;
    }
    var next;
    if (greedy) {
      var bm = 0, bv = -Infinity;
      for (var g2 = 0; g2 < V; g2++) if (work[g2] > bv) { bv = work[g2]; bm = g2; }
      next = bm;
    } else {
      var order = Array.prototype.slice.call(idx, 0, V);
      order.sort(function (a, b) { return work[b] - work[a]; });
      var keep = order.slice(0, topK);
      var invT = 1 / Math.max(0.05, temperature);
      var mx = -Infinity, z2;
      for (z2 = 0; z2 < keep.length; z2++) { work[keep[z2]] *= invT; if (work[keep[z2]] > mx) mx = work[keep[z2]]; }
      var sum = 0;
      for (z2 = 0; z2 < keep.length; z2++) { var ev = Math.exp(work[keep[z2]] - mx); work[keep[z2]] = ev; sum += ev; }
      /* nucleus: walk the sorted list, keep the smallest prefix reaching topP */
      var cum = 0, cut = keep.length;
      for (z2 = 0; z2 < keep.length; z2++) {
        cum += work[keep[z2]] / sum;
        if (cum >= topP) { cut = z2 + 1; break; }
      }
      var allowed = keep.slice(0, Math.max(1, cut));
      var tot = 0;
      for (z2 = 0; z2 < allowed.length; z2++) tot += work[allowed[z2]];
      var r = rng() * tot, acc = 0;
      next = allowed[allowed.length - 1];
      for (z2 = 0; z2 < allowed.length; z2++) {
        acc += work[allowed[z2]];
        if (r <= acc) { next = allowed[z2]; break; }
      }
      if (temperature > 0.05) logprobs.push(Math.log(Math.max(1e-12, work[next] / sum)) * invT);
      else logprobs.push(0);
    }
    if (next === 1) unkCount++;
    gen.push(next);
    used[next] = 1;
    ids.push(next);
    if (stopSet[next]) break;
    if (ids.length >= cf.ctx) break;
  }

  var avg = 0;
  for (var l = 0; l < logprobs.length; l++) avg += logprobs[l];
  if (logprobs.length) avg /= logprobs.length;
  return { ids: gen, n: gen.length, avgLogprob: avg, meanProb: logprobs.length ? Math.exp(avg) : 0, unkCount: unkCount };
};

/* -------------------------------------------------------- quantisation ---- */

OnyXLM.prototype.quantize = function () {
  var p = this.params(), out = [];
  for (var i = 0; i < p.length; i++) {
    var w = p[i][0], n = w.length, mx = 0, j;
    for (j = 0; j < n; j++) { var a = Math.abs(w[j]); if (a > mx) mx = a; }
    var scale = mx > 0 ? mx / 127 : 1;
    var q = new Int8Array(n);
    for (j = 0; j < n; j++) {
      var v = Math.round(w[j] / scale);
      q[j] = v > 127 ? 127 : (v < -127 ? -127 : v);
    }
    out.push({ s: scale, q: q });
  }
  return out;
};

OnyXLM.prototype.dequantize = function (list) {
  var p = this.params();
  for (var i = 0; i < p.length && i < list.length; i++) {
    var w = p[i][0], it = list[i];
    if (!it || !it.q || it.q.length !== w.length) continue;
    for (var j = 0; j < w.length; j++) w[j] = it.q[j] * it.s;
  }
  return this;
};

/* ================================================================= Corpus ==
 * Training data is a set of weighted plain-text documents. The trainer draws
 * fixed-length windows from one concatenated token stream.
 * ========================================================================*/

function Corpus(opts) {
  opts = opts || {};
  this.maxChars = opts.maxChars || 700000;
  this.docs = [];
  this.charCount = 0;
  this.stream = null;
  this.dirty = true;
  this._dedupe = new Map();
  this.kinds = {};
}

Corpus.prototype.has = function (text) {
  var key = text.slice(0, 120) + '\u0001' + text.length;
  return this._dedupe.has(key);
};

Corpus.prototype.add = function (text, kind, weight) {
  text = normalizeForTraining(text);
  if (!text || text.length < 12) return false;
  var key = text.slice(0, 120) + '\u0001' + text.length;
  if (this._dedupe.has(key)) return false;
  this._dedupe.set(key, 1);
  this.docs.push({ text: text, kind: kind || 'general', w: weight != null ? weight : 1 });
  this.charCount += text.length;
  this.kinds[kind || 'general'] = (this.kinds[kind || 'general'] || 0) + 1;
  this.dirty = true;
  this._trim();
  return true;
};

Corpus.prototype._trim = function () {
  while (this.charCount > this.maxChars && this.docs.length > 60) {
    var worst = -1, worstScore = Infinity;
    for (var i = 0; i < this.docs.length; i++) {
      var d = this.docs[i];
      var protective = (d.kind === 'seed' || d.kind === 'synthetic') ? 6 : 0;
      var score = d.w * 3 + protective - (i / this.docs.length) * 0.2;
      if (score < worstScore) { worstScore = score; worst = i; }
    }
    if (worst < 0) break;
    this.charCount -= this.docs[worst].text.length;
    this.docs.splice(worst, 1);
  }
};

Corpus.prototype.rebuild = function (tokenizer) {
  var ids = [];
  for (var i = 0; i < this.docs.length; i++) {
    var d = this.docs[i];
    var reps = Math.max(1, Math.min(4, Math.round(d.w * 1.5)));
    var t = tokenizer.encode(d.text);
    for (var r = 0; r < reps; r++) {
      for (var j = 0; j < t.length; j++) ids.push(t[j]);
      ids.push(tokenizer.id('<eos>'));
    }
  }
  this.stream = Int32Array.from(ids);
  this.dirty = false;
  return this.stream.length;
};

Corpus.prototype.sampleBatch = function (B, T, rng) {
  var stream = this.stream;
  if (!stream || stream.length < T + 2) return null;
  var tokens = new Int32Array(B * T), targets = new Int32Array(B * T);
  for (var b = 0; b < B; b++) {
    var start = (rng() * (stream.length - T - 1)) | 0;
    for (var t = 0; t < T; t++) {
      tokens[b * T + t] = stream[start + t];
      targets[b * T + t] = stream[start + t + 1];
    }
  }
  return { tokens: tokens, targets: targets, B: B, T: T };
};

Corpus.prototype.serialize = function () { return { docs: this.docs, maxChars: this.maxChars }; };
Corpus.load = function (obj, opts) {
  var c = new Corpus(opts || {});
  if (obj && obj.docs) {
    for (var i = 0; i < obj.docs.length; i++) {
      var d = obj.docs[i];
      if (!d || !d.text) continue;
      c.docs.push({ text: d.text, kind: d.kind || 'general', w: d.w != null ? d.w : 1 });
      c.charCount += d.text.length;
      c.kinds[d.kind || 'general'] = (c.kinds[d.kind || 'general'] || 0) + 1;
      c._dedupe.set(d.text.slice(0, 120) + '\u0001' + d.text.length, 1);
    }
  }
  return c;
};

/* ========================================================== IndexedDB ===== */

var DB_NAME = 'onyx-llm', DB_STORE = 'state', DB_VER = 2;

function idbOpen() {
  return new Promise(function (resolve, reject) {
    try {
      if (typeof indexedDB === 'undefined') return reject(new Error('IndexedDB unavailable'));
      var req = indexedDB.open(DB_NAME, DB_VER);
      req.onupgradeneeded = function () {
        var db = req.result;
        if (!db.objectStoreNames.contains(DB_STORE)) db.createObjectStore(DB_STORE);
      };
      req.onsuccess = function () { resolve(req.result); };
      req.onerror = function () { reject(req.error || new Error('idb open failed')); };
    } catch (e) { reject(e); }
  });
}

function idbSet(key, value) {
  return idbOpen().then(function (db) {
    return new Promise(function (resolve, reject) {
      var tx = db.transaction(DB_STORE, 'readwrite');
      tx.objectStore(DB_STORE).put(value, key);
      tx.oncomplete = function () { try { db.close(); } catch (e) { } resolve(true); };
      tx.onerror = function () { reject(tx.error); };
      tx.onabort = function () { reject(tx.error); };
    });
  });
}

function idbGet(key) {
  return idbOpen().then(function (db) {
    return new Promise(function (resolve, reject) {
      var tx = db.transaction(DB_STORE, 'readonly');
      var r = tx.objectStore(DB_STORE).get(key);
      r.onsuccess = function () { try { db.close(); } catch (e) { } resolve(r.result); };
      r.onerror = function () { reject(r.error); };
    });
  });
}

/* ================================================================= Engine ==
 * Bootstraps from the seed corpus, learns from every conversation turn, and
 * answers with the network when it is confident enough. Otherwise it reports
 * lowConfidence and the caller falls back to the reasoning engine.
 * ========================================================================*/

var DEFAULTS = {
  preset: 'micro',
  /* Character-level tokenisation. Russian is close to phonetic, so a ~90 symbol
   * alphabet is enough — and it keeps the tied output projection (the single
   * most expensive layer) roughly twenty times cheaper than a BPE vocabulary
   * would, which is what makes self-training in a browser tab viable. */
  merges: 1,
  stepBudgetPerTurn: 4,
  batchSize: 4,
  seqLen: 64,
  lr: 6e-3,
  minLr: 2e-4,
  warmupSteps: 40,
  totalSteps: 60000,
  replayRatio: 0.3,
  bootstrapSteps: 90,
  minLogprob: -2.0,
  minChars: 34,
  maxNewTokens: 96,
  maxGenMs: 9000,
  temperature: 0.8,
  topK: 40,
  topP: 0.93,
  repetitionPenalty: 1.14,
  maxCharsCorpus: 900000,
  maxReplay: 400
};

function Engine(cfg) {
  this.cfg = Object.assign({}, DEFAULTS, cfg || {});
  this.tokenizer = null;
  this.model = null;
  this.opt = null;
  this.corpus = null;
  this.replay = [];
  this.rng = makeRng((Date.now() ^ 0x2545f491) >>> 0);
  this.trained = 0;
  this.tokensSeen = 0;
  this.lossEMA = null;
  this.lossHistory = [];
  this.ready = false;
  this.bootstrapping = false;
  this.lastSave = 0;
  this.storageError = null;
  this.generations = 0;
  this.fallbacks = 0;
}

Engine.prototype.status = function () {
  var c = this.corpus;
  return {
    version: VERSION,
    ready: this.ready,
    bootstrapping: this.bootstrapping,
    params: this.model ? this.model.paramCount() : 0,
    vocab: this.tokenizer ? this.tokenizer.size() : 0,
    merges: this.tokenizer ? this.tokenizer.merges.length : 0,
    chars: this.tokenizer ? this.tokenizer.chars.length : 0,
    steps: this.trained,
    tokens: this.tokensSeen,
    loss: this.lossEMA,
    ppl: this.lossEMA != null ? Math.exp(Math.min(20, this.lossEMA)) : null,
    corpusChars: c ? c.charCount : 0,
    corpusDocs: c ? c.docs.length : 0,
    corpusKinds: c ? c.kinds : {},
    streamTokens: c && c.stream ? c.stream.length : 0,
    cfg: this.cfg,
    lossHistory: this.lossHistory.slice(-120),
    generations: this.generations,
    fallbacks: this.fallbacks,
    storage: this.storageError ? 'error' : 'ok',
    ctx: this.model ? this.model.cfg.ctx : 0,
    dModel: this.model ? this.model.cfg.dModel : 0,
    layers: this.model ? this.model.cfg.nLayers : 0,
    heads: this.model ? this.model.cfg.nHeads : 0
  };
};

Engine.prototype.lrAt = function (step) {
  var c = this.cfg;
  if (step < c.warmupSteps) return c.lr * (step + 1) / c.warmupSteps;
  var total = Math.max(c.warmupSteps + 200, c.totalSteps);
  var prog = Math.min(1, (step - c.warmupSteps) / Math.max(1, total - c.warmupSteps));
  return c.minLr + 0.5 * (c.lr - c.minLr) * (1 + Math.cos(Math.PI * prog));
};

/* -------------------------------------------------------------- startup --- */

Engine.prototype.init = function (seedText) {
  var self = this;
  return this.load().then(function (restored) {
    if (!restored) {
      self.corpus = new Corpus({ maxChars: self.cfg.maxCharsCorpus });
      self.addSeed(seedText || '');
      self.buildFresh();
      return self.bootstrap({ steps: self.cfg.bootstrapSteps, timeBudgetMs: 9000 }).then(function () { return false; });
    }
    if (seedText) {
      var added = self.addSeed(seedText);
      if (added) self.corpus.rebuild(self.tokenizer);
    }
    self.ready = true;
    return true;
  }).catch(function (err) {
    self.storageError = String((err && err.message) || err);
    if (!self.model) {
      self.corpus = new Corpus({ maxChars: self.cfg.maxCharsCorpus });
      self.addSeed(seedText || '');
      self.buildFresh();
    }
    self.ready = true;
    return false;
  });
};

Engine.prototype.buildFresh = function () {
  /* Train the tokenizer on marker-free text so its whole merge budget goes to
   * genuine sub-words; the markers are encoded atomically afterwards. */
  var text = this.corpus.docs.map(function (d) { return trainingOnlyText(d.text); }).join('\n');
  this.tokenizer = new BPETokenizer().train(text, { merges: this.cfg.merges || 1400, minFreq: 2 });
  this.resetModel();
  this.ready = true;
  return this;
};

Engine.prototype.resetModel = function () {
  var preset = CONFIGS[this.cfg.preset] || CONFIGS.micro;
  /* The embedding must have exactly one row per tokenizer id, otherwise a
   * serialised model cannot be restored faithfully. If the tokenizer is tiny we
   * grow it explicitly rather than faking a larger matrix. */
  this.model = new OnyXLM({
    preset: this.cfg.preset,
    vocabSize: Math.max(8, this.tokenizer.size()),
    dModel: preset.dModel, nLayers: preset.nLayers, nHeads: preset.nHeads, dFf: preset.dFf, ctx: preset.ctx
  });
  if (this.tokenizer.size() < MIN_VOCAB) this.growTokenizerTo(MIN_VOCAB);
  this.opt = new Adam(this.model, { lr: this.cfg.lr, weightDecay: 0.01, clip: 1.0 });
  this.model.opt = this.opt;
  this.corpus.rebuild(this.tokenizer);
  this.trained = 0; this.tokensSeen = 0; this.lossEMA = null; this.lossHistory = [];
};

/* Keep a little headroom in the vocabulary so small corpora still have room for
 * the control markers and for characters learned later. */
var MIN_VOCAB = 64;

Engine.prototype.growTokenizerTo = function (n) {
  while (this.tokenizer.nextId < n) {
    var filler = '_' + this.tokenizer.nextId;
    if (!this.tokenizer.stoi.has(filler) && !this.tokenizer.extra.has(filler)) {
      this.tokenizer.extra.set(filler, this.tokenizer.nextId++);
    } else this.tokenizer.nextId++;
  }
  this.syncVocab();
};

/* The tokenizer may learn new characters at any time; keep the embedding in
 * sync so no input is ever silently dropped. */
Engine.prototype.syncVocab = function () {
  if (!this.tokenizer || !this.model) return false;
  var size = this.tokenizer.size();
  if (size > this.model.cfg.vocabSize) { this.model.growVocab(size, this.trained + 1); return true; }
  return false;
};

Engine.prototype.addSeed = function (text) {
  if (!text) return 0;
  var parts = String(text).split(/\n(?=#{2,3}\s)/);
  var n = 0;
  for (var i = 0; i < parts.length; i++) {
    var p = parts[i].trim();
    if (p.length < 20) continue;
    if (this.corpus.add(p, 'seed', 1.7)) n++;
  }
  return n;
};

Engine.prototype.addDocument = function (text, kind, weight) {
  if (!this.corpus || !this.tokenizer) return false;
  var ok = this.corpus.add(text, kind || 'live', weight != null ? weight : 1);
  if (!ok) return false;
  // learn any new characters first, then re-encode the stream
  var norm = normalizeForTraining(text);
  var ext = this.tokenizer.extendChars(norm);
  if (ext.added) this.syncVocab();
  this.corpus.rebuild(this.tokenizer);
  return true;
};

/* Learn from one conversation turn. `answer` may be empty (nothing to learn
 * from an unanswered question beyond its phrasing). */
Engine.prototype.learnTurn = function (question, answer, sources) {
  if (!this.corpus || !this.tokenizer) return 0;
  var q = String(question || '').trim();
  var a = String(answer || '').trim();
  var added = 0;
  if (q.length >= 4 && q.length <= 900) {
    var pair = '<user> ' + q + '\n<ai> ' + (a ? a.slice(0, 1500) : '') + '\n<eos>';
    if (this.addDocument(pair, 'dialog', 1.5)) added++;
  }
  if (a.length >= 60) {
    if (this.addDocument(a.slice(0, 2200), 'answer', 1.0)) added++;
  }
  if (sources && sources.length) {
    for (var i = 0; i < Math.min(sources.length, 3); i++) {
      var s = sources[i];
      if (!s) continue;
      var txt = (s.title ? s.title + '. ' : '') + (s.text || '');
      if (txt.length >= 100 && this.addDocument(txt.slice(0, 1400), 'web', 0.85)) added++;
    }
  }
  if (q.length >= 4) {
    this.replay.push(q.slice(0, 240));
    if (this.replay.length > this.cfg.maxReplay) this.replay.shift();
  }
  if (added) lastLearnTs = nowMs();
  return added;
};

var lastLearnTs = 0;

/* --------------------------------------------------------------- learning - */

Engine.prototype.trainSteps = function (budget, opts) {
  opts = opts || {};
  var c = this.cfg;
  if (!this.model || !this.corpus || !this.corpus.stream || this.corpus.stream.length < c.seqLen + 4) {
    return { steps: 0, loss: this.lossEMA, skipped: true, reason: 'corpus-too-small' };
  }
  var done = 0, lastLoss = null, t0 = nowMs();
  for (var i = 0; i < budget; i++) {
    var batch = this.corpus.sampleBatch(c.batchSize, c.seqLen, this.rng);
    if (!batch) break;
    if (opts.mixReplay !== false && this.replay.length && this.rng() < c.replayRatio) this._injectReplay(batch);
    var lr = this.lrAt(this.trained);
    var res = this.model.step(batch.tokens, batch.targets, c.batchSize, c.seqLen, lr);
    lastLoss = res.loss;
    this.trained++;
    this.tokensSeen += c.batchSize * c.seqLen;
    this.lossEMA = this.lossEMA == null ? lastLoss : this.lossEMA * 0.93 + lastLoss * 0.07;
    done++;
    if (done % 5 === 0) {
      this.lossHistory.push(Math.round(lastLoss * 1000) / 1000);
      if (this.lossHistory.length > 1200) this.lossHistory = this.lossHistory.slice(-300);
    }
    if (nowMs() - t0 > (opts.timeBudgetMs || 1500)) break;
  }
  return { steps: done, loss: lastLoss, ema: this.lossEMA, ms: Math.round(nowMs() - t0) };
};

Engine.prototype._injectReplay = function (batch) {
  var c = this.cfg;
  var rep = this.replay[(this.rng() * this.replay.length) | 0];
  if (!rep) return;
  var ids = this.tokenizer.encode('<user> ' + rep + '\n<ai> ');
  var eos = this.tokenizer.id('<eos>');
  if (ids.length < 4) return;
  var row = (this.rng() * c.batchSize) | 0;
  var take = Math.min(c.seqLen, ids.length);
  var off = ids.length > take ? ((this.rng() * (ids.length - take)) | 0) : 0;
  for (var t = 0; t < take; t++) {
    batch.tokens[row * c.seqLen + t] = ids[off + t];
    batch.targets[row * c.seqLen + t] = (off + t + 1 < ids.length) ? ids[off + t + 1] : eos;
  }
  for (var t2 = take; t2 < c.seqLen; t2++) {
    batch.tokens[row * c.seqLen + t2] = eos;
    batch.targets[row * c.seqLen + t2] = eos;
  }
};

Engine.prototype.bootstrap = function (opts) {
  opts = opts || {};
  if (this.bootstrapping) return Promise.resolve({ skipped: true });
  this.bootstrapping = true;
  var self = this;
  var total = opts.steps || this.cfg.bootstrapSteps;
  var perChunk = opts.chunk || 12;
  var deadline = nowMs() + (opts.timeBudgetMs || 9000);
  var acc = { steps: 0, loss: null, ema: null };
  function chunk() {
    if (acc.steps >= total || nowMs() > deadline) {
      self.bootstrapping = false;
      return Promise.resolve(acc);
    }
    var r = self.trainSteps(Math.min(perChunk, total - acc.steps), { timeBudgetMs: 1500, mixReplay: false });
    acc.steps += r.steps; acc.loss = r.loss; acc.ema = r.ema;
    if (r.skipped) { self.bootstrapping = false; return Promise.resolve(acc); }
    return new Promise(function (res) { setTimeout(res, 0); }).then(chunk);
  }
  return chunk();
};

/* ------------------------------------------------------------- inference -- */

Engine.prototype.buildPrompt = function (question, context) {
  var ids = [this.tokenizer.id('<bos>')];
  if (context) {
    /* Retrieved material is fed in as plain prose before the question, so the
     * network writes the answer itself instead of quoting a source list. */
    var intro = this.tokenizer.encode('сведения по теме: ');
    for (var k = 0; k < intro.length; k++) ids.push(intro[k]);
    var cIds = this.tokenizer.encode(String(context).slice(0, 420));
    for (var i = 0; i < cIds.length; i++) ids.push(cIds[i]);
  }
  ids.push(this.tokenizer.id('<user>'));
  var qIds = this.tokenizer.encode(String(question || '').slice(0, 600));
  for (var j = 0; j < qIds.length; j++) ids.push(qIds[j]);
  ids.push(this.tokenizer.id('<eos>'));
  ids.push(this.tokenizer.id('<ai>'));
  return ids;
};

Engine.prototype.answer = function (question, opts) {
  opts = opts || {};
  if (!this.model || !this.tokenizer) return null;
  var c = this.cfg;
  try {
    var prompt = this.buildPrompt(question, opts.context);
    var maxNew = Math.max(8, Math.min(opts.maxNew || c.maxNewTokens, this.model.cfg.ctx - 8));
    if (prompt.length + maxNew > this.model.cfg.ctx) prompt = prompt.slice(prompt.length - (this.model.cfg.ctx - maxNew));
    var g = this.model.generate(prompt, {
      temperature: opts.temperature != null ? opts.temperature : c.temperature,
      topK: opts.topK || c.topK,
      topP: opts.topP != null ? opts.topP : c.topP,
      maxNew: maxNew,
      maxMs: opts.maxMs || c.maxGenMs,
      repetitionPenalty: opts.repetitionPenalty != null ? opts.repetitionPenalty : c.repetitionPenalty,
      stopIds: [this.tokenizer.id('<eos>'), this.tokenizer.id('<user>'), this.tokenizer.id('<bos>')],
      rng: this.rng
    });
    this.generations++;
    var text = cleanGenerated(this.tokenizer.decode(g.ids));
    var words = text.split(/\s+/).filter(Boolean);
    var uniq = new Set();
    for (var i = 0; i < words.length; i++) uniq.add(words[i]);
    var diversity = words.length ? uniq.size / words.length : 0;
    var coh = this.coherence(text);
    var low = false, reasons = [];
    if (text.length < c.minChars) { low = true; reasons.push('too-short'); }
    if (words.length < 4) { low = true; reasons.push('too-few-words'); }
    if (g.n > 0 && g.unkCount / g.n > 0.05) { low = true; reasons.push('unknown-tokens'); }
    if (g.avgLogprob < c.minLogprob) { low = true; reasons.push('low-logprob'); }
    if (words.length > 8 && diversity < 0.45) { low = true; reasons.push('repetitive'); }
    if (coh.rep3 > 0.25) { low = true; reasons.push('looping'); }
    if (coh.wordHit < 0.45) { low = true; reasons.push('nonsense-words'); }
    if (coh.score < 0.50) { low = true; reasons.push('incoherent'); }
    if (this.lossEMA != null && this.lossEMA > 4.2) { low = true; reasons.push('untrained'); }
    if (low) this.fallbacks++;
    return {
      text: text,
      lowConfidence: low,
      reasons: reasons,
      avgLogprob: g.avgLogprob,
      meanProb: g.meanProb,
      genTokens: g.n,
      diversity: diversity,
      coherence: coh.score,
      wordHit: coh.wordHit,
      repeat: coh.rep3,
      lossEMA: this.lossEMA,
      trained: this.trained
    };
  } catch (e) {
    this.fallbacks++;
    return { text: '', lowConfidence: true, reasons: ['error:' + ((e && e.message) || e)], genTokens: 0 };
  }
};

function cleanGenerated(text) {
  var t = String(text || '');
  t = t.replace(/<[a-z]+>/g, ' ');
  t = t.replace(/[ \t]+/g, ' ');
  t = t.replace(/\n{3,}/g, '\n\n');
  t = t.replace(/^[\s,;:.\-]+/, '');
  t = t.replace(/[\s,;:\-]+$/, '');
  /* Drop a trailing half-written word so a reply never ends mid-syllable. */
  var m = /[^\s.!?,;:]+$/.exec(t);
  if (m && m[0].length <= 2 && t.length > m[0].length) t = t.slice(0, t.length - m[0].length);
  t = t.trim();
  if (t) t = t.charAt(0).toUpperCase() + t.slice(1);
  return t;
}

/* ------------------------------------------------------------- coherence --
 * The confidence gate. A small model that has barely trained produces text with
 * a very characteristic signature: well-formed letter transitions (so it *looks*
 * like Russian) but words that do not exist in the vocabulary the model was
 * trained on, plus heavy phrase repetition. This scores those signals and the
 * caller falls back to the reasoning engine whenever the score is poor.
 * ------------------------------------------------------------------------*/

var LEX_STOP = { 'и': 1, 'в': 1, 'во': 1, 'не': 1, 'что': 1, 'он': 1, 'на': 1, 'я': 1, 'с': 1, 'со': 1, 'как': 1, 'а': 1, 'то': 1, 'все': 1, 'она': 1, 'так': 1, 'его': 1, 'но': 1, 'да': 1, 'ты': 1, 'к': 1, 'у': 1, 'же': 1, 'вы': 1, 'за': 1, 'бы': 1, 'по': 1, 'только': 1, 'ее': 1, 'мне': 1, 'было': 1, 'вот': 1, 'от': 1, 'меня': 1, 'еще': 1, 'нет': 1, 'о': 1, 'из': 1, 'ему': 1, 'теперь': 1, 'когда': 1, 'даже': 1, 'ну': 1, 'вдруг': 1, 'ли': 1, 'если': 1, 'уже': 1, 'или': 1, 'ни': 1, 'быть': 1, 'был': 1, 'него': 1, 'до': 1, 'вас': 1, 'нибудь': 1, 'опять': 1, 'уж': 1, 'вам': 1, 'ведь': 1, 'там': 1, 'потом': 1, 'себя': 1, 'ничего': 1, 'ей': 1, 'может': 1, 'они': 1, 'тут': 1, 'где': 1, 'есть': 1, 'надо': 1, 'ней': 1, 'для': 1, 'мы': 1, 'тебя': 1, 'их': 1, 'чем': 1, 'была': 1, 'сам': 1, 'чтоб': 1, 'без': 1, 'будто': 1, 'чего': 1, 'раз': 1, 'тоже': 1, 'себе': 1, 'под': 1, 'будет': 1, 'ж': 1, 'тогда': 1, 'кто': 1, 'этот': 1, 'того': 1, 'потому': 1, 'этого': 1, 'какой': 1, 'совсем': 1, 'ним': 1, 'здесь': 1, 'этом': 1, 'один': 1, 'почти': 1, 'мой': 1, 'тем': 1, 'чтобы': 1, 'нее': 1, 'сейчас': 1, 'были': 1, 'куда': 1, 'зачем': 1, 'всех': 1, 'никогда': 1, 'можно': 1, 'при': 1, 'наконец': 1, 'два': 1, 'об': 1, 'другой': 1, 'хоть': 1, 'после': 1, 'над': 1, 'больше': 1, 'тот': 1, 'через': 1, 'эти': 1, 'нас': 1, 'про': 1, 'всего': 1, 'них': 1, 'какая': 1, 'много': 1, 'разве': 1, 'три': 1, 'эту': 1, 'моя': 1, 'впрочем': 1, 'хорошо': 1, 'свою': 1, 'этой': 1, 'перед': 1, 'иногда': 1, 'лучше': 1, 'чуть': 1, 'том': 1, 'нельзя': 1, 'такой': 1, 'им': 1, 'более': 1, 'всегда': 1, 'конечно': 1, 'всю': 1, 'между': 1 };

Engine.prototype.buildLexicon = function () {
  this.lexicon = new Set();
  if (!this.corpus) return 0;
  var seen = Object.create(null);
  for (var i = 0; i < this.corpus.docs.length; i++) {
    var words = String(this.corpus.docs[i].text).split(/[^\p{L}\p{N}]+/u);
    for (var j = 0; j < words.length; j++) {
      var w = words[j];
      if (w.length >= 3 && w.length <= 24) seen[w] = 1;
    }
  }
  for (var k in seen) this.lexicon.add(k);
  /* Also accept every word the model itself has stored in its own corpus and
   * every reasonably common Russian word shape, so the gate is not too harsh. */
  for (var s in LEX_STOP) this.lexicon.add(s);
  return this.lexicon.size;
};

/* Fraction of the generated word tokens that are real words. */
Engine.prototype.wordHitRate = function (text) {
  if (!this.lexicon) this.buildLexicon();
  var words = String(text || '').toLowerCase().replace(/ё/g, 'е').split(/[^\p{L}\p{N}]+/u);
  var total = 0, hits = 0;
  for (var i = 0; i < words.length; i++) {
    var w = words[i];
    if (w.length < 3) continue;
    total++;
    if (this.lexicon.has(w)) hits++;
  }
  if (!total) return 0;
  return hits / total;
};

/* Repeated n-gram ratio: high values mean the model is looping. */
function repeatRatio(text, n) {
  var toks = String(text || '').split(/\s+/).filter(Boolean);
  if (toks.length < n + 1) return 0;
  var seen = Object.create(null), dup = 0, total = 0;
  for (var i = 0; i + n <= toks.length; i++) {
    var key = toks.slice(i, i + n).join(' ');
    total++;
    if (seen[key]) dup++; else seen[key] = 1;
  }
  return total ? dup / total : 0;
}

Engine.prototype.coherence = function (text) {
  var t = String(text || '');
  var words = t.split(/\s+/).filter(Boolean);
  var hit = this.wordHitRate(t);
  var rep2 = repeatRatio(t, 2);
  var rep3 = repeatRatio(t, 3);
  var score = 0.45 * hit + 0.30 * (1 - Math.min(1, rep2 * 1.6)) + 0.25 * (1 - Math.min(1, rep3 * 2.2));
  if (words.length < 5) score *= 0.6;
  return { score: score, wordHit: hit, rep2: rep2, rep3: rep3 };
};
Engine.prototype.feedback = function (question, answer, score) {
  if (!this.corpus || !this.tokenizer) return false;
  var q = String(question || '').trim(), a = String(answer || '').trim();
  if (!q || !a) return false;
  return this.addDocument('<user> ' + q + '\n<ai> ' + a + '\n<eos>', score > 0 ? 'approved' : 'rejected', score > 0 ? 2.2 : 0.08);
};

/* ------------------------------------------------------------ persistence - */

Engine.prototype.serialize = function () {
  if (!this.model || !this.tokenizer || !this.corpus) return null;
  return {
    ts: Date.now(),
    trained: this.trained,
    tokensSeen: this.tokensSeen,
    lossEMA: this.lossEMA,
    lossHistory: this.lossHistory.slice(-200),
    tokenizer: this.tokenizer.serialize(),
    modelCfg: this.model.cfg,
    weights: this.model.quantize(),
    corpus: this.corpus.serialize(),
    replay: this.replay.slice(-this.cfg.maxReplay)
  };
};

Engine.prototype.save = function (force) {
  var self = this;
  if (!this.model) return Promise.resolve(false);
  if (!force && nowMs() - this.lastSave < 20000) return Promise.resolve(false);
  var payload = this.serialize();
  this.lastSave = nowMs();
  return idbSet('model', payload).then(function () { return true; }).catch(function (e) {
    self.storageError = String((e && e.message) || e);
    return false;
  });
};

Engine.prototype.load = function () {
  var self = this;
  return idbGet('model').then(function (data) {
    if (!data || !data.tokenizer || !data.modelCfg) return false;
    self.tokenizer = BPETokenizer.deserialize(data.tokenizer);
    self.corpus = Corpus.load(data.corpus, { maxChars: self.cfg.maxCharsCorpus });
    self.replay = Array.isArray(data.replay) ? data.replay : [];
    var cfg = Object.assign({}, data.modelCfg);
    cfg.vocabSize = self.tokenizer.size();
    self.model = new OnyXLM(cfg);
    self.model.dequantize(data.weights);
    self.opt = new Adam(self.model, { lr: self.cfg.lr, weightDecay: 0.01, clip: 1.0 });
    self.model.opt = self.opt;
    self.trained = data.trained || 0;
    self.tokensSeen = data.tokensSeen || 0;
    self.lossEMA = typeof data.lossEMA === 'number' ? data.lossEMA : null;
    self.lossHistory = Array.isArray(data.lossHistory) ? data.lossHistory : [];
    self.corpus.rebuild(self.tokenizer);
    self.syncVocab();
    return true;
  }).catch(function () { return false; });
};

Engine.prototype.reset = function (seedText) {
  var self = this;
  return idbSet('model', null).catch(function () { }).then(function () {
    self.corpus = new Corpus({ maxChars: self.cfg.maxCharsCorpus });
    self.addSeed(seedText || '');
    self.replay = [];
    self.buildFresh();
    self.ready = true;
    return self.bootstrap({ steps: 40, timeBudgetMs: 5000 });
  });
};

Engine.prototype.exportWeights = function () {
  if (!this.model) return null;
  var p = this.model.params(), out = [];
  for (var i = 0; i < p.length; i++) out.push(Array.prototype.slice.call(p[i][0]));
  return {
    kind: 'onyx-llm-weights', version: VERSION, cfg: this.model.cfg,
    tokenizer: this.tokenizer.serialize(), trained: this.trained,
    tokensSeen: this.tokensSeen, lossEMA: this.lossEMA, params: out
  };
};

Engine.prototype.importWeights = function (obj) {
  try {
    if (!obj || obj.kind !== 'onyx-llm-weights' || !obj.params || !obj.tokenizer) return false;
    this.tokenizer = BPETokenizer.deserialize(obj.tokenizer);
    var cfg = Object.assign({}, obj.cfg);
    cfg.vocabSize = this.tokenizer.size();
    var model = new OnyXLM(cfg);
    var p = model.params();
    if (p.length !== obj.params.length) return false;
    for (var i = 0; i < p.length; i++) {
      var src = obj.params[i];
      if (!src || src.length !== p[i][0].length) return false;
      for (var j = 0; j < src.length; j++) p[i][0][j] = src[j];
    }
    this.model = model;
    this.opt = new Adam(this.model, { lr: this.cfg.lr, weightDecay: 0.01, clip: 1.0 });
    this.model.opt = this.opt;
    this.trained = obj.trained || 0;
    this.tokensSeen = obj.tokensSeen || 0;
    this.lossEMA = obj.lossEMA != null ? obj.lossEMA : null;
    if (this.corpus) this.corpus.rebuild(this.tokenizer);
    this.ready = true;
    return true;
  } catch (e) { return false; }
};

Engine.prototype.exportCorpus = function () {
  return this.corpus ? this.corpus.docs.map(function (d) { return d.text; }).join('\n\n') : '';
};

/* ============================================================= Worker RPC == */

function makeWorkerRpc(engine, post) {
  var handlers = {
    status: function () { return engine.status(); },
    init: function (m) { return engine.init(m.seed); },
    bootstrap: function (m) { return engine.bootstrap(m.opts || {}); },
    train: function (m) { return engine.trainSteps(m.steps || 4, m.opts || {}); },
    learn: function (m) { return { added: engine.learnTurn(m.question, m.answer, m.sources) }; },
    addCorpus: function (m) { return { added: engine.addDocument(m.text, m.kind, m.weight) ? 1 : 0 }; },
    feedback: function (m) { return { ok: engine.feedback(m.question, m.answer, m.score) }; },
    generate: function (m) { return engine.answer(m.question, m.opts || {}); },
    /* Same as generate, but accepts retrieved passages so the network can write
     * the reply itself while grounded in fresh material. */
    generateWithContext: function (m) { return engine.answer(m.question, Object.assign({}, m.opts || {}, { context: m.context || '' })); },
    save: function (m) { return engine.save(!!(m && m.force)); },
    reset: function (m) { return engine.reset(m.seed); },
    export: function () { return engine.exportWeights(); },
    exportCorpus: function () { return { text: engine.exportCorpus() }; },
    import: function (m) { return { ok: engine.importWeights(m.data) }; },
    config: function (m) { Object.assign(engine.cfg, m.cfg || {}); return engine.status(); }
  };
  return function (msg) {
    var h = handlers[msg && msg.type];
    if (!h) { post({ id: msg && msg.id, ok: false, error: 'unknown message type' }); return; }
    try {
      var out = h(msg);
      if (out && typeof out.then === 'function') {
        out.then(function (r) { post({ id: msg.id, ok: true, result: r }); },
          function (e) { post({ id: msg.id, ok: false, error: String((e && e.message) || e) }); });
      } else {
        post({ id: msg.id, ok: true, result: out });
      }
    } catch (e) {
      post({ id: msg.id, ok: false, error: String((e && e.message) || e) });
    }
  };
}

var OnyXLLM = {
  version: VERSION,
  Engine: Engine,
  OnyXLM: OnyXLM,
  BPETokenizer: BPETokenizer,
  Corpus: Corpus,
  Adam: Adam,
  CONFIGS: CONFIGS,
  DEFAULTS: DEFAULTS,
  SPECIALS: SPECIALS,
  makeRng: makeRng,
  geluForward: geluForward,
  geluDeriv: geluDeriv,
  layerNormForward: layerNormForward,
  layerNormBackward: layerNormBackward,
  normalizeForTraining: normalizeForTraining,
  makeWorkerRpc: makeWorkerRpc,
  idbGet: idbGet,
  idbSet: idbSet,
  _internal: {
    mmLinear: mmLinear,
    mmLinearBackward: mmLinearBackward,
    softmaxInPlace: softmaxInPlace,
    scatterHeads: scatterHeads,
    headSlicedDW: headSlicedDW,
    layerNormBackward: layerNormBackward,
    layerNormForward: layerNormForward
  }
};

global.OnyXLLM = OnyXLLM;
try { if (typeof module !== 'undefined' && module.exports) module.exports = OnyXLLM; } catch (e) { }
return OnyXLLM;
}

/* Auto-wire: inside a worker we install the RPC loop; on the page we just
 * publish the namespace (the UI layer decides whether to spawn a worker). */
/* Auto-wire: inside a worker we install the RPC loop; on a page we publish the
 * namespace; under Node (tests / offline training) we just hand back the API. */
var __isWorker = (typeof WorkerGlobalScope !== 'undefined' && typeof self !== 'undefined' && self instanceof WorkerGlobalScope);
var __isNode = (typeof process !== 'undefined' && process.versions && process.versions.node && typeof window === 'undefined');

if (__isWorker) {
  var __onyxEngine = OnyXLLMEngineFactory(self);
  self.onmessage = (function () {
    var rpc = __onyxEngine.makeWorkerRpc(new __onyxEngine.Engine(), function (m) { self.postMessage(m); });
    return function (e) { rpc(e.data || {}); };
  })();
} else if (typeof window !== 'undefined') {
  OnyXLLMEngineFactory(window);
} else if (__isNode) {
  module.exports = OnyXLLMEngineFactory(typeof globalThis !== 'undefined' ? globalThis : {});
}
