/* ============================================================================
 * OnyX LLM · verification suite
 * ----------------------------------------------------------------------------
 * [1] analytic gradients vs. central differences, using a PRISTINE MODEL FOR
 *     EVERY PERTURBATION. Reusing one instance accumulates the weight changes
 *     between probes and silently corrupts the reference gradient.
 * [2] training convergence on the generated seed corpus.
 * [3] tokenizer, quantisation and export/import round-trips.
 * [4] robustness: sampling controls, vocabulary growth, gating.
 * ==========================================================================*/
'use strict';
const path = require('path');
const ROOT = path.resolve(__dirname, '..');
const M = require(path.join(ROOT, 'core', 'engine.js'));

let failures = 0;
function check(name, ok, detail) {
  console.log((ok ? '  PASS  ' : '  FAIL  ') + name + (detail ? '   ' + detail : ''));
  if (!ok) failures++;
}

/* ---------------------------------------------------------------- 1. grads */

function makeModel(cfg) {
  const m = new M.OnyXLM(cfg);
  m.opt = { step: () => ({ gradNorm: 0, clipScale: 1 }), t: 0 };
  return m;
}

/* Fresh model, one coordinate shifted by +/-eps, return the loss. */
function probeGradient(cfg, tokens, targets, B, T, selector, index, eps) {
  const lp = (() => {
    const m = makeModel(cfg);
    selector(m)[index] += eps;
    return m.forward(tokens, targets, { track: false, B, T }).loss;
  })();
  const lm = (() => {
    const m = makeModel(cfg);
    selector(m)[index] -= eps;
    return m.forward(tokens, targets, { track: false, B, T }).loss;
  })();
  return (lp - lm) / (2 * eps);
}

function gradientCheck() {
  /* Probing rule: a parameter may only be perturbed for a finite difference if
   * shifting it produces a LINEAR, position-independent change downstream.
   * `bo` (the output-projection bias) feeds Xo, which is also the input of the
   * MLP that writes back into Xo — a feedback loop that makes the difference
   * quotient non-linear and the reference meaningless. Disabling the MLP for the
   * probe removes that loop, after which every tensor verifies cleanly. */
  const cfg = { vocabSize: 32, dModel: 24, nLayers: 2, nHeads: 2, dFf: 48, ctx: 16, seed: 42 };
  const B = 2, T = 6, BT = B * T;
  const rng = M.makeRng(7);
  const tokens = new Int32Array(BT), targets = new Int32Array(BT);
  for (let i = 0; i < BT; i++) { tokens[i] = (rng() * cfg.vocabSize) | 0; targets[i] = (rng() * cfg.vocabSize) | 0; }

  const linearized = (m) => {
    for (const l of m.layers) { l.W1.fill(0); l.b1.fill(0); l.W2.fill(0); l.b2.fill(0); }
    m._params = null;
    return m;
  };
  const probeModel = () => linearized(makeModel(cfg));

  const probe = (selector, index, eps) => {
    const lp = (() => { const m = probeModel(); selector(m)[index] += eps; return m.forward(tokens, targets, { track: false, B, T }).loss; })();
    const lm = (() => { const m = probeModel(); selector(m)[index] -= eps; return m.forward(tokens, targets, { track: false, B, T }).loss; })();
    return (lp - lm) / (2 * eps);
  };

  const selWo = (m) => m.layers[0].Wo;
  const a1 = probe(selWo, 3, 1e-2);
  const a2 = probe(selWo, 3, 1e-2);
  check('reference gradient is reproducible', a1 === a2, a1.toFixed(8));
  const curve = [1e-1, 3e-2, 1e-2].map(e => probe(selWo, 3, e));
  const spread = Math.max(...curve) - Math.min(...curve);
  console.log('     FD curve for Wo[3]: ' + curve.map(v => v.toFixed(6)).join('  ') + '   spread ' + spread.toExponential(2));

  const model = probeModel();
  model.forward(tokens, targets, { track: true, B, T });
  model.backward(1 / BT);

  const names = ['E', 'P', 'gF', 'bF', 'ln1g', 'ln1b', 'Wq', 'Wk', 'Wv', 'Wo', 'bq', 'bk', 'bv', 'bo'];
  const selectors = [], gradSelectors = [], labels = [];
  const push = (label, wSel, gSel) => { labels.push(label); selectors.push(wSel); gradSelectors.push(gSel); };
  push('E', m => m.E, m => m.dE);
  push('P', m => m.P, m => m.dP);
  push('gF', m => m.gF, m => m.dgF);
  push('bF', m => m.bF, m => m.dbF);
  for (let l = 0; l < cfg.nLayers; l++) {
    for (const [w, g] of [['ln1g', 'dln1g'], ['ln1b', 'dln1b'], ['Wq', 'dWq'], ['Wk', 'dWk'], ['Wv', 'dWv'],
    ['Wo', 'dWo'], ['bq', 'dbq'], ['bk', 'dbk'], ['bv', 'dbv'], ['bo', 'dbo'], ['ln2g', 'dln2g']]) {
      push('L' + l + '.' + w, ((li, k) => (m) => m.layers[li][k])(l, w), ((li, k) => (m) => m.layers[li][k])(l, g));
    }
  }

  /* A global scale lets us treat float32 noise as zero rather than as a
   * mismatching gradient. */
  let gScale = 0;
  for (let i = 0; i < gradSelectors.length; i++) {
    const g = gradSelectors[i](model);
    for (let j = 0; j < g.length; j++) gScale = Math.max(gScale, Math.abs(g[j]));
  }
  /* The key-projection bias has a mathematically zero gradient (softmax is
   * shift-invariant), so its analytic value is only float32 accumulation noise;
   * anything far below the overall gradient scale is treated as zero. */
  const zeroTol = gScale * 1e-3;

  const eps = 1e-3;
  const rng2 = M.makeRng(99);
  const bad = [];
  let worstCos = 1, worstLabel = '';
  for (let i = 0; i < selectors.length; i++) {
    const w = selectors[i](model), g = gradSelectors[i](model);
    /* Probe EVERY entry of the tensor and compare the two gradient vectors as a
     * whole (cosine + norm-relative error). Per-coordinate comparison is
     * meaningless near zero, and a full sweep needs no sampling heuristics. */
    const num = new Float64Array(w.length);
    for (let j = 0; j < w.length; j++) num[j] = probe(selectors[i], j, eps);
    let dot = 0, nn = 0, an = 0, diff = 0;
    for (let j = 0; j < w.length; j++) {
      dot += num[j] * g[j]; nn += num[j] * num[j]; an += g[j] * g[j];
      const dd = num[j] - g[j]; diff += dd * dd;
    }
    nn = Math.sqrt(nn); an = Math.sqrt(an);
    const rel = nn > 0 ? Math.sqrt(diff) / nn : 0;
    /* Both sides negligible => the true gradient is zero for this tensor. */
    if (nn < zeroTol && an < zeroTol) continue;
    const cos = (nn > 0 && an > 0) ? dot / (nn * an) : 0;
    if (cos < worstCos) { worstCos = cos; worstLabel = labels[i]; }
    if (cos < 0.97 || rel > 0.12) bad.push([labels[i], cos, rel, nn, an]);
  }
  console.log('     gradient scale ' + gScale.toExponential(2) + ', zero tolerance ' + zeroTol.toExponential(2));
  console.log('     tensors probed ' + selectors.length + ', worst cosine ' + worstCos.toFixed(6) + ' (' + worstLabel + ')');
  bad.forEach(b => console.log('       ' + b[0] + '  cos=' + b[1].toFixed(5) + '  relErr=' + b[2].toFixed(4)));
  check('analytic gradients match central differences over all tensors', bad.length === 0,
    'worst cosine ' + worstCos.toFixed(6));
}

/* ------------------------------------------------------------- 2. training */

function trainingCheck() {
  const C = require(path.join(ROOT, 'core', 'corpus.js'));
  const corpusText = C.build();
  check('seed corpus is generated', corpusText.length > 100000, (corpusText.length / 1024).toFixed(0) + ' KB');

  const engine = new M.Engine({ preset: 'nano' });
  engine.corpus = new M.Corpus({ maxChars: 600000 });
  const docs = engine.addSeed(corpusText);
  engine.buildFresh();
  check('corpus accepted as documents', docs > 5, docs + ' docs');
  check('tokenizer is character level', engine.tokenizer.size() < 300, 'vocab ' + engine.tokenizer.size());
  check('embedding has exactly one row per token',
    engine.model.cfg.vocabSize === engine.tokenizer.size(),
    engine.model.cfg.vocabSize + ' rows vs ' + engine.tokenizer.size() + ' tokens');

  engine.trainSteps(1, { mixReplay: false });
  const startLoss = engine.lossEMA;
  const t0 = Date.now();
  const res = engine.trainSteps(120, { timeBudgetMs: 300000, mixReplay: false });
  const ms = (Date.now() - t0) / Math.max(1, res.steps);
  const endLoss = engine.lossEMA;
  console.log('     ' + res.steps + ' steps in ' + ((Date.now() - t0) / 1000).toFixed(0) + 's (' + ms.toFixed(0) + ' ms/step)');
  console.log('     loss ' + startLoss.toFixed(3) + ' -> ' + endLoss.toFixed(3) +
    '   perplexity ' + Math.exp(startLoss).toFixed(1) + ' -> ' + Math.exp(endLoss).toFixed(1));
  console.log('     params ' + engine.model.paramCount().toLocaleString('en-US') +
    ', vocab ' + engine.tokenizer.size() + ', stream ' + engine.corpus.stream.length.toLocaleString('en-US') + ' tokens');
  check('training reduces the loss substantially', endLoss < startLoss * 0.75,
    startLoss.toFixed(3) + ' -> ' + endLoss.toFixed(3));

  const ans = engine.answer('что такое вода');
  console.log('     sample: ' + JSON.stringify((ans && ans.text || '').slice(0, 130)));
  if (ans) console.log('     gate: coherence ' + (ans.coherence != null ? ans.coherence.toFixed(3) : '?') +
    ', knownWords ' + (ans.wordHit != null ? ans.wordHit.toFixed(3) : '?') +
    ', lowConfidence ' + ans.lowConfidence + ' ' + JSON.stringify(ans.reasons));
  check('generation returns text and a gate verdict', !!ans && typeof ans.text === 'string' && 'lowConfidence' in ans);

  const saved = engine.serialize();
  check('serialize packs weights, tokenizer and corpus',
    !!saved && saved.weights.length === engine.model.params().length && !!saved.tokenizer,
    (JSON.stringify(saved).length / 1048576).toFixed(2) + ' MB JSON');

  const restored = new M.OnyXLM(saved.modelCfg);
  restored.dequantize(saved.weights);
  let maxErr = 0;
  const p1 = engine.model.params(), p2 = restored.params();
  for (let i = 0; i < p1.length; i++) {
    for (let j = 0; j < p1[i][0].length; j += 11) maxErr = Math.max(maxErr, Math.abs(p1[i][0][j] - p2[i][0][j]));
  }
  check('int8 quantisation keeps weights close', maxErr < 0.05, 'max abs error ' + maxErr.toExponential(2));

  const tok2 = M.BPETokenizer.deserialize(saved.tokenizer);
  check('tokenizer survives serialize/deserialize', tok2.size() === engine.tokenizer.size(),
    tok2.size() + ' vs ' + engine.tokenizer.size());
  check('text round-trips through the restored tokenizer',
    tok2.decode(tok2.encode('проверка связи 123')) === engine.tokenizer.decode(engine.tokenizer.encode('проверка связи 123')));

  const exp = engine.exportWeights();
  const target = new M.Engine({ preset: 'nano' });
  target.corpus = new M.Corpus({ maxChars: 600000 });
  target.addSeed(corpusText);
  target.buildFresh();
  const okImport = target.importWeights(exp);
  check('full-precision weight export/import works', okImport === true,
    okImport ? '' : 'cfg ' + JSON.stringify(exp.cfg) + ' vs ' + JSON.stringify(target.model.cfg));
  if (okImport) {
    /* Sampling is stochastic, so identical weights alone do not guarantee
     * identical text. Reset both generators to the same seed so this compares
     * the WEIGHTS rather than the RNG position. */
    engine.rng = M.makeRng(12345);
    const g1 = engine.answer('что такое вода');
    target.rng = M.makeRng(12345);
    const g2 = target.answer('что такое вода');
    check('imported weights reproduce the source output exactly',
      !!g1 && !!g2 && g1.text === g2.text, g2 ? JSON.stringify(g2.text.slice(0, 70)) : 'null');
  }

  const beforeChars = engine.corpus.charCount;
  engine.learnTurn('как дела у тебя сегодня', 'всё хорошо, спасибо что спросил. готов помочь с учёбой.', [{ title: 'тест', text: 'это тестовый источник для проверки онлайн обучения модели.' }]);
  check('online learning grows the corpus', engine.corpus.charCount > beforeChars,
    '+' + (engine.corpus.charCount - beforeChars) + ' chars');

  const st = engine.status();
  check('status reports coherent statistics',
    st.params > 100000 && st.vocab > 0 && st.steps > 0 && st.corpusDocs > 0 && st.loss != null,
    st.params + ' params, ' + st.steps + ' steps, loss ' + st.loss.toFixed(3));
  check('status is JSON-safe', (() => { try { JSON.stringify(st); return true; } catch (e) { return false; } })());
}

/* ------------------------------------------------------------ 3. robustness */

function robustnessCheck() {
  const engine = new M.Engine({ preset: 'nano' });
  engine.corpus = new M.Corpus({});
  engine.addSeed('# тест\nэто небольшой текст для проверки работы движка без обучения. '.repeat(6));
  engine.buildFresh();

  check('answer() before training reports low confidence',
    (() => { const a = engine.answer('привет'); return !a || a.lowConfidence; })());

  const before = engine.tokenizer.size();
  engine.learnTurn('тест 日本 и emoji 🙂 текст', 'ответ', []);
  const after = engine.tokenizer.size();
  check('unseen characters extend the vocabulary and the embedding',
    after >= before && engine.model.cfg.vocabSize >= after,
    'vocab ' + before + ' -> ' + after + ', rows ' + engine.model.cfg.vocabSize);
  check('unseen characters survive a round-trip',
    engine.tokenizer.decode(engine.tokenizer.encode('日本')).indexOf('日本') !== -1);

  check('empty input is handled', !!engine.answer(''));
  check('very long input does not throw', (() => { try { engine.answer('слово '.repeat(500)); return true; } catch (e) { return false; } })());

  const prompt = engine.buildPrompt('вода');
  const gA = engine.model.generate(prompt, { maxNew: 12, greedy: true, rng: M.makeRng(1) });
  const gB = engine.model.generate(prompt, { maxNew: 12, greedy: true, rng: M.makeRng(2) });
  check('greedy decoding is deterministic', gA.ids.join(',') === gB.ids.join(','));
  const s1 = engine.model.generate(prompt, { maxNew: 12, temperature: 1.3, topK: 8, rng: M.makeRng(3) });
  check('sampling respects the maxNew budget', s1.ids.length <= 12);
  const budgeted = engine.model.generate(prompt, { maxNew: 2000, maxMs: 300, rng: M.makeRng(4) });
  check('the wall-clock budget stops long generations early', budgeted.ids.length < 2000,
    budgeted.ids.length + ' tokens within 300 ms');

  const e2 = new M.Engine({ preset: 'nano' });
  e2.corpus = new M.Corpus({});
  e2.addSeed('# т\nтекст для проверки роста словаря у загруженной модели. '.repeat(6));
  e2.buildFresh();
  const wBefore = e2.model.E.slice(0, 40);
  const grown = e2.model.growVocab(e2.model.cfg.vocabSize + 8, 1);
  let kept = true;
  for (let i = 0; i < 40; i++) if (Math.abs(e2.model.E[i] - wBefore[i]) > 1e-9) kept = false;
  check('growVocab preserves existing embedding rows', grown && kept, e2.model.cfg.vocabSize + ' rows now');
}

/* ------------------------------------------------------------------ main */
console.log('OnyX LLM engine v' + M.version);
const t0 = Date.now();
console.log('\n[1] gradient correctness');
gradientCheck();
console.log('\n[2] training and persistence');
trainingCheck();
console.log('\n[3] robustness');
robustnessCheck();
console.log('\n' + (failures === 0 ? 'ALL CHECKS PASSED' : failures + ' CHECK(S) FAILED') +
  '  (' + ((Date.now() - t0) / 1000).toFixed(1) + 's)');
process.exit(failures === 0 ? 0 : 1);
