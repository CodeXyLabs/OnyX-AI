/* ============================================================================
 * OnyX · site integration patch
 * ============================================================================
 * Applies the additive changes to the site shell (index.html):
 *
 *   1. a new "V0.5" entry in the model list, wired for video generation;
 *   2. image generation that stamps the OnyX watermark into the picture;
 *   3. video generation for the V0.5 model, with a moving watermark;
 *   4. open sources folded into the answer as prose instead of source chips;
 *   5. the startup splash clip.
 *
 * Every edit is anchored on an exact source string, and the script refuses to
 * apply twice. Run it after restoring index.html from site.original.html.
 * ==========================================================================*/

'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const FILE = path.join(ROOT, 'index.html');

let src = fs.readFileSync(FILE, 'utf8');
let applied = 0;
let failed = [];

function replaceOnce(label, from, to) {
  const count = src.split(from).length - 1;
  if (count === 0) { failed.push(label + ' (anchor not found)'); return; }
  if (count > 1) { failed.push(label + ' (anchor appears ' + count + ' times)'); return; }
  src = src.replace(from, to);
  applied++;
}

/* ---------------------------------------------------------------- 1. V0.5 */

replaceOnce('model list entry',
  "var VERSIONS = {",
  "var VERSIONS = {\n'0.5': { name:'OnyX Ai V0.5 VISION', short:'V0.5', badge:'v 0.5 VISION', tagline:'OnyX V0.5 · генерирует видео · изображения с водяным знаком', tagline2:'V0.5 · видео + изображения', placeholder:'Спроси OnyX V0.5 или попроси создать видео…', depth:5, decay:0.72, maxHypotheses:8, maxAnswerLength:6000, useWeb:true, useImages:true, useVideo:true, cls:'ultimate' },");

replaceOnce('version card order',
  "var order = ['4.3','4.2','4.0','3.1','2.5'];",
  "var order = ['0.5','4.3','4.2','4.0','3.1','2.5'];");

replaceOnce('version card features',
  "        if(key === '2.5') feats = '<li>Краткие ответы</li><li>Только граф знаний</li><li>Без интернета</li>';",
  "        if(key === '0.5') feats = '<li>Генерация видео</li><li>Водяной знак на видео и картинках</li><li>Источники вплетены в ответ</li><li>Все источники</li>';\n        else if(key === '2.5') feats = '<li>Краткие ответы</li><li>Только граф знаний</li><li>Без интернета</li>';");

replaceOnce('capability sync on model switch',
  "function applyVersion(key){\n  try{\n    if(!VERSIONS[key]) return;\n    currentVersion = key;\n    var meta = VERSIONS[key];",
  "function applyVersion(key){\n  try{\n    if(!VERSIONS[key]) return;\n    currentVersion = key;\n    var meta = VERSIONS[key];\n    /* Tell the bridge which capabilities this model unlocks. */\n    try{ if(typeof OnyXLLM !== 'undefined' && OnyXLLM.capabilities) OnyXLLM.bridge.caps = OnyXLLM.capabilities(key); }catch(e){}");

/* --------------------------------------------------- 2 & 3. images, videos */

replaceOnce('image/video generation block',
  "      var stImg = addStage(ui, 'Нейросеть рисует изображение…');\n      await sleep(rndInt(300, 500));\n      if(myRun !== runId) return;\n      var imgUrl = generateImageUrl(text);\n      clearTyping(ui);\n      ui.content.innerHTML = 'Готово — вот что получилось:';\n      doneStage(stImg);\n      reasoningLog.push('Изображение сгенерировано');\n      addImageNode(ui, imgUrl, text, true);\n      if(showReasoning) renderReasoningBlock(ui, reasoningLog);\n      var imgAns = 'Готово — вот что получилось:';\n      var imgOutTok = countTokens(imgAns);\n      setTokenStats(ui, inTok, imgOutTok, 0);\n      if(chat){ chat.messages.push({ role:'ai', text:imgAns, imageUrl:imgUrl, imagePrompt:text, reasoning:reasoningLog, inTok:inTok, outTok:imgOutTok, thinkTok:0 }); saveChats(); renderHistory(); }\n      return;",
  `      var wantsVideo = /(видео|видеоролик|ролик|анимаци|video|клип)/i.test(text);
      if(wantsVideo && vmeta.useVideo){
        if(typeof OnyXVideo === 'undefined' || !OnyXVideo.supported()){
          var noVid = 'Этот браузер не умеет записывать видео: нужны MediaRecorder и canvas.captureStream. Попробуй Chrome, Edge или свежий Firefox.';
          clearTyping(ui);
          await new Promise(function(resolve){ streamText(ui, noVid, resolve); });
          if(myRun !== runId) return;
          return;
        }
        var stVid = addStage(ui, 'Нейросеть рендерит видео… 0%');
        var vidProg = function(p){ try{ stVid.querySelector('.lbl').textContent = 'Нейросеть рендерит видео… ' + Math.round(p * 100) + '%'; }catch(e){} };
        var vid = null;
        try{
          vid = await OnyXVideo.generate(text, { seconds: 8, width: 960, height: 540, fps: 30, cornerEvery: 2.2, onProgress: vidProg });
        }catch(e){ vid = null; }
        if(myRun !== runId) return;
        if(!vid){
          var vidErr = 'Не удалось создать видео. Попробуй ещё раз или опиши сцену иначе.';
          clearTyping(ui);
          await new Promise(function(resolve){ streamText(ui, vidErr, resolve); });
          if(myRun !== runId) return;
          return;
        }
        doneStage(stVid);
        clearTyping(ui);
        var vidAns = 'Готово — вот видео по твоему описанию:';
        ui.content.innerHTML = fmtInline(vidAns);
        reasoningLog.push('Видео сгенерировано: ' + vid.seconds + ' с, ' + vid.width + '×' + vid.height + ', ' + vid.ext.toUpperCase());
        if(vid.watermarked) reasoningLog.push('Водяной знак встроен в кадры и перемещается по углам');
        addVideoNode(ui, vid, text);
        if(showReasoning) renderReasoningBlock(ui, reasoningLog);
        var vidTok = countTokens(vidAns);
        setTokenStats(ui, inTok, vidTok, 0);
        if(chat){ chat.messages.push({ role:'ai', text:vidAns, videoUrl:vid.url, videoExt:vid.ext, imagePrompt:text, reasoning:reasoningLog, inTok:inTok, outTok:vidTok, thinkTok:0 }); saveChats(); renderHistory(); }
        if(typeof OnyXLLM !== 'undefined' && OnyXLLM.learn) try{ OnyXLLM.learn(text, vidAns, null); }catch(e){}
        return;
      }
      var stImg = addStage(ui, 'Нейросеть рисует изображение…');
      await sleep(rndInt(300, 500));
      if(myRun !== runId) return;
      var imgUrl = generateImageUrl(text), imgStamped = false;
      /* Generate through the canvas service so the OnyX watermark ends up inside
       * the picture itself and survives saving. */
      if(typeof OnyXImage !== 'undefined' && OnyXImage.generate){
        try{
          var made = await OnyXImage.generate(text, { size: 1024 });
          if(made && made.url){ imgUrl = made.url; imgStamped = !!made.watermarked; }
        }catch(e){ imgStamped = false; }
        if(myRun !== runId) return;
      }
      clearTyping(ui);
      ui.content.innerHTML = 'Готово — вот что получилось:';
      doneStage(stImg);
      reasoningLog.push('Изображение сгенерировано');
      reasoningLog.push(imgStamped ? 'Водяной знак OnyX добавлен в правый нижний угол' : 'Водяной знак показан поверх изображения');
      addImageNode(ui, imgUrl, text, !imgStamped, imgStamped);
      if(showReasoning) renderReasoningBlock(ui, reasoningLog);
      var imgAns = 'Готово — вот что получилось:';
      var imgOutTok = countTokens(imgAns);
      setTokenStats(ui, inTok, imgOutTok, 0);
      if(chat){ chat.messages.push({ role:'ai', text:imgAns, imageUrl:imgUrl, imagePrompt:text, reasoning:reasoningLog, inTok:inTok, outTok:imgOutTok, thinkTok:0 }); saveChats(); renderHistory(); }
      if(typeof OnyXLLM !== 'undefined' && OnyXLLM.learn) try{ OnyXLLM.learn(text, imgAns, null); }catch(e){}
      return;`);

/* ------------------------------------------- 4. woven sources, no link chips */

replaceOnce('ask bridge with intent',
  "    var llmRes = null;\n    if(typeof OnyXLLM !== 'undefined' && OnyXLLM.ask){\n      try{ llmRes = await OnyXLLM.ask(text); }catch(e){ llmRes = null; }\n    }",
  `    var llmAnswer = null;
    if(typeof OnyXLLM !== 'undefined' && OnyXLLM.bridge && OnyXLLM.bridge.ask){
      try{ llmAnswer = await OnyXLLM.bridge.ask(text, intent); }catch(e){ llmAnswer = null; }
    }
    var llmRes = llmAnswer ? llmAnswer.answer : null;
    /* Facts the assistant gathered on its own. They are spoken as part of the
     * reply, never listed as links. */
    var woven = (llmAnswer && llmAnswer.woven) ? llmAnswer.woven : '';`);

replaceOnce('llm answer without source chips',
  `      reasoningLog.push('Шагов обучения модели: ' + (llmRes.trained || 0));
      if(showReasoning) renderReasoningBlock(ui, reasoningLog);
      addSources(ui, [{ label:'нейросеть onyx llm' }]);
      var llmTok = countTokens(llmRes.text);
      setTokenStats(ui, inTok, llmTok, 0);
      if(chat){
        chat.messages.push({ role:'ai', text:llmRes.text, reasoning:reasoningLog, sources:[{ label:'нейросеть onyx llm' }], inTok:inTok, outTok:llmTok, thinkTok:0 });
        saveChats(); renderHistory();
      }
      if(typeof OnyXLLM.learn === 'function') OnyXLLM.learn(text, llmRes.text, null);
      return;`,
  `      reasoningLog.push('Шагов обучения модели: ' + (llmRes.trained || 0));
      if(woven) reasoningLog.push('Дополнено сведениями из открытых источников');
      var llmText = llmRes.text;
      if(woven){
        clearTyping(ui);
        await new Promise(function(resolve){ streamText(ui, llmText + '\\n\\n' + woven, resolve); });
        if(myRun !== runId) return;
        llmText = llmText + '\\n\\n' + woven;
      }
      if(showReasoning) renderReasoningBlock(ui, reasoningLog);
      var llmTok = countTokens(llmText);
      setTokenStats(ui, inTok, llmTok, 0);
      if(chat){
        chat.messages.push({ role:'ai', text:llmText, reasoning:reasoningLog, inTok:inTok, outTok:llmTok, thinkTok:0 });
        saveChats(); renderHistory();
      }
      if(typeof OnyXLLM.learn === 'function') OnyXLLM.learn(text, llmText, null);
      return;`);

replaceOnce('woven fallback in the reasoning answer',
  "    var answer = buildStructuredAnswer(intent, concepts, hypotheses, derived, webResults, version);\n    doneStage(st7);",
  `    var answer = buildStructuredAnswer(intent, concepts, hypotheses, derived, webResults, version);
    /* Fold what was retrieved into the prose instead of appending a source list. */
    if(woven) answer = answer + '\\n\\n' + woven;
    doneStage(st7);`);

replaceOnce('no source chips on the reasoning answer',
  "    for(var c = 0; c < Math.min(concepts.length, 2); c++) srcList.push({ label:concepts[c].label });\n    if(derived.length) srcList.push({ label:'вывод по графу' });\n    if(srcList.length) addSources(ui, srcList);",
  "    /* Sources are woven into the text, so nothing is listed here. */\n    srcList = [];");

/* ------------------------------------------------------------- 5. splash */

replaceOnce('startup splash',
  "  safeCall('bootOnyXLLM', function(){",
  `  safeCall('showSplash', function(){
    if(typeof OnyXSplash !== 'undefined'){
      OnyXSplash.onDone = function(){ try{ var i = $('#input'); if(i) i.focus(); }catch(e){} };
      OnyXSplash.show();
    }
  });
  safeCall('bootOnyXLLM', function(){`);

/* --------------------------------------------------- media node renderers */

replaceOnce('image node and video node',
  "function openLightbox(url){",
  `function addVideoNode(ui, vid, prompt){
  try{
    var wrap = document.createElement('div');
    wrap.className = 'ai-image-wrap';
    wrap.style.maxWidth = '520px';
    var v = document.createElement('video');
    v.src = vid.url;
    v.poster = '';
    v.controls = true;
    v.loop = true;
    v.muted = true;
    v.setAttribute('playsinline', '');
    v.style.width = '100%';
    v.style.display = 'block';
    v.addEventListener('loadeddata', function(){ scrollChat(); });
    wrap.appendChild(v);
    var bar = document.createElement('div');
    bar.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;padding:8px 10px;border-top:1px solid rgba(255,255,255,.08)';
    bar.innerHTML = '<button class="pbtn" data-act="save">💾 Скачать ' + String(vid.ext || 'webm').toUpperCase() + '</button>' +
      '<button class="pbtn" data-act="open">↗ Открыть</button>';
    wrap.appendChild(bar);
    bar.addEventListener('click', function(e){
      var act = e.target && e.target.getAttribute ? e.target.getAttribute('data-act') : null;
      if(!act) return;
      if(act === 'save' && typeof OnyXVideo !== 'undefined') OnyXVideo.save(vid, 'onyx_' + Date.now() + '.' + (vid.ext || 'webm'));
      if(act === 'open') window.open(vid.url, '_blank');
    });
    ui.bubble.appendChild(wrap); scrollChat();
  }catch(e){}
}
function openLightbox(url){`);

replaceOnce('image node signature',
  "function addImageNode(ui, url, prompt, animate){",
  "function addImageNode(ui, url, prompt, animate, stamped){" );

replaceOnce('image node watermark overlay',
  "    img.addEventListener('click', function(){ openLightbox(url); });\n    img.src = url;",
  `    /* When the mark could not be composited (a cross-origin image), overlay it
     * so the picture is still signed. */
    if(!stamped && url.indexOf('data:') !== 0){
      var mark = document.createElement('img');
      mark.src = (typeof ONYX_WATERMARK_DATA_URI !== 'undefined' && ONYX_WATERMARK_DATA_URI) ? ONYX_WATERMARK_DATA_URI : 'assets/watermark/onyx_logo.png';
      mark.alt = '';
      mark.style.cssText = 'position:absolute;right:2.6%;bottom:2.6%;width:26%;max-width:170px;pointer-events:none;opacity:.95;filter:drop-shadow(0 4px 14px rgba(0,0,0,.6))';
      wrap.style.position = 'relative';
      mark.onerror = function(){ if(mark.parentNode) mark.parentNode.removeChild(mark); };
      wrap.appendChild(mark);
    }
    img.addEventListener('click', function(){ openLightbox(url); });
    img.src = url;`);

replaceOnce('restore image nodes with video',
  "        if(m.imageUrl) addImageNode(pair.ui, m.imageUrl, m.imagePrompt || '', false);",
  "        if(m.imageUrl) addImageNode(pair.ui, m.imageUrl, m.imagePrompt || '', false, m.imageUrl.indexOf('data:') === 0);\n        if(m.videoUrl){ var vdata = { url:m.videoUrl, ext:m.videoExt || 'webm', seconds:0, width:0, height:0, watermarked:true }; addVideoNode(pair.ui, vdata, m.imagePrompt || ''); }");

/* ------------------------------------------------------------ write back */

if (failed.length) {
  console.error('integration patch FAILED:');
  failed.forEach(f => console.error('  · ' + f));
  process.exit(1);
}

const already = src.indexOf('ONYX_INTEGRATION_MARKER') !== -1;
if (already) { console.error('integration already applied'); process.exit(1); }

src = src.replace('<!DOCTYPE html>', '<!DOCTYPE html>\n<!-- ONYX_INTEGRATION_MARKER -->');
fs.writeFileSync(FILE, src, 'utf8');
console.log('integration applied: ' + applied + ' edits, index.html now ' + (src.length / 1024).toFixed(1) + ' KB');
