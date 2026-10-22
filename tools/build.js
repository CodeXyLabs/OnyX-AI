/* ============================================================================
 * OnyX · build
 * ============================================================================
 * Assembles the single-file site from the sources in core/, web/ and assets/.
 *
 *   index.html
 *     ├─ <script id="onyx-llm-engine-source">   core/engine.js
 *     │     Read back at runtime via textContent and handed to a Web Worker
 *     │     through a Blob URL. That indirection is what keeps the whole product
 *     │     a single portable .html file that also works from file://, where a
 *     │     worker created from an external file would be blocked by CORS.
 *     ├─ <script>  core/corpus.js     bootstrap knowledge
 *     ├─ <script>  web/search.js      quiet retrieval and prose blending
 *     ├─ <script>  web/image.js       pictures, watermark baked in
 *     ├─ <script>  web/video.js       clips, moving watermark baked in
 *     ├─ <script>  web/splash.js      startup intro
 *     └─ <script>  web/bridge.js      worker, statistics, model capabilities
 *
 * The image watermark is also embedded as a data URI so pictures can be signed
 * even when the page is opened straight from disk.
 *
 * Restore index.html from site.original.html before rebuilding.
 * ==========================================================================*/

'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const FILE = path.join(ROOT, 'index.html');

const MODULES = [
  { file: 'core/corpus.js', note: 'базовый корпус, на котором модель учится до первого сообщения' },
  { file: 'web/python_bridge.js', note: 'мост к Python-серверу (python_core/server.py)' },
  { file: 'web/search.js', note: 'поиск в открытых источниках и вплетение фактов в ответ' },
  { file: 'web/image.js', note: 'генерация изображений, водяной знак впечатывается в картинку' },
  { file: 'web/video.js', note: 'генерация видео, водяной знак перемещается по углам' },
  { file: 'web/splash.js', note: 'стартовая заставка' },
  { file: 'web/bridge.js', note: 'воркер, статистика, возможности моделей' }
];

const ENGINE = 'core/engine.js';
const ENGINE_MARK = '<script id="onyx-llm-engine-source" type="text/onyx-engine">';
const APP_ANCHOR = "<script>\n'use strict';";
const MODULES_START = '<!-- ============================ OnyX · нейросеть ============================ -->';

function read(rel) {
  const p = path.join(ROOT, rel);
  if (!fs.existsSync(p)) throw new Error('missing source: ' + rel);
  return fs.readFileSync(p, 'utf8');
}

/* A literal "</script" inside a script body would close the block early. */
function safe(code) {
  return code.split('</script').join('<\\/script');
}

function dataUri(rel, mime) {
  const p = path.join(ROOT, rel);
  if (!fs.existsSync(p)) throw new Error('missing asset: ' + rel);
  const buf = fs.readFileSync(p);
  return 'data:' + mime + ';base64,' + buf.toString('base64');
}

function main() {
  let html = read('index.html');

  if (html.indexOf('ONYX_INTEGRATION_MARKER') === -1) {
    console.error('site integration is missing — run tools/integrate.js first');
    process.exit(1);
  }

  /* Idempotent: if a previous build already injected the modules, cut them out
   * first so rebuilding refreshes them instead of stacking a second copy. */
  const strip = html.indexOf(MODULES_START);
  if (strip !== -1) {
    const anchor = html.indexOf(APP_ANCHOR, strip);
    if (anchor < 0) {
      console.error('found the modules block but not the application anchor after it');
      process.exit(1);
    }
    html = html.slice(0, strip) + html.slice(anchor);
    console.log('removed the previous module block (' + ((anchor - strip) / 1024).toFixed(1) + ' KB)');
  }

  const engine = read(ENGINE);
  const logoUri = dataUri('assets/watermark/onyx_logo.png', 'image/png');
  const markUri = dataUri('assets/watermark/onyx_mark.png', 'image/png');
  const posterUri = dataUri('assets/splash/onyx_intro_poster.jpg', 'image/jpeg');

  let block = '';
  block += '<!-- ============================ OnyX · нейросеть ============================ -->\n';
  block += '<!-- Исходник движка. На странице не выполняется: bridge читает его\n';
  block += '     textContent и запускает в Web Worker через Blob URL, чтобы обучение\n';
  block += '     шло в фоне и не подвешивало интерфейс. -->\n';
  block += ENGINE_MARK + '\n' + safe(engine) + '\n</script>\n';

  /* Assets the page needs no matter where it is opened from. */
  block += '<!-- Водяные знаки и постер заставки, встроенные как data URI. -->\n';
  block += '<script>\n';
  block += 'window.ONYX_WATERMARK_DATA_URI = ' + JSON.stringify(logoUri) + ';\n';
  block += 'window.ONYX_MARK_DATA_URI = ' + JSON.stringify(markUri) + ';\n';
  block += 'window.ONYX_SPLASH_POSTER = ' + JSON.stringify(posterUri) + ';\n';
  block += '</script>\n';

  for (const mod of MODULES) {
    block += '<!-- ' + mod.note + ' -->\n';
    block += '<script>\n' + safe(read(mod.file)) + '\n</script>\n';
  }

  const at = html.indexOf(APP_ANCHOR);
  if (at < 0) {
    console.error('could not find the application script in index.html');
    process.exit(1);
  }
  html = html.slice(0, at) + block + html.slice(at);

  fs.writeFileSync(FILE, html, 'utf8');

  const kb = n => (n / 1024).toFixed(1) + ' KB';
  console.log('index.html assembled');
  console.log('  engine        ' + kb(engine.length));
  for (const mod of MODULES) console.log('  ' + mod.file.padEnd(14) + kb(read(mod.file).length));
  console.log('  watermark png ' + kb(logoUri.length));
  console.log('  video mark    ' + kb(markUri.length));
  console.log('  splash poster ' + kb(posterUri.length));
  console.log('  total page    ' + kb(html.length));
}

main();
