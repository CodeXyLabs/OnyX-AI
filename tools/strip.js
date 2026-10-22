/* ============================================================================
 * OnyX · strip injected modules
 * ============================================================================
 * Removes every injected module block from index.html while keeping the site
 * application script, the V0.5 model entry and the rest of the integration
 * hooks intact. Use it whenever a build has left a stale copy behind, then run
 * tools/build.js again.
 *
 * The boundary is exact: everything from the first module banner up to the
 * application script anchor is removed. The application script is the last
 * <script> block in the file, so this cannot eat page content.
 * ==========================================================================*/

'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const FILE = path.join(ROOT, 'index.html');

let html = fs.readFileSync(FILE, 'utf8');
const before = html.length;

const MODULES_START = '<!-- ============================ OnyX · нейросеть ============================ -->';
const APP_ANCHOR = "<script>\n'use strict';";

const appAt = html.lastIndexOf(APP_ANCHOR);
if (appAt < 0) {
  console.error('could not find the application script — refusing to modify the page');
  process.exit(1);
}

/* Anything injected sits between the last <<</html>>> body content and the app
 * script, so remove only blocks that appear before the app anchor. */
const head = html.slice(0, appAt);
const tail = html.slice(appAt);

let removed = 0;
const start = head.indexOf(MODULES_START);
if (start !== -1) {
  removed = head.length - start;
  html = head.slice(0, start) + tail;
  console.log('removed ' + (removed / 1024).toFixed(1) + ' KB of injected modules');
} else {
  console.log('no module banner found; checking for stray engine blocks');
}

/* Belt and braces: drop any leftover engine placeholder blocks. */
let guard = 0;
while (guard++ < 12) {
  const i = html.indexOf('<script id="onyx-llm-engine-source"');
  if (i === -1) break;
  const close = html.indexOf('</script>', i);
  if (close === -1) break;
  let end = close + '</script>'.length;
  if (html[end] === '\n') end++;
  html = html.slice(0, i) + html.slice(end);
  console.log('removed a stray engine block');
}

fs.writeFileSync(FILE, html, 'utf8');
const left = (html.match(/<script id="onyx-llm-engine-source"/g) || []).length;
console.log('index.html: ' + (before / 1024).toFixed(1) + ' KB -> ' + (html.length / 1024).toFixed(1) + ' KB');
console.log('engine blocks remaining: ' + left);
console.log('integration marker kept : ' + (html.indexOf('ONYX_INTEGRATION_MARKER') !== -1));
