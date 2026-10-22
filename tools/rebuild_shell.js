/* ============================================================================
 * OnyX · rebuild the site shell
 * ============================================================================
 * Reconstructs index.html as "static shell + application script", discarding
 * every injected module and every stale copy. The application script is taken
 * verbatim from the current file, so the V0.5 model entry and all the other
 * integration hooks are preserved exactly.
 *
 * Run this when the page has accumulated duplicate module blocks, then run
 * tools/build.js to inject fresh modules.
 * ==========================================================================*/

'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const FILE = path.join(ROOT, 'index.html');

const html = fs.readFileSync(FILE, 'utf8');
const APP_ANCHOR = "<script>\n'use strict';";

/* The application script is the final <script> block: it owns the page's whole
 * behaviour. Everything after it is closing markup. */
const appAt = html.lastIndexOf(APP_ANCHOR);
if (appAt < 0) {
  console.error('application script not found — nothing was changed');
  process.exit(1);
}
const closeAt = html.indexOf('</script>', appAt);
if (closeAt < 0) {
  console.error('application script is not terminated — nothing was changed');
  process.exit(1);
}
const appScript = html.slice(appAt, closeAt + '</script>'.length);

/* Sanity checks on what we are about to keep. The document-level marker lives
 * in the head rather than the application script. */
const must = [
  ['V0.5 model entry', "'0.5': { name:'OnyX Ai V0.5"],
  ['video node renderer', 'function addVideoNode'],
  ['image service call', 'OnyXImage.generate'],
  ['video service call', 'OnyXVideo.generate'],
  ['splash call', 'OnyXSplash.show'],
  ['woven sources', 'if(woven) answer'],
  ['settings tile', 'stileLlm']
];
let bad = false;
for (const [label, needle] of must) {
  const ok = appScript.indexOf(needle) !== -1;
  console.log('  ' + (ok ? 'ok  ' : 'MISS') + '  ' + label);
  if (!ok) bad = true;
}
if (bad) {
  console.error('the application script is missing integration hooks — aborting so nothing is lost');
  process.exit(1);
}

/* Rebuild: everything before the first injected module, plus the app script and
 * the closing markup after it. Any other factory block that survived an earlier
 * partial build is dropped as well, so the result is exactly one application
 * script and nothing else. */
const MODULES_START = '<!-- ============================ OnyX · нейросеть ============================ -->';
const FACTORY_RE = /function\s+OnyX\w+Factory\s*\(/;

function stripStrayFactories(text, keepScript) {
  const re = /<script([^>]*)>([\s\S]*?)<\/script>/g;
  let out = '';
  let last = 0;
  let m;
  let dropped = 0;
  while ((m = re.exec(text)) !== null) {
    const body = m[2];
    if (body === keepScript) continue;                 // the application script
    if (!FACTORY_RE.test(body)) continue;              // not a module
    out += text.slice(last, m.index);
    last = re.lastIndex;
    if (text[last] === '\n') last++;
    dropped++;
  }
  out += text.slice(last);
  return { text: out, dropped: dropped };
}

let head = html.slice(0, appAt);
const cut = head.indexOf(MODULES_START);
if (cut !== -1) head = head.slice(0, cut);
const headClean = stripStrayFactories(head, null);
console.log('dropped ' + headClean.dropped + ' stray module block(s) before the application script');

const tail = html.slice(closeAt + '</script>'.length);
const tailClean = stripStrayFactories(tail, null);
if (tailClean.dropped) console.log('dropped ' + tailClean.dropped + ' stray module block(s) after the application script');

const rebuilt = headClean.text + appScript + tailClean.text;

fs.writeFileSync(FILE, rebuilt, 'utf8');
console.log('\nshell rebuilt: ' + (html.length / 1024).toFixed(1) + ' KB -> ' + (rebuilt.length / 1024).toFixed(1) + ' KB');
console.log('engine blocks : ' + (rebuilt.match(/<script id="onyx-llm-engine-source"/g) || []).length);
console.log('next step     : node tools/build.js');
