/* A correct <script> block extractor.
 *
 * A plain /<script[^>]*>([\s\S]*?)<\/script>/ regex is wrong here: the page's
 * own code contains the literal "<script id=..." inside JavaScript strings, and
 * a regex closes the block at the first "</script>" it sees, which can land in
 * the middle of unrelated code.
 *
 * This scanner walks the file and only treats a "<script" as a real opening tag
 * when it is followed by attributes that look like a tag and terminated by ">",
 * and it never starts a new block inside one it is already reading.
 */
'use strict';

function extractScripts(html) {
  const blocks = [];
  let i = 0;
  const n = html.length;
  while (i < n) {
    const open = html.indexOf('<script', i);
    if (open === -1) break;
    const gt = html.indexOf('>', open);
    if (gt === -1) break;
    const attrs = html.slice(open + '<script'.length, gt);
    /* A real opening tag's attributes contain no quotes-with-angle-brackets and
     * no "[" or "(" (which would betray a JS expression). */
    if (/[\[\]()]/.test(attrs)) { i = gt + 1; continue; }
    const close = html.indexOf('</script>', gt + 1);
    if (close === -1) break;
    blocks.push({ attrs, body: html.slice(gt + 1, close), start: open, end: close + '</script>'.length });
    i = close + '</script>'.length;
  }
  return blocks;
}

module.exports = { extractScripts };
