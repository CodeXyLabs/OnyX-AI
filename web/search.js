/* ============================================================================
 * OnyX · web knowledge
 * ============================================================================
 * Retrieval that stays out of the way.
 *
 * The assistant answers from its own model and knowledge graph first. Only when
 * that is not enough does it consult open sources, and the retrieved text is
 * then spoken as part of the answer — no citation chips, no list of links. This
 * module gathers the material and rewrites it into flowing Russian prose.
 * ==========================================================================*/

function OnyXWebFactory(global) {
'use strict';

var TIMEOUT = 9000;

function fetchJson(url, timeoutMs) {
  var timeout = timeoutMs || TIMEOUT;
  var controller = null, timer = null;
  var opts = { headers: { 'Accept': 'application/json' } };
  try {
    if (typeof AbortController !== 'undefined') {
      controller = new AbortController();
      timer = setTimeout(function () { controller.abort(); }, timeout);
    }
    return fetch(url, controller ? Object.assign({}, opts, { signal: controller.signal }) : opts)
      .then(function (r) { if (timer) clearTimeout(timer); return r.ok ? r.json() : null; })
      .catch(function () { if (timer) clearTimeout(timer); return null; });
  } catch (e) { return Promise.resolve(null); }
}

function stripHtml(html) {
  if (!html) return '';
  var t = String(html);
  t = t.replace(/<script[\s\S]*?<\/script>/gi, '').replace(/<style[\s\S]*?<\/style>/gi, '');
  t = t.replace(/<[^>]+>/g, ' ');
  t = t.replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&hellip;/g, '…')
    .replace(/&mdash;/g, '—').replace(/&ndash;/g, '–').replace(/&laquo;/g, '«').replace(/&raquo;/g, '»');
  t = t.replace(/\[\d+\]/g, '').replace(/\{\{[^{}]*\}\}/g, '');
  t = t.replace(/\s+/g, ' ').trim();
  return t;
}

/* Cut at a sentence boundary so a snippet never ends mid-thought. */
function trimToSentence(text, max) {
  var t = String(text || '').trim();
  if (t.length <= max) return t;
  var c = t.slice(0, max);
  var cut = Math.max(c.lastIndexOf('. '), c.lastIndexOf('! '), c.lastIndexOf('? '));
  if (cut > max * 0.55) return c.slice(0, cut + 1).trim();
  return c.replace(/\s+\S*$/, '').trim();
}

/* ---------------------------------------------------------------- sources */

function wikipedia(query, lang, limit) {
  var url = 'https://' + lang + '.wikipedia.org/w/api.php?action=query&list=search&srsearch=' +
    encodeURIComponent(query) + '&format=json&origin=*&srlimit=' + (limit || 2) + '&srprop=snippet';
  return fetchJson(url).then(function (j) {
    var hits = (j && j.query && j.query.search) || [];
    if (!hits.length) return [];
    return Promise.all(hits.map(function (hit) {
      var sum = 'https://' + lang + '.wikipedia.org/api/rest_v1/page/summary/' + encodeURIComponent(hit.title);
      return fetchJson(sum, 7000).then(function (s) {
        if (!s || !s.extract) return null;
        return {
          source: 'Википедия',
          title: s.title || hit.title,
          text: trimToSentence(s.extract, 700),
          url: (s.content_urls && s.content_urls.desktop && s.content_urls.desktop.page) || ''
        };
      }).catch(function () { return null; });
    })).then(function (list) { return list.filter(Boolean); });
  });
}

function duckduckgo(query) {
  var url = 'https://api.duckduckgo.com/?q=' + encodeURIComponent(query) +
    '&format=json&no_html=1&skip_disambig=1&t=onyx';
  return fetchJson(url).then(function (j) {
    if (!j) return [];
    var text = j.AbstractText || j.Definition || j.Answer || '';
    if (!text || text.length < 60) return [];
    return [{ source: 'открытые источники', title: j.Heading || query, text: trimToSentence(text, 600), url: j.AbstractURL || '' }];
  });
}

function stackExchange(query) {
  var url = 'https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&pagesize=2&q=' +
    encodeURIComponent(query) + '&site=stackoverflow&filter=withbody';
  return fetchJson(url).then(function (j) {
    if (!j || j.backoff) return [];
    return (j.items || []).map(function (item) {
      var title = stripHtml(item.title || '');
      var body = trimToSentence(stripHtml(item.body || ''), 520);
      if (!title || body.length < 70) return null;
      return { source: 'технические обсуждения', title: title, text: body, url: item.link || '' };
    }).filter(Boolean);
  });
}

/* Gather from every source in parallel and drop near-duplicates. */
function search(query, opts) {
  opts = opts || {};
  if (!query || !String(query).trim()) return Promise.resolve([]);
  var tasks = [];
  if (opts.wiki !== false) {
    tasks.push(wikipedia(query, 'ru', 2));
    tasks.push(wikipedia(query, 'en', 1));
  }
  if (opts.general !== false) tasks.push(duckduckgo(query));
  if (opts.tech) tasks.push(stackExchange(query));
  if (!tasks.length) return Promise.resolve([]);

  return Promise.all(tasks.map(function (p) { return p.catch(function () { return []; }); }))
    .then(function (groups) {
      var all = [];
      for (var i = 0; i < groups.length; i++) {
        for (var j = 0; j < groups[i].length; j++) all.push(groups[i][j]);
      }
      var seen = Object.create(null);
      var out = [];
      for (var k = 0; k < all.length; k++) {
        var r = all[k];
        if (!r || !r.text) continue;
        var key = r.text.slice(0, 70).toLowerCase();
        if (seen[key]) continue;
        seen[key] = 1;
        out.push(r);
        if (out.length >= (opts.max || 4)) break;
      }
      return out;
    });
}

/* ------------------------------------------------------------- blending ---
 * Turn retrieved passages into sentences that read as part of the answer.
 * Different question shapes want different framings.
 * ------------------------------------------------------------------------*/

function firstSentences(text, count) {
  var t = String(text || '').replace(/\s+/g, ' ').trim();
  if (!t) return [];
  var parts = t.split(/(?<=[.!?])\s+/);
  var out = [];
  for (var i = 0; i < parts.length && out.length < count; i++) {
    var s = parts[i].trim();
    if (s.length < 35) continue;
    out.push(s.replace(/^[-–—•*]\s*/, ''));
  }
  return out.length ? out : [t];
}

function lowerFirst(s) {
  var t = String(s || '').trim();
  if (!t) return t;
  /* Only lower a leading capital if it is not an acronym or a proper noun we
   * cannot recognise; Russian sentences normally start with a capital. */
  var first = t.split(/\s+/)[0];
  if (first.length > 1 && first === first.toUpperCase()) return t;
  return t.charAt(0).toLowerCase() + t.slice(1);
}

var OPENERS = {
  definition: ['Если смотреть по существу, ', 'Точнее говоря, ', 'Определение здесь такое: '],
  how: ['Механизм здесь такой: ', 'По сути это работает так: ', ''],
  why: ['Причина в том, что ', 'Дело в том, что ', ''],
  compare: ['Если добавить к этому сравнение, ', 'В дополнение к этому, ', ''],
  quantity: ['Если нужны числа, ', 'По данным справочников, ', ''],
  who: ['Известно, что ', 'Здесь уместно вспомнить, что ', ''],
  where: ['Если говорить о месте, ', 'Географически ', ''],
  when: ['По времени это так: ', 'Исторически ', ''],
  default: ['Дополню: ', 'Ещё можно добавить: ', 'К этому стоит добавить: ']
};

function cleanFact(text) {
  var t = String(text || '')
    .replace(/\[\d+\]/g, '')
    .replace(/\s*\([^)]{0,40}(?:archived|retrieved|дата обращения)[^)]*\)/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
  t = t.replace(/[,;:]\s*$/, '');
  return t;
}

/* Build 1-3 natural sentences from the retrieved material. */
function blend(results, question, intent) {
  if (!results || !results.length) return '';
  var opener = (OPENERS[intent] || OPENERS.default)[0];
  var facts = [];
  for (var i = 0; i < results.length && facts.length < 3; i++) {
    var sents = firstSentences(results[i].text, 1);
    for (var j = 0; j < sents.length; j++) {
      var s = cleanFact(sents[j]);
      if (s.length < 45) continue;
      /* Skip passages that merely restate the question. */
      if (facts.indexOf(s) !== -1) continue;
      facts.push(s);
      if (facts.length >= 3) break;
    }
  }
  if (!facts.length) return '';

  var out = [];
  var head = opener + lowerFirst(facts[0]);
  out.push(/[.!?]$/.test(head) ? head : head + '.');
  if (facts[1]) {
    var second = facts[1];
    var connector = /^(он|она|оно|они|это|этот|эта|эти)\b/i.test(second) ? ' ' : ' Кроме того, ';
    var body = connector + lowerFirst(second);
    out.push(/[.!?]$/.test(body.trim()) ? body.trim() : body.trim() + '.');
  }
  if (facts[2]) {
    var third = lowerFirst(facts[2]);
    var tail = ' Добавлю, что ' + third;
    out.push(/[.!?]$/.test(tail.trim()) ? tail.trim() : tail.trim() + '.');
  }
  return out.join(' ');
}

/* A short hint used as extra context when the local model writes the answer. */
function contextFor(results, maxChars) {
  if (!results || !results.length) return '';
  var budget = maxChars || 380;
  var parts = [];
  for (var i = 0; i < results.length; i++) {
    var piece = (results[i].title ? results[i].title + ': ' : '') + results[i].text;
    if (budget - piece.length < 60) break;
    parts.push(piece);
    budget -= piece.length;
  }
  return parts.join(' ');
}

/* Should we even search? Conversational turns should not hit the network. */
var CHITCHAT = /^(привет|здравствуй|хай|прив|добрый (день|вечер|утро)|как дела|как ты|спасибо|благодарю|пока|до свидания|кто ты|что ты умеешь|какая у тебя версия|сколько времени|который час)\b/i;
var TRIVIAL = /^\s*\d+\s*[-+*/^×÷%]\s*\d+/;

function shouldSearch(question) {
  var q = String(question || '').trim();
  if (q.length < 12) return false;
  if (CHITCHAT.test(q)) return false;
  if (TRIVIAL.test(q)) return false;
  return true;
}

var OnyXWeb = {
  search: search,
  blend: blend,
  contextFor: contextFor,
  shouldSearch: shouldSearch,
  stripHtml: stripHtml,
  trimToSentence: trimToSentence,
  firstSentences: firstSentences
};

global.OnyXWeb = OnyXWeb;
try { if (typeof module !== 'undefined' && module.exports) module.exports = OnyXWeb; } catch (e) { }
return OnyXWeb;
}

if (typeof window !== 'undefined') OnyXWebFactory(window);
else if (typeof process !== 'undefined' && process.versions && process.versions.node && typeof module !== 'undefined' && !module.parent) module.exports = OnyXWebFactory(globalThis);
