/* ============================================================================
 * OnyX · image generation
 * ============================================================================
 * Renders a picture from a text description and stamps the OnyX watermark into
 * its bottom-right corner. The watermark is composited on a canvas and the
 * result is exported as a data URL, so the mark is part of the image itself and
 * survives saving, copying and sharing.
 * ==========================================================================*/

function OnyXImageFactory(global) {
'use strict';

var WATERMARK_SRC = null;      // resolved from the embedded data URI or the file
var watermarkImage = null;     // decoded HTMLImageElement
var watermarkPromise = null;

/* The page may embed the mark as a data URI (so everything works from file://)
 * or leave it as a relative path served from assets/watermark/. */
function watermarkSource() {
  if (WATERMARK_SRC) return WATERMARK_SRC;
  if (global.ONYX_WATERMARK_DATA_URI) { WATERMARK_SRC = global.ONYX_WATERMARK_DATA_URI; return WATERMARK_SRC; }
  WATERMARK_SRC = 'assets/watermark/onyx_logo.png';
  return WATERMARK_SRC;
}

function loadWatermark() {
  if (watermarkPromise) return watermarkPromise;
  watermarkPromise = new Promise(function (resolve) {
    try {
      var img = new Image();
      img.onload = function () { watermarkImage = img; resolve(img); };
      img.onerror = function () { watermarkImage = null; resolve(null); };
      img.src = watermarkSource();
    } catch (e) { resolve(null); }
  });
  return watermarkPromise;
}

/* ------------------------------------------------------------ generation -- */

function buildPrompt(userPrompt) {
  var p = String(userPrompt || '').trim();
  p = p.replace(/^(нарисуй|сгенерируй|создай|покажи|сделай|нарисовать|изобрази|начерти|draw|generate|paint|render|create)\s+/i, '').trim();
  if (!p) p = 'abstract composition, deep black background, red accents';
  return 'masterpiece, highly detailed, professional digital art, ' + p +
    ', cinematic lighting, sharp focus, rich colours, intricate detail';
}

function rngInt(a, b) { return Math.floor(Math.random() * (b - a + 1)) + a; }

function endpoint(prompt, size, seed, model) {
  var p = encodeURIComponent(buildPrompt(prompt));
  return 'https://image.pollinations.ai/prompt/' + p +
    '?width=' + size + '&height=' + size + '&seed=' + seed +
    '&nologo=true&model=' + (model || 'flux');
}

/* ------------------------------------------------------------- watermark -- */

/* Draw the mark into the bottom-right corner, scaled to the picture. */
function stampWatermark(ctx, width, height, img) {
  if (!img || !img.width) return false;
  var target = Math.max(96, Math.round(width * 0.26));
  var scale = target / img.width;
  var w = target;
  var h = Math.max(1, Math.round(img.height * scale));
  var margin = Math.max(12, Math.round(width * 0.028));
  var x = width - w - margin;
  var y = height - h - margin;

  ctx.save();
  /* A soft dark plate keeps the mark readable over bright artwork without
   * hiding it. */
  var pad = Math.round(h * 0.14);
  var radius = Math.round(h * 0.3);
  var g = ctx.createLinearGradient(x - pad, y - pad, x + w + pad, y + h + pad);
  g.addColorStop(0, 'rgba(0,0,0,0.42)');
  g.addColorStop(1, 'rgba(0,0,0,0.22)');
  ctx.fillStyle = g;
  roundRect(ctx, x - pad, y - pad, w + pad * 2, h + pad * 2, radius);
  ctx.fill();

  ctx.globalAlpha = 0.94;
  ctx.shadowColor = 'rgba(0,0,0,0.55)';
  ctx.shadowBlur = Math.round(h * 0.12);
  ctx.drawImage(img, x, y, w, h);
  ctx.restore();
  return true;
}

function roundRect(ctx, x, y, w, h, r) {
  var rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  ctx.lineTo(x + rr, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x + rr, y);
  ctx.closePath();
}

/* --------------------------------------------------------------- loading -- */

function loadImage(url, timeoutMs) {
  return new Promise(function (resolve, reject) {
    var done = false;
    var img = new Image();
    var timer = setTimeout(function () {
      if (done) return;
      done = true;
      reject(new Error('image load timed out'));
    }, timeoutMs || 45000);
    img.onload = function () { if (done) return; done = true; clearTimeout(timer); resolve(img); };
    img.onerror = function () { if (done) return; done = true; clearTimeout(timer); reject(new Error('image load failed')); };
    img.crossOrigin = 'anonymous';
    img.src = url;
  });
}

/* Fetch through fetch() first: that gives a same-origin blob which can always be
 * read back from a canvas. Falls back to a direct <img> load. */
function fetchAsImage(url, timeoutMs) {
  var controller = null, timer = null;
  var init = {};
  try {
    if (typeof AbortController !== 'undefined') {
      controller = new AbortController();
      timer = setTimeout(function () { controller.abort(); }, timeoutMs || 45000);
      init.signal = controller.signal;
    }
    return fetch(url, init)
      .then(function (r) {
        if (timer) clearTimeout(timer);
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.blob();
      })
      .then(function (blob) {
        return new Promise(function (resolve, reject) {
          var fr = new FileReader();
          fr.onload = function () { loadImage(String(fr.result), 20000).then(resolve, reject); };
          fr.onerror = function () { reject(new Error('blob read failed')); };
          fr.readAsDataURL(blob);
        });
      })
      .catch(function () { if (timer) clearTimeout(timer); return loadImage(url, 45000); });
  } catch (e) {
    return loadImage(url, 45000);
  }
}

/* ------------------------------------------------------------------- API -- */

var ImageService = {
  ready: false,

  warmup: function () {
    var self = this;
    return loadWatermark().then(function () { self.ready = !!watermarkImage; return self.ready; });
  },

  /* Generate and stamp. Resolves to { dataUrl, width, height, watermarked }. */
  generate: function (prompt, opts) {
    opts = opts || {};
    var size = opts.size || 1024;
    var seed = opts.seed || rngInt(1, 999999);
    var model = opts.model;
    var attempts = opts.attempts || 3;

    var go = function (n) {
      var url = endpoint(prompt, size, n === 1 ? seed : rngInt(1, 999999), model);
      return fetchAsImage(url, opts.timeout || 50000).then(function (img) {
        return stamp(img, prompt);
      }).catch(function (err) {
        if (n < attempts) return go(n + 1);
        throw err;
      });
    };
    return loadWatermark().then(function () { return go(1); });
  },

  /* Draw the image onto a canvas and put the mark in the corner. If the canvas
   * is tainted (a cross-origin image without CORS headers) we fall back to the
   * raw URL, and the caller can still overlay the mark with CSS. */
  stamp: function (img, prompt) {
    var w = img.naturalWidth || img.width;
    var h = img.naturalHeight || img.height;
    var canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);
    var stamped = stampWatermark(ctx, w, h, watermarkImage);
    var url;
    try {
      url = canvas.toDataURL('image/jpeg', 0.92);
    } catch (e) {
      return { url: img.src, dataUrl: null, width: w, height: h, watermarked: false, tainted: true };
    }
    return { url: url, dataUrl: url, width: w, height: h, watermarked: stamped, tainted: false };
  },

  /* Produce a placeholder picture locally when the network is unavailable, so
   * the feature degrades into something real instead of an error box. */
  placeholder: function (prompt, opts) {
    opts = opts || {};
    var size = opts.size || 768;
    var canvas = document.createElement('canvas');
    canvas.width = size; canvas.height = size;
    var ctx = canvas.getContext('2d');
    var seed = 0;
    var text = String(prompt || 'onyx');
    for (var i = 0; i < text.length; i++) seed = (seed * 31 + text.charCodeAt(i)) >>> 0;
    var rnd = function () { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

    var base = ctx.createLinearGradient(0, 0, size, size);
    base.addColorStop(0, '#050505');
    base.addColorStop(0.55, 'hsl(' + Math.floor(rnd() * 30) + ', 45%, 12%)');
    base.addColorStop(1, '#0b0b0b');
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, size, size);

    ctx.globalCompositeOperation = 'lighter';
    for (var k = 0; k < 90; k++) {
      var x = rnd() * size, y = rnd() * size;
      var r = 20 + rnd() * size * 0.35;
      var g = ctx.createRadialGradient(x, y, 0, x, y, r);
      var hue = rnd() < 0.45 ? 6 : 0;
      var alpha = 0.05 + rnd() * 0.16;
      g.addColorStop(0, 'hsla(' + hue + ', 90%, ' + (35 + rnd() * 35) + '%, ' + alpha + ')');
      g.addColorStop(1, 'hsla(' + hue + ', 90%, 40%, 0)');
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalCompositeOperation = 'source-over';

    stampWatermark(ctx, size, size, watermarkImage);
    return { url: canvas.toDataURL('image/jpeg', 0.9), dataUrl: canvas.toDataURL('image/jpeg', 0.9), width: size, height: size, watermarked: !!watermarkImage, placeholder: true };
  },

  /* Small preview used by the "enhance" / regenerate buttons. */
  variantUrl: function (prompt, size) {
    return endpoint(prompt, size || 1024, rngInt(1, 999999));
  }
};

global.OnyXImage = ImageService;
try { if (typeof module !== 'undefined' && module.exports) module.exports = ImageService; } catch (e) { }
return ImageService;
}

if (typeof window !== 'undefined') OnyXImageFactory(window);
