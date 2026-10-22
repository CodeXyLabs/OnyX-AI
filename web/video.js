/* ============================================================================
 * OnyX · video generation
 * ============================================================================
 * Produces a real, savable video file from a text description.
 *
 * The clip is rendered frame by frame onto a canvas — procedural motion, depth
 * and colour derived deterministically from the prompt — and encoded with
 * MediaRecorder. The OnyX watermark is drawn into the frames themselves and
 * jumps to a random corner on a timer, so the mark is part of the video rather
 * than an overlay in the page.
 *
 * The animation is generated, not sampled from a video diffusion model: that is
 * what is achievable with the CPU of a browser tab, and it is stated plainly in
 * the UI. The output is a genuine video with a genuine moving watermark.
 * ==========================================================================*/

function OnyXVideofactory(global) {
'use strict';

/* ------------------------------------------------------------- watermark -- */

var markPromise = null;
var markImage = null;

function markSource() {
  return global.ONYX_MARK_DATA_URI || 'assets/watermark/onyx_mark.png';
}

function loadMark() {
  if (markPromise) return markPromise;
  markPromise = new Promise(function (resolve) {
    try {
      var img = new Image();
      img.onload = function () { markImage = img; resolve(img); };
      img.onerror = function () { markImage = null; resolve(null); };
      img.src = markSource();
    } catch (e) { resolve(null); }
  });
  return markPromise;
}

/* --------------------------------------------------------------- helpers -- */

function hashSeed(text) {
  var h = 2166136261;
  var s = String(text || 'onyx');
  for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

function makeRng(seed) {
  var s = seed >>> 0 || 1;
  return function () {
    s = (s + 0x6D2B79F5) >>> 0;
    var t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* Pick the encoder the browser actually supports. */
function pickMimeType() {
  var candidates = [
    'video/mp4;codecs=avc1.42E01E',
    'video/mp4',
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm'
  ];
  if (typeof MediaRecorder === 'undefined') return null;
  for (var i = 0; i < candidates.length; i++) {
    try {
      if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(candidates[i])) return candidates[i];
    } catch (e) { }
  }
  return '';   // let the browser choose
}

function supported() {
  try {
    return typeof MediaRecorder !== 'undefined' &&
      typeof HTMLCanvasElement !== 'undefined' &&
      !!HTMLCanvasElement.prototype.captureStream;
  } catch (e) { return false; }
}

/* ----------------------------------------------------------- scene design --
 * The prompt decides palette, motion and structure, so two different
 * descriptions really do produce visibly different clips.
 * ------------------------------------------------------------------------*/

var PALETTES = [
  { name: 'cyber', bg: ['#05060a', '#0d1b2a', '#1b263b'], accents: [190, 320, 12] },
  { name: 'ember', bg: ['#0a0403', '#2b0f08', '#431407'], accents: [12, 28, 45] },
  { name: 'aurora', bg: ['#03110f', '#062a24', '#0b3b33'], accents: [150, 175, 200] },
  { name: 'violet', bg: ['#08040f', '#1c1030', '#2c1a4a'], accents: [265, 285, 310] },
  { name: 'mono', bg: ['#050505', '#141414', '#1f1f1f'], accents: [0, 0, 8] },
  { name: 'ocean', bg: ['#02060f', '#04203d', '#06325c'], accents: [205, 220, 190] },
  { name: 'sunset', bg: ['#120207', '#3a0a1c', '#5c1430'], accents: [340, 15, 40] },
  { name: 'forest', bg: ['#030805', '#0a1f12', '#123321'], accents: [120, 140, 95] }
];

var KEYWORDS = {
  city: ['город', 'city', 'улиц', 'неон', 'cyberpunk', 'киберпанк', 'здани', 'urban'],
  nature: ['природ', 'лес', 'горы', 'море', 'океан', 'закат', 'рассвет', 'поле', 'nature', 'forest', 'mountain', 'ocean'],
  space: ['космос', 'звезд', 'звёзд', 'планет', 'галакт', 'space', 'star', 'galaxy', 'nebula', 'туманност'],
  abstract: ['абстракт', 'abstract', 'градиент', 'линии', 'форма', 'геометр'],
  water: ['вода', 'вод', 'дожд', 'река', 'озер', 'water', 'rain', 'river', 'lake'],
  fire: ['огон', 'пожар', 'пламя', 'fire', 'flame', 'лава']
};

function classify(prompt) {
  var p = String(prompt || '').toLowerCase();
  for (var key in KEYWORDS) {
    var list = KEYWORDS[key];
    for (var i = 0; i < list.length; i++) if (p.indexOf(list[i]) !== -1) return key;
  }
  return 'abstract';
}

function designScene(prompt) {
  var rng = makeRng(hashSeed(prompt));
  var kind = classify(prompt);
  var palette = PALETTES[Math.floor(rng() * PALETTES.length)];
  if (kind === 'space' || kind === 'abstract') palette = PALETTES[3];
  if (kind === 'fire') palette = PALETTES[1];
  if (kind === 'water') palette = PALETTES[5];
  if (kind === 'nature') palette = rng() < 0.5 ? PALETTES[7] : PALETTES[2];
  if (kind === 'city') palette = PALETTES[0];

  return {
    kind: kind,
    palette: palette,
    rng: rng,
    layers: 3 + Math.floor(rng() * 4),
    particles: 60 + Math.floor(rng() * 140),
    drift: 0.4 + rng() * 1.6,
    pulse: 0.5 + rng() * 1.5,
    seed: hashSeed(prompt)
  };
}

/* ---------------------------------------------------------------- drawing -- */

var CORNERS = ['tl', 'tr', 'bl', 'br'];

function cornerPosition(corner, width, height, w, h) {
  var margin = Math.round(width * 0.028);
  var x, y;
  if (corner === 'tl') { x = margin; y = margin; }
  else if (corner === 'tr') { x = width - w - margin; y = margin; }
  else if (corner === 'bl') { x = margin; y = height - h - margin; }
  else { x = width - w - margin; y = height - h - margin; }
  return { x: x, y: y };
}

function drawMark(ctx, width, height, corner, alpha) {
  if (!markImage || !markImage.width) return;
  var target = Math.max(120, Math.round(width * 0.30));
  var scale = target / markImage.width;
  var w = target;
  var h = Math.max(1, Math.round(markImage.height * scale));
  var pos = cornerPosition(corner, width, height, w, h);
  ctx.save();
  ctx.globalAlpha = alpha == null ? 0.95 : alpha;
  ctx.shadowColor = 'rgba(0,0,0,0.6)';
  ctx.shadowBlur = Math.round(h * 0.16);
  ctx.drawImage(markImage, pos.x, pos.y, w, h);
  ctx.restore();
}

function drawFrame(ctx, width, height, scene, t, elapsed, corner) {
  var p = scene.palette;
  var rng = scene.rng;
  var phase = t * scene.pulse;

  /* Background wash. */
  var g = ctx.createLinearGradient(0, 0, width, height);
  g.addColorStop(0, p.bg[0]);
  g.addColorStop(0.55, p.bg[1]);
  g.addColorStop(1, p.bg[2]);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, width, height);

  ctx.globalCompositeOperation = 'lighter';

  /* Slow drifting clouds of colour. */
  for (var i = 0; i < scene.layers; i++) {
    var seed = (scene.seed + i * 977) >>> 0;
    var r2 = makeRng(seed);
    var cx = (r2() * width + Math.sin(phase * (0.3 + i * 0.17) + i) * width * 0.22 * scene.drift);
    var cy = (r2() * height + Math.cos(phase * (0.24 + i * 0.13) + i) * height * 0.22 * scene.drift);
    var rad = width * (0.24 + r2() * 0.42);
    var hue = p.accents[i % p.accents.length];
    var alpha = 0.10 + 0.10 * (0.5 + 0.5 * Math.sin(phase + i));
    var rg = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
    rg.addColorStop(0, 'hsla(' + hue + ', 85%, ' + (42 + i * 6) + '%, ' + alpha.toFixed(3) + ')');
    rg.addColorStop(0.5, 'hsla(' + hue + ', 80%, 35%, ' + (alpha * 0.4).toFixed(3) + ')');
    rg.addColorStop(1, 'hsla(' + hue + ', 80%, 30%, 0)');
    ctx.fillStyle = rg;
    ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2); ctx.fill();
  }

  /* Foreground structure depends on what was asked for. */
  if (scene.kind === 'city') {
    var groundY = height * 0.72;
    for (var b = 0; b < 26; b++) {
      var br = makeRng((scene.seed + b * 131) >>> 0);
      var bw = width * (0.02 + br() * 0.05);
      var bx = br() * width - width * 0.02 + Math.sin(phase * 0.2 + b) * width * 0.01;
      var bh = height * (0.08 + br() * 0.42) * (1 + 0.05 * Math.sin(phase + b));
      ctx.fillStyle = 'rgba(6,8,14,0.86)';
      ctx.fillRect(bx, groundY - bh, bw, bh + height);
      for (var win = 0; win < 18; win++) {
        var wr = makeRng((scene.seed + b * 31 + win * 7) >>> 0);
        if (wr() < 0.45) continue;
        ctx.fillStyle = 'hsla(' + p.accents[win % 3] + ', 90%, 65%, ' + (0.25 + 0.5 * Math.abs(Math.sin(phase + win))) + ')';
        ctx.fillRect(bx + bw * 0.18 + (win % 3) * bw * 0.26, groundY - bh + 14 + Math.floor(win / 3) * (bh / 7), Math.max(2, bw * 0.12), Math.max(2, bh / 16));
      }
    }
  } else if (scene.kind === 'space') {
    for (var s = 0; s < 240; s++) {
      var sr = makeRng((scene.seed + s * 29) >>> 0);
      var sx = sr() * width, sy = sr() * height;
      var tw = 0.4 + 0.6 * Math.abs(Math.sin(phase * (0.5 + sr()) + s));
      ctx.fillStyle = 'rgba(255,255,255,' + (0.12 + tw * 0.7).toFixed(3) + ')';
      var sz = sr() < 0.94 ? 1.2 : 2.6;
      ctx.beginPath(); ctx.arc(sx, sy, sz, 0, Math.PI * 2); ctx.fill();
    }
  } else if (scene.kind === 'water') {
    for (var wv = 0; wv < 9; wv++) {
      ctx.strokeStyle = 'hsla(' + p.accents[wv % 3] + ', 80%, 60%, ' + (0.05 + 0.06 * (wv % 3)) + ')';
      ctx.lineWidth = 1 + wv * 0.6;
      ctx.beginPath();
      for (var x = 0; x <= width; x += 12) {
        var y = height * (0.42 + wv * 0.055) + Math.sin(x * 0.006 + phase * (0.6 + wv * 0.2) + wv) * (10 + wv * 3);
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  } else if (scene.kind === 'fire') {
    for (var f = 0; f < 70; f++) {
      var fr = makeRng((scene.seed + f * 53) >>> 0);
      var fy = height - ((elapsed * (0.25 + fr() * 0.5) + fr()) % 1) * height;
      var fx = width * (0.15 + fr() * 0.7) + Math.sin(phase * 2 + f) * 26;
      var frad = width * (0.02 + fr() * 0.07);
      var fg = ctx.createRadialGradient(fx, fy, 0, fx, fy, frad);
      fg.addColorStop(0, 'hsla(' + (18 + fr() * 26) + ', 100%, 62%, 0.28)');
      fg.addColorStop(1, 'hsla(10, 100%, 50%, 0)');
      ctx.fillStyle = fg;
      ctx.beginPath(); ctx.arc(fx, fy, frad, 0, Math.PI * 2); ctx.fill();
    }
  } else {
    /* abstract: rotating ribbons */
    for (var k = 0; k < 7; k++) {
      ctx.strokeStyle = 'hsla(' + p.accents[k % 3] + ', 85%, 62%, ' + (0.10 + 0.05 * k) + ')';
      ctx.lineWidth = 2 + (k % 3) * 2;
      ctx.beginPath();
      for (var xx = 0; xx <= width; xx += 10) {
        var yy = height * 0.5 + Math.sin(xx * 0.004 + phase * (0.5 + k * 0.2) + k) * height * 0.28;
        if (xx === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
      }
      ctx.stroke();
    }
  }

  /* Particles give every clip a sense of depth. */
  for (var q = 0; q < scene.particles; q++) {
    var pr = makeRng((scene.seed + q * 7919) >>> 0);
    var depth = 0.3 + pr() * 0.7;
    var px = ((pr() * width) + Math.sin(phase * (0.2 + depth) + q) * 40 * depth + width) % width;
    var py = ((pr() * height) - elapsed * (12 + depth * 40) + height * 4) % height;
    var pa = 0.10 + depth * 0.5 * Math.abs(Math.sin(phase + q));
    ctx.fillStyle = 'rgba(255,255,255,' + pa.toFixed(3) + ')';
    ctx.beginPath(); ctx.arc(px, py, 0.7 + depth * 2.1, 0, Math.PI * 2); ctx.fill();
  }

  ctx.globalCompositeOperation = 'source-over';

  /* Vignette keeps the frame from looking flat. */
  var vg = ctx.createRadialGradient(width / 2, height / 2, Math.min(width, height) * 0.25, width / 2, height / 2, Math.max(width, height) * 0.75);
  vg.addColorStop(0, 'rgba(0,0,0,0)');
  vg.addColorStop(1, 'rgba(0,0,0,0.55)');
  ctx.fillStyle = vg;
  ctx.fillRect(0, 0, width, height);

  /* The watermark is part of the frame, and it moves between corners. */
  drawMark(ctx, width, height, corner, 0.95);
}

/* ----------------------------------------------------------------- record -- */

var VideoService = {
  ready: false,
  lastError: null,

  supported: supported,

  warmup: function () {
    var self = this;
    return loadMark().then(function () { self.ready = !!markImage; return self.ready; });
  },

  /* Render a clip. opts: { seconds, width, height, fps, cornerEvery, onProgress }
   * Resolves to { blob, url, mime, ext, width, height, seconds, watermarked }. */
  generate: function (prompt, opts) {
    opts = opts || {};
    var self = this;
    if (!supported()) {
      this.lastError = 'браузер не поддерживает запись видео (MediaRecorder + captureStream)';
      return Promise.reject(new Error(this.lastError));
    }
    var mime = pickMimeType();
    if (mime === null) {
      this.lastError = 'браузер не поддерживает запись видео';
      return Promise.reject(new Error(this.lastError));
    }

    var seconds = Math.max(2, Math.min(24, opts.seconds || 8));
    var width = opts.width || 960;
    var height = opts.height || 540;
    var fps = opts.fps || 30;
    var cornerEvery = opts.cornerEvery || 2.2;
    var onProgress = opts.onProgress || function () { };

    return loadMark().then(function () {
      var scene = designScene(prompt);
      var canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      var ctx = canvas.getContext('2d');

      var stream = canvas.captureStream(fps);
      var rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 6000000 } : { videoBitsPerSecond: 6000000 });
      var chunks = [];
      rec.ondataavailable = function (e) { if (e.data && e.data.size) chunks.push(e.data); };

      var finished = new Promise(function (resolve, reject) {
        rec.onstop = function () {
          var type = (rec.mimeType || mime || 'video/webm').split(';')[0];
          var blob = new Blob(chunks, { type: type });
          resolve({
            blob: blob,
            url: URL.createObjectURL(blob),
            mime: type,
            ext: type.indexOf('mp4') !== -1 ? 'mp4' : 'webm',
            width: width,
            height: height,
            seconds: seconds,
            bytes: blob.size,
            watermarked: !!markImage,
            cornerEvery: cornerEvery
          });
        };
        rec.onerror = function (e) { reject(e && e.error ? e.error : new Error('recording failed')); };
      });

      /* Choose a random starting corner and rotate on a timer. */
      var rng = makeRng(hashSeed(prompt) ^ 0x5f3759df);
      var corner = CORNERS[Math.floor(rng() * CORNERS.length)];
      var nextSwitch = cornerEvery;

      var t0 = performance.now();
      var raf = 0;
      rec.start(250);

      var loop = function () {
        var now = performance.now();
        var elapsed = (now - t0) / 1000;
        if (elapsed >= seconds) {
          drawFrame(ctx, width, height, scene, elapsed, elapsed, corner);
          try { rec.stop(); } catch (e) { }
          return;
        }
        if (elapsed >= nextSwitch) {
          var pick;
          do { pick = CORNERS[Math.floor(rng() * CORNERS.length)]; } while (pick === corner && CORNERS.length > 1);
          corner = pick;
          nextSwitch = elapsed + cornerEvery;
        }
        drawFrame(ctx, width, height, scene, elapsed, elapsed, corner);
        onProgress(Math.min(1, elapsed / seconds));
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);

      return finished.then(function (res) {
        cancelAnimationFrame(raf);
        try { stream.getTracks().forEach(function (tr) { tr.stop(); }); } catch (e) { }
        return res;
      });
    });
  },

  /* One still frame, handy for a poster or a quick preview. */
  poster: function (prompt, opts) {
    opts = opts || {};
    var width = opts.width || 960, height = opts.height || 540;
    var canvas = document.createElement('canvas');
    canvas.width = width; canvas.height = height;
    var ctx = canvas.getContext('2d');
    var scene = designScene(prompt);
    drawFrame(ctx, width, height, scene, 3.5, 3.5, 'br');
    return canvas.toDataURL('image/jpeg', 0.9);
  },

  save: function (result, filename) {
    var a = document.createElement('a');
    a.href = result.url;
    a.download = filename || ('onyx_video.' + (result.ext || 'webm'));
    document.body.appendChild(a);
    a.click();
    setTimeout(function () {
      try { URL.revokeObjectURL(a.href); } catch (e) { }
      if (a.parentNode) a.parentNode.removeChild(a);
    }, 2000);
  }
};

global.OnyXVideo = VideoService;
try { if (typeof module !== 'undefined' && module.exports) module.exports = VideoService; } catch (e) { }
return VideoService;
}

if (typeof window !== 'undefined') OnyXVideofactory(window);
