/* ============================================================================
 * OnyX · startup splash
 * ============================================================================
 * Plays the CodeXy intro clip once, full screen, with no controls, then fades
 * away. The video is muted so autoplay is allowed by every browser, and it is
 * skipped entirely for users who asked for reduced motion.
 *
 * If the clip cannot be loaded (for example the page was opened from file://
 * with the assets folder missing) the splash shows an animated logo instead of
 * an error, so startup is never broken.
 * ==========================================================================*/

function OnyXSplashFactory(global) {
'use strict';

var VIDEO_SRC = null;
var POSTER_SRC = null;

function videoSource() {
  if (VIDEO_SRC) return VIDEO_SRC;
  VIDEO_SRC = global.ONYX_SPLASH_DATA_URI || 'assets/splash/onyx_intro.mp4';
  return VIDEO_SRC;
}
function posterSource() {
  if (POSTER_SRC) return POSTER_SRC;
  POSTER_SRC = global.ONYX_SPLASH_POSTER || 'assets/splash/onyx_intro_poster.jpg';
  return POSTER_SRC;
}

function prefersReducedMotion() {
  try {
    return !!(global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches);
  } catch (e) { return false; }
}

var Splash = {
  el: null,
  video: null,
  done: false,
  maxMs: 14000,

  /* Build the overlay. It covers the app until the clip ends or is dismissed. */
  show: function (opts) {
    opts = opts || {};
    if (this.done || this.el) return this;
    if (prefersReducedMotion()) { this.finish(true); return this; }

    var overlay = document.createElement('div');
    overlay.id = 'onyxSplash';
    overlay.setAttribute('aria-hidden', 'true');
    overlay.style.cssText = [
      'position:fixed', 'inset:0', 'z-index:2000',
      'background:#000',
      'display:flex', 'align-items:center', 'justify-content:center',
      'overflow:hidden',
      'opacity:1',
      'transition:opacity .9s cubic-bezier(.22,1,.36,1)',
      'cursor:pointer'
    ].join(';');

    var video = document.createElement('video');
    video.setAttribute('playsinline', '');
    video.setAttribute('muted', '');
    video.muted = true;
    video.defaultMuted = true;
    video.autoplay = true;
    video.preload = 'auto';
    video.controls = false;
    video.disablePictureInPicture = true;
    video.style.cssText = [
      'width:100%', 'height:100%',
      'object-fit:cover',
      'background:#000',
      'pointer-events:none'
    ].join(';');
    try { video.poster = posterSource(); } catch (e) { }

    /* A subtle loading line so a slow first paint still feels intentional. */
    var bar = document.createElement('div');
    bar.style.cssText = [
      'position:absolute', 'left:50%', 'bottom:7%',
      'transform:translateX(-50%)',
      'width:min(240px,44vw)', 'height:2px',
      'background:rgba(255,255,255,.14)', 'border-radius:2px', 'overflow:hidden'
    ].join(';');
    var fill = document.createElement('div');
    fill.style.cssText = [
      'width:0%', 'height:100%',
      'background:linear-gradient(90deg,#ff2b1f,#ff6b5f)',
      'transition:width .25s linear'
    ].join(';');
    bar.appendChild(fill);

    overlay.appendChild(video);
    overlay.appendChild(bar);
    document.body.appendChild(overlay);

    this.el = overlay;
    this.video = video;
    this.fill = fill;
    var self = this;

    var cleanupTimer = null;
    var bound = false;

    var fail = function () {
      /* Swap in the animated fallback logo. */
      try { video.remove(); } catch (e) { }
      try { bar.remove(); } catch (e) { }
      var fallback = document.createElement('div');
      fallback.style.cssText = [
        'display:flex', 'flex-direction:column', 'align-items:center', 'gap:18px',
        'font-family:"Doto","Space Grotesk",sans-serif', 'color:#f5f5f5'
      ].join(';');
      fallback.innerHTML =
        '<div style="display:flex;align-items:center;gap:14px">' +
        '<span style="width:34px;height:34px;border-radius:50%;background:#ff2b1f;' +
        'box-shadow:0 0 40px rgba(255,43,31,.8);animation:onyxSplashPulse 2s ease-in-out infinite"></span>' +
        '<span style="font-size:clamp(34px,7vw,62px);font-weight:900;letter-spacing:.14em">OnyX Ai</span>' +
        '</div>' +
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#8f8f8f;letter-spacing:.18em">CodeXy™ Labs</div>' +
        '<style>@keyframes onyxSplashPulse{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(.72);opacity:.55}}</style>';
      overlay.appendChild(fallback);
      setTimeout(function () { self.finish(); }, 2200);
    };

    var start = function () {
      var p = video.play();
      if (p && typeof p.catch === 'function') p.catch(function () { fail(); });
      bound = true;
      clearTimeout(cleanupTimer);
      cleanupTimer = setTimeout(function () { self.finish(); }, self.maxMs);
    };

    video.addEventListener('ended', function () { self.finish(); });
    video.addEventListener('error', fail);
    video.addEventListener('loadeddata', function () { if (!bound) start(); });
    video.addEventListener('timeupdate', function () {
      if (!video.duration || !isFinite(video.duration)) return;
      fill.style.width = Math.min(100, (video.currentTime / video.duration) * 100) + '%';
    });

    overlay.addEventListener('click', function () { self.finish(); });
    document.addEventListener('keydown', this._keyHandler = function (e) {
      if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') self.finish();
    });

    /* If nothing has started within four seconds, assume the clip is missing. */
    setTimeout(function () { if (!bound) fail(); }, 4000);
    try { video.load(); } catch (e) { }
    return this;
  },

  /* Fade out and remove. Safe to call repeatedly and from every exit path. */
  finish: function (immediate) {
    if (this.done) return;
    this.done = true;
    var self = this;
    var el = this.el;
    if (this._keyHandler) {
      try { document.removeEventListener('keydown', this._keyHandler); } catch (e) { }
    }
    if (!el) return;
    var remove = function () {
      try { if (self.video) { self.video.pause(); self.video.removeAttribute('src'); } } catch (e) { }
      try { if (el.parentNode) el.parentNode.removeChild(el); } catch (e) { }
      self.el = null;
      self.video = null;
      if (typeof self.onDone === 'function') { try { self.onDone(); } catch (e) { } }
    };
    if (immediate) { remove(); return; }
    el.style.opacity = '0';
    setTimeout(remove, 950);
  }
};

global.OnyXSplash = Splash;
try { if (typeof module !== 'undefined' && module.exports) module.exports = Splash; } catch (e) { }
return Splash;
}

if (typeof window !== 'undefined') OnyXSplashFactory(window);
