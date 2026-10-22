"""
OnyX Ai · media asset preparation
=================================
Turns the three source media files the user supplied into the compact assets the
site needs:

  66d8bed9-...png      -> assets/watermark/onyx_logo.png      (image watermark)
  20557508184803.mp4   -> assets/watermark/onyx_mark.png      (video watermark,
                                                                blue keyed out)
  20557508315875.mp4   -> assets/splash/onyx_intro.mp4        (startup splash)
                          assets/splash/onyx_intro_poster.jpg (first frame)

The image watermark keeps its alpha channel so it can be baked into generated
pictures; the video watermark has its flat blue background (RGB 0,69,185)
removed so it can float over generated video frames.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC_LOGO = ROOT / "assets" / "watermark" / "onyx_logo_source.png"
SRC_MARK_VIDEO = ROOT / "assets" / "watermark" / "onyx_mark_source.mp4"
SRC_SPLASH_VIDEO = ROOT / "assets" / "splash" / "onyx_intro_source.mp4"

OUT_MARK_DIR = ROOT / "assets" / "watermark"
OUT_SPLASH_DIR = ROOT / "assets" / "splash"

FFMPEG_CANDIDATES = [
    Path(r"C:\Users\Admin\AppData\Local\CapCut\Apps\9.0.0.3858\ffmpeg.exe"),
]


def find_ffmpeg() -> str | None:
    for c in FFMPEG_CANDIDATES:
        if c.exists():
            return str(c)
    from shutil import which
    return which("ffmpeg")


def trim_alpha(img: Image.Image, pad: int = 6) -> Image.Image:
    """Crop to the non-transparent content box, with a small padding."""
    a = np.array(img)
    alpha = a[..., 3]
    ys, xs = np.where(alpha > 12)
    if len(xs) == 0:
        return img
    x0, x1 = max(0, xs.min() - pad), min(img.width, xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(img.height, ys.max() + 1 + pad)
    return img.crop((x0, y0, x1, y1))


def drop_faint_alpha(img: Image.Image, threshold: int = 10) -> Image.Image:
    """Remove the coloured halo the chroma key leaves along the edges."""
    a = np.array(img).astype(np.int16)
    alpha = a[..., 3]
    # Kill isolated near-transparent pixels.
    alpha = np.where(alpha < threshold, 0, alpha)
    # Re-premultiply-free edge cleanup: fade pixels that still carry blue.
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    blueness = np.clip((b.astype(np.int16) - np.maximum(r, g)) / 60.0, 0, 1)
    alpha = np.clip(alpha * (1.0 - blueness * 0.9), 0, 255)
    a[..., 3] = alpha
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def build_image_watermark() -> dict:
    """The PNG watermark, trimmed and scaled for a corner overlay."""
    OUT_MARK_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.open(SRC_LOGO).convert("RGBA")
    trimmed = trim_alpha(img, pad=4)
    # A corner mark should not exceed ~22% of a 1024px picture's width.
    target_w = 300
    scale = target_w / trimmed.width
    resized = trimmed.resize(
        (target_w, max(1, round(trimmed.height * scale))), Image.LANCZOS
    )
    out = OUT_MARK_DIR / "onyx_logo.png"
    resized.save(out, "PNG", optimize=True)
    # Lossless WebP is usually smaller and fully supported by browsers.
    webp = OUT_MARK_DIR / "onyx_logo.webp"
    try:
        resized.save(webp, "WEBP", lossless=True, quality=100, method=6)
    except Exception:
        webp = None
    return {
        "source": SRC_LOGO.name,
        "size": resized.size,
        "png_bytes": out.stat().st_size,
        "webp_bytes": webp.stat().st_size if webp and webp.exists() else None,
    }


def build_video_watermark(ffmpeg: str) -> dict:
    """Key the flat blue background out of the animated mark and keep the frame
    that produces the cleanest alpha, then trim and scale it."""
    OUT_MARK_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_MARK_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    results = []
    # A handful of timestamps across the 3 s clip; the mark cycles through
    # several states, so pick whichever frame keys out most cleanly.
    for i, ts in enumerate([0.2, 1.0, 1.6, 2.2, 2.9]):
        raw = tmp_dir / f"frame_{i}.png"
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(ts),
             "-i", str(SRC_MARK_VIDEO), "-frames:v", "1",
             "-vf", "colorkey=0x0045B9:0.20:0.10", "-y", str(raw)],
            check=True,
        )
        img = Image.open(raw).convert("RGBA")
        cleaned = drop_faint_alpha(img, threshold=24)
        no_blue = np.array(cleaned)[..., 3]
        # How much of the blue field survived?
        leftover = int(np.count_nonzero(no_blue > 40))
        results.append((leftover, i, ts, cleaned))

    results.sort(key=lambda r: r[0])
    leftover, idx, ts, best = results[0]
    trimmed = trim_alpha(best, pad=8)
    target_w = 420
    scale = target_w / trimmed.width
    resized = trimmed.resize(
        (target_w, max(1, round(trimmed.height * scale))), Image.LANCZOS
    )
    out = OUT_MARK_DIR / "onyx_mark.png"
    resized.save(out, "PNG", optimize=True)

    for f in tmp_dir.glob("*.png"):
        f.unlink()
    tmp_dir.rmdir()
    # The intermediate keyed frames are no longer needed.
    for stray in OUT_MARK_DIR.glob("onyx_mark_*.png"):
        if stray.name != "onyx_mark.png":
            stray.unlink()

    return {
        "source": SRC_MARK_VIDEO.name,
        "chosen_frame": ts,
        "opaque_pixels": leftover,
        "size": resized.size,
        "bytes": out.stat().st_size,
    }


def build_splash(ffmpeg: str) -> dict:
    """Copy the splash clip next to the site and grab a poster frame.

    A straight stream copy keeps the original quality and avoids a re-encode;
    `+faststart` moves the index to the front so playback can begin before the
    whole file has loaded, which is exactly what a startup splash needs."""
    OUT_SPLASH_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_SPLASH_DIR / "onyx_intro.mp4"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(SRC_SPLASH_VIDEO),
         "-c", "copy", "-movflags", "+faststart", "-y", str(dst)],
        check=True,
    )
    poster = OUT_SPLASH_DIR / "onyx_intro_poster.jpg"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", "5.0",
         "-i", str(SRC_SPLASH_VIDEO), "-frames:v", "1", "-q:v", "4",
         "-y", str(poster)],
        check=True,
    )
    return {
        "source": SRC_SPLASH_VIDEO.name,
        "mp4_bytes": dst.stat().st_size,
        "poster_bytes": poster.stat().st_size,
        "mp4": str(dst.relative_to(ROOT)).replace(os.sep, "/"),
    }


def main() -> int:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("ffmpeg not found — cannot process the video assets", file=sys.stderr)
        return 1
    for p in (SRC_LOGO, SRC_MARK_VIDEO, SRC_SPLASH_VIDEO):
        if not p.exists():
            print(f"missing source file: {p.name}", file=sys.stderr)
            return 1

    print("ffmpeg:", ffmpeg)
    logo = build_image_watermark()
    print("image watermark :", logo)
    mark = build_video_watermark(ffmpeg)
    print("video watermark :", mark)
    splash = build_splash(ffmpeg)
    print("splash          :", splash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
