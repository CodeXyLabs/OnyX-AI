"""
Refine the keyed video watermark into a clean matte.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "assets" / "watermark" / "_raw"
OUT = ROOT / "assets" / "watermark" / "onyx_mark.png"


def build_alpha(rgb: np.ndarray, lo: float = 55.0, hi: float = 135.0) -> np.ndarray:
    """Distance-from-key matte.

    The key colour is a flat blue (0,69,185). For every pixel we measure the
    Euclidean distance to it: at 0 distance the pixel is background, far away it
    is solid artwork, and in between we keep a soft ramp so the round edges stay
    smooth instead of aliased.
    """
    key = np.array([0.0, 69.0, 185.0])
    d = np.sqrt(((rgb.astype(np.float32) - key) ** 2).sum(axis=2))
    a = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    return (a * 255.0).astype(np.uint8)


def despill(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Remove the blue fringe that survives around edges.

    Only pixels close to the edge are touched: the mark's own grey and red parts
    are already blue-free, and clamping them would wash the artwork out.
    """
    out = rgb.astype(np.float32).copy()
    # Edge mask: alpha is partial, or the pixel sits next to transparency.
    a_img = Image.fromarray(alpha, "L")
    eroded = np.array(a_img.filter(ImageFilter.MinFilter(5)))
    edge = (alpha > 0) & ((alpha < 250) | (eroded < 200))
    if edge.any():
        r, g, b = out[..., 0], out[..., 1], out[..., 2]
        excess = np.clip(b - np.maximum(r, g), 0, None)
        scale = np.ones_like(b)
        scale[edge] = 1.0 - np.clip(excess[edge] / 160.0, 0.0, 0.95)
        out[..., 1] = np.where(edge, g + excess * 0.35, g)   # pull some blue back into green
        out[..., 2] = np.where(edge, b * scale, b)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    if not TMP.exists():
        print("run tools/prepare_assets.py first (missing", TMP, ")")
        return 1
    frames = sorted(TMP.glob("frame_*.png"))
    if not frames:
        print("no raw frames found")
        return 1

    best = None
    for f in frames:
        img = Image.open(f).convert("RGB")
        rgb = np.array(img)
        alpha = build_alpha(rgb)
        solid = int(np.count_nonzero(alpha > 200))
        clear = int(np.count_nonzero(alpha < 16))
        total = alpha.size
        # We want a mark that is mostly solid, with a clean background.
        score = solid - 0
        print(f"  {f.name}: solid={solid/total:.1%} clear={clear/total:.1%}")
        if best is None or score > best[0]:
            best = (score, f, rgb, alpha)

    _, path, rgb, alpha = best
    print("chosen:", path.name)

    rgba = np.dstack([despill(rgb, alpha), alpha])
    img = Image.fromarray(rgba, "RGBA")

    # Trim to content.
    ys, xs = np.where(alpha > 12)
    pad = 8
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
           min(img.width, xs.max() + 1 + pad), min(img.height, ys.max() + 1 + pad))
    img = img.crop(box)

    target_w = 420
    scale = target_w / img.width
    img = img.resize((target_w, max(1, round(img.height * scale))), Image.LANCZOS)
    img.save(OUT, "PNG", optimize=True)

    a = np.array(img)[..., 3]
    print("written", OUT, img.size, OUT.stat().st_size, "bytes")
    print(f"  alpha: mean {a.mean():.0f}, fully transparent {np.mean(a == 0):.1%}, solid {np.mean(a > 200):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
