# -*- coding: utf-8 -*-
"""
OnyX Python Core · media_video.py
==================================
Генератор видео с анимированным водяным знаком.

Особенности:
  - процедурная анимация, зависящая от текста (палитра, тип сцены, частицы);
  - водяной знак берётся из видео-исходника (20557512903395.mp4 / onyx_mark_source.mp4),
    при этом СИНИЙ ФОН УБИРАЕТСЯ хромакеем;
  - знак перемещается по углам случайным образом каждые ~2 секунды;
  - итоговый файл кодируется в MP4 (H.264) через ffmpeg (imageio-ffmpeg);
  - при недоступности ffmpeg создаётся анимированный GIF (fallback).
"""

import io
import math
import os
import random
import struct
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = None

MEDIA_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(MEDIA_DIR)
GEN_DIR = os.path.join(CORE_DIR, "generated")
os.makedirs(GEN_DIR, exist_ok=True)

MARK_FRAMES_DIR = os.path.join(CORE_DIR, "mark_frames")
MARK_FRAMES = None


def hash_seed(text):
    h = 2166136261
    for ch in str(text):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def make_rng(seed):
    s = seed
    def r():
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s / 4294967296.0
    return r


def hsv_to_rgb(h, s, v):
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


# ------------------------------------------------------------------- палитры и классификация
PALETTES = [
    {"bg": [(5, 6, 10), (13, 27, 42), (27, 38, 59)], "acc": [190, 320, 12]},
    {"bg": [(10, 4, 3), (43, 15, 8), (67, 20, 7)], "acc": [12, 28, 45]},
    {"bg": [(3, 17, 15), (6, 42, 36), (11, 59, 51)], "acc": [150, 175, 200]},
    {"bg": [(8, 4, 15), (28, 16, 48), (44, 26, 74)], "acc": [265, 285, 310]},
    {"bg": [(5, 5, 5), (20, 20, 20), (31, 31, 31)], "acc": [0, 0, 8]},
    {"bg": [(2, 6, 15), (4, 32, 61), (6, 50, 92)], "acc": [205, 220, 190]},
    {"bg": [(18, 2, 7), (58, 10, 28), (92, 20, 48)], "acc": [340, 15, 40]},
    {"bg": [(3, 8, 5), (10, 31, 18), (18, 51, 33)], "acc": [120, 140, 95]},
]

KEYWORDS = {
    "city": ["город", "city", "улиц", "неон", "cyberpunk", "киберпанк", "здани", "urban", "мегаполис"],
    "nature": ["природ", "лес", "горы", "гор", "поле", "луг", "дерев", "nature", "forest", "mountain", "meadow", "туман"],
    "space": ["космос", "звезд", "звёзд", "планет", "галакт", "space", "star", "galaxy", "nebula", "туманност", "вселенн", "лун"],
    "abstract": ["абстракт", "abstract", "градиент", "линии", "форма", "геометр", "паттерн"],
    "water": ["вода", "вод", "дожд", "река", "озер", "море", "океан", "закат", "восход", "водоём", "water", "rain", "river", "lake", "sea", "ocean", "sunset", "sunrise"],
    "fire": ["огон", "пожар", "пламя", "лава", "вулкан", "fire", "flame", "lava"],
}


def classify(prompt):
    p = prompt.lower()
    for key, lst in KEYWORDS.items():
        for kw in lst:
            if kw in p:
                return key
    return "abstract"


def design_scene(prompt):
    rng = make_rng(hash_seed(prompt))
    kind = classify(prompt)
    palette = PALETTES[int(rng() * len(PALETTES))]
    if kind in ("space", "abstract"):
        palette = PALETTES[3]
    if kind == "fire":
        palette = PALETTES[1]
    if kind == "water":
        palette = PALETTES[5]
    if kind == "nature":
        palette = PALETTES[7] if rng() < 0.5 else PALETTES[2]
    if kind == "city":
        palette = PALETTES[0]
    # объекты из запроса (для объектного рендера кадров)
    try:
        from .objects import parse_objects
        objects = parse_objects(prompt)
    except Exception:
        objects = []
    return {
        "kind": kind, "palette": palette, "rng": rng,
        "layers": 3 + int(rng() * 4),
        "particles": 60 + int(rng() * 140),
        "drift": 0.4 + rng() * 1.6,
        "pulse": 0.5 + rng() * 1.5,
        "seed": hash_seed(prompt),
        "prompt": prompt,
        "objects": objects,
    }


# ------------------------------------------------------------------- кадры
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def draw_realistic_scene(width, height, scene, t, elapsed, kind):
    """Реалистичные сцены с градиентами: закат/водная гладь, горы, космос.
    Возвращает небо-изображение (RGBA) или None, если сцена не поддерживается."""
    rng = scene["rng"]
    img = Image.new("RGB", (width, height), (10, 10, 20))
    dr = ImageDraw.Draw(img)

    if kind == "water" or kind == "nature":
        # ГОРИЗОНТ + НЕБО с градиентом заката
        horizon = int(height * 0.52)
        # небо: от тёмного верха к тёплому у горизонта
        sky_top = (12, 14, 30)
        sky_mid = (255, 138, 66)
        sky_low = (255, 210, 120)
        for y in range(horizon):
            tt = y / max(1, horizon)
            if tt < 0.55:
                col = mix(sky_top, sky_mid, tt / 0.55)
            else:
                col = mix(sky_mid, sky_low, (tt - 0.55) / 0.45)
            dr.line([(0, y), (width, y)], fill=col)
        # солнце с ореолом
        sun_x = width * (0.5 + 0.18 * math.sin(t * 0.5))
        sun_y = int(horizon * 0.78)
        sun_r = int(height * 0.09)
        glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for rr in range(sun_r * 3, 0, -4):
            a = int(120 * (1 - rr / (sun_r * 3)))
            gd.ellipse([sun_x - rr, sun_y - rr, sun_x + rr, sun_y + rr], fill=(255, 210, 120, a))
        img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
        dr = ImageDraw.Draw(img)
        dr.ellipse([sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r], fill=(255, 240, 200))
        # вода: отражающий градиент + дорожка света
        water_top = (60, 40, 70)
        water_bot = (8, 10, 25)
        for y in range(horizon, height):
            tt = (y - horizon) / max(1, height - horizon)
            col = mix(water_top, water_bot, tt)
            dr.line([(0, y), (width, y)], fill=col)
        # блики солнца на воде
        for i in range(16):
            ry = horizon + (i / 16) * (height - horizon)
            wl = int(width * (0.12 + 0.01 * i))
            bx = sun_x - wl + (i % 3) * int(width * 0.03)
            alpha = max(20, int(90 * (1 - i / 16)))
            dr.rectangle([bx, ry, bx + wl, ry + 2], fill=(255, 220, 160, alpha))
        # облака тонкие
        for c in range(5):
            ox = (c * 731 + int(t * 20)) % (width + 400) - 200
            oy = int(horizon * (0.15 + 0.3 * (c % 3) / 3)) + int(8 * math.sin(t * 0.7 + c))
            for oi in range(5):
                dr.ellipse([ox + oi * 40, oy, ox + oi * 40 + 70, oy + 12], fill=(255, 255, 255, 40))
        return img.convert("RGBA")

    if kind == "space":
        # космос с туманностью и планетой
        nebula_colors = [(30, 20, 60), (80, 30, 90), (40, 60, 120), (20, 40, 80)]
        neb = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        nd = ImageDraw.Draw(neb)
        for i in range(5):
            nx = (i * 463 + int(t * 10)) % width
            ny = (i * 233 + int(t * 6)) % height
            nr = width * (0.18 + 0.08 * i)
            col = nebula_colors[i % 4]
            for rr in range(int(nr), 0, -8):
                a = int(70 * (1 - rr / nr))
                nd.ellipse([nx - rr, ny - rr, nx + rr, ny + rr], fill=col + (a,))
        base = Image.new("RGB", (width, height), (5, 5, 14))
        img = Image.alpha_composite(base.convert("RGBA"), neb).convert("RGB")
        dr = ImageDraw.Draw(img)
        # звёзды разных размеров
        for s in range(240):
            sr = make_rng((scene["seed"] + s * 29) & 0xFFFFFFFF)
            sx, sy = sr() * width, sr() * height
            tw = 0.4 + 0.6 * abs(math.sin(t * (0.5 + sr()) + s))
            sz = 1.0 if sr() < 0.95 else 2.2
            br = int(60 + tw * 190)
            dr.ellipse([sx, sy, sx + sz, sy + sz], fill=(br, br, min(255, br + 20)))
        # планета с кольцом
        px, py = width * 0.72, height * 0.30
        pr = int(height * 0.16)
        dr.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(210, 170, 120))
        dr.ellipse([px - pr, py - pr, px + pr, py + pr], outline=(255, 230, 180), width=2)
        # кольцо — эллипс поверх планеты (без сложной маски)
        dr.arc([px - int(pr * 1.9), py - int(pr * 0.6), px + int(pr * 1.9), py + int(pr * 0.6)],
               start=170, end=370, fill=(240, 210, 170), width=3)
        return img.convert("RGBA")

    if kind == "city":
        # неоновая панорама города
        for y in range(height):
            tt = y / height
            col = mix((8, 10, 20), (25, 20, 45), tt)
            dr.line([(0, y), (width, y)], fill=col)
        ground = int(height * 0.74)
        for b in range(30):
            br = make_rng((scene["seed"] + b * 131) & 0xFFFFFFFF)
            bw = width * (0.02 + br() * 0.04)
            bx = br() * width + math.sin(t * 0.3 + b) * 6
            bh = height * (0.1 + br() * 0.4)
            dr.rectangle([bx, ground - bh, bx + bw, height], fill=(12, 14, 22))
            for win in range(10):
                wr = make_rng((scene["seed"] + b * 31 + win * 7) & 0xFFFFFFFF)
                if wr() < 0.4:
                    continue
                hue = [195, 320, 30][win % 3]
                wcol = hsv_to_rgb(hue, 0.9, 0.7)
                dr.rectangle([bx + 2 + (win % 3) * (bw / 3), ground - bh + 4 + (win // 3) * (bh / 4), bx + 2 + (win % 3) * (bw / 3) + bw / 5, ground - bh + 4 + (win // 3) * (bh / 4) + bh / 8], fill=wcol)
        # луны
        dr.ellipse([width * 0.15, height * 0.12, width * 0.15 + 26, height * 0.12 + 26], fill=(240, 240, 255))
        return img.convert("RGBA")

    return None


def draw_frame(width, height, scene, t, elapsed, corner, mark=None):
    # 1) Объектный движок: если запрос разобран на объекты — рисуем сцену из них
    if scene.get("objects"):
        try:
            from .objects import render_scene
            img = render_scene(width, height, scene.get("prompt", ""), time_t=t, seed=(scene["seed"] + int(t * 10)) & 0xFFFF)
            img = img.convert("RGB")
            # виньетка
            v = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            vd = ImageDraw.Draw(v)
            for i in range(22):
                alpha = int(4 * i)
                if alpha <= 255:
                    vd.rectangle([i, i, width - i, height - i], outline=(0, 0, 0, alpha))
            img = Image.alpha_composite(img, v).convert("RGB")
            if mark is not None:
                img = draw_mark(img, mark, width, height, corner)
            return img
        except Exception:
            pass
    # 2) пробуем реалистичную сцену; если не поддерживается — рисуем абстракцию
    realistic = draw_realistic_scene(width, height, scene, t, elapsed, scene["kind"])
    if realistic is not None:
        img = realistic.convert("RGB")
        draw = ImageDraw.Draw(img)
        # лёгкая дымка
        haze = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        hh = int(height * 0.12)
        for yy in range(hh):
            a = int(22 * (1 - yy / hh))
            hd.line([(0, height - hh + yy), (width, height - hh + yy)], fill=(255, 255, 255, a))
        img = Image.alpha_composite(img.convert("RGBA"), haze).convert("RGB")
        draw = ImageDraw.Draw(img)
        # виньетка
        v = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        vd = ImageDraw.Draw(v)
        for i in range(24):
            alpha = int(4 * i)
            if alpha <= 255:
                vd.rectangle([i, i, width - i, height - i], outline=(0, 0, 0, alpha))
        img = Image.alpha_composite(img.convert("RGBA"), v).convert("RGB")
        if mark is not None:
            img = draw_mark(img, mark, width, height, corner)
        return img

    # --- старая процедурная отрисовка (абстракция) ---
    p = scene["palette"]
    rng = scene["rng"]
    phase = t * scene["pulse"]
    img = Image.new("RGB", (width, height), p["bg"][0])
    draw = ImageDraw.Draw(img)

    # градиентный фон
    for y in range(height):
        tt = y / height
        if tt < 0.55:
            col = mix(p["bg"][0], p["bg"][1], tt / 0.55)
        else:
            col = mix(p["bg"][1], p["bg"][2], (tt - 0.55) / 0.45)
        draw.line([(0, y), (width, y)], fill=col)

    # цветные «облака»
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for i in range(scene["layers"]):
        seed = (scene["seed"] + i * 977) & 0xFFFFFFFF
        r2 = make_rng(seed)
        cx = r2() * width + math.sin(phase * (0.3 + i * 0.17) + i) * width * 0.22 * scene["drift"]
        cy = r2() * height + math.cos(phase * (0.24 + i * 0.13) + i) * height * 0.22 * scene["drift"]
        rad = width * (0.24 + r2() * 0.42)
        hue = p["acc"][i % len(p["acc"])]
        alpha = 0.10 + 0.10 * (0.5 + 0.5 * math.sin(phase + i))
        col = hsv_to_rgb(hue, 0.85, 0.5)
        for rr in range(int(rad), 0, -6):
            a = int(alpha * 255 * (1 - rr / rad))
            od.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=col + (a,))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # структура по типу
    if scene["kind"] == "city":
        ground_y = height * 0.72
        for b in range(26):
            br = make_rng((scene["seed"] + b * 131) & 0xFFFFFFFF)
            bw = width * (0.02 + br() * 0.05)
            bx = br() * width - width * 0.02 + math.sin(phase * 0.2 + b) * width * 0.01
            bh = height * (0.08 + br() * 0.42) * (1 + 0.05 * math.sin(phase + b))
            dark = (6, 8, 14)
            draw.rectangle([bx, ground_y - bh, bx + bw, height], fill=dark)
            for win in range(12):
                wr = make_rng((scene["seed"] + b * 31 + win * 7) & 0xFFFFFFFF)
                if wr() < 0.45:
                    continue
                wcol = hsv_to_rgb(p["acc"][win % 3], 0.9, 0.65)
                draw.rectangle([bx + bw * 0.18 + (win % 3) * bw * 0.26,
                                ground_y - bh + 14 + (win // 3) * (bh / 7),
                                bx + bw * 0.18 + (win % 3) * bw * 0.26 + max(2, bw * 0.12),
                                ground_y - bh + 14 + (win // 3) * (bh / 7) + max(2, bh / 16)],
                               fill=wcol)
    elif scene["kind"] == "space":
        for s in range(220):
            sr = make_rng((scene["seed"] + s * 29) & 0xFFFFFFFF)
            sx, sy = sr() * width, sr() * height
            tw = 0.4 + 0.6 * abs(math.sin(phase * (0.5 + sr()) + s))
            br = int(30 + tw * 220)
            draw.ellipse([sx, sy, sx + 1.5, sy + 1.5], fill=(br, br, min(255, br + 20)))
    elif scene["kind"] == "water":
        for wv in range(8):
            wcol = hsv_to_rgb(p["acc"][wv % 3], 0.8, 0.6)
            for x in range(0, width, 12):
                y = height * (0.42 + wv * 0.055) + math.sin(x * 0.006 + phase * (0.6 + wv * 0.2) + wv) * (10 + wv * 3)
                draw.rectangle([x, y, x + 6, y + 2], fill=wcol)
    elif scene["kind"] == "fire":
        for f in range(60):
            fr = make_rng((scene["seed"] + f * 53) & 0xFFFFFFFF)
            fy = height - ((elapsed * (0.25 + fr() * 0.5) + fr()) % 1) * height
            fx = width * (0.15 + fr() * 0.7) + math.sin(phase * 2 + f) * 26
            frad = width * (0.02 + fr() * 0.07)
            col = hsv_to_rgb(18 + fr() * 26, 1.0, 0.65)
            for rr in range(int(frad), 0, -3):
                a = int(90 * (1 - rr / frad))
                od2 = ImageDraw.Draw(Image.new("RGBA", (1, 1), col + (a,))).ellipse
            draw.ellipse([fx - frad, fy - frad, fx + frad, fy + frad], fill=col)
    else:
        # abstract: линии
        for k in range(7):
            col = hsv_to_rgb(p["acc"][k % 3], 0.85, 0.62)
            for xx in range(0, width, 10):
                yy = height * 0.5 + math.sin(xx * 0.004 + phase * (0.5 + k * 0.2) + k) * height * 0.28
                draw.rectangle([xx, yy, xx + 4, yy + 2], fill=col)

    # частицы (векторизованные: генерируем случайные позиции заранее, считаем np-массивом)
    particles = scene.get("particles", 90)
    pseed = (scene["seed"] + 7919) & 0xFFFFFFFF
    pr = make_rng(pseed)
    px_arr = np.array([pr() for _ in range(particles)])
    py_arr = np.array([pr() for _ in range(particles)])
    pd_arr = 0.3 + np.array([pr() for _ in range(particles)]) * 0.7
    px = (px_arr * width + np.sin(phase * (0.2 + pd_arr) + np.arange(particles)) * 40 * pd_arr + width) % width
    py = (py_arr * height - elapsed * (12 + pd_arr * 40) + height * 4) % height
    pa = 0.10 + pd_arr * 0.5 * np.abs(np.sin(phase + np.arange(particles)))
    for q in range(particles):
        sz = (0.7 + pd_arr[q] * 2.1) * 2
        br = int(255 * pa[q])
        draw.rectangle([px[q], py[q], px[q] + sz, py[q] + sz], fill=(br, br, br))

    # лёгкая «дымка» — полупрозрачная волна поверх нижней трети кадра
    haze = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    hd = ImageDraw.Draw(haze)
    hh = int(height * 0.16)
    for yy in range(hh):
        a = int(30 * (1 - yy / hh))
        offset = int(math.sin(elapsed * 0.3 + yy * 0.01) * width * 0.03)
        hd.line([(offset, height - hh + yy), (width + offset, height - hh + yy)], fill=(p["acc"][0] % 360, 40, 60, a) if False else (255, 255, 255, a))
    img = Image.alpha_composite(img.convert("RGBA"), haze).convert("RGB")
    draw = ImageDraw.Draw(img)

    # кинематографическая виньетка (градиентная маска: тёмные углы)
    v = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    vd = ImageDraw.Draw(v)
    for i in range(26):
        alpha = int(4.5 * i)
        if alpha <= 255:
            vd.rectangle([i, i, width - i, height - i], outline=(0, 0, 0, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), v).convert("RGB")
    draw = ImageDraw.Draw(img)

    # водяной знак
    if mark is not None:
        img = draw_mark(img, mark, width, height, corner)
    return img


def mix(c1, c2, t):
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


def draw_mark(img, mark, width, height, corner):
    """Вставляет водяной знак (RGBA) в заданный угол."""
    mark = mark.copy()
    tw = max(110, int(width * 0.30))
    ratio = tw / mark.width
    th = max(1, int(mark.height * ratio))
    mark = mark.resize((tw, th), Image.LANCZOS)
    margin = max(12, int(width * 0.028))
    if corner == "tl":
        x, y = margin, margin
    elif corner == "tr":
        x, y = width - tw - margin, margin
    elif corner == "bl":
        x, y = margin, height - th - margin
    else:
        x, y = width - tw - margin, height - th - margin
    # мягкая тень
    img = img.convert("RGBA")
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([x - 8, y - 8, x + tw + 8, y + th + 8], radius=10, fill=(0, 0, 0, 110))
    img = Image.alpha_composite(img, shadow)
    img.paste(mark, (x, y), mark)
    return img.convert("RGB")


def load_mark_frames():
    """Загружает кадры водяного знака с убранным синим фоном (RGBA)."""
    global MARK_FRAMES
    if MARK_FRAMES is not None:
        return MARK_FRAMES
    frames = []
    if os.path.isdir(MARK_FRAMES_DIR):
        names = sorted(os.listdir(MARK_FRAMES_DIR))
        for n in names:
            if n.endswith(".png"):
                try:
                    frames.append(Image.open(os.path.join(MARK_FRAMES_DIR, n)).convert("RGBA"))
                except Exception:
                    continue
    if not frames:
        # fallback: статика
        for p in (os.path.join(CORE_DIR, "watermark_video_clean.png"),
                  os.path.join(CORE_DIR, "watermark_logo_clean.png"),
                  os.path.join(MEDIA_DIR, "..", "assets", "watermark", "onyx_mark.png")):
            if os.path.exists(p):
                try:
                    frames.append(Image.open(p).convert("RGBA"))
                    break
                except Exception:
                    continue
    MARK_FRAMES = frames
    return frames


def save_mp4(frames, out_path, fps=24):
    """Записывает кадры (PIL Images) в MP4 через ffmpeg."""
    if FFMPEG is None:
        return save_gif(frames, out_path.replace(".mp4", ".gif"))
    h, w = frames[0].size[1], frames[0].size[0]
    cmd = [
        FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait(timeout=60)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        return save_gif(frames, out_path.replace(".mp4", ".gif"))
    if proc.returncode != 0:
        return save_gif(frames, out_path.replace(".mp4", ".gif"))
    return out_path


def save_gif(frames, out_path, fps=15):
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return out_path


def generate_video(prompt, seconds=8, width=960, height=540, fps=24, on_progress=None, corner_every=2.0, use_online=True):
    """Генерирует видео и сохраняет файл. Возвращает информацию о файле."""
    scene = design_scene(prompt)
    marks = load_mark_frames()
    mark = marks[0] if marks else None
    corners = ["tl", "tr", "bl", "br"]
    rng = make_rng(hash_seed(prompt) ^ 0x5F3759DF)
    corner = corners[int(rng() * 4)]
    next_switch = corner_every
    total = int(seconds * fps)
    frames = []

    # Онлайн-режим: пробуем получить серию сгенерированных кадров от Pollinations,
    # чтобы сделать «живое» видео с движением камеры по реальным изображениям.
    online_base = None
    if use_online and not scene.get("objects"):
        try:
            import services as svc
            shots = svc.pollinations_image_variants(prompt, count=min(3, max(2, total // 20)), size=max(width, height))
            if len(shots) >= 2:
                online_base = []
                for data in shots:
                    try:
                        from PIL import Image as _I
                        im = _I.open(io.BytesIO(data)).convert("RGB").resize((width, height), _I.BILINEAR)
                        online_base.append(im)
                    except Exception:
                        continue
        except Exception:
            online_base = None

    frame_id = 0
    for i in range(total):
        t = i / fps
        elapsed = t
        if t >= next_switch:
            pick = corners[int(rng() * 4)]
            while pick == corner and len(corners) > 1:
                pick = corners[int(rng() * 4)]
            corner = pick
            next_switch = t + corner_every
        # вращаем кадры знака, чтобы знак был «живым»
        cur_mark = mark
        if marks and len(marks) > 1:
            cur_mark = marks[i % len(marks)]
        if online_base is not None:
            # панорама по generated-кадрам: плавный сдвиг/зум
            nf = len(online_base)
            base = online_base[min(nf - 1, int(elapsed / max(0.5, seconds) * nf))]
            zoom = 1.06 + 0.04 * math.sin(elapsed * 0.5)
            w2, h2 = int(width * zoom), int(height * zoom)
            frame = base.resize((w2, h2), Image.BILINEAR)
            x0 = int((w2 - width) * (0.5 + 0.4 * math.sin(elapsed * 0.7)))
            y0 = int((h2 - height) * (0.5 + 0.4 * math.cos(elapsed * 0.5)))
            frame = frame.crop((max(0, x0), max(0, y0), max(0, x0 + width), max(0, y0 + height)))
            frame = frame.resize((width, height), Image.BILINEAR)
            # виньетка
            v = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            vd = ImageDraw.Draw(v)
            for vi in range(20):
                a = int(3.5 * vi)
                if a <= 255:
                    vd.rectangle([vi, vi, width - vi, height - vi], outline=(0, 0, 0, a))
            frame = Image.alpha_composite(frame.convert("RGBA"), v).convert("RGB")
        else:
            frame = draw_frame(width, height, scene, t, elapsed, corner, cur_mark)
            # кинематографическое «дыхание камеры»: очень медленный zoom
            zoom = 1.0 + 0.02 * math.sin(elapsed * 0.4)
            if abs(zoom - 1.0) > 0.001:
                w2, h2 = int(width * zoom), int(height * zoom)
                frame = frame.resize((w2, h2), Image.BILINEAR)
                x0, y0 = (w2 - width) // 2, (h2 - height) // 2
                frame = frame.crop((x0, y0, x0 + width, y0 + height))
        if mark is not None:
            frame = draw_mark(frame, mark, width, height, corner)
        frames.append(frame)
        if on_progress and i % 4 == 0:
            on_progress(i / total)
    ts = int(time.time() * 1000)
    out_path = os.path.join(GEN_DIR, f"onyx_video_{ts}.mp4")
    final = save_mp4(frames, out_path, fps=fps)
    ext = os.path.splitext(final)[1].lstrip(".")
    mode = "online-ai" if online_base is not None else ("objects" if scene.get("objects") else "procedural")
    return {
        "path": final,
        "url": "/generated/" + os.path.basename(final),
        "ext": ext,
        "seconds": seconds,
        "width": width,
        "height": height,
        "watermarked": mark is not None,
        "cornerEvery": corner_every,
        "mode": mode,
        "bytes": os.path.getsize(final),
    }


def poster(prompt, width=960, height=540):
    """Один кадр-постер."""
    scene = design_scene(prompt)
    mark = load_mark_frames()
    cur_mark = mark[0] if mark else None
    img = draw_frame(width, height, scene, 3.5, 3.5, "br", cur_mark)
    ts = int(time.time() * 1000)
    out = os.path.join(GEN_DIR, f"onyx_poster_{ts}.jpg")
    img.save(out, "JPEG", quality=88)
    return {"path": out, "url": "/generated/" + os.path.basename(out), "width": width, "height": height}


def extract_mark_frames_from_video(video_path, out_dir=None):
    """Извлекает кадры из видео и убирает синий фон (хромакей). Используется при первом запуске."""
    out_dir = out_dir or MARK_FRAMES_DIR
    os.makedirs(out_dir, exist_ok=True)
    try:
        import imageio.v3 as iio
    except ImportError:
        return False
    frames = []
    try:
        for i, f in enumerate(iio.imiter(video_path)):
            frames.append(f)
            if len(frames) >= 30:
                break
    except Exception:
        return False
    if not frames:
        return False
    # убираем синий фон в каждом кадре
    saved = 0
    for i, f in enumerate(frames):
        rgb = np.array(f)[..., :3].astype(int)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        blue = (b > 60) & (b > r + 30) & (b > g + 20)
        dark = (r < 12) & (g < 12) & (b < 12)
        alpha = np.where(blue | dark, 0, 255).astype(np.uint8)
        out = np.dstack([np.array(f)[..., :3], alpha]).astype(np.uint8)
        Image.fromarray(out, "RGBA").save(os.path.join(out_dir, f"frame_{i:03d}.png"))
        saved += 1
    # обрезаем по общему bbox содержимого
    try:
        xs, ys = [], []
        for n in sorted(os.listdir(out_dir)):
            if not n.endswith(".png"):
                continue
            im = np.array(Image.open(os.path.join(out_dir, n)).convert("RGBA"))
            sel = im[..., 3] > 40
            yy, xx = np.where(sel)
            if len(xx):
                xs.extend([xx.min(), xx.max()])
                ys.extend([yy.min(), yy.max()])
        if xs and ys:
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            for n in sorted(os.listdir(out_dir)):
                if n.endswith(".png"):
                    p = os.path.join(out_dir, n)
                    Image.open(p).convert("RGBA").crop((x0, y0, x1 + 1, y1 + 1)).save(p)
    except Exception:
        pass
    # сброс кеша
    global MARK_FRAMES
    MARK_FRAMES = None
    return saved > 0