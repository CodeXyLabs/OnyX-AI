# -*- coding: utf-8 -*-
"""
OnyX Python Core · media_image.py
==================================
Генератор изображений с водяным знаком OnyX.

Режим A (основной, офлайн): процедурная генерация изображения из текста —
градиенты, фигуры, полутона — на чистом Python + Pillow. Водяной знак
обязательно впечатывается в правый нижний угол каждого изображения.

Режим B (онлайн, при доступности сети): запрос к pollinations.ai; результат
так же проходит через водяной знак.

Использует два доступных водяных знака:
  - python_core/watermark_logo_clean.png   (из файла 66d8bed9-.../логотип OnyX)
  - python_core/watermark_video_clean.png  (из видео-водяного знака)
"""

import hashlib
import io
import math
import os
import random
import time
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WM_LOGO = None
WM_VIDEO = None

# каталог с медиа-файлами
MEDIA_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(MEDIA_DIR)
LOGO_PATH = os.path.join(CORE_DIR, "watermark_logo_clean.png")
VIDEO_WM_PATH = os.path.join(CORE_DIR, "watermark_video_clean.png")
GEN_DIR = os.path.join(CORE_DIR, "generated")

os.makedirs(GEN_DIR, exist_ok=True)


def _load_logo():
    global WM_LOGO
    if WM_LOGO is None and os.path.exists(LOGO_PATH):
        try:
            WM_LOGO = Image.open(LOGO_PATH).convert("RGBA")
        except Exception:
            WM_LOGO = None
    elif WM_LOGO is None and os.path.exists(os.path.join(CORE_DIR, "watermark_logo.png")):
        try:
            WM_LOGO = Image.open(os.path.join(CORE_DIR, "watermark_logo.png")).convert("RGBA")
        except Exception:
            WM_LOGO = None
    return WM_LOGO


def _load_video_mark():
    global WM_VIDEO
    if WM_VIDEO is None:
        for p in (VIDEO_WM_PATH, os.path.join(MEDIA_DIR, "mark_frames", "frame_000.png"),
                  os.path.join(CORE_DIR, "assets", "watermark", "onyx_mark.png")):
            if os.path.exists(p):
                try:
                    WM_VIDEO = Image.open(p).convert("RGBA")
                    break
                except Exception:
                    continue
    return WM_VIDEO


def _hash_seed(text):
    h = 2166136261
    for ch in str(text):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _rng(seed):
    s = seed
    def r():
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s / 4294967296.0
    return r


def _find_font(size):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------- процедурная генерация
THEMES = {
    "abstract": {"hue": [0, 30, 300], "dark": True},
    "cyber": {"hue": [185, 265, 320], "dark": True},
    "space": {"hue": [220, 265, 290], "dark": True},
    "nature": {"hue": [90, 140, 170], "dark": False},
    "warm": {"hue": [10, 30, 45], "dark": False},
    "cool": {"hue": [190, 210, 240], "dark": False},
    "mono": {"hue": [0, 0, 0], "dark": True},
}


def classify_prompt(prompt):
    p = prompt.lower()
    if any(k in p for k in ["космос", "звезд", "планет", "галактик", "nebula", "space"]):
        return "space"
    if any(k in p for k in ["кибер", "неон", "город", "city", "cyber"]):
        return "cyber"
    if any(k in p for k in ["природ", "лес", "гор", "море", "океан", "река", "nature", "forest", "mountain", "ocean", "поле"]):
        return "nature"
    if any(k in p for k in ["закат", "огонь", "пламя", "костёр", "fire", "sunset"]):
        return "warm"
    if any(k in p for k in ["лёд", "снег", "вода", "зима", "мороз", "ice", "winter"]):
        return "cool"
    if any(k in p for k in ["тихий", "тёмный", "чёрный", "градиент", "монохром", "bw", "black"]):
        return "mono"
    return "abstract"


def procedural_image(prompt, size=768):
    """Рисует абстрактную «картину» по ключевым словам промпта."""
    theme = THEMES[classify_prompt(prompt)]
    seed = _hash_seed(prompt + str(size))
    rnd = _rng(seed)
    img = Image.new("RGB", (size, size), (8, 8, 12))
    draw = ImageDraw.Draw(img)

    # фон: радиальный/линейный градиент
    base = (rnd(), rnd(), rnd())
    c1 = (int(26 * (base[0] + 0.15)), int(30 * (base[1] + 0.1)), int(44 * (base[2] + 0.2)))
    c2 = (int(3 + 18 * base[0]), int(3 + 20 * base[1]), int(6 + 30 * base[2]))
    for y in range(size):
        t = y / size
        mixed = tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))
        draw.line([(0, y), (size, y)], fill=mixed)

    # большие «облака» — цветовые пятна
    for k in range(26):
        hue = theme["hue"][k % len(theme["hue"])] + rnd() * 30 - 15
        cx = rnd() * size
        cy = rnd() * size
        rad = size * (0.12 + rnd() * 0.4)
        alpha = 0.06 + rnd() * 0.18
        # рисуем градиентное пятно через оверлей
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for r in range(int(rad), 0, -4):
            a = int(alpha * 255 * (1 - r / rad))
            color = hsv_to_rgb(hue, 0.65 + rnd() * 0.2, 0.5 + rnd() * 0.35)
            ld.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (a,))
        img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
        draw = ImageDraw.Draw(img)

    # тонкие линии / структура
    for k in range(int(14 + rnd() * 26)):
        hue = theme["hue"][k % len(theme["hue"])] + rnd() * 20
        x1 = rnd() * size
        y1 = rnd() * size
        x2 = x1 + (rnd() - 0.5) * size * 0.8
        y2 = y1 + (rnd() - 0.5) * size * 0.8
        color = hsv_to_rgb(hue, 0.4 + rnd() * 0.3, 0.6 + rnd() * 0.3)
        width = 1 + rnd() * 2.5
        draw.line([(x1, y1), (x2, y2)], fill=color + (40,), width=int(width))

    # точечные «звёзды» в тёмных темах
    if theme["dark"]:
        for k in range(180):
            x, y = rnd() * size, rnd() * size
            r = rnd()
            br = 90 + int(r * 165)
            draw.ellipse([x, y, x + 1.5, y + 1.5], fill=(br, br, min(255, br + 20)))

    # виньетка
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for i in range(40):
        inset = i
        alpha = int(2.2 * i)
        od.rectangle([inset, inset, size - inset, size - inset], outline=(0, 0, 0, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay)

    return img.convert("RGB")


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


def stamp_watermark(img, mark=None, scale=0.26, corner="br"):
    """Впечатывает водяной знак в указанный угол. resized: исходное изображение."""
    mark = mark if mark is not None else _load_logo()
    if mark is None:
        return img, False
    w, h = img.size
    target_w = max(96, int(w * scale))
    ratio = target_w / mark.width
    target_h = max(1, int(mark.height * ratio))
    mark = mark.resize((target_w, target_h), Image.LANCZOS)
    margin = max(12, int(w * 0.028))
    if corner == "br":
        x, y = w - target_w - margin, h - target_h - margin
    elif corner == "tr":
        x, y = w - target_w - margin, margin
    elif corner == "bl":
        x, y = margin, h - target_h - margin
    else:
        x, y = margin, margin
    # полупрозрачная подложка
    img_rgba = img.convert("RGBA")
    pad = max(8, int(target_h * 0.12))
    plate = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    pd = ImageDraw.Draw(plate)
    pd.rounded_rectangle([x - pad, y - pad, x + target_w + pad, y + target_h + pad],
                         radius=int(target_h * 0.3), fill=(0, 0, 0, 100))
    img_rgba = Image.alpha_composite(img_rgba, plate)
    img_rgba.paste(mark, (x, y), mark)
    return img_rgba.convert("RGB"), True


def _apply_style(img, style):
    """Применяет стиль к изображению: насыщенность, контраст, цветовой тон."""
    try:
        from PIL import ImageEnhance
        import numpy as np
        if style == "cinematic":
            img = ImageEnhance.Contrast(img).enhance(1.15)
            img = ImageEnhance.Color(img).enhance(1.1)
            # лёгкий сине-оранжевый тон
            arr = np.array(img).astype(int)
            arr[..., 0] = np.clip(arr[..., 0] * 0.96 + 12, 0, 255)
            arr[..., 2] = np.clip(arr[..., 2] * 1.06 - 6, 0, 255)
            img = Image.fromarray(arr.astype("uint8"), "RGB")
        elif style == "anime":
            img = ImageEnhance.Color(img).enhance(1.5)
            img = ImageEnhance.Contrast(img).enhance(1.1)
        elif style == "photo":
            img = ImageEnhance.Sharpness(img).enhance(1.2)
        elif style == "neon":
            img = ImageEnhance.Color(img).enhance(1.55)
            img = ImageEnhance.Contrast(img).enhance(1.2)
            arr = np.array(img).astype(int)
            arr[..., 0] = np.clip(arr[..., 0] * 1.05, 0, 255)
            arr[..., 2] = np.clip(arr[..., 2] * 1.12, 0, 255)
            img = Image.fromarray(arr.astype("uint8"), "RGB")
        elif style == "minimal":
            img = ImageEnhance.Color(img).enhance(0.7)
    except Exception:
        pass
    return img


def generate_image(prompt, size=768, use_online=True, timeout=40, style=None):
    """Генерирует изображение и возвращает путь к файлу + данные о водяном знаке.
    Приоритет источников: Pollinations → Openverse → объектный движок → процедурный."""
    ts = int(time.time() * 1000)
    mode = "procedural"
    img = None

    # 1) Онлайн-генерация: Pollinations (если доступен)
    if use_online:
        try:
            import services as svc
            data = svc.pollinations_image(prompt, size=size, seed=_hash_seed(prompt) % 10_000_000)
            if data:
                img = Image.open(io.BytesIO(data)).convert("RGB")
                mode = "online-ai"
        except Exception:
            pass

    # 2) Онлайн-подбор реальных изображений: Openverse (фото/арт по запросу)
    if img is None and use_online:
        try:
            import services as svc
            hits = svc.openverse_search_images(prompt, limit=3)
            if hits:
                data = svc._fetch(hits[0]["url"], timeout=30)
                if data:
                    try:
                        cand = Image.open(io.BytesIO(data)).convert("RGB")
                        # масштабируем к нужному размеру с сохранением пропорций
                        cand.thumbnail((size, size))
                        canvas = Image.new("RGB", (size, size), (20, 20, 30))
                        ox, oy = (size - cand.width) // 2, (size - cand.height) // 2
                        canvas.paste(cand, (ox, oy))
                        img = canvas
                        mode = "openverse"
                    except Exception:
                        img = None
        except Exception:
            pass

    # 3) Объектный движок: рисует сцену по объектам из запроса
    if img is None:
        try:
            from .objects import render_scene, parse_objects
            objs = parse_objects(prompt)
            if objs:
                img = render_scene(size, size, prompt, time_t=0.5, seed=_hash_seed(prompt) % 999983).convert("RGB")
                mode = "objects"
        except Exception:
            pass

    # 4) Процедурная абстракция (самый надёжный fallback)
    if img is None:
        img = procedural_image(prompt, size)

    if style:
        img = _apply_style(img, style)

    img, stamped = stamp_watermark(img, corner="br")
    out_path = os.path.join(GEN_DIR, f"onyx_img_{ts}.jpg")
    img.save(out_path, "JPEG", quality=92)
    return {
        "path": out_path,
        "url": "/generated/" + os.path.basename(out_path),
        "width": img.width,
        "height": img.height,
        "watermarked": stamped,
        "mode": mode,
        "style": style,
    }


def placeholder_image(prompt="onyx", size=512):
    img = Image.new("RGB", (size, size), (10, 10, 16))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / size
        draw.line([(0, y), (size, y)], fill=(int(22 * (1 - t) + 6 * t), int(26 * (1 - t) + 6 * t), int(38 * (1 - t) + 10 * t)))
    for i in range(40):
        x, y = _rng(_hash_seed(prompt + str(i)))(), _hash_seed(prompt + str(i)) % size
        r = 15 + _rng(_hash_seed(prompt + str(i)))() * 40
        hue = 0 if i % 2 else 6
        for rr in range(int(r), 0, -2):
            a = int(30 * (1 - rr / r))
            draw.ellipse([x - rr, y - rr, x + rr, y + rr], fill=hsv_to_rgb(hue, 0.9, 0.4) + (a,))
    draw.text((size // 2 - 40, size // 2 - 10), "OnyX", fill=(230, 230, 230), font=_find_font(28))
    img, _ = stamp_watermark(img, corner="br")
    out_path = os.path.join(GEN_DIR, f"onyx_placeholder_{ts_now()}.jpg")
    img.save(out_path, "JPEG", quality=85)
    return {"path": out_path, "url": "/generated/" + os.path.basename(out_path),
            "width": img.width, "height": img.height, "watermarked": True, "mode": "placeholder"}


_ts_counter = 0
def ts_now():
    global _ts_counter
    _ts_counter += 1
    return int(time.time() * 1000) + _ts_counter