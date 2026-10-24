# -*- coding: utf-8 -*-
"""
OnyX Python Core · objects.py
==============================
Библиотека процедурных объектов для генерации изображений и видео.
Каждый объект умеет отрисовывать себя на канвасе в заданной позиции/масштабе.
Парсер запроса разбивает описание на объекты и строит сцену.

Объекты: небо, солнце, луна, облака, горы, деревья, дома, машины, люди,
животные (собака, кошка, птица, рыба, бабочка), цветы, трава, вода,
ракета, корабль, звезды, снежинки, огонь и другие.

Рисование чисто процедурное (PIL), без внешних зависимостей.
"""

import math

from PIL import Image, ImageDraw

try:
    import numpy as np
except ImportError:
    np = None


# ------------------------------------------------------------------ хелперы
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


def mix(c1, c2, t):
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def iround(x):
    return int(round(x))


# ------------------------------------------------------------------ объекты
def draw_sky(draw, w, h, palette, flavor=None):
    top, mid, low = palette
    for y in range(h):
        tt = y / max(1, h)
        if tt < 0.5:
            col = mix(top, mid, tt / 0.5)
        else:
            col = mix(mid, low, (tt - 0.5) / 0.5)
        draw.line([(0, y), (w, y)], fill=col)


def draw_sun(draw, w, h, x, y, r, alpha=255):
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for rr in range(r * 2, 0, -3):
        a = int(alpha * (1 - rr / (r * 2)) * 0.6)
        gd.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(255, 220, 140, a))
    return glow


def draw_moon(draw, w, h, x, y, r):
    draw.ellipse([x - r, y - r, x + r, y + r], fill=(235, 235, 250))
    # кратеры
    craters = [(0.3, 0.2, 0.35), (-0.3, 0.4, 0.22), (0.1, -0.3, 0.18)]
    for cx, cy, cr in craters:
        draw.ellipse([x + cx * r - cr * r, y + cy * r - cr * r, x + cx * r + cr * r, y + cy * r + cr * r], fill=(205, 205, 225))


def draw_cloud(draw, x, y, w, scale=1.0, shade=(255, 255, 255)):
    for i in range(5):
        cw = w * (0.35 + 0.12 * i)
        ch = cw * 0.4
        cx = x + i * w * 0.18 - w * 0.25
        cy = y - (0.15 - 0.05 * i) * ch
        draw.ellipse([cx, cy, cx + cw, cy + ch], fill=shade + (35,))


def draw_mountain(draw, x, w, base_y, height, color):
    points = [(x, base_y), (x + w / 2, base_y - height), (x + w, base_y)]
    draw.polygon(points, fill=color)
    # снеговая вершина
    points2 = [(x + w * 0.42, base_y - height * 0.86), (x + w / 2, base_y - height), (x + w * 0.58, base_y - height * 0.86)]
    draw.polygon(points2, fill=(245, 245, 250))


def draw_tree(draw, x, base_y, scale=1.0, leaf_color=(30, 110, 70), trunk=(110, 80, 50)):
    h = 70 * scale
    # ствол
    draw.rectangle([x - 5 * scale, base_y - h * 0.45, x + 5 * scale, base_y], fill=trunk)
    # крона из трёх кругов
    for i in range(3):
        cx = x + (i - 1) * 16 * scale
        cy = base_y - h * 0.55 - (i % 2) * 8 * scale
        r = 20 * scale
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=leaf_color)


def draw_house(draw, x, base_y, scale=1.0, wall=(190, 150, 100), roof=(140, 60, 50), window=(255, 215, 120)):
    w = 70 * scale
    h = 45 * scale
    draw.rectangle([x - w / 2, base_y - h, x + w / 2, base_y], fill=wall)
    draw.polygon([(x - w / 2 - 6 * scale, base_y - h), (x, base_y - h - 30 * scale), (x + w / 2 + 6 * scale, base_y - h)], fill=roof)
    # дверь
    draw.rectangle([x - 8 * scale, base_y - 22 * scale, x + 8 * scale, base_y], fill=(80, 50, 30))
    # окна
    for i in range(2):
        wx = x - w / 2 + 12 * scale + i * 30 * scale
        draw.rectangle([wx, base_y - h + 10 * scale, wx + 14 * scale, base_y - h + 26 * scale], fill=window)


def draw_car(draw, x, y, scale=1.0, color=(200, 60, 60)):
    w = 50 * scale
    h = 18 * scale
    draw.rounded_rectangle([x - w / 2, y - h, x + w / 2, y], radius=int(6 * scale), fill=color)
    # кабина
    draw.rounded_rectangle([x - w * 0.15, y - h * 1.9, x + w * 0.25, y - h * 1.1], radius=int(5 * scale), fill=(140, 170, 200))
    # колёса
    for wx in (-w * 0.3, w * 0.3):
        draw.ellipse([x + wx - 5 * scale, y - 5 * scale, x + wx + 5 * scale, y + 5 * scale], fill=(30, 30, 30))


def draw_person(draw, x, y, scale=1.0, shirt=(60, 90, 160), pants=(60, 60, 70)):
    head_r = 6 * scale
    draw.ellipse([x - head_r, y - head_r - 22 * scale, x + head_r, y - head_r - 8 * scale], fill=(230, 190, 150))
    draw.rectangle([x - 7 * scale, y - 8 * scale, x + 7 * scale, y + 10 * scale], fill=shirt)
    draw.rectangle([x - 7 * scale, y + 10 * scale, x + 7 * scale, y + 22 * scale], fill=pants)
    # руки
    draw.line([x - 7 * scale, y - 2 * scale, x - 13 * scale, y + 8 * scale], fill=shirt, width=int(3 * scale))
    draw.line([x + 7 * scale, y - 2 * scale, x + 13 * scale, y + 8 * scale], fill=shirt, width=int(3 * scale))


def draw_dog(draw, x, y, scale=1.0, color=(140, 100, 60)):
    body_w, body_h = 26 * scale, 14 * scale
    draw.ellipse([x - body_w / 2, y - body_h, x + body_w / 2, y], fill=color)
    # голова
    draw.ellipse([x + body_w * 0.35, y - body_h - 8 * scale, x + body_w * 0.65, y - body_h + 10 * scale], fill=color)
    # уши
    draw.ellipse([x + body_w * 0.32, y - body_h - 14 * scale, x + body_w * 0.42, y - body_h - 4 * scale], fill=(90, 60, 40))
    draw.ellipse([x + body_w * 0.58, y - body_h - 14 * scale, x + body_w * 0.68, y - body_h - 4 * scale], fill=(90, 60, 40))
    # лапы
    for px in (-body_w * 0.35, body_w * 0.1):
        draw.rectangle([x + px, y - 3 * scale, x + px + 7 * scale, y + 3 * scale], fill=color)
    # хвост
    draw.line([x - body_w / 2, y - body_h * 0.8, x - body_w / 2 - 12 * scale, y - body_h * 1.3], fill=color, width=int(4 * scale))


def draw_cat(draw, x, y, scale=1.0, color=(90, 90, 100)):
    body_w, body_h = 22 * scale, 13 * scale
    draw.ellipse([x - body_w / 2, y - body_h, x + body_w / 2, y], fill=color)
    # голова
    head_r = 7 * scale
    draw.ellipse([x + body_w * 0.4, y - body_h - 6 * scale, x + body_w * 0.4 + head_r * 2, y - body_h - 6 * scale + head_r * 2], fill=color)
    # уши
    draw.polygon([(x + body_w * 0.45, y - body_h - 18 * scale), (x + body_w * 0.55, y - body_h - 10 * scale), (x + body_w * 0.68, y - body_h - 16 * scale)], fill=color)
    # хвост
    draw.line([x - body_w / 2, y - 4 * scale, x - body_w / 2 - 16 * scale, y - 16 * scale], fill=color, width=int(3 * scale))


def draw_bird(draw, x, y, scale=1.0, color=(40, 80, 140)):
    # тело
    draw.ellipse([x - 6 * scale, y - 4 * scale, x + 6 * scale, y + 4 * scale], fill=color)
    # голова
    draw.ellipse([x + 5 * scale, y - 6 * scale, x + 10 * scale, y - 2 * scale], fill=color)
    # клюв
    draw.polygon([(x + 9 * scale, y - 4 * scale), (x + 14 * scale, y - 3 * scale), (x + 9 * scale, y - 2 * scale)], fill=(230, 180, 60))
    # крылья (машущие — амплитуда по времени)
    wing = 6 * scale + 4 * scale * math.sin(10 * x)
    draw.polygon([(x - 2 * scale, y - 1 * scale), (x - 8 * scale, y - wing), (x - 1 * scale, y + 2 * scale)], fill=mix(color, (30, 30, 30), 0.3))


def draw_fish(draw, x, y, scale=1.0, color=(230, 140, 60)):
    draw.ellipse([x - 9 * scale, y - 5 * scale, x + 9 * scale, y + 5 * scale], fill=color)
    draw.polygon([(x + 8 * scale, y - 4 * scale), (x + 15 * scale, y), (x + 8 * scale, y + 4 * scale)], fill=color)
    draw.ellipse([x - 5 * scale, y - 2 * scale, x - 3 * scale, y], fill=(20, 20, 30))
    # плавник
    draw.polygon([(x - 2 * scale, y - 5 * scale), (x + 1 * scale, y - 9 * scale), (x + 3 * scale, y - 4 * scale)], fill=mix(color, (0, 0, 0), 0.2))


def draw_butterfly(draw, x, y, scale=1.0, color=(230, 100, 160)):
    flap = 2 * scale * math.sin(6 * x)
    for side in (-1, 1):
        x0, x1 = min(x - side * 12 * scale, x - side * 5 * scale), max(x - side * 12 * scale, x - side * 5 * scale)
        draw.ellipse([x0, y - 9 * scale - flap, x1, y - 2 * scale - flap], fill=color)
        x2, x3 = min(x - side * 10 * scale, x - side * 4 * scale), max(x - side * 10 * scale, x - side * 4 * scale)
        draw.ellipse([x2, y + 1 * scale + flap, x3, y + 8 * scale + flap], fill=mix(color, (0, 0, 0), 0.15))
    draw.line([x, y - 8 * scale, x, y + 8 * scale], fill=(60, 40, 50), width=2)


def draw_flower(draw, x, y, scale=1.0, color=(220, 80, 120)):
    stem_h = 24 * scale
    draw.line([x, y, x, y - stem_h], fill=(50, 130, 60), width=int(2 * scale))
    for i in range(6):
        ang = i * math.pi / 3
        px = x + math.cos(ang) * 6 * scale
        py = y - stem_h + math.sin(ang) * 6 * scale
        draw.ellipse([px - 3 * scale, py - 3 * scale, px + 3 * scale, py + 3 * scale], fill=color)
    draw.ellipse([x - 3 * scale, y - stem_h - 3 * scale, x + 3 * scale, y - stem_h + 3 * scale], fill=(255, 220, 80))


def draw_grass(draw, x, base_y, scale=1.0, color=(60, 150, 60)):
    for i in range(7):
        gx = x + (i - 3) * 3 * scale
        gh = (8 + (i % 3) * 4) * scale
        draw.line([gx, base_y, gx + 1 * scale, base_y - gh], fill=color, width=int(2 * scale))


def draw_water(draw, w, h, horizon, palette, time_t=0):
    top, mid, bot = palette
    for y in range(horizon, h):
        tt = (y - horizon) / max(1, h - horizon)
        col = mix(top, bot, tt)
        draw.line([(0, y), (w, y)], fill=col)
    # рябь
    for i in range(10):
        ry = horizon + (i / 10) * (h - horizon)
        for j in range(0, w, 18):
            dx = math.sin(time_t * 2 + ry * 0.1 + j * 0.2) * 3
            draw.line([j + dx, ry, j + 8 + dx, ry], fill=(255, 255, 255, 30))


def draw_rocket(draw, x, y, scale=1.0, color=(220, 220, 230)):
    draw.polygon([(x, y - 26 * scale), (x - 6 * scale, y + 6 * scale), (x + 6 * scale, y + 6 * scale)], fill=color)
    draw.rectangle([x - 6 * scale, y + 6 * scale, x + 6 * scale, y + 10 * scale], fill=(180, 180, 190))
    # иллюминатор
    draw.ellipse([x - 3 * scale, y - 8 * scale, x + 3 * scale, y - 2 * scale], fill=(120, 200, 255))
    # пламя
    fl = 8 * scale + 3 * scale * abs(math.sin(8 * x))
    draw.polygon([(x - 5 * scale, y + 10 * scale), (x, y + 10 * scale + fl), (x + 5 * scale, y + 10 * scale)], fill=(255, 160, 60))


def draw_ship(draw, x, y, scale=1.0, color=(140, 90, 50)):
    draw.polygon([(x - 22 * scale, y), (x + 22 * scale, y), (x + 12 * scale, y - 14 * scale), (x - 10 * scale, y - 14 * scale)], fill=color)
    # мачта
    draw.rectangle([x - 1 * scale, y - 30 * scale, x + 1 * scale, y - 14 * scale], fill=(100, 70, 40))
    # парус
    draw.polygon([(x, y - 28 * scale), (x + 10 * scale, y - 16 * scale), (x, y - 14 * scale)], fill=(240, 240, 250))


def draw_snowflake(draw, x, y, scale=1.0):
    for i in range(6):
        ang = i * math.pi / 3
        dx, dy = math.cos(ang) * 5 * scale, math.sin(ang) * 5 * scale
        draw.line([x, y, x + dx, y + dy], fill=(255, 255, 255), width=1)
        draw.line([x + dx * 0.6, y + dy * 0.6, x + dx * 0.8 + dy * 0.3, y + dy * 0.8 - dx * 0.3], fill=(255, 255, 255), width=1)
        draw.line([x + dx * 0.6, y + dy * 0.6, x + dx * 0.8 - dy * 0.3, y + dy * 0.8 + dx * 0.3], fill=(255, 255, 255), width=1)


def draw_fire(draw, x, y, scale=1.0, size=1.0):
    fh = 14 * scale * size
    fl = 6 * scale * size
    draw.polygon([(x - fl, y), (x, y - fh), (x + fl, y)], fill=(255, 140, 40))
    draw.polygon([(x - fl * 0.6, y), (x, y - fh * 0.7), (x + fl * 0.6, y)], fill=(255, 220, 80))


def draw_star(draw, x, y, scale=1.0, bright=200):
    draw.ellipse([x, y, x + max(1, scale), y + max(1, scale)], fill=(bright, bright, bright))


# ------------------------------------------------ реестр объектов
OBJECT_FUNCS = {
    "солнце": draw_sun, "sun": draw_sun,
    "луна": draw_moon, "moon": draw_moon, "месяц": draw_moon,
    "облак": draw_cloud, "cloud": draw_cloud, "туч": draw_cloud,
    "гор": draw_mountain, "mountain": draw_mountain, "холм": draw_mountain,
    "дерев": draw_tree, "tree": draw_tree, "ель": draw_tree, "сосн": draw_tree, "берёз": draw_tree, "дуб": draw_tree,
    "дом": draw_house, "house": draw_house, "коттедж": draw_house, "изб": draw_house, "здан": draw_house, "building": draw_house,
    "машин": draw_car, "car": draw_car, "автомобил": draw_car, "грузовик": draw_car,
    "человек": draw_person, "person": draw_person, "люд": draw_person, "мужчин": draw_person, "женщин": draw_person, "девочк": draw_person, "мальчик": draw_person,
    "собак": draw_dog, "dog": draw_dog, "щенок": draw_dog,
    "кошк": draw_cat, "cat": draw_cat, "кот": draw_cat, "котён": draw_cat,
    "птиц": draw_bird, "bird": draw_bird, "ворон": draw_bird, "воробей": draw_bird,
    "рыб": draw_fish, "fish": draw_fish, "дельфин": draw_fish,
    "бабочк": draw_butterfly, "butterfly": draw_butterfly, "мотылёк": draw_butterfly,
    "цвет": draw_flower, "flower": draw_flower, "тюльпан": draw_flower, "роза": draw_flower, "подсолнух": draw_flower,
    "трав": draw_grass, "grass": draw_grass,
    "ракет": draw_rocket, "rocket": draw_rocket,
    "корабл": draw_ship, "ship": draw_ship, "лодк": draw_ship, "парусник": draw_ship,
    "снежинк": draw_snowflake, "snowflake": draw_snowflake,
    "огон": draw_fire, "fire": draw_fire, "костёр": draw_fire, "пламя": draw_fire,
    "звезд": draw_star, "star": draw_star,
}

# Палитры фона по времени суток
SKIES = {
    "day": ((80, 140, 210), (130, 180, 230), (190, 220, 240)),
    "sunset": ((40, 30, 90), (230, 110, 60), (255, 200, 110)),
    "night": ((8, 8, 25), (20, 18, 50), (40, 30, 70)),
    "dawn": ((70, 90, 150), (255, 170, 110), (255, 220, 160)),
}


def detect_time(prompt):
    p = prompt.lower()
    if any(k in p for k in ("ночь", "ночной", "вечер", "тёмн", "night", "evening")):
        return "night"
    if any(k in p for k in ("закат", "вечер", "заходящ", "sunset")):
        return "sunset"
    if any(k in p for k in ("рассвет", "утренн", "восход", "sunrise", "dawn")):
        return "dawn"
    return "day"


# ------------------------------------------------ парсер запроса на объекты
def parse_objects(prompt):
    """Разбивает запрос на список объектов с позициями и масштабом.
    Возвращает список dict: {type, x, y, scale, count}."""
    import re
    p = prompt.lower()
    found = []
    # ландшафтные «наборы»: парк/лес/луг автоматически добавляют деревья и траву
    landscape_hint = None
    for kw in ("парк", "лес", "рощ", "луг", "поле", "саад"):
        if kw in p:
            landscape_hint = kw
            break
    for key, func in OBJECT_FUNCS.items():
        if key in p:
            # количество: число перед словом
            count = 1
            m = re.search(r"(\d+)\s*[а-я]*\s*" + key, p)
            if m:
                count = int(m.group(1))
            found.append({"key": key, "func": func, "count": max(1, min(count, 8))})
    # если ландшафт упомянут, но деревьев/травы нет — добавить
    if landscape_hint and not any(o["key"] in ("дерев", "ель", "сосн", "берёз", "дуб") for o in found):
        found.append({"key": "дерев", "func": draw_tree, "count": 4})
    if landscape_hint and not any(o["key"] in ("трав", "grass") or o["key"] == "цвет" for o in found):
        found.append({"key": "трав", "func": draw_grass, "count": 8})
    return found


# ------------------------------------------------ рендер сцены из объектов
def render_scene(width, height, prompt, time_t=0.0, seed=0):
    """Рисует сцену, состоящую из объектов. Возвращает RGBA Image."""
    import random
    rnd = random.Random(seed)
    img = Image.new("RGB", (width, height), (10, 10, 20))
    draw = ImageDraw.Draw(img)

    sky = SKIES[detect_time(prompt)]
    draw_sky(draw, width, height, sky)

    # звёзды ночью
    if sky is SKIES["night"]:
        for _ in range(120):
            sx, sy = rnd.random() * width, rnd.random() * height
            draw.ellipse([sx, sy, sx + 1.5, sy + 1.5], fill=(255, 255, 255))

    objects = parse_objects(prompt)
    if not objects:
        # ничего не нашли — рисуем типовой пейзаж
        objects = [{"key": "дерев", "func": draw_tree, "count": 3},
                   {"key": "трав", "func": draw_grass, "count": 6}]

    horizon = int(height * 0.6)
    water_keys = {"корабл", "лодк", "парусник", "рыб", "дельфин"}
    has_water = any(k in water_keys for k in [x["key"] for x in objects]) or \
                any(w in prompt.lower() for w in ("море", "океан", "река", "озер", "вода"))
    if has_water:
        horizon = int(height * 0.52)

    # объекты по слоям: небо -> дальний план -> средний -> передний
    layer_far = []      # облака, луна, солнце, горы, звёзды
    layer_mid = []      # деревья, дома
    layer_near = []     # люди, животные, машины, растения

    for obj in objects:
        key = obj["key"]
        for i in range(obj["count"]):
            # позиция по типу
            if key in ("облак", "туч"):
                x = rnd.random() * width * 0.9 + width * 0.05
                y = rnd.random() * horizon * 0.35 + 10
                layer_far.append((obj["func"], x, y, 0.6 + rnd.random() * 0.5))
            elif key in ("солнце", "sun"):
                x = rnd.random() * width * 0.4 + width * 0.3
                y = rnd.random() * horizon * 0.5 + horizon * 0.15
                layer_far.append((obj["func"], x, y, 0.8 + rnd.random() * 0.3))
            elif key in ("луна", "moon", "месяц"):
                x = rnd.random() * width * 0.5 + width * 0.3
                y = rnd.random() * horizon * 0.4 + horizon * 0.1
                layer_far.append((obj["func"], x, y, 0.6 + rnd.random() * 0.3))
            elif key in ("гор", "холм"):
                x = rnd.random() * width * 0.7
                h = int((0.25 + rnd.random() * 0.2) * height)
                layer_far.append((obj["func"], x, width * (0.4 + rnd.random() * 0.5), horizon, h, (70 + rnd.randint(0, 20), 80 + rnd.randint(0, 20), 100)))
            elif key in ("звезд"):
                x = rnd.random() * width
                y = rnd.random() * horizon
                layer_far.append((obj["func"], x, y, 1 + rnd.random() * 2, 150 + rnd.randint(0, 100)))
            elif key in ("дерев", "ель", "сосн", "берёз", "дуб"):
                x = rnd.random() * width * 0.9 + width * 0.05
                y = horizon + rnd.random() * (height - horizon) * 0.25
                layer_mid.append((obj["func"], x, y, 0.7 + rnd.random() * 0.8, (30 + rnd.randint(0, 30), 90 + rnd.randint(0, 40), 60)))
            elif key in ("дом", "house", "здан"):
                x = rnd.random() * width * 0.8 + width * 0.1
                y = horizon + rnd.random() * (height - horizon) * 0.15
                layer_mid.append((obj["func"], x, y, 0.8 + rnd.random() * 0.5))
            elif key in ("корабл", "лодк", "парусник"):
                x = rnd.random() * width * 0.7 + width * 0.1
                y = horizon + rnd.random() * (height - horizon) * 0.4
                layer_mid.append((obj["func"], x, y, 0.7 + rnd.random() * 0.4))
            elif key in ("машин", "автомобил"):
                x = rnd.random() * width * 0.8 + width * 0.1
                y = height - 20 - rnd.random() * 10
                layer_near.append((obj["func"], x, y, 0.8 + rnd.random() * 0.4, (rnd.randint(120, 220), rnd.randint(40, 120), rnd.randint(40, 120))))
            elif key in ("человек", "люд"):
                x = rnd.random() * width * 0.8 + width * 0.1
                y = height - 8 - rnd.random() * 6
                layer_near.append((obj["func"], x, y, 0.7 + rnd.random() * 0.5))
            elif key in ("собак", "щенок"):
                x = rnd.random() * width * 0.8 + width * 0.1
                y = height - 8 - rnd.random() * 5
                layer_near.append((obj["func"], x, y, 0.7 + rnd.random() * 0.4))
            elif key in ("кошк", "кот"):
                x = rnd.random() * width * 0.7 + width * 0.1
                y = height - 8 - rnd.random() * 5
                layer_near.append((obj["func"], x, y, 0.6 + rnd.random() * 0.4))
            elif key in ("птиц", "ворон"):
                x = rnd.random() * width
                y = rnd.random() * horizon * 0.7
                layer_far.append((obj["func"], x, y, 0.6 + rnd.random() * 0.5))
            elif key in ("рыб"):
                x = rnd.random() * width * 0.8 + width * 0.1
                y = rnd.random() * (height - horizon) * 0.6 + horizon + 20
                layer_near.append((obj["func"], x, y, 0.8 + rnd.random() * 0.5))
            elif key in ("бабочк", "мотылёк"):
                x = rnd.random() * width
                y = rnd.random() * horizon * 0.8
                layer_far.append((obj["func"], x, y, 0.5 + rnd.random() * 0.4))
            elif key in ("цвет", "flower"):
                x = rnd.random() * width * 0.9 + width * 0.05
                y = horizon + rnd.random() * (height - horizon) * 0.8
                layer_near.append((obj["func"], x, y, 0.8 + rnd.random() * 0.5, (rnd.randint(180, 240), rnd.randint(60, 130), rnd.randint(100, 180))))
            elif key in ("трав", "grass"):
                x = rnd.random() * width * 0.9 + width * 0.05
                y = height - rnd.random() * 6
                layer_near.append((obj["func"], x, y, 0.8 + rnd.random() * 0.6))
            elif key in ("снежинк"):
                x = rnd.random() * width
                y = rnd.random() * height
                layer_far.append((obj["func"], x, y, 0.8 + rnd.random() * 0.6))
            elif key in ("огон", "пламя"):
                x = rnd.random() * width * 0.8 + width * 0.1
                y = height - 10 - rnd.random() * 10
                layer_near.append((obj["func"], x, y, 0.7 + rnd.random() * 0.5, 0.8 + rnd.random() * 0.4))
            elif key in ("ракет"):
                x = rnd.random() * width * 0.6 + width * 0.2
                y = height - 30 - rnd.random() * 40
                layer_mid.append((obj["func"], x, y, 0.8 + rnd.random() * 0.6))
            else:
                x = rnd.random() * width
                y = rnd.random() * height
                layer_mid.append((obj["func"], x, y, 0.6 + rnd.random() * 0.6))

    # вода
    if has_water:
        draw_water(draw, width, height, horizon, ((60, 80, 110), (40, 60, 90), (10, 20, 40)), time_t)
    else:
        # земля под объектами ближнего плана
        ground_top = (50, 90, 50) if sky is not SKIES["night"] else (25, 40, 30)
        ground_bot = (20, 50, 25) if sky is not SKIES["night"] else (12, 18, 14)
        for y in range(horizon, height):
            tt = (y - horizon) / max(1, height - horizon)
            draw.line([(0, y), (width, y)], fill=mix(ground_top, ground_bot, tt))

    # отрисовка по слоям
    def render_layer(items, animated=False):
        for item in items:
            fn = item[0]
            args = list(item[1:])
            if animated and len(bytes(str(type(fn)).encode())) > 0:
                pass
            fn(draw, *args)

    render_layer(layer_far)
    render_layer(layer_mid)
    # трава/цветы на переднем плане рисуются после воды, но до людей
    render_layer([it for it in layer_near if it[0] in (draw_flower, draw_grass)])
    render_layer([it for it in layer_near if it[0] not in (draw_flower, draw_grass)])

    return img.convert("RGBA")