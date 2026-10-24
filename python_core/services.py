# -*- coding: utf-8 -*-
"""
Onyx Python Core · services.py
===============================
Внешние бесплатные API-интеграции:
  · поиск: Wikipedia (RU/EN), DuckDuckGo, Bing HTML, Openverse (изображения)
  · генерация изображений: Pollinations (без ключа)
  · генерация видео: анимация кадров из Pollinations + поиск видео в Wikimedia Commons
  · обучение: результаты поиска/генерации возвращаются как обучающий корпус

Все вызовы безопасно обрабатывают ошибки сети и таймауты — при сбое
нейросеть продолжает работать на локальных движках.
"""

import json
import re
import time
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def _fetch(url, timeout=10, retries=2):
    """Выполняет GET-запрос и возвращает bytes. Возвращает None при ошибке."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    return None


def _fetch_json(url, timeout=10):
    data = _fetch(url, timeout)
    if not data:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------- поиск
def wikipedia_search(query, lang="ru", limit=3):
    """Поиск по Википедии (без ключа)."""
    url = ("https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}"
           "&format=json&origin=*&srlimit={n}&srprop=snippet").format(lang=lang, q=urllib.parse.quote(query), n=limit)
    j = _fetch_json(url)
    if not j:
        return []
    hits = (j.get("query", {}).get("search", []) or [])[:limit]
    out = []
    for h in hits:
        title = h.get("title", "")
        sum_url = ("https://{lang}.wikipedia.org/api/rest_v1/page/summary/{t}").format(lang=lang, t=urllib.parse.quote(title))
        s = _fetch_json(sum_url, timeout=8)
        extract = (s or {}).get("extract", "")
        if extract:
            out.append({
                "source": "Википедия", "title": title,
                "text": extract[:650],
                "url": (s or {}).get("content_urls", {}).get("desktop", {}).get("page", ""),
            })
    return out


def duckduckgo_search(query):
    """Инстанс-ответ DuckDuckGo (без ключа)."""
    url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote(query) + \
          "&format=json&no_html=1&skip_disambig=1&t=onyx"
    j = _fetch_json(url)
    if not j:
        return []
    text = j.get("AbstractText") or j.get("Definition") or j.get("Answer") or ""
    if len(text) < 60:
        return []
    return [{"source": "открытые источники", "title": j.get("Heading") or query,
             "text": text[:550], "url": j.get("AbstractURL", "")}]


def bing_search(query):
    """Bing HTML-поиск (парсинг, без ключа)."""
    url = "https://www.bing.com/search?q=" + urllib.parse.quote(query) + "&setlang=ru"
    html = _fetch(url, timeout=10)
    if not html:
        return []
    try:
        text = html.decode("utf-8", "ignore")
    except Exception:
        return []
    # вытаскиваем сниппеты из блока результатов
    results = []
    pattern = re.compile(r'<li class="b_algo".*?<h2><a href="([^"]+)"[^>]*>(.*?)</a>.*?<p[^>]*>(.*?)</p>', re.S)
    for m in pattern.finditer(text):
        link = m.group(1)
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        snippet = re.sub(r"<[^>]+>", "", m.group(3)).strip()
        if title and snippet:
            results.append({"source": "Bing", "title": title, "text": snippet[:500], "url": link})
        if len(results) >= 3:
            break
    return results


def openverse_search_images(query, limit=5):
    """Openverse: бесплатные изображения с открытыми лицензиями (без ключа)."""
    url = "https://api.openverse.org/v1/images/?q=" + urllib.parse.quote(query) + "&page_size=" + str(limit)
    j = _fetch_json(url)
    if not j:
        return []
    out = []
    for r in (j.get("results") or [])[:limit]:
        img_url = r.get("url") or r.get("direct_url") or ""
        if not img_url:
            continue
        out.append({
            "source": "Openverse", "title": r.get("title") or query,
            "text": (r.get("tags") and ", ".join(t.get("name", "") for t in r.get("tags", [])[:6])) or "",
            "url": img_url, "license": r.get("license", ""),
            "width": r.get("width", 0), "height": r.get("height", 0),
        })
    return out


def wikimedia_video_search(query, limit=3):
    """Поиск готовых видео на Wikimedia Commons (CC-лицензии, без ключа)."""
    url = ("https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch=" +
           urllib.parse.quote(query) + "&gsrnamespace=6&gsrlimit=" + str(limit) +
           "&prop=imageinfo&iiprop=url|size|mime&format=json&origin=*")
    j = _fetch_json(url)
    if not j:
        return []
    pages = (j.get("query", {}).get("pages", {}) or {})
    out = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [{}])[0]
        mime = ii.get("mime", "")
        if "video" not in mime:
            continue
        vid_url = ii.get("url", "")
        if not vid_url:
            continue
        out.append({"source": "Wikimedia Commons", "title": p.get("title", "").replace("File:", ""),
                    "text": mime, "url": vid_url, "width": ii.get("width", 0), "height": ii.get("height", 0)})
        if len(out) >= limit:
            break
    return out


BLOCKED_WORDS = ("порнограф", "эротик", "xxx", "сексшоп", "инцест", "педофил", "наркотрафик", "оружие массового поражения", "взрывчатк", "изготовление бомб", "child porn", "sex video", "escort")


def _is_blocked(text):
    t = (text or "").lower()
    return any(w in t for w in BLOCKED_WORDS)


def multi_search(query, max_results=6):
    """Объединяет результаты из всех доступных источников (с фильтрацией)."""
    results = []
    for fn in (lambda: wikipedia_search(query, "ru", 3),
               lambda: wikipedia_search(query, "en", 1),
               lambda: duckduckgo_search(query),
               lambda: bing_search(query),
               lambda: openverse_search_images(query, 3)):
        try:
            results += fn()
        except Exception:
            continue
    seen = set()
    out = []
    for r in results:
        if not r or not r.get("text"):
            continue
        # пропускаем нежелательный контент
        if _is_blocked(r.get("text", "")) or _is_blocked(r.get("title", "")):
            continue
        key = (r["text"][:60] + (r.get("url") or "")[:40]).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= max_results:
            break
    return out


# ---------------------------------------------------------------- генерация изображений
def pollinations_image(prompt, size=512, seed=None):
    """Генерирует изображение через Pollinations (без ключа). Возвращает bytes или None."""
    p = re.sub(r"^(нарисуй|сгенерируй|создай|покажи|сделай|нарисовать|изобрази|draw|generate|paint|render|create)\s+", "", prompt.strip(), flags=re.I)
    if seed is None:
        import hashlib
        seed = int(hashlib.md5(p.encode("utf-8")).hexdigest()[:8], 16) % 10_000_000
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(p) +
           "?width={w}&height={h}&seed={s}&nologo=true&model=flux").format(w=size, h=size, s=seed)
    return _fetch(url, timeout=30)


def pollinations_image_variants(prompt, count=4, size=256):
    """Генерирует несколько кадров-вариантов (для анимации видео)."""
    frames = []
    for i in range(count):
        img = pollinations_image(prompt, size=size, seed=(i + 1) * 12345)
        if img:
            frames.append(img)
    return frames


# ---------------------------------------------------------------- вспомогательные
def strip_html(html):
    t = html or ""
    t = re.sub(r"<script[\s\S]*?</script>", "", t, flags=re.I)
    t = re.sub(r"<style[\s\S]*?</style>", "", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    import html as htmlmod
    t = htmlmod.unescape(t)
    return re.sub(r"\s+", " ", t).strip()