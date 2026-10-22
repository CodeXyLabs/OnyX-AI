# -*- coding: utf-8 -*-
"""
OnyX Python Core · engine/composer.py
======================================
Сборка ответа из найденных фактов под конкретное намерение вопроса.

Маленький трансформер плохо держит структуру, поэтому там, где есть
проверенные факты, ответ собирается детерминированно: определение,
сравнение, пошаговая инструкция, объяснение причины, перечисление или
короткий факт. Это даёт предсказуемое качество и отсутствие «воды».
"""

import re

try:
    from ..knowledge.knowledge_base import _stem as kb_stem
except Exception:
    try:
        from knowledge.knowledge_base import _stem as kb_stem
    except Exception:
        def kb_stem(word):
            return word


VOWELS = set("аеёиоуыэюяaeiou")

IMPERATIVES = (
    "начни", "начните", "используй", "используйте", "добавь", "добавьте",
    "проверь", "проверьте", "читай", "читайте", "решай", "решайте",
    "делай", "делайте", "строй", "стройте", "установи", "установите",
    "настрой", "настройте", "выучи", "выучите", "пиши", "пишите",
    "изучай", "изучайте", "попробуй", "попробуйте", "смотри", "смотрите",
    "разбей", "разбейте", "повторяй", "повторяйте", "заведи", "заведите",
)

CAUSE_MARKERS = ("потому что", "так как", "поэтому", "из-за", "причин",
                 "связано с тем", "дело в том")


# ------------------------------------------------------------------ текст

def split_sentences(text):
    t = (text or "").replace("\n", " ").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def _cap(s):
    return s[0].upper() + s[1:] if s else s


def _lower(s):
    if not s:
        return s
    first = s.split(" ")[0]
    if len(first) > 1 and first == first.upper():
        return s          # аббревиатуры не трогаем
    return s[0].lower() + s[1:]


def words(text):
    return re.findall(r"[а-яa-z0-9]+", (text or "").lower().replace("ё", "е"))


def stems(text):
    return {kb_stem(w) for w in words(text) if len(w) > 2}


def polish(text):
    """Русская типографика и чистота: тире, кавычки, пробелы, конец фразы."""
    t = text or ""
    t = re.sub(r"\s*-\s*", " — ", t)
    t = t.replace('"', "«", 1)
    # закрывающие кавычки: каждая вторая
    out, opened = [], False
    for ch in t:
        if ch == '"':
            out.append("»" if opened else "«")
            opened = not opened
        else:
            out.append(ch)
    t = "".join(out)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,.;:!?])\1+", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    if t:
        t = _cap(t)
        if not t.endswith((".", "!", "?", "…")):
            t += "."
    return t


def _is_good_sentence(s):
    ws = words(s)
    # предлоги и союзы («в», «у», «и») не должны портить оценку предложения
    significant = [w for w in ws if len(w) >= 3]
    if len(significant) < 3:
        return False
    good = sum(1 for w in significant if any(c in VOWELS for c in w))
    return good / len(significant) >= 0.8


def _similar(a, b):
    """Похожи ли предложения по составу слов (для отсева дублей)."""
    wa, wb = set(words(a)), set(words(b))
    if not wa or not wb:
        return False
    return len(wa & wb) / max(1, len(wa | wb)) >= 0.4


def _shared_content(a, b):
    """Сколько значимых слов общего у двух предложений."""
    wa = {w for w in words(a) if len(w) >= 4}
    wb = {w for w in words(b) if len(w) >= 4}
    return len(wa & wb)


# ------------------------------------------------------------------ выбор фактов

def _pool(facts):
    """Плоский список предложений с темой и ссылкой на факт."""
    out = []
    for idx, f in enumerate(facts or []):
        text = (f.get("fact") or "").strip()
        topic = f.get("topic") or ""
        for s in split_sentences(text):
            if _is_good_sentence(s):
                out.append({"s": s.strip(), "topic": topic, "fact": idx})
    return out


def _score(item, entities, keywords, intent):
    s = item["s"]
    low = s.lower().replace("ё", "е")
    st = stems(s)
    score = 0.0
    for e in entities or []:
        if not e:
            continue
        el = e.lower().replace("ё", "е")
        if el in low:
            score += 8
        elif stems(e) & st:
            score += 5
    if "— это" in low or " - это" in low:
        score += 3
    for k in (keywords or [])[:8]:
        if k and k.lower() in low:
            score += 1
        elif kb_stem(k) in st:
            score += 0.5
    if intent in ("definition", "comparison") and re.match(r"^[а-яёa-z0-9 \-]+ — это", low):
        score += 2
    if 40 <= len(s) <= 240:
        score += 1
    return score


def _best(pool, entities, keywords, intent, used=None, require_entity=False):
    used = used if used is not None else set()
    best, best_score = None, 0.0
    for item in pool:
        if item["s"] in used:
            continue
        sc = _score(item, entities, keywords, intent)
        if require_entity and sc < 4:
            continue
        if sc > best_score:
            best, best_score = item, sc
    return best, best_score


def _head_token(entity):
    toks = [t for t in words(entity) if len(t) > 2]
    return toks[-1] if toks else ""


def _has_word(text, word):
    """Совпадение целого слова, а не подстроки: «нейрон» не равно «нейросеть»."""
    if not word:
        return False
    return re.search(rf"(?<![а-яёa-z0-9]){re.escape(word)}(?![а-яёa-z0-9])", text) is not None


def _entity_hits(sentence, entity):
    toks = [t for t in words(entity) if len(t) > 2]
    if not toks:
        return 0
    st = stems(sentence)
    low = sentence.lower().replace("ё", "е")
    hits = 0
    for t in toks:
        if _has_word(low, t) or kb_stem(t) in st:
            hits += 1
    return hits


def _matches_entity(sentence, entity):
    """Предложение действительно про эту сущность, а не про соседнюю тему."""
    if not entity:
        return False
    hits = _entity_hits(sentence, entity)
    if hits == 0:
        return False
    low = sentence.lower().replace("ё", "е")
    ent_low = entity.lower().replace("ё", "е")
    if _has_word(low, ent_low):
        return True
    toks = [t for t in words(entity) if len(t) > 2]
    definition = bool(re.match(r"^[а-яёa-z0-9 \-]{2,60} — это", low))
    if definition and hits >= 1:
        return True
    if hits >= 2:
        return True
    # однословная сущность: достаточно совпадения головного слова
    if len(toks) == 1:
        head = _head_token(entity)
        return bool(head) and (_has_word(low, head) or kb_stem(head) in stems(sentence))
    return False


def _about(entity, pool, used):
    """Лучшее предложение именно про сущность (с проверкой совпадения)."""
    if not entity:
        return None
    best, best_score = None, 0.0
    for item in pool:
        if item["s"] in used:
            continue
        if not _matches_entity(item["s"], entity):
            continue
        sc = 6 + _entity_hits(item["s"], entity) * 2
        if re.match(r"^[а-яёa-z0-9 \-]+ — это", item["s"].lower()):
            sc += 3
        if 40 <= len(item["s"]) <= 240:
            sc += 1
        if sc > best_score:
            best, best_score = item, sc
    return best


def _strip_definition(sentence, entity):
    """«нейросеть — это X» -> «X», чтобы вставить в свой шаблон."""
    s = sentence.strip()
    m = re.match(r"^[^—]{0,60}—\s*это\s+(.+)", s, re.I)
    if m:
        return m.group(1).strip()
    m = re.match(rf"^{re.escape(entity)}\s*[—-]\s*(.+)", s, re.I)
    if m:
        return m.group(1).strip()
    # «александр пушкин — русский поэт...» при сущности «пушкин»
    if entity:
        m = re.match(rf"^[^—]{{0,40}}\b{re.escape(entity)}\b[^—]{{0,10}}[—-]\s*(.+)", s, re.I)
        if m:
            return m.group(1).strip()
    return s


# ------------------------------------------------------------------ сценарии

def _definition(analysis, pool):
    ents = analysis.get("entities") or []
    ent = ents[0] if ents else ""
    first = _about(ent, pool, set()) if ent else None
    if not first:
        # без прямого совпадения сущности определение не выдумываем
        return None
    body = _strip_definition(first["s"], ent)
    if not body.lower().startswith(ent.lower()):
        head = f"{_cap(ent)} — это {_lower(body)}"
    else:
        head = body
    used = {first["s"]}
    second = _about(ent, pool, used)
    if second and second["s"] != first["s"] and len(head) < 420 \
            and not _similar(head, second["s"]) \
            and _shared_content(head, second["s"]) < 2:
        head += " " + _cap(second["s"])
        used.add(second["s"])
    return {"answer": head, "used": used, "confidence": 0.9}


def _comparison(analysis, pool):
    ents = (analysis.get("entities") or [])[:2]
    if len(ents) < 2:
        return None
    a, b = ents[0], ents[1]
    used = set()
    sa = _about(a, pool, used)
    if sa:
        used.add(sa["s"])
    sb = _about(b, pool, used)

    parts = [f"Сравнение: «{a}» и «{b}»."]
    if sa:
        parts.append("Первое: " + _cap(_strip_definition(sa["s"], a)))
    else:
        parts.append("Первое понятие в базе знаний не описано.")
    if sb:
        parts.append("Второе: " + _cap(_strip_definition(sb["s"], b)))
    else:
        parts.append("Второе понятие в базе знаний не описано.")

    contrast = None
    for item in pool:
        low = item["s"].lower()
        if any(m in low for m in ("в отличие", "разница", "тогда как", "напротив",
                                  "отличается", "а не ")):
            contrast = item["s"]
            break
    if contrast:
        parts.append("Ключевое отличие: " + _lower(contrast))
    elif sa and sb and sa["topic"] != sb["topic"]:
        parts.append(f"Они из разных тем: «{a}» — про «{sa['topic']}», «{b}» — про «{sb['topic']}».")
    elif sa and sb:
        parts.append("Оба понятия из одной темы, но описывают разные вещи.")

    found = int(bool(sa)) + int(bool(sb))
    conf = 0.9 if found == 2 else (0.6 if found == 1 else 0.4)
    return {"answer": " ".join(parts), "used": used, "confidence": conf,
            "low": found < 2}


def _howto(analysis, pool):
    qn = analysis.get("normalized", "")
    # «как работает/устроен» — это объяснение, а не инструкция
    if re.search(r"\bработает\b|\bработают\b|\bустроен", qn) and not re.search(
            r"сделать|выучить|начать|настроить|установить|создать|написать|проверить", qn):
        return _definition(analysis, pool)

    steps = []
    for item in pool:
        low = item["s"].lower()
        if any(low.startswith(v + " ") or f" {v} " in low for v in IMPERATIVES):
            # запятая внутри такой фразы часто разделяет шаги
            chunks = [c.strip() for c in re.split(r",\s+", item["s"]) if len(c.split()) >= 2]
            steps.extend(chunks if len(chunks) > 1 else [item["s"]])
    steps = [s for s in dict.fromkeys(steps) if len(s.split()) >= 2][:5]
    # одного шага мало: это ещё не инструкция
    if len(steps) < 2:
        return None
    body = "Коротко, по шагам:\n" + "\n".join(f"{i}. {_cap(s.rstrip('.!'))}" for i, s in enumerate(steps, 1))
    return {"answer": body, "used": set(steps), "confidence": 0.85}


def _why(analysis, pool):
    ents = analysis.get("entities") or []
    kws = analysis.get("keywords") or []

    def relevant(item):
        s = item["s"]
        if ents and any(_matches_entity(s, e) for e in ents):
            return True
        st = stems(s)
        return sum(1 for k in kws if kb_stem(k) in st) >= 2

    cause = None
    for item in pool:
        low = item["s"].lower()
        if any(m in low for m in CAUSE_MARKERS) and relevant(item):
            cause = item
            break
    if cause:
        return {"answer": _cap(cause["s"]), "used": {cause["s"]}, "confidence": 0.85}
    best, score = _best(pool, ents, kws, "why")
    if not best or not relevant(best):
        return None
    return {"answer": "Объяснение: " + _lower(best["s"]), "used": {best["s"]},
            "confidence": 0.7 if score >= 4 else 0.6}


def _list(analysis, pool):
    ents = analysis.get("entities") or []
    kws = analysis.get("keywords") or []
    ent = ents[0] if ents else ""
    toks = [t for t in words(ent) if len(t) > 2] if ent else []
    need_overlap = 2 if len(toks) >= 2 else 1
    head = _head_token(ent)
    items = []
    for item in pool:
        low = item["s"].lower().replace("ё", "е")
        # «Программирование — это ...» описывает саму категорию, а не её виды
        if len(toks) >= 2 and head and re.match(rf"^{re.escape(head)}\s*[—-]\s*это", low):
            continue
        overlap = len(stems(item["topic"]) & stems(ent)) if ent else 0
        hits = _entity_hits(item["s"], ent) if ent else 0
        keep = overlap >= need_overlap or hits >= max(2, need_overlap)
        if not keep:
            continue
        sc = _score(item, [ent] if ent else [], kws, "list") + overlap * 4
        if sc >= 6 and item["s"] not in items:
            items.append(item["s"])
    items = items[:5]
    if len(items) < 2:
        return None
    body = "Вот что можно перечислить:\n" + "\n".join("— " + _cap(s.rstrip(".!")) for s in items)
    return {"answer": body, "used": set(items), "confidence": 0.8}


def _short_fact(analysis, pool, intent):
    ents = analysis.get("entities") or []
    kws = analysis.get("keywords") or []
    first, score = _best(pool, ents, kws, intent)
    if not first:
        return None
    used = {first["s"]}
    text = _cap(first["s"])
    second, sscore = _best(pool, ents, kws, intent, used)
    if second and sscore >= 3 and len(text) < 260:
        text += " " + _cap(second["s"])
        used.add(second["s"])
    return {"answer": text, "used": used, "confidence": 0.8 if score >= 4 else 0.65}


NO_FACTS = {
    "definition": "В моей базе знаний пока нет определения «{e}». Могу поискать сведения в открытых источниках — уточни, в каком контексте тебе это важно.",
    "comparison": "Про «{a}» и «{b}» у меня нет точных описаний в базе знаний. Если хочешь, найду сведения в сети — скажи, что именно сравниваем: применение, устройство или стоимость.",
    "howto": "Пошагового материала по теме «{e}» у меня нет. Опиши, что уже сделано и какой результат нужен, — тогда предложу конкретный план.",
    "why": "Точной причины по вопросу «{e}» в базе знаний нет. Могу поискать объяснение в открытых источниках — уточни, о каком явлении речь.",
    "list": "Готового списка по теме «{e}» у меня нет. Уточни, что именно перечислить: виды, примеры или инструменты.",
    "fact": "Прямого ответа в базе знаний нет. Уточни вопрос или попроси поискать в сети.",
    "detail": "Подробностей по этой теме в базе знаний мало. Спроси конкретнее — расскажу, что знаю.",
    "unknown": "Прямого ответа в базе знаний нет. Уточни вопрос — и я постараюсь помочь или найти сведения.",
}


def compose(analysis, facts, question=""):
    """Собирает ответ под намерение. Возвращает dict с answer/confidence/low."""
    intent = analysis.get("intent", "unknown")
    ents = analysis.get("entities") or []
    pool = _pool(facts)

    if not pool:
        e = ents[0] if ents else (analysis.get("keywords") or [""])[0]
        a = ents[0] if len(ents) > 0 else e
        b = ents[1] if len(ents) > 1 else ""
        template = NO_FACTS.get(intent, NO_FACTS["unknown"])
        text = template.format(e=e or "этой теме", a=a or "этой теме", b=b or "второму понятию")
        return {"answer": polish(text), "confidence": 0.3, "low": True, "used": set()}

    if intent == "definition":
        res = _definition(analysis, pool)
    elif intent == "comparison":
        res = _comparison(analysis, pool)
    elif intent == "howto":
        res = _howto(analysis, pool)
    elif intent == "why":
        res = _why(analysis, pool)
    elif intent == "list":
        res = _list(analysis, pool)
    elif intent == "detail":
        res = _short_fact(analysis, pool, "detail")
    else:
        res = _short_fact(analysis, pool, "fact")

    if not res:
        return {"answer": "", "confidence": 0.0, "low": True, "used": set()}

    text = polish(res["answer"])
    low = bool(res.get("low", False)) or res["confidence"] < 0.6
    return {
        "answer": text[:900],
        "confidence": res["confidence"],
        "low": low,
        "used": res.get("used", set()),
    }
