# -*- coding: utf-8 -*-
"""
OnyX Python Core · engine/nlu.py
=================================
Понимание запроса: нормализация, исправление типичных опечаток,
определение намерения, извлечение сущностей, расширение синонимами
и разрешение уточняющих вопросов («а почему?», «подробнее»).

Модуль ничего не генерирует — он готовит структуру для композитора ответов.
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


# ------------------------------------------------------------------ лексика

# Частые опечатки и разговорные сокращения → нормальная форма.
TYPO_MAP = {
    "програмирование": "программирование",
    "программированние": "программирование",
    "алгоримт": "алгоритм",
    "алгоритмн": "алгоритм",
    "нейросет": "нейросеть",
    "нейронка": "нейросеть",
    "нейронки": "нейросеть",
    "нейронку": "нейросеть",
    "нейронок": "нейросеть",
    "интернетт": "интернет",
    "инет": "интернет",
    "комп": "компьютер",
    "компьютерр": "компьютер",
    "ии": "искусственный интеллект",
    "иишный": "искусственный интеллект",
    "мозги": "мозг",
    "память": "память",
    "база": "база данных",
    "бд": "база данных",
    "ос": "операционная система",
    "по": "программное обеспечение",
    "железо": "аппаратное обеспечение",
    "софт": "программное обеспечение",
    "фотосинтезз": "фотосинтез",
    "гравитаций": "гравитация",
    "вселеннная": "вселенная",
}

SYNONYMS = {
    "искусственный интеллект": ["нейросеть", "машинное обучение"],
    "нейросеть": ["искусственный интеллект", "машинное обучение"],
    "машинное обучение": ["нейросеть", "искусственный интеллект"],
    "компьютер": ["процессор", "память"],
    "интернет": ["сеть", "протокол"],
    "программирование": ["код", "язык программирования"],
    "база данных": ["sql", "таблица"],
    "операционная система": ["windows", "linux"],
    "мозг": ["нейрон", "память"],
    "вселенная": ["космос", "галактика"],
    "космос": ["вселенная", "планета"],
    "гравитация": ["притяжение", "масса"],
    "энергия": ["мощность", "работа"],
    "фотосинтез": ["растение", "хлорофилл"],
}

# Слова, не несущие смысла для поиска фактов.
STOP = {
    "что", "как", "это", "такое", "такой", "такая", "такие", "кто", "чем", "про",
    "для", "его", "её", "ее", "они", "она", "оно", "мне", "тебе", "нам", "вам",
    "можно", "нужно", "надо", "есть", "был", "была", "было", "быть", "будет",
    "расскажи", "объясни", "скажи", "покажи", "напиши", "дай", "подскажи",
    "почему", "зачем", "когда", "где", "какой", "какая", "какие", "каких",
    "самый", "самая", "очень", "просто", "вообще", "между", "отличается",
    "отличие", "разница", "лучше", "хуже", "пример", "примеры", "значит",
    "означает", "определение", "понятие", "суть", "работает", "работают",
    "устроен", "устроена", "устроено", "устроены", "помоги", "помогите",
    "подробнее", "дальше", "поясни", "ещё", "еще", "именно", "ладно", "окей",
}

INTENT_RULES = [
    ("comparison", [r"чем\s+отлича", r"в\s+чём\s+разниц", r"в\s+чем\s+разниц",
                    r"разниц\w*\s+между", r"отличи\w*\s+между", r"сравни\w*",
                    r"что\s+лучше", r"или\s+лучше"]),
    ("definition", [r"что\s+такое", r"кто\s+так(ой|ая|ие)", r"что\s+значит",
                    r"что\s+означает", r"дай\s+определение", r"\bопределение\b",
                    r"что\s+представляет\s+собой"]),
    ("detail", [r"подробнее", r"расскажи\s+(?:мне\s+)?(?:про|о|об)\s", r"^расскажи\b"]),
    ("why", [r"^почему", r"\bпочему\b", r"зачем", r"по\s+какой\s+причине",
             r"в\s+чём\s+причин", r"в\s+чем\s+причин", r"отчего"]),
    ("howto", [r"^как\s", r"\bкак\s+(сделать|работает|работают|устроен|выучить|"
                       r"начать|настроить|установить|решить|написать|создать|"
                       r"проверить|исправить)", r"каким\s+образом", r"инструкц",
               r"пошагов", r"с\s+чего\s+начать"]),
    ("list", [r"какие\s+(бывают|есть|существуют)", r"перечисли", r"список",
              r"приведи\s+пример", r"примеры\s+чего", r"виды\s+"]),
    ("fact", [r"^кто\s", r"^когда\s", r"^где\s", r"^сколько\s", r"^какой\s",
              r"^какая\s", r"^какие\s", r"какого\s+года"]),
]

FOLLOWUP_CUES = {"а", "ещё", "еще", "подробнее", "дальше", "почему", "зачем",
                 "как", "и", "то", "это", "что", "именно", "поясни"}


# ------------------------------------------------------------------ базовое

def normalize(text):
    """Приводит текст к единому виду, сохраняя смысл."""
    t = str(text or "").strip().lower().replace("ё", "е")
    t = t.replace("\u2212", "-").replace("«", '"').replace("»", '"')
    t = re.sub(r"[\u2018\u2019\u201c\u201d]", '"', t)
    t = re.sub(r"[\u2013\u2014]", "-", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" \t\n?!.,;:")


def _collapse_repeats(text):
    """нейроссеееть -> нейросеть (3+ одинаковых символа подряд)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def fix_typos(text):
    """Исправляет частые опечатки и приводит основы слов к словарной форме."""
    t = _collapse_repeats(text)
    # устойчивые основы: любые словоформы и опечатки → каноническое слово
    STEM_FIXES = (
        (r"\bнейрос\w*", "нейросеть"),
        (r"\bпрограм\w*ирован\w*", "программирование"),
        (r"\bалгорит\w*", "алгоритм"),
        (r"\bинтернет\w*", "интернет"),
        (r"\bфотосинтез\w*", "фотосинтез"),
    )
    for pattern, repl in STEM_FIXES:
        t = re.sub(pattern, repl, t)
    words = t.split()
    out = []
    for w in words:
        bare = re.sub(r"[^а-яa-z0-9-]", "", w)
        if bare in TYPO_MAP:
            out.append(TYPO_MAP[bare])
        else:
            out.append(w)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def words_of(text):
    return re.findall(r"[а-яa-z0-9]+", normalize(text))


def stems_of(text):
    return {kb_stem(w) for w in words_of(text) if len(w) > 2}


def keywords_of(text):
    """Значимые слова: без стоп-слов и слишком коротких."""
    return [w for w in words_of(text) if len(w) > 2 and w not in STOP and kb_stem(w) not in STOP]


# ------------------------------------------------------------------ намерение

def detect_intent(q):
    """Определяет тип вопроса.

    Сначала явные конструкции определения (они однозначны), затем
    эллиптические уточнения без собственных значимых слов, затем правила.
    """
    if re.search(r"\bчто\s+такое\b", q) or \
       re.search(r"^(кто\s+так(ой|ая|ие)|что\s+значит|что\s+означает|"
                 r"дай\s+определение|определение\s+|что\s+за\b)", q) or \
       re.search(r"\bэто\s+что\b", q):
        return "definition"

    ws = q.split()
    kws = keywords_of(q)
    if len(ws) <= 4 and len(kws) == 0:
        return "followup"

    for intent, patterns in INTENT_RULES:
        for p in patterns:
            if re.search(p, q):
                return intent
    if len(ws) <= 2:
        return "followup"
    return "unknown"


# ------------------------------------------------------------------ сущности

LEAD_VERBS = ("выучить", "изучить", "сделать", "работает", "работают", "устроен",
              "устроена", "настроить", "установить", "решить", "написать", "создать",
              "проверить", "исправить", "начать", "понять", "объяснить", "используют")
TAIL_PHRASES = ("что лучше", "что такое", "это что", "что это", "и что", "вообще",
                "на самом деле")


def _clean_entity(s, stop_tail=()):
    s = s.strip(" \t.,;:?!-\"'")
    s = re.sub(r"\s+", " ", s)
    for tail in tuple(stop_tail) + TAIL_PHRASES:
        s = re.sub(rf"\s+{tail}$", "", s, flags=re.I)
    # отрезаем ведущие глаголы-связки: «как работает интернет» -> «интернет»
    parts = s.split()
    while parts and parts[0].lower() in LEAD_VERBS:
        parts.pop(0)
    while parts and parts[0].lower() in ("это", "такое", "вообще", "именно"):
        parts.pop(0)
    while parts and parts[-1] in ("это", "такое", "такой", "такая", "вообще", "именно"):
        parts.pop()
    return " ".join(parts).strip(" ,.-")


def extract_entities(q, intent):
    """Возвращает список сущностей вопроса (обычно 1–2)."""
    text = q

    if intent == "comparison":
        for pattern in (
            r"чем\s+отлича\w*\s+(.+?)\s+от\s+(.+)",
            r"разниц\w*\s+между\s+(.+?)\s+и\s+(.+)",
            r"отличи\w*\s+между\s+(.+?)\s+и\s+(.+)",
            r"сравни\w*\s+(.+?)\s+и\s+(.+)",
            r"что\s+лучше[:\s]+(.+?)\s+или\s+(.+)",
            r"(.+?)\s+или\s+(.+)",
        ):
            m = re.search(pattern, text)
            if m:
                a = _clean_entity(m.group(1), ("это", "такое"))
                b = _clean_entity(m.group(2), ("это", "такое"))
                if a and b:
                    return [a, b]

    if intent == "definition":
        for pattern in (r"что\s+такое\s+(.+)", r"кто\s+так(?:ой|ая|ие)\s+(.+)",
                        r"что\s+значит\s+(.+)", r"что\s+означает\s+(.+)",
                        r"определение\s+(.+)"):
            m = re.search(pattern, text)
            if m:
                e = _clean_entity(m.group(1))
                if e:
                    return [e]

    if intent == "howto":
        for pattern in (r"(?:как|каким\s+образом)\s+(.+)", r"с\s+чего\s+начать\s+(.+)"):
            m = re.search(pattern, text)
            if m:
                e = _clean_entity(m.group(1), ("работает", "работают", "устроен"))
                if e:
                    return [e]

    if intent == "why":
        m = re.search(r"(?:почему|зачем|отчего)\s+(.+)", text)
        if m:
            e = _clean_entity(m.group(1))
            if e:
                return [e]

    if intent == "list":
        m = re.search(r"(?:какие\s+(?:бывают|есть|существуют)|виды)\s+(.+)", text)
        if m:
            e = _clean_entity(m.group(1))
            if e:
                return [e]

    if intent == "fact":
        m = re.search(r"(?:кто|когда|где|сколько|какой|какая|какие)\s+(.+)", text)
        if m:
            e = _clean_entity(m.group(1))
            if e:
                return [e]

    # общий случай: самое длинное значимое словосочетание
    kws = keywords_of(text)
    return [" ".join(kws[:3])] if kws else []


# ------------------------------------------------------------------ расширение

def expand_query(q, entities=None):
    """Добавляет синонимы и сущности, чтобы поиск фактов ловил больше релевантного."""
    base = q
    extra = []
    for ent in (entities or []):
        if ent:
            extra.append(ent)
            for syn in SYNONYMS.get(ent, []):
                extra.append(syn)
    for w in words_of(q):
        for syn in SYNONYMS.get(w, []):
            extra.append(syn)
    # ключевые слова без служебных
    kws = keywords_of(q)
    joined = " ".join([base] + extra + kws)
    return re.sub(r"\s+", " ", joined).strip()


# ------------------------------------------------------------------ уточнения

def is_followup(q, history):
    intent = detect_intent(q)
    if intent != "followup":
        return False
    return bool(history)


def resolve_followup(q, history):
    """«а почему?» после вопроса о небе → «почему небо голубое».

    Строит полноценный вопрос из темы предыдущего запроса и подсказки
    уточнения, чтобы дальше работал обычный разбор намерения.
    """
    if not history:
        return q, None
    prev = normalize(history[-1].get("user", ""))
    if not prev:
        return q, None
    topic = " ".join(keywords_of(prev)[:4])
    if not topic:
        return q, None
    qn = normalize(q)
    if re.search(r"\bпочему\b|\bзачем\b|\bотчего\b", qn):
        merged = "почему " + topic
    elif re.search(r"\bкак\b", qn):
        merged = "как " + topic
    elif "подробнее" in qn or "еще" in qn:
        merged = "расскажи подробнее про " + topic
    elif "сравни" in qn:
        merged = "сравни " + topic
    elif re.search(r"\bчто\b", qn):
        merged = "что такое " + topic
    else:
        merged = qn + " " + topic
    return merged.strip(), prev


# ------------------------------------------------------------------ точка входа

def analyze(question, history=None):
    """Полный разбор запроса.

    Возвращает словарь:
      original, normalized, intent, entities, keywords, query_for_search,
      followup_of, typo_fixed
    """
    original = str(question or "")
    raw = normalize(original)
    fixed = fix_typos(raw)
    typo_fixed = fixed != raw

    resolved_from = None
    if history and detect_intent(fixed) == "followup" and is_followup(fixed, history):
        fixed, resolved_from = resolve_followup(fixed, history)

    intent = detect_intent(fixed)
    entities = extract_entities(fixed, intent)
    keywords = keywords_of(fixed)
    search_query = expand_query(fixed, entities)

    return {
        "original": original,
        "normalized": fixed,
        "intent": intent,
        "entities": entities,
        "keywords": keywords,
        "query_for_search": search_query,
        "followup_of": resolved_from,
        "typo_fixed": typo_fixed,
    }


INTENT_TITLES = {
    "definition": "определение понятия",
    "comparison": "сравнение двух понятий",
    "howto": "объяснение «как сделать»",
    "why": "объяснение причины",
    "list": "перечисление",
    "fact": "фактический вопрос",
    "followup": "уточняющий вопрос",
    "unknown": "общий вопрос",
}
