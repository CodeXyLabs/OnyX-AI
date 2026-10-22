# -*- coding: utf-8 -*-
"""
OnyX Python Core · brain.py
============================
«Мозг»: связывает токенизатор, трансформер, знания и диалоговую память.
Отвечает на вопрос гибридом:
  1. если вопрос тривиальный (приветствие, вычисление) — отвечает сразу;
  2. ищет факты в базе знаний и строит компактный контекст;
  3. модель генерирует ответ, получая контекст как «память»;
  4. если генерация мусорная (низкая связность) — используется факт-ответ;
  5. результат сохраняется в историю диалога и используется для дообучения.
"""

import json
import os
import re
import time

try:
    import numpy as np
except ImportError:
    np = None

from .tokenizer import BPETokenizer
from .transformer import TransformerLM, PRESETS, estimate_params
try:
    from .solver import solve_task
except ImportError:
    from solver import solve_task
try:
    from .nlu import analyze, INTENT_TITLES
    from .composer import compose
except ImportError:
    from nlu import analyze, INTENT_TITLES
    from composer import compose
try:
    from ..knowledge.knowledge_base import (
        build_corpus_text, search_facts, context_for_facts, CHITCHAT, TOPIC_HINTS, DIALOGS,
        search_dialogs,
    )
    from ..knowledge.programming import lookup_programming, programming_stats
except ImportError:
    from knowledge.knowledge_base import (
        build_corpus_text, search_facts, context_for_facts, CHITCHAT, TOPIC_HINTS, DIALOGS,
        search_dialogs,
    )
    from knowledge.programming import lookup_programming, programming_stats


def normalize_question(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def is_greeting(q):
    p = re.compile(r"^(привет|здравствуй|здравствуйте|хай|прив|добрый (день|вечер|утро)|ку|hello|hi)\b")
    return bool(p.match(q))


def is_thanks(q):
    return bool(re.match(r"^(спасибо|благодарю|спс|спасиб)\b", q))


def is_identity(q):
    return bool(re.search(r"(кто ты|что ты|как тебя зовут|твоё имя|ты кто)", q))


def is_what_can_do(q):
    return bool(re.search(r"(что.*(умеешь|можешь)|умеешь|можешь помоч|возможност)", q))


def is_version(q):
    return bool(re.search(r"(какая.*(версия|модель)|версия|v0\.5|v 0\.5|какой ты)", q))


def is_calc(q):
    return bool(re.match(r"^\s*(\d+\s*[-+*/^×÷%()\s]+\d+)|(посчитай|вычисли|реши|сколько будет)\s*\d", q))


def eval_math(expr):
    try:
        expr = re.sub(r"^(посчитай|вычисли|реши|сколько будет)\s*", "", expr.strip(), flags=re.I)
        expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None


def match_dialog(question):
    """Ищет подходящий готовый диалог по ключевым словам (только для коротких вопросов)."""
    q = normalize_question(question)
    if len(q) > 42:
        return None
    # стоп-слова, из-за которых не стоит матчить готовый ответ
    STOP = {"такое", "расскажи", "что", "как", "про", "почему", "зачем", "сколько", "будет", "можно", "состоит", "из", "чем"}
    words = [w for w in q.split() if len(w) > 3 and w not in STOP]
    if not words:
        return None
    best_match = None
    best_score = 0
    for qq, aa in DIALOGS:
        qq_words = set(qq.split())
        score = sum(1 for w in words if w in qq_words)
        # штраф, если в вопросе есть слова, которых нет в примере диалога
        missing = sum(1 for w in words if len(w) > 5 and w not in qq_words)
        score -= missing * 0.6
        if score > best_score:
            best_score = score
            best_match = aa
    # для коротких вопросов достаточно одно точного совпадения значимого слова
    if best_match and best_score >= max(1, min(2, len(words))):
        return best_match
    return None


def compute_confidence(text):
    """Оценка связности текста: доля слов, похожих на настоящие (частотный словарь),
    минус штраф за повторы n-грамм и «склейки»."""
    text = text or ""
    words = re.findall(r"[а-яёa-z]+", text.lower())
    if not words:
        return 0.0
    vowels = set("аеёиоуыэюяaeiou")
    # частотный словарь настоящих русских слов
    common = set("""
        и в не на я что он с к а как это по но они у мы вы за то все же так вот
        из его или были быть при от о для она также можно который когда если
        нейросеть модель обучение содержит создаёт один компьютер язык работа
        число время функция слово значение программа алгоритм система данные
        информация процесс энергия скорость движение свет звук человек клетка
        организм геном история страна земля океан город год день неделя месяц
        свой первый другой наш ваш их нам вас него себя этом этом счёт вопрос
        ответ правила блок кода язык python база знаний сеть интернет протокол
        математика физика химия биология география история литература музыка
        искусство спорт здоровье питание психология экономика путешествия
        столица страна москва париж лондон токио пекин рим берлин сша китай
        россия индия бразилия канада австралия энергия гравитация орбита
        планета звезда галактика вселенная свет скорость частота длина волна
        клетка вирус бактерия геном днк белок жир углевод витамин вода солнце
        луна год месяц час минута секунда время путь скорость масса сила
        работа энергия заряд ток напряжение сопротивление процессор память
        диск файл система драйвер сеть маршрут адрес порт протокол сервер
        клиент приложение разработка тест отладка ошибка рефакторинг код
    """.split())
    real = 0
    for w in words:
        if w in common:
            real += 1
        elif len(w) >= 4 and any(c in vowels for c in w):
            # слово похоже на настоящее: есть гласные и правдоподобная длина
            vc = sum(1 for c in w if c in vowels)
            if vc >= 2 and vc <= len(w) * 0.8:
                real += 1
    ratio = real / len(words)
    # штраф за повторы
    ngram_penalty = 1.0
    if len(words) >= 4:
        tri = set()
        dup = 0
        n = 0
        for i in range(len(words) - 2):
            t = tuple(words[i:i+3])
            n += 1
            if t in tri:
                dup += 1
            tri.add(t)
        if n > 0:
            ngram_penalty = max(0.15, 1.0 - dup / n)
    # штраф за очень короткие слова-обрубки (ae, gh и т.п.)
    chopped = sum(1 for w in words if len(w) <= 2)
    chop_penalty = max(0.2, 1.0 - chopped * 0.35 / max(1, len(words)))
    length_bonus = 0.6 if len(text) < 30 else 1.0
    return max(0.0, min(1.0, ratio * ngram_penalty * chop_penalty * length_bonus))


class OnyxBrain:
    def __init__(self, workdir=None, preset="nano"):
        self.workdir = workdir or os.path.dirname(os.path.abspath(__file__))
        self.preset = preset
        # Папка моделей: явный workdir изолирует чекпоинт (тесты и эксперименты
        # не должны затирать рабочие веса). Без workdir — общий python_core/models.
        core_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(os.path.abspath(workdir), "models") if workdir \
            else os.path.join(core_root, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.model_dir, "onyx_python_brain.json")
        self.knowledge_path = os.path.join(core_root, "knowledge", "knowledge_base.py")
        self.history = []
        self.max_history = 12
        self.tokenizer_ready = False
        self.model_ready = False
        self.init_model()

    # ------------------------------------------------------------ setup
    def init_model(self):
        corpus = build_corpus_text()
        self.tokenizer = BPETokenizer()
        self.tokenizer.train(corpus, max_merges=1500, min_freq=2)
        self.model = TransformerLM(self.tokenizer.vocab_size, preset=self.preset, dtype=np.float64)
        self.tokenizer_ready = True
        # кешируем токены корпуса — при обучении берём случайные срезы
        self.corpus_ids = self.tokenizer.encode(corpus)
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tok_data = data.get("tokenizer", {})
                if tok_data:
                    self.tokenizer = BPETokenizer.from_json(json.dumps(tok_data))
                    self.corpus_ids = self.tokenizer.encode(build_corpus_text())
                if data.get("model"):
                    self.model.import_weights(data["model"])
                    self.model.steps = data.get("steps", 0)
                self.model_ready = True
            except Exception as e:
                print("[brain] checkpoint load failed:", e)
                self.model_ready = False
        # короткий «прогрев» если не было чекпоинта
        if not self.model_ready:
            self.warm_start()

    def _make_batch(self, ids, n_chunks, ctx_len, seed=None):
        """Собирает батч из случайных срезов корпуса (на границах пробелов,
        чтобы не разрезать слова)."""
        rng = np.random.default_rng(seed)
        chunks = []
        available = max(0, len(ids) - ctx_len - 1)
        if available < 1:
            ids = ids + ids
            available = len(ids) - ctx_len - 1
        # заранее индексируем позиции после пробела — старт с границы слова
        breaks = [i for i in range(0, available) if self._at_word_boundary(ids, i)]
        if len(breaks) < n_chunks:
            breaks = list(range(0, available))
        for _ in range(n_chunks):
            st = int(breaks[rng.integers(0, len(breaks))])
            chunks.append(np.array(ids[st:st + ctx_len + 1], dtype=np.int64)[None, :])
        return np.concatenate(chunks, axis=0)

    def _at_word_boundary(self, ids, pos):
        """True, если предыдущий токен — пробел или спецсимвол (граница слова)."""
        if pos <= 0:
            return True
        prev = ids[pos - 1]
        SPACE = self.tokenizer.char_to_id.get(" ")
        return prev < self.tokenizer.next_special or prev == SPACE

    def warm_start(self, steps=40):
        """Быстрое обучение на корпусе при первом запуске (без сохранения)."""
        T = min(96, self.model.ctx)
        B = 6
        for s in range(steps):
            try:
                X = self._make_batch(self.corpus_ids, B, T, seed=s)
                self.model.train_step(X, lr=8e-4, clip=1.0)
            except Exception:
                break
        self.model_ready = True

    def train_minutes(self, minutes=1.0):
        """Фоновое обучение на корпусе в течение заданного времени."""
        T = min(96, self.model.ctx)
        B = 8
        total = len(self.corpus_ids)
        end = time.time() + minutes * 60
        n = 0
        losses = []
        seed = self.model.steps
        while time.time() < end:
            try:
                X = self._make_batch(self.corpus_ids, B, T, seed=(seed + n) & 0xFFFF)
                loss = self.model.train_step(X, lr=5e-4, clip=1.0)
                losses.append(float(loss))
                n += 1
            except Exception:
                break
        self.save_checkpoint(force=True)
        avg = sum(losses) / len(losses) if losses else 0
        return {"steps": n, "avg_loss": avg, "last_loss": losses[-1] if losses else None}

    # ------------------------------------------------------------ save/load
    def save_checkpoint(self, force=False):
        os.makedirs(self.model_dir, exist_ok=True)
        data = {
            "tokenizer": json.loads(self.tokenizer.to_json()),
            "model": self.model.export_weights(quantize=False),
            "steps": self.model.steps,
            "saved_at": time.time(),
        }
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return self.checkpoint_path

    # ------------------------------------------------------------ answering
    def answer(self, question, temperature=0.8, max_tokens=220, with_context=True):
        q = normalize_question(question)
        start = time.time()

        # --- быстрые тривиальные ответы ---
        if is_greeting(q):
            return self._quick(random_choice(CHITCHAT), q, start)
        if is_thanks(q):
            return self._quick("Пожалуйста! Рад помочь. Если что-то ещё нужно — спрашивай.", q, start)
        if is_identity(q):
            return self._quick("Я OnyX — самообучаемая нейросеть. Отвечаю на вопросы, генерирую изображения и видео, умею искать сведения в сети и учиться на диалогах. Моя версия Python-ядра 0.5.", q, start)
        if is_version(q):
            return self._quick("Я работаю на ядре OnyX V0.5: трансформер, обученный на моём корпусе, плюс гибрид знаний и поиска. В интерфейсе можешь выбрать и другие модели — 2.5, 3.1, 4.0, 4.2, 4.3.", q, start)

        # --- вычислительные задачи: точный решатель (уравнения, проценты, прогрессии) ---
        solved = solve_task(q)
        if solved:
            return self._solved_answer(solved, q, start)

        # --- программирование: точная база знаний вместо догадок модели ---
        prog = lookup_programming(q, limit=1)
        if prog:
            return self._programming_answer(prog[0], q, start)

        # --- понимание запроса и сборка ответа из проверенных фактов ---
        analysis = analyze(q, self.history)
        composed = self._compose_answer(analysis, q, start)
        if composed:
            return composed

        if is_calc(q):
            val = eval_math(q)
            if val is not None:
                return self._quick("Результат: " + format_number(val), q, start)

        # быстрый матч по готовому диалогу (если вопрос похож на пример из обучения)
        dialog_hit = match_dialog(q)
        if dialog_hit:
            return self._quick(dialog_hit, q, start)

        # --- сбор фактов ---
        facts = search_facts(q, limit=7)
        fact_ctx = context_for_facts(facts, max_chars=700) if with_context else ""

        # --- контекст диалога (короткая выжимка) ---
        hist_ctx = self._history_ctx()

        # --- генерация моделью только для творческих запросов ---
        # На фактовых вопросах маленькая модель даёт ответ «не в тему» и стоит
        # 0.7–1.7 с. Там быстрее и точнее собрать ответ из фактов.
        creative = bool(re.search(
            r"(напиши|сочини|придумай|стих|рассказ|истори|слоган|идею|"
            r"варианты назван|представь|пофантазируй|подбери слов)", q))
        gen_text = ""
        if creative and (facts or fact_ctx):
            prompt_parts = []
            if hist_ctx:
                prompt_parts.append("контекст: " + hist_ctx[:300])
            if fact_ctx:
                prompt_parts.append("факты: " + fact_ctx[:700])
            prompt_question = "<user> " + q + " <ai>"
            prompt_parts.append(prompt_question)
            prompt_text = " ".join(prompt_parts)
            prompt_ids = self.tokenizer.encode(prompt_text)

            try:
                gen_ids = self.model.generate(
                    self.tokenizer, prompt_ids, max_new=min(max_tokens, 48),
                    temperature=temperature, top_k=45, top_p=0.92,
                    repetition_penalty=1.12, seed=None,
                )
                gen_text = self.tokenizer.decode(gen_ids)
            except Exception:
                gen_text = ""

        # пост-обработка: обрезать до разумного конца предложения
        gen_text = cleanup_generation(gen_text)
        # пытаемся сделать текст связным (убрать тарабарщину и обрывочные фразы)
        coherent = improve_coherence(gen_text, facts)
        if coherent:
            gen_text = coherent
        conf = compute_confidence(gen_text)
        # признак «склейки»: длинные цепочки букв без пробелов почти всегда мусор
        glued = bool(re.search(r"[а-яёa-z]{16,}", gen_text.replace(" ", "")))
        too_short = len(gen_text) < 40
        # мусорная генерация: низкая связность, склейка или слишком короткий текст
        broken = conf < 0.75 or glued or too_short

        # --- если генерация плохая, используем фактический ответ ---
        if broken:
            gen_text = build_fact_answer(facts, q)
            conf = max(0.6, 0.85 if facts else 0.45)

        self.history.append({"user": q, "ai": gen_text})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # --- режим мышления: строим «цепочку рассуждений» ---
        thinking = self.think_chain(q, intent=q, facts=facts, conf=conf, answer=gen_text)

        # -- автообучение на каждом диалоге: (вопрос → ответ) — фоновая дообучение
        try:
            blocked = ("порнограф", "эротик", "xxx", "наркотик", "педофил", "child porn", "патроны", "взрывчатк")
            if not any(b in (q + " " + gen_text).lower() for b in blocked):
                self.learn(q, gen_text, score=conf)
        except Exception:
            pass

        return {
            "text": gen_text,
            "confidence": conf,
            "facts": facts[:3],
            "model": f"onyx-python-{self.preset}",
            "ms": int((time.time() - start) * 1000),
            "lowConfidence": conf < 0.45,
            "thinking": thinking,
        }

    def _quick(self, text, q, start):
        self.history.append({"user": q, "ai": text})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        return {"text": text, "confidence": 0.99, "facts": [], "model": "onyx-python-fact",
                "ms": int((time.time() - start) * 1000), "lowConfidence": False,
                "thinking": self.think_chain(q, intent=q, facts=[], conf=0.99, answer=text)}

    def _direct(self, text, q, start, thinking, conf=0.97, tag="solver", low_confidence=False):
        """Прямой ответ из детерминированного модуля (решатель/база знаний).

        Такие ответы не «сочиняются» моделью: их не нужно проверять на связность,
        и обучение на них не имеет смысла (там точные данные, а не свободный текст).
        """
        self.history.append({"user": q, "ai": text})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        return {"text": text, "confidence": conf, "facts": [], "model": "onyx-python-" + tag,
                "ms": int((time.time() - start) * 1000), "lowConfidence": bool(low_confidence),
                "thinking": thinking[:8]}

    def _solved_answer(self, res, q, start):
        """Формирует ответ по результату решателя: ответ + проверяемые шаги."""
        kind_names = {
            "arithmetic": "арифметическое выражение",
            "equation": "уравнение",
            "system": "система уравнений",
            "percent": "задача на проценты",
            "progression": "прогрессия",
            "number": "теория чисел",
            "combinatorics": "комбинаторика",
            "units": "перевод единиц",
            "motion": "задача на движение",
        }
        steps = res.get("steps") or []
        text = "Ответ: " + res["answer"]
        if steps:
            text += "\n\nРешение:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))

        thinking = [f"Задача распознана как {kind_names.get(res.get('kind'), 'вычисление')} — решаю точно, а не генерирую."]
        thinking.extend(steps)
        thinking.append("Проверяю результат подстановкой и арифметикой.")
        return self._direct(text, q, start, thinking, conf=0.99, tag="solver")

    def _programming_answer(self, hit, q, start):
        """Ответ по базе знаний программирования: тема, объяснение, пример кода."""
        if isinstance(hit, list):
            hit = hit[0] if hit else {}
        title = hit.get("title", "Программирование")
        answer = (hit.get("answer") or "").strip()
        code = hit.get("code", "")

        # инструкции («как …», «с чего начать …») отдаём нумерованными шагами
        if re.match(r"^(как|с чего начать|каким образом)\b", q):
            plan = re.search(r"план:\s*(.+)", answer, re.I)
            if plan and "→" in plan.group(1):
                items = [x.strip() for x in plan.group(1).split("→") if x.strip()]
                if len(items) >= 2:
                    answer = "План по шагам:\n" + "\n".join(
                        f"{i}. {x}" for i, x in enumerate(items[:8], 1))
            if not answer.startswith("План по шагам"):
                steps = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer)
                         if len(s.split()) >= 3]
                if len(steps) >= 2:
                    answer = "\n".join(f"{i}. {s.rstrip('.!')}"
                                       for i, s in enumerate(steps[:6], 1))
        if answer and not answer.endswith((".", "!", "?", "…")):
            answer += "."

        text = title + "\n\n" + answer
        if code:
            text += "\n\nПример:\n" + code

        thinking = [
            "Вопрос относится к программированию — беру точную статью из базы знаний.",
            f"Совпавшая тема: «{title}» (вес совпадения {hit.get('score', '—')}).",
            "Отвечаю фактами и рабочим примером кода, без свободной генерации модели.",
        ]
        return self._direct(text, q, start, thinking, conf=0.97, tag="code")

    def _gather_facts(self, analysis):
        """Факты из трёх источников: расширенный запрос, сущности, диалоги."""
        q = analysis.get("query_for_search") or ""
        facts = list(search_facts(q, limit=8))
        seen = {f["fact"] for f in facts}
        for ent in analysis.get("entities") or []:
            if not ent:
                continue
            for f in search_facts(ent, limit=4):
                if f["fact"] not in seen:
                    seen.add(f["fact"])
                    facts.append(f)
        try:
            for f in search_dialogs(analysis.get("normalized", ""), limit=3):
                if f["fact"] not in seen:
                    seen.add(f["fact"])
                    facts.append(f)
        except Exception:
            pass
        return facts

    def _compose_answer(self, analysis, q, start):
        """Собирает ответ под намерение. None — если фактов нет и нужен поиск/модель."""
        intent = analysis.get("intent", "unknown")
        facts = self._gather_facts(analysis)
        res = compose(analysis, facts, q)
        if not res or not res.get("answer"):
            return None
        conf = float(res.get("confidence") or 0.0)
        structured = intent in ("definition", "comparison", "howto", "why", "list", "detail")
        # для общих вопросов берём сборку только при хорошей опоре на факты
        if not structured and conf < 0.75:
            return None

        thinking = ["Намерение: " + INTENT_TITLES.get(intent, intent) + "."]
        if analysis.get("entities"):
            thinking.append("Сущности вопроса: " + ", ".join(f"«{e}»" for e in analysis["entities"]) + ".")
        if analysis.get("typo_fixed"):
            thinking.append("Исправил опечатку/словоформу: «" + analysis.get("normalized", "") + "».")
        if analysis.get("followup_of"):
            thinking.append("Это уточнение — подставил тему прошлого вопроса: «" + analysis["followup_of"] + "».")
        topics = " · ".join(dict.fromkeys(f.get("topic", "") for f in facts[:4]))
        used = len(res.get("used") or [])
        thinking.append(f"Нашёл фактов: {len(facts)} (темы: {topics or '—'}); в ответ взял {used}.")
        thinking.append(f"Уверенность: {int(conf * 100)}%.")
        return self._direct(res["answer"], q, start, thinking,
                            conf=max(0.3, min(0.99, conf)), tag="answer",
                            low_confidence=bool(res.get("low")))

    def think_chain(self, question, intent=None, facts=None, conf=0.5, answer="", max_steps=6):
        """Строит «режим мышления»: последовательность внутренних шагов,
        которые привели к ответу. Шаги понятные и проверяемые."""
        facts = facts or []
        q = normalize_question(question)
        steps = []

        # 1. распознаём намерение
        if is_greeting(q):
            steps.append("Пользователь поздоровался — это социальный запрос, развёрнутый анализ не нужен.")
            steps.append("Отвечаю приветствием и предлагаю помощь.")
            return steps
        if is_calc(q):
            steps.append("В вопросе есть арифметическое выражение — вычисляю напрямую.")
            steps.append("Применяю правила арифметики к выражению.")
            return steps
        if is_identity(q):
            steps.append("Вопрос о моей идентичности — отвечаю фактами о своей цели и возможностях.")
            return steps
        if is_version(q):
            steps.append("Спрашивают о версии — сообщаю текущее ядро и варианты моделей.")
            return steps

        if "что" in q and "такое" in q:
            steps.append("Вопрос формата «что такое X» — ищу определение в базе знаний.")
        elif q.startswith(("как", "почему", "зачем")):
            steps.append("Вопрос-объяснение — собираю причинно-следственные факты.")
        elif "сравн" in q or "отличи" in q:
            steps.append("Вопрос сравнения — ищу пары фактов про оба объекта.")
        else:
            steps.append("Распознаю ключевые понятия вопроса и ищу связанные факты.")

        key_words = [w for w in q.split() if len(w) > 3]
        if key_words:
            steps.append("Ключевые понятия: " + ", ".join(key_words[:5]) + ".")
        else:
            steps.append("Отдельных ключевых понятий не выделено.")

        if facts:
            top_topics = " · ".join(dict.fromkeys(f["topic"] for f in facts[:3]))
            steps.append(f"Найдено фактов: {len(facts)} — по темам: {top_topics}.")
            steps.append(f"Главный факт: «{facts[0]['fact'][:110]}…».")
        else:
            steps.append("В базе знаний точных фактов не нашлось — использую готовый ответ из базы диалогов.")

        steps.append(f"Уверенность в ответе: {int(conf * 100)}%.")

        if answer:
            brief = answer[:120].rstrip()
            steps.append(f"Формулирую ответ: «{brief}{'…' if len(answer) > 120 else ''}».")

        return steps[:max_steps]

    def _history_ctx(self):
        parts = []
        for h in self.history[-3:]:
            parts.append(h["user"] + " -> " + h["ai"][:90])
        return " ".join(parts)

    # ------------------------------------------------------------ learning from dialog
    def learn(self, question, answer, score=1):
        """Дообучение на диалоге: один быстрый шаг, чтобы не задерживать ответ.
        Прогресс накапливается и сохраняется при следующем /api/save или перезапуске."""
        try:
            text = normalize_question(question) + " " + normalize_question(answer)
            ids = self.tokenizer.encode(text)
            if len(ids) < 12:
                return False
            # один компактный батч из последних 96 токенов диалога
            if len(ids) > self.model.ctx:
                ids = ids[-self.model.ctx:]
            import numpy as _np
            X2 = _np.array([ids[:1] + [0] * (len(ids) - 1)], dtype=_np.int64)  # placeholder
            # строим пару (вход, цель) для предсказания следующего токена
            inp = _np.array(ids[:-1], dtype=_np.int64)
            X = _np.zeros((1, len(inp) + 1), dtype=_np.int64)
            X[0, :-1] = inp
            self.model.train_step(X, lr=2e-4, clip=1.0, weight_decay=1e-4)
            return True
        except Exception:
            return False

    def status(self):
        try:
            kstats = programming_stats()
        except Exception:
            kstats = {"entries": 0, "topics": 0}
        return {
            "model": f"onyx-python-{self.preset}",
            "params": self.model.num_params(),
            "steps": self.model.steps,
            "vocab": self.tokenizer.vocab_size,
            "layers": self.model.n_layers,
            "dModel": self.model.d_model,
            "heads": self.model.n_heads,
            "ctx": self.model.ctx,
            "history": len(self.history),
            "preset": self.preset,
            "ready": self.model_ready and self.tokenizer_ready,
            "knowledge": {
                "programming_entries": kstats.get("entries", 0),
                "programming_topics": kstats.get("topics", 0),
                "solver": True,
            },
            "presets": {k: estimate_params(self.tokenizer.vocab_size, k) for k in PRESETS},
        }


# ------------------------------------------------------------------ helpers
def random_choice(lst):
    import random
    return random.choice(lst)


def format_number(v):
    if isinstance(v, float):
        return "{:.10g}".format(v)
    return str(v)


def cleanup_generation(text):
    """Убирает мусор из сгенерированного текста: служебные токены, повторяющиеся пробелы."""
    t = text or ""
    t = re.sub(r"\[(bos|eos|user|ai)\]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    # если внутри остались маркеры угловых скобок — убираем их содержимое
    t = re.sub(r"<[^>]{0,12}>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # обрезаем по последней точке/вопросительному, если текст длинный
    if len(t) > 160:
        for sep in (". ", "! ", "? "):
            idx = t.rfind(sep, 40, len(t))
            if idx > 60:
                t = t[:idx + 1]
                break
    return t


def split_sentences(text):
    """Делит текст на предложения (грубо, но для русского достаточно)."""
    t = (text or "").replace("\n", " ").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", t)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out if out else [t]


def sentence_quality(sentence):
    """Оценка связности одного предложения: 0..1."""
    words = re.findall(r"[а-яёa-z]+", sentence.lower())
    if not words:
        return 0.0
    vowels = set("аеёиоуыэюяaeiou")
    # нормальные русские слова имеют гласные и не состоят из одних согласных
    good = 0
    for w in words:
        vc = sum(1 for c in w if c in vowels)
        if len(w) >= 3 and 1 <= vc <= len(w) * 0.8:
            good += 1
    ratio = good / len(words)
    # штраф за длинные «склейки» без пробелов уже отсеяны; штраф за обрывочные слова
    shorts = sum(1 for w in words if len(w) <= 2)
    return max(0.0, ratio * (1.0 - shorts * 0.4 / max(1, len(words))))


def improve_coherence(text, facts):
    """Делает сгенерированный текст связным: отсекает мусорные предложения,
    переставляет их в логичный порядок и соединяет естественными переходами."""
    if not text:
        return None
    sentences = split_sentences(text)
    # оставляем предложения, которые выглядят осмысленно
    good = [s for s in sentences if sentence_quality(s) >= 0.6 and len(s.split()) >= 2]
    if len(good) >= 2:
        # первым ставим самое «описательное» предложение (с содержательными словами)
        good.sort(key=lambda s: -len(re.findall(r"[а-яёa-z]{4,}", s)))
        # убираем дубликаты (похожие начала)
        unique = []
        seen_starts = set()
        for s in good:
            key = s[:30].lower()
            if key in seen_starts:
                continue
            seen_starts.add(key)
            unique.append(s)
        if len(unique) >= 2:
            def _cap2(s):
                return s[0].upper() + s[1:] if s else s
            head = unique[0]
            body = unique[1:]
            connectors = ["Кроме того, ", "Дополнительно: ", "Важно и то, что ", ""]
            parts = [head]
            for i, s in enumerate(body):
                conn = connectors[i % len(connectors)]
                parts.append((conn + _cap2(s)) if conn else s)
            return " ".join(parts).strip()
    # не получилось сделать связный текст — вернём None, чтобы взять факты
    return None


def build_fact_answer(facts, question):
    """Собирает ответ из релевантных фактов, если модель не справилась."""
    if not facts:
        # пытаемся удержать диалог: переформулируем общий ответ
        q = normalize_question(question)
        if "что" in q and "такое" in q:
            return ("По этому вопросу у меня пока нет точных данных в базе знаний. "
                    "Могу объяснить в общих чертах, найти сведения в открытых источниках "
                    "или уточнить, что именно тебя интересует?")
        return ("Я — OnyX, самообучаемая нейросеть. Прямого ответа в моей базе знаний "
                "сейчас нет, но я могу поискать сведения в сети или обсудить тему. "
                "Что именно тебя интересует?")
    lines = []
    if len(facts) == 1:
        lines.append(facts[0]["fact"])
    else:
        seen = set()
        for f in facts:
            fact = f["fact"]
            if fact in seen:
                continue
            seen.add(fact)
            lines.append(fact.capitalize() if not fact[0].isupper() else fact)
        lines = lines[:3]
    # соединяем факты естественными переходами, а не просто склейкой
    def _cap(s):
        return s[0].upper() + s[1:] if s else s

    def _lower(s):
        # не переводим в нижний регистр акронимы/сокращения
        first = s.split(" ")[0] if s else ""
        if len(first) > 1 and first == first.upper():
            return s
        return s[0].lower() + s[1:] if s else s

    if len(lines) == 1:
        answer = lines[0]
    elif len(lines) == 2:
        answer = _cap(lines[0]) + " " + _cap(lines[1])
    else:
        answer = _cap(lines[0]) + " " + _cap(lines[1]) + \
                 " Кроме того, " + _lower(lines[2].lstrip())
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer.endswith((".", "!", "?")):
        answer += "."
    if len(answer) < 120:
        answer += " Если нужны подробности, могу рассказать больше."
    return answer[:700]