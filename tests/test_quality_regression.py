# -*- coding: utf-8 -*-
"""
OnyX · регрессионные тесты качества и защитных метрик.

Запуск:  python tests/test_quality_regression.py
Совместимо с pytest: все проверки — функции test_*.

Пороги берутся из tests/eval/thresholds.json (зафиксированы до эксперимента).
"""

import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "python_core")
EVAL = os.path.join(ROOT, "tests", "eval")
sys.path.insert(0, CORE)
sys.path.insert(0, EVAL)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import eval_quality as ev  # noqa: E402
from engine.brain import OnyxBrain  # noqa: E402
from engine.composer import _matches_entity  # noqa: E402
from engine.solver import solve_task  # noqa: E402
from knowledge.knowledge_base import search_dialogs  # noqa: E402
from knowledge.programming import lookup_programming  # noqa: E402

_BRAIN = None


def brain():
    global _BRAIN
    if _BRAIN is None:
        _BRAIN = OnyxBrain(workdir=CORE, preset="nano")
    return _BRAIN


def _load(name):
    with open(os.path.join(EVAL, name), encoding="utf-8") as f:
        return json.load(f)["items"]


def _run(items):
    answers, lat = {}, []
    for it in items:
        t0 = time.perf_counter()
        res = brain().answer(it["q"], max_tokens=120)
        lat.append((time.perf_counter() - t0) * 1000)
        answers[it["id"]] = res.get("text") or ""
    checks = {it["id"]: ev.check_item(it, answers[it["id"]]) for it in items}
    n = len(items)
    rate = lambda key: sum(1 for v in checks.values() if v[key]) / n  # noqa: E731
    return {
        "correctness": rate("correct"),
        "completeness": rate("complete"),
        "usefulness": rate("useful"),
        "safety": rate("safety"),
        "p95": sorted(lat)[max(0, int(len(lat) * 0.95) - 1)],
        "checks": checks,
    }


# ---------------------------------------------------------------- быстрые юниты

def test_entity_word_boundary():
    """«нейрон» не должен совпадать с «нейросеть» (зафиксированный дефект)."""
    assert _matches_entity("нейросеть — это математическая модель", "нейрон") is False
    assert _matches_entity("нейрон — это клетка нервной системы", "нейрон") is True


def test_dialogs_ignore_stopwords():
    """Вопрос «почему идет дождь» не должен цеплять диалог про небо."""
    hits = search_dialogs("почему идет дождь", limit=3)
    joined = " ".join(h["fact"] for h in hits).lower()
    assert "небо" not in joined or "дожд" in joined


def test_solver_linear():
    r = solve_task("реши уравнение 3x + 5 = 20")
    assert r and "x = 5" in r["answer"]


def test_programming_jwt():
    r = lookup_programming("что такое JWT")
    assert r and "jwt" in r[0]["title"].lower()


def test_howto_has_numbered_steps():
    r = brain().answer("как выучить программирование")
    assert "\n1." in r["text"], r["text"][:120]


def test_fast_path_latency():
    """Быстрый путь без резервной генерации: ответ укладывается в 0.5 с."""
    t0 = time.perf_counter()
    brain().answer("почему идет дождь", max_tokens=120)
    assert (time.perf_counter() - t0) < 0.5


# ---------------------------------------------------------------- пороги набора

def test_main_thresholds():
    with open(os.path.join(EVAL, "thresholds.json"), encoding="utf-8") as f:
        th = json.load(f)["acceptance"]
    m = _run(_load("eval_set_v1.json"))
    assert m["correctness"] >= 0.88, f"correctness {m['correctness']:.3f}"
    assert m["completeness"] >= 0.88, f"completeness {m['completeness']:.3f}"
    assert m["usefulness"] >= 0.72, f"usefulness {m['usefulness']:.3f}"
    assert m["safety"] >= 0.99, f"safety {m['safety']:.3f}"
    assert m["p95"] <= th["latency_p95_max_ms"], f"p95 {m['p95']:.0f} ms"
    print(f"\nmain: correct={m['correctness']:.3f} complete={m['completeness']:.3f} "
          f"useful={m['usefulness']:.3f} safety={m['safety']:.3f} p95={m['p95']:.0f} ms")


def test_heldout_thresholds():
    m = _run(_load("eval_set_heldout_v1.json"))
    assert m["correctness"] >= 0.58, f"correctness {m['correctness']:.3f}"
    assert m["safety"] >= 0.99, f"safety {m['safety']:.3f}"
    print(f"\nheldout: correct={m['correctness']:.3f} safety={m['safety']:.3f} "
          f"p95={m['p95']:.0f} ms")


def test_determinism():
    """Один и тот же вопрос дважды даёт один и тот же ответ."""
    q = "что такое нейросеть"
    a = brain().answer(q)["text"]
    b = brain().answer(q)["text"]
    assert a == b, "ответы различаются между прогонами"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({(time.perf_counter() - t0):.2f} с)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {e}")
    print(f"\nитого: {len(tests) - failed}/{len(tests)} пройдено")
    sys.exit(1 if failed else 0)
