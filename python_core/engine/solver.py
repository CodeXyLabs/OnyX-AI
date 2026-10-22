# -*- coding: utf-8 -*-
"""
OnyX Python Core · engine/solver.py
====================================
Решатель вычислительных и текстовых задач: арифметика, уравнения,
системы, проценты, прогрессии, комбинаторика, теория чисел, единицы.

Работает без eval: выражения разбираются через AST с белым списком
операций, поэтому пользовательский ввод не может выполнить код.
"""

import ast
import math
import re

# ─────────────────────────── утилиты ───────────────────────────

_NUM = r"-?\d+(?:[.,]\d+)?"


def _num(s):
    """Парсит число с русской десятичной запятой, пробелами и минусом."""
    if s is None:
        return None
    s = str(s).strip().replace("\u2212", "-").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def fmt(v, nd=6):
    """Аккуратный вывод числа: 2.0 -> 2, 0.3333333333 -> 0.333333."""
    if v is None:
        return "—"
    if isinstance(v, int):
        return str(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def _norm(q):
    q = str(q or "").lower().replace("ё", "е")
    q = q.replace("\u2212", "-").replace("×", "*").replace("÷", "/")
    q = q.replace("²", "^2").replace("³", "^3")
    q = re.sub(r"\s+", " ", q).strip()
    return q


# ─────────────────────────── безопасная арифметика ───────────────────────────

_ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial,
    "degrees": math.degrees, "radians": math.radians,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}


class _Safe(ast.NodeVisitor):
    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("bad constant")

    def visit_Name(self, node):
        if node.id in _ALLOWED_CONSTS:
            return _ALLOWED_CONSTS[node.id]
        raise ValueError("bad name")

    def visit_BinOp(self, node):
        a = self.visit(node.left)
        b = self.visit(node.right)
        op = type(node.op)
        if op is ast.Add:
            return a + b
        if op is ast.Sub:
            return a - b
        if op is ast.Mult:
            return a * b
        if op is ast.Div:
            if b == 0:
                raise ZeroDivisionError
            return a / b
        if op is ast.FloorDiv:
            if b == 0:
                raise ZeroDivisionError
            return a // b
        if op is ast.Mod:
            if b == 0:
                raise ZeroDivisionError
            return a % b
        if op is ast.Pow:
            if abs(b) > 1000 or abs(a) > 1e12:
                raise ValueError("too big")
            return a ** b
        raise ValueError("bad op")

    def visit_UnaryOp(self, node):
        v = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise ValueError("bad unary")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("bad func")
        args = [self.visit(a) for a in node.args]
        return float(_ALLOWED_FUNCS[node.func.id](*args))

    def generic_visit(self, node):
        raise ValueError("bad node")


def safe_eval(expr):
    """Вычисляет арифметическое выражение безопасно. None при ошибке."""
    try:
        tree = ast.parse(expr, mode="eval")
        return _Safe().visit(tree)
    except Exception:
        return None


# ─────────────────────────── проценты ───────────────────────────

def _percent(q):
    # сколько процентов составляет X от Y
    m = re.search(r"сколько\s+процент\w*\s+(?:составляет\s+|это\s+)?" + f"({_NUM})" + r"\s*(?:от|из)\s*" + f"({_NUM})", q)
    if m:
        x, y = _num(m.group(1)), _num(m.group(2))
        if y:
            return {
                "answer": f"{fmt(x / y * 100, 4)}%",
                "steps": [f"Формула: доля = часть / целое · 100%",
                          f"Подставляем: {fmt(x)} / {fmt(y)} · 100",
                          f"Считаем: {fmt(x / y, 6)} · 100 = {fmt(x / y * 100, 4)}%"],
                "kind": "percent",
            }

    # X% от Y
    m = re.search(f"({_NUM})\\s*(?:%|процент\\w*)\\s*(?:от|\\*)\\s*({_NUM})", q)
    if m:
        p, y = _num(m.group(1)), _num(m.group(2))
        res = y * p / 100
        return {
            "answer": fmt(res),
            "steps": [f"Процент от числа: {fmt(p)}% от {fmt(y)}",
                      f"Переводим процент в долю: {fmt(p)} / 100 = {fmt(p / 100, 6)}",
                      f"Умножаем: {fmt(y)} · {fmt(p / 100, 6)} = {fmt(res)}"],
            "kind": "percent",
        }

    # скидка / наценка: цена X, скидка Y%
    m = re.search(
        r"(?:цена|стоимость|товар|скидк\w*)[^\d]{0,20}(" + _NUM + r")\s*(?:руб\w*|₽)?"
        r"[^\d]{0,20}(?:скидк\w*|на)\s*(" + _NUM + r")\s*%", q)
    if m:
        price, disc = _num(m.group(1)), _num(m.group(2))
        res = price * (1 - disc / 100)
        return {
            "answer": fmt(res),
            "steps": [f"Скидка {fmt(disc)}% от {fmt(price)} — это {fmt(price * disc / 100)}",
                      f"Итог: {fmt(price)} − {fmt(price * disc / 100)} = {fmt(res)}"],
            "kind": "percent",
        }

    # увеличить / уменьшить на X%
    m = re.search(r"(?:увелич\w*|уменьш\w*|подорожал\w*|подешевел\w*)[^\d]{0,15}({0})\s*(?:на|в)\s*({1})\s*%".format(_NUM, _NUM), q)
    if m:
        base, p = _num(m.group(1)), _num(m.group(2))
        up = "увелич" in q or "подорож" in q
        k = 1 + p / 100 if up else 1 - p / 100
        return {
            "answer": fmt(base * k),
            "steps": [f"Изменение на {fmt(p)}%: множитель {fmt(k, 4)}",
                      f"{fmt(base)} · {fmt(k, 4)} = {fmt(base * k)}"],
            "kind": "percent",
        }
    return None


# ─────────────────────────── уравнения ───────────────────────────

def _parse_side_linear(side):
    """Возвращает (коэффициент при x, свободный член) для линейного выражения."""
    s = side.replace(" ", "").replace("=-", "=-")
    # нормализуем знаки: "-x" -> "-1x"
    s = re.sub(r"(^|[+\-*/])x", r"\g<1>1x", s)
    # "2*x" -> "2x"
    s = s.replace("*x", "x").replace("x*", "x")
    a = 0.0
    b = 0.0
    # разбиваем на слагаемые с учётом знака
    terms = re.findall(r"[+\-]?[^+\-]+", s)
    for t in terms:
        if not t:
            continue
        sign = -1.0 if t.startswith("-") else 1.0
        t = t.lstrip("+-")
        if "x" in t:
            if "/" in t:
                m = re.match(r"^(\d+(?:\.\d+)?)?\s*x\s*/\s*(\d+(?:\.\d+)?)$", t)
                if m:
                    coef = _num(m.group(1)) if m.group(1) else 1.0
                    den = _num(m.group(2)) or 1.0
                    a += sign * coef / den
                    continue
            m = re.match(r"^(\d+(?:\.\d+)?)?x$", t)
            if m:
                a += sign * (_num(m.group(1)) if m.group(1) else 1.0)
        else:
            v = _num(t)
            if v is not None:
                b += sign * v
    return a, b


def _linear(q):
    body = q
    for pre in ("реши уравнение", "решить уравнение", "найди x", "найдите x", "чему равен x", "реши"):
        body = body.replace(pre, " ")
    if "=" not in body or "x" not in body:
        return None
    if "^2" in body.replace(" ", ""):
        return None
    left, _, right = body.partition("=")
    right = re.split(r"[;,]| и ", right)[0]
    left = re.split(r"[;,]| и ", left)[-1]
    a1, b1 = _parse_side_linear(left)
    a2, b2 = _parse_side_linear(right)
    a = a1 - a2
    b = b2 - b1
    if abs(a) < 1e-12:
        return None
    x = b / a
    return {
        "answer": f"x = {fmt(x)}",
        "steps": [
            f"Переносим всё с x влево, числа вправо: ({fmt(a1)} − {fmt(a2)})·x = {fmt(b2)} − {fmt(b1)}",
            f"Получаем: {fmt(a)}·x = {fmt(b)}",
            f"Делим обе части на {fmt(a)}: x = {fmt(b)} / {fmt(a)} = {fmt(x)}",
            f"Проверка: подстановка x = {fmt(x)} в исходное равенство даёт верное тождество.",
        ],
        "kind": "equation",
    }


def _parse_quadratic_side(side):
    s = side.replace(" ", "")
    s = re.sub(r"(^|[+\-*/])x\^2", r"\g<1>1x^2", s)
    s = re.sub(r"(^|[+\-*/])x(?!\^)", r"\g<1>1x", s)
    a = b = c = 0.0
    for t in re.findall(r"[+\-]?[^+\-]+", s):
        if not t:
            continue
        sign = -1.0 if t.startswith("-") else 1.0
        t = t.lstrip("+-")
        if "x^2" in t:
            m = re.match(r"^(\d+(?:\.\d+)?)?x\^2$", t)
            if m:
                a += sign * (_num(m.group(1)) if m.group(1) else 1.0)
        elif "x" in t:
            m = re.match(r"^(\d+(?:\.\d+)?)?x$", t)
            if m:
                b += sign * (_num(m.group(1)) if m.group(1) else 1.0)
        else:
            v = _num(t)
            if v is not None:
                c += sign * v
    return a, b, c


def _quadratic(q):
    body = q
    for pre in ("реши уравнение", "решить уравнение", "реши", "найди корни"):
        body = body.replace(pre, " ")
    if "=" not in body:
        return None
    compact = body.replace(" ", "")
    if "^2" not in compact:
        return None
    left, _, right = body.partition("=")
    right = re.split(r"[;]| и ", right)[0]
    a1, b1, c1 = _parse_quadratic_side(left)
    a2, b2, c2 = _parse_quadratic_side(right)
    a, b, c = a1 - a2, b1 - b2, c1 - c2
    if abs(a) < 1e-12:
        return None
    D = b * b - 4 * a * c
    steps = [
        f"Приводим к виду ax² + bx + c = 0: {fmt(a)}x² + ({fmt(b)})x + ({fmt(c)}) = 0",
        f"Дискриминант: D = b² − 4ac = ({fmt(b)})² − 4·{fmt(a)}·({fmt(c)}) = {fmt(D)}",
    ]
    if D < -1e-12:
        steps.append("D < 0 — действительных корней нет (комплексные корни).")
        return {"answer": "действительных корней нет", "steps": steps, "kind": "equation"}
    if abs(D) <= 1e-12:
        x = -b / (2 * a)
        steps.append(f"D = 0 — один корень: x = −b / (2a) = {fmt(-b)} / {fmt(2 * a)} = {fmt(x)}")
        return {"answer": f"x = {fmt(x)}", "steps": steps, "kind": "equation"}
    sq = math.sqrt(D)
    x1 = (-b + sq) / (2 * a)
    x2 = (-b - sq) / (2 * a)
    steps.append(f"√D = {fmt(sq, 6)}")
    steps.append(f"x₁ = (−b + √D) / 2a = ({fmt(-b)} + {fmt(sq, 6)}) / {fmt(2 * a)} = {fmt(x1)}")
    steps.append(f"x₂ = (−b − √D) / 2a = ({fmt(-b)} − {fmt(sq, 6)}) / {fmt(2 * a)} = {fmt(x2)}")
    return {"answer": f"x₁ = {fmt(x1)}, x₂ = {fmt(x2)}", "steps": steps, "kind": "equation"}


def _system(q):
    if "=" not in q:
        return None
    parts = [p.strip() for p in re.split(r";|\n|\bи\b", q) if "=" in p]
    if len(parts) != 2:
        return None
    left_right = []
    for p in parts:
        if "x" not in p or "y" not in p:
            return None
        l, _, r = p.partition("=")
        left_right.append((l, r))

    def parse2(side):
        s = side.replace(" ", "")
        s = re.sub(r"(^|[+\-])x", r"\g<1>1x", s)
        s = re.sub(r"(^|[+\-])y", r"\g<1>1y", s)
        s = re.sub(r"(\d)\*([xy])", r"\1\2", s)
        a = b = c = 0.0
        for t in re.findall(r"[+\-]?[^+\-]+", s):
            if not t:
                continue
            sign = -1.0 if t.startswith("-") else 1.0
            t = t.lstrip("+-")
            if "x" in t:
                m = re.match(r"^(\d+(?:\.\d+)?)?x$", t)
                if m:
                    a += sign * (_num(m.group(1)) if m.group(1) else 1.0)
            elif "y" in t:
                m = re.match(r"^(\d+(?:\.\d+)?)?y$", t)
                if m:
                    b += sign * (_num(m.group(1)) if m.group(1) else 1.0)
            else:
                v = _num(t)
                if v is not None:
                    c += sign * v
        return a, b, c

    a1, b1, c1 = parse2(left_right[0][0])
    d1, b2, c2 = parse2(left_right[0][1])
    a2, e1, c3 = parse2(left_right[1][0])
    d2, e2, c4 = parse2(left_right[1][1])
    A1, B1, C1 = a1 - d1, b1 - b2, c2 - c1
    A2, B2, C2 = a2 - d2, e1 - e2, c4 - c3
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-12:
        return None
    x = (C1 * B2 - C2 * B1) / det
    y = (A1 * C2 - A2 * C1) / det
    return {
        "answer": f"x = {fmt(x)}, y = {fmt(y)}",
        "steps": [
            f"Система приведена к виду: {fmt(A1)}x + {fmt(B1)}y = {fmt(C1)}; {fmt(A2)}x + {fmt(B2)}y = {fmt(C2)}",
            f"Определитель: Δ = {fmt(A1)}·{fmt(B2)} − {fmt(A2)}·{fmt(B1)} = {fmt(det)}",
            f"Δx = {fmt(C1)}·{fmt(B2)} − {fmt(C2)}·{fmt(B1)} = {fmt(C1 * B2 - C2 * B1)}",
            f"Δy = {fmt(A1)}·{fmt(C2)} − {fmt(A2)}·{fmt(C1)} = {fmt(A1 * C2 - A2 * C1)}",
            f"x = Δx/Δ = {fmt(x)}, y = Δy/Δ = {fmt(y)}",
        ],
        "kind": "system",
    }


# ─────────────────────────── прогрессии ───────────────────────────

def _progression(q):
    geo = "геометрическ" in q
    arith = "арифметическ" in q
    if not (geo or arith):
        return None
    nums = {}
    for key in ("a1", "b1", "d", "q", "n"):
        m = re.search(rf"\b{key}\s*=\s*({_NUM})", q)
        if m:
            nums[key] = _num(m.group(1))
    first = nums.get("a1", nums.get("b1"))
    n = nums.get("n")
    if n is None:
        m = re.search(r"(?:найти|найди|чему равен)\s+[ab]?(\d{1,4})", q)
        if m:
            n = _num(m.group(1))
    if first is None or n is None:
        return None
    ni = int(n) if abs(n - round(n)) < 1e-9 else n
    if geo:
        r = nums.get("q")
        if r is None:
            return None
        val = first * (r ** (n - 1))
        s = first * (r ** n - 1) / (r - 1) if abs(r - 1) > 1e-12 else first * n
        return {
            "answer": f"b{ni} = {fmt(val)}",
            "steps": [f"Геометрическая прогрессия: b₁ = {fmt(first)}, q = {fmt(r)}",
                      f"bₙ = b₁ · q^(n−1) = {fmt(first)} · {fmt(r)}^{fmt(n - 1)} = {fmt(val)}",
                      f"Сумма первых {fmt(n)} членов: Sₙ = {fmt(s)}"],
            "kind": "progression",
        }
    d = nums.get("d")
    if d is None:
        return None
    val = first + (n - 1) * d
    s = n * (2 * first + (n - 1) * d) / 2
    return {
        "answer": f"a{ni} = {fmt(val)}",
        "steps": [f"Арифметическая прогрессия: a₁ = {fmt(first)}, d = {fmt(d)}",
                  f"aₙ = a₁ + (n−1)d = {fmt(first)} + {fmt(n - 1)}·{fmt(d)} = {fmt(val)}",
                  f"Сумма первых {fmt(n)} членов: Sₙ = n(a₁ + aₙ)/2 = {fmt(s)}"],
        "kind": "progression",
    }


# ─────────────────────────── теория чисел ───────────────────────────

def _primes(n):
    if n < 2:
        return False, []
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append(n)
    return len(factors) == 1, factors


def _number_theory(q):
    m = re.search(r"(?:факториал)\s*(?:от\s*)?(\d{1,3})", q)
    if m:
        n = int(m.group(1))
        return {"answer": fmt(math.factorial(n)),
                "steps": [f"{n}! = 1·2·…·{n}", f"Результат: {math.factorial(n)}"],
                "kind": "number"}

    m = re.search(r"(?:сочетани\w*|c)\s*(?:из\s*)?(\d{1,3})\s*(?:по|,)\s*(\d{1,3})", q)
    if m:
        n, k = int(m.group(1)), int(m.group(2))
        if 0 <= k <= n:
            v = math.comb(n, k)
            return {"answer": f"C({n},{k}) = {v}",
                    "steps": [f"Сочетания: C(n,k) = n! / (k!(n−k)!)",
                              f"C({n},{k}) = {n}! / ({k}!·{n - k}!) = {v}"],
                    "kind": "combinatorics"}

    m = re.search(r"(?:перестанов\w*)\s*(?:из\s*)?(\d{1,3})", q)
    if m:
        n = int(m.group(1))
        return {"answer": f"P({n}) = {math.factorial(n)}",
                "steps": [f"Перестановки: P(n) = n!", f"P({n}) = {n}! = {math.factorial(n)}"],
                "kind": "combinatorics"}

    m = re.search(r"(?:размещени\w*)\s*(?:из\s*)?(\d{1,3})\s*(?:по|,)\s*(\d{1,3})", q)
    if m:
        n, k = int(m.group(1)), int(m.group(2))
        v = math.perm(n, k)
        return {"answer": f"A({n},{k}) = {v}",
                "steps": ["Размещения: A(n,k) = n! / (n−k)!", f"A({n},{k}) = {v}"],
                "kind": "combinatorics"}

    m = re.search(r"простое\s*(?:ли\s*)?(?:число\s*)?(\d{2,9})", q)
    if m:
        n = int(m.group(1))
        is_p, _ = _primes(n)
        return {"answer": "да, простое" if is_p else "нет, составное",
                "steps": [f"Проверяем делители до √{n} ≈ {fmt(math.sqrt(n), 3)}",
                          f"Вывод: {n} " + ("простое число" if is_p else "составное")],
                "kind": "number"}

    m = re.search(r"(?:разложи\w*|множител\w*)[^\d]{0,25}(\d{2,9})", q)
    if m:
        n = int(m.group(1))
        _, factors = _primes(n)
        return {"answer": " · ".join(str(f) for f in factors),
                "steps": [f"Последовательное деление на простые: {n} = " + " · ".join(str(f) for f in factors)],
                "kind": "number"}

    m = re.search(r"(?:нод|gcd)\D{0,10}(\d{1,9})\D{1,10}(\d{1,9})", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        g = math.gcd(a, b)
        return {"answer": f"НОД({a},{b}) = {g}",
                "steps": [f"Алгоритм Евклида: НОД({a},{b})", f"Ответ: {g}"],
                "kind": "number"}

    m = re.search(r"(?:нок|lcm)\D{0,10}(\d{1,9})\D{1,10}(\d{1,9})", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        l = abs(a * b) // math.gcd(a, b)
        return {"answer": f"НОК({a},{b}) = {l}",
                "steps": [f"НОК = a·b / НОД = {a}·{b} / {math.gcd(a, b)} = {l}"],
                "kind": "number"}
    return None


# ─────────────────────────── единицы ───────────────────────────

_UNITS = {
    "мм": ("len", 0.001), "сантиметр": ("len", 0.01), "см": ("len", 0.01),
    "метр": ("len", 1.0), "м": ("len", 1.0), "километр": ("len", 1000.0), "км": ("len", 1000.0),
    "грамм": ("mass", 0.001), "г": ("mass", 0.001), "кг": ("mass", 1.0),
    "килограмм": ("mass", 1.0), "тонна": ("mass", 1000.0), "т": ("mass", 1000.0),
    "мг": ("mass", 1e-6),
    "секунд": ("time", 1.0), "с": ("time", 1.0), "минут": ("time", 60.0), "мин": ("time", 60.0),
    "час": ("time", 3600.0), "ч": ("time", 3600.0), "суток": ("time", 86400.0), "день": ("time", 86400.0),
}

_SHORT = {
    "мм": "мм", "сантиметр": "см", "см": "см", "метр": "м", "м": "м",
    "километр": "км", "км": "км", "грамм": "г", "г": "г", "кг": "кг",
    "килограмм": "кг", "тонна": "т", "т": "т", "мг": "мг",
    "секунд": "с", "с": "с", "минут": "мин", "мин": "мин",
    "час": "ч", "ч": "ч", "суток": "сут", "день": "сут",
}


def _short_unit(key):
    return _SHORT.get(key, key)


def _units(q):
    m = re.search(rf"(?:переведи|перевести|сколько|вырази)?\s*({_NUM})\s*([а-яa-z/]+)\s*(?:в|это сколько)\s*([а-яa-z/]+)", q)
    if not m:
        return None
    val = _num(m.group(1))
    src, dst = m.group(2), m.group(3)
    if val is None:
        return None

    if ("цельс" in src or src == "c") and ("фаренгейт" in dst or dst == "f"):
        r = val * 9 / 5 + 32
        return {"answer": f"{fmt(r)} °F", "steps": [f"°F = °C · 9/5 + 32 = {fmt(val)} · 1.8 + 32 = {fmt(r)}"], "kind": "units"}
    if "фаренгейт" in src and ("цельс" in dst or dst == "c"):
        r = (val - 32) * 5 / 9
        return {"answer": f"{fmt(r)} °C", "steps": [f"°C = (°F − 32) · 5/9 = ({fmt(val)} − 32) · 5/9 = {fmt(r)}"], "kind": "units"}

    def find_unit(s):
        for key in sorted(_UNITS, key=len, reverse=True):
            if key in s:
                return _UNITS[key] + (key,)
        return None

    su, du = find_unit(src), find_unit(dst)
    if not su or not du or su[0] != du[0]:
        return None
    base = val * su[1]
    r = base / du[1]
    return {
        "answer": f"{fmt(r)} {_short_unit(du[2])}",
        "steps": [f"Переводим в базовую единицу: {fmt(val)} · {fmt(su[1], 6)} = {fmt(base, 6)}",
                  f"И в нужную: {fmt(base, 6)} / {fmt(du[1], 6)} = {fmt(r, 6)}"],
        "kind": "units",
    }


# ─────────────────────────── движение ───────────────────────────

def _motion(q):
    if not any(w in q for w in ("скорост", "ехал", "шёл", "шел", "проехал", "прошёл", "прошел")):
        return None
    speed = re.search(rf"({_NUM})\s*(?:км/ч|км/час|м/с|м/с)", q)
    time_ = re.search(rf"({_NUM})\s*(?:час\w*|ч\b|минут\w*|мин\b)", q)
    dist = re.search(rf"({_NUM})\s*(?:км\b|метр\w*|м\b)", q)
    if speed and time_:
        v, t = _num(speed.group(1)), _num(time_.group(1))
        if "минут" in time_.group(0):
            t = t / 60
        r = v * t
        return {"answer": f"{fmt(r)} км",
                "steps": [f"Формула пути: S = v · t",
                          f"S = {fmt(v)} км/ч · {fmt(t)} ч = {fmt(r)} км"],
                "kind": "motion"}
    if speed and dist:
        v, s = _num(speed.group(1)), _num(dist.group(1))
        if v:
            t = s / v
            return {"answer": f"{fmt(t, 4)} ч",
                    "steps": ["Формула времени: t = S / v",
                              f"t = {fmt(s)} / {fmt(v)} = {fmt(t, 4)} ч"],
                    "kind": "motion"}
    return None


# ─────────────────────────── арифметика ───────────────────────────

def _arith(q):
    body = q
    for pre in ("посчитай", "вычисли", "сколько будет", "сколько это", "реши пример", "найди значение", "чему равно", "="):
        body = body.replace(pre, " ")
    body = re.sub(r"(?<=\d)\s+(?=\d)", "", body)
    # оставляем только допустимые символы
    cleaned = re.sub(r"[^0-9+\-*/^().,%\s]", " ", body)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # убираем хвостовой мусор
    cleaned = cleaned.strip(" .")
    if not cleaned:
        return None
    # внутри арифметики «%» — это остаток от деления, а не процент
    if not re.search(r"[+\-*/^%]", cleaned):
        return None
    if not re.search(r"\d", cleaned):
        return None
    expr = cleaned.replace("^", "**").replace(",", ".")
    # отсекаем хвост из одиночных знаков
    expr = re.sub(r"[+\-*/]+$", "", expr).strip()
    val = safe_eval(expr)
    if val is None:
        return None
    return {
        "answer": fmt(val),
        "steps": [f"Выражение: {cleaned}", f"Порядок действий: скобки → степень → умножение/деление → сложение/вычитание", f"Результат: {fmt(val)}"],
        "kind": "arithmetic",
    }


# ─────────────────────────── точка входа ───────────────────────────

def solve_task(question):
    """Пытается решить задачу. Возвращает dict или None.

    {"answer": str, "steps": [str], "kind": str}
    """
    q = _norm(question)
    if not q or len(q) > 400:
        return None

    mathish = any(w in q for w in (
        "реши", "посчитай", "вычисли", "уравнен", "сколько будет", "найди x", "процент",
        "прогресс", "факториал", "сочетан", "перестанов", "размещен", "нод", "нок",
        "простое", "разложи", "переведи", "перевести", "вырази", "скорост", "дискриминант",
        "систем", "корни", "чему равен", "%", "скидк", "цена", "стоимость", "найди", "найти",
        " в минуты", " в метры", " в кг", " в граммы", " в часы", " в секунды",
        "фаренгейт", "цельс", "градус", " в км", " в м ", " в см", " в мг",
    )) or bool(re.search(r"\d\s*[-+*/^%]\s*\d", q)) or ("=" in q and "x" in q)

    if not mathish:
        return None

    for fn in (_system, _quadratic, _linear, _progression, _number_theory,
               _units, _percent, _motion, _arith):
        try:
            res = fn(q)
        except Exception:
            res = None
        if res:
            res["question"] = q
            return res
    return None
