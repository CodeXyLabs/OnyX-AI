# -*- coding: utf-8 -*-
"""
OnyX · интеграционные тесты LLM-API.

Требует запущенного сервера:  python python_core/server.py 8787
Запуск:  python tests/test_llm_api.py
Совместимо с pytest (функции test_*).
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8787"
TIMEOUT = 600


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path, payload, timeout=TIMEOUT):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _server_up():
    try:
        get("/api/status")
        return True
    except Exception:
        return False


def test_status():
    st = get("/api/llm/status")
    assert st["model"] == "Qwen2.5-1.5B-Instruct"
    assert "loaded" in st and "loading" in st
    assert st.get("error") in (None, "")


def test_models_endpoint():
    data = get("/v1/models")
    assert data["object"] == "list"
    assert any(m["id"] == "onyx-v0.5" for m in data["data"])


def test_simple_answer():
    t0 = time.time()
    r = post("/api/llm", {"question": "Ответь одним словом: да", "max_tokens": 8})
    dt = time.time() - t0
    assert r.get("ok") is True, r
    assert isinstance(r.get("text"), str) and r["text"].strip()
    assert 0 < r["tokens"] <= 8
    assert dt < 300, f"слишком долго: {dt:.0f} с"


def test_token_cap():
    r = post("/api/llm", {"question": "Скажи: раз", "max_tokens": 999})
    assert r["tokens"] <= 512, f"лимит не соблюдён: {r['tokens']}"


def test_openai_nonstream():
    r = post("/v1/chat/completions", {
        "model": "onyx-v0.5",
        "messages": [{"role": "user", "content": "Скажи: проверка"}],
        "max_tokens": 8,
    })
    assert r["object"] == "chat.completion"
    assert r["choices"][0]["message"]["content"].strip()
    assert r["usage"]["completion_tokens"] > 0


def test_openai_stream():
    payload = {
        "model": "onyx-v0.5",
        "messages": [{"role": "user", "content": "Скажи одно слово: раз"}],
        "max_tokens": 6,
        "stream": True,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    pieces = 0
    done = False
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        assert "text/event-stream" in (r.headers.get("Content-Type") or "")
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                done = True
                break
            obj = json.loads(data)
            delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            if delta:
                pieces += 1
    assert done, "поток не завершился маркером [DONE]"
    assert pieces >= 1, "поток не отдал ни одного куска"


def test_old_chat_still_fast():
    """Регрессия: старый быстрый путь не сломан загрузкой LLM."""
    t0 = time.time()
    r = post("/api/chat", {"question": "что такое нейросеть", "max_tokens": 40})
    dt = time.time() - t0
    assert r.get("text")
    assert dt < 5, f"быстрый путь деградировал: {dt:.2f} с"


def main():
    if not _server_up():
        print("СЛОЖНО: сервер не запущен, тесты LLM-API не выполнены")
        return 2
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f} с)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {e}")
    print(f"\nитого: {len(tests) - failed}/{len(tests)} пройдено")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
