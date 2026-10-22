# -*- coding: utf-8 -*-
"""
OnyX V0.5 · клиент базовой модели
==================================
Подключение к локальному серверу OnyX из любого Python-приложения.
Только стандартная библиотека — ставится копированием файла.

    from tools.onyx_client import OnyxClient
    c = OnyxClient()
    print(c.ask("Что такое нейросеть?", max_tokens=96))
    for piece in c.stream("Перечисли три цвета"):
        print(piece, end="")

CLI:
    python tools/onyx_client.py "вопрос"
    python tools/onyx_client.py --stream "вопрос"
    python tools/onyx_client.py --status
"""

import json
import sys
import time
import urllib.request


class OnyxClient:
    def __init__(self, base="http://127.0.0.1:8787", timeout=600):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def status(self):
        """Состояние модели: загружена, грузится, ошибка."""
        with urllib.request.urlopen(self.base + "/api/llm/status", timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    def ask(self, question, max_tokens=128, temperature=0.7):
        """Один ответ целиком. Возвращает текст."""
        res = self._post("/api/llm", {
            "question": question,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        if not res.get("ok"):
            raise RuntimeError(res.get("error") or "LLM вернула ошибку")
        return res["text"]

    def ask_full(self, question, max_tokens=128, temperature=0.7):
        """Ответ вместе с метриками (токены, секунды, ток/с)."""
        return self._post("/api/llm", {
            "question": question,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

    def stream(self, question, max_tokens=128, temperature=0.7):
        """Генератор кусочков текста по мере генерации."""
        payload = {
            "model": "onyx-v0.5",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if "error" in obj:
                    raise RuntimeError(str(obj["error"]))
                delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                if delta:
                    yield delta


def main():
    args = [a for a in sys.argv[1:]]
    if not args or "--help" in args:
        print(__doc__)
        return
    client = OnyxClient()
    if "--status" in args:
        print(json.dumps(client.status(), ensure_ascii=False, indent=2))
        return
    stream = "--stream" in args
    question = " ".join(a for a in args if not a.startswith("--"))
    t0 = time.time()
    if stream:
        out = []
        for piece in client.stream(question):
            out.append(piece)
            print(piece, end="", flush=True)
        print()
        print(f"\n[поток: {time.time() - t0:.1f} с, {len(''.join(out))} символов]")
    else:
        res = client.ask_full(question)
        print(res["text"])
        print(f"\n[{res['tokens']} токенов за {res['seconds']} с = {res['tok_per_s']} ток/с]")


if __name__ == "__main__":
    main()
