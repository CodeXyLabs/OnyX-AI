# -*- coding: utf-8 -*-
"""
OnyX Python Core · server.py
=============================
Локальный HTTP-сервер для OnyX:
  · обслуживает index.html и все ассеты (работает и с file://, и по http);
  · /api/chat   — ответ нейросети на вопрос (гибрид: трансформер + факты + поиск);
  · /api/image  — генерация изображения с водяным знаком;
  · /api/video  — генерация видео с анимированным водяным знаком;
  · /api/search — тихий поиск по открытым источникам, вплетение в ответ;
  · /api/status — состояние мозга (шаги, loss, параметры);
  · /api/train  — фоновое обучение;
  · /api/learn  — дообучение на диалоге.

Запуск:  python python_core/server.py  [порт]
По умолчанию порт 8787.
"""

import json
import mimetypes
import os
import re
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# OpenBLAS ускоряет маленькие матрицы в однопоточном режиме: без этой
# переменной numpy тратит секунды на управление потоками на каждый matmul.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# чтобы импорты работали и из корня проекта, и из python_core
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
CORE_DIR = os.path.join(ROOT, "python_core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from engine.brain import OnyxBrain  # noqa: E402
import llm_engine  # noqa: E402  (Qwen2.5-1.5B, Apache-2.0)

# ---------------------------------------------------------------- глобальный мозг
_brain = None
_brain_lock = threading.Lock()
# состояние инициализации: сервер отвечает сразу, модель грузится в фоне
_brain_state = {"loading": False, "error": None}


def get_brain():
    global _brain
    with _brain_lock:
        if _brain is None:
            _brain_state["loading"] = True
            _brain_state["error"] = None
            try:
                _brain = OnyxBrain(workdir=CORE_DIR, preset="nano")
            except Exception as e:
                _brain_state["error"] = str(e)
                _brain_state["loading"] = False
                raise
            _brain_state["loading"] = False
        return _brain


def brain_status_stub():
    """Быстрый ответ /api/status, пока модель ещё не загружена."""
    return {
        "ready": False,
        "loading": bool(_brain_state["loading"]) or not _brain_state["error"],
        "error": _brain_state["error"],
        "engine": "onyx-python",
        "storage": "python",
        "model": "onyx-python-nano",
        "preset": "nano",
        "params": 0,
        "steps": 0,
        "history": 0,
    }


def guess_mime(path):
    m = mimetypes.guess_type(path)[0]
    return m or "application/octet-stream"


def send_file(handler, path, cache=True):
    if not os.path.exists(path) or os.path.isdir(path):
        handler.send_error(404)
        return
    with open(path, "rb") as f:
        data = f.read()
    handler.send_response(200)
    handler.send_header("Content-Type", guess_mime(path))
    if cache:
        handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def json_response(handler, obj, status=200):
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_body(handler):
    try:
        length = int(handler.headers.get("Content-Length", 0))
    except ValueError:
        length = 0
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------- поиск по открытым источникам
def search_sources(query, max_results=6):
    """Поиск по множеству бесплатных источников + автообучение на результатах."""
    try:
        import services as svc
        results = svc.multi_search(query, max_results=max_results)
    except Exception:
        results = []
    # обучаем нейросеть на найденных текстах (тихо, без влияния на ответ)
    if results:
        try:
            brain = get_brain()
            texts = [r.get("text", "") for r in results if r.get("text")]
            if texts:
                corpus_piece = " ".join(texts)[:2000]
                brain.learn(query, corpus_piece, score=0.5)
        except Exception:
            pass
    return results


def blend_results(results, question):
    """Превращает найденные источники в естественные предложения ответа."""
    if not results:
        return ""
    sentences = []
    for r in results[:3]:
        t = r["text"]
        parts = re.split(r"(?<=[.!?])\s+", t)
        for p in parts:
            p = p.strip()
            if len(p) < 45:
                continue
            sentences.append(p)
            break
    if not sentences:
        return ""
    openers = ["Если смотреть по существу, ", "Точнее говоря, ", "К этому стоит добавить: ", "Известно, что "]
    opener = openers[(hash(question) % len(openers))]
    out = opener + sentences[0][0].lower() + sentences[0][1:] if sentences[0] else ""
    if not out.endswith((".", "!", "?")):
        out += "."
    if len(sentences) > 1:
        out += " " + "Дополнительно: " + sentences[1][0].lower() + sentences[1][1:]
        if not out.endswith((".", "!", "?")):
            out += "."
    return out


def context_for(results, max_chars=380):
    parts = []
    used = 0
    for r in results:
        piece = (r["title"] + ": " if r.get("title") else "") + r["text"]
        if used + len(piece) > max_chars:
            break
        parts.append(piece)
        used += len(piece)
    return " ".join(parts)


# ---------------------------------------------------------------- handler
class OnyxHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path in ("/api/status", "/api/status/"):
            # Пока модель грузится, отвечаем мгновенно: сайт поймёт, что нужно
            # подождать, и повторит запрос сам.
            if _brain is None:
                json_response(self, brain_status_stub())
                return
            st = get_brain().status()
            st["storage"] = "python"
            st["engine"] = "onyx-python"
            json_response(self, st)
            return
        if path in ("/api/llm/status", "/api/llm/status/"):
            st = llm_engine.status()
            st["engine"] = "onyx-llm"
            json_response(self, st)
            return
        if path in ("/v1/models", "/v1/models/"):
            json_response(self, {"object": "list", "data": [{
                "id": llm_engine.MODEL_ID,
                "object": "model",
                "owned_by": "onyx",
                "name": llm_engine.MODEL_NAME,
            }]})
            return
        if path.startswith("/api/search"):
            q = qs.get("q", [""])[0]
            maxr = int(qs.get("max", ["4"])[0])
            results = search_sources(q, max_results=maxr)
            woven = blend_results(results, q)
            ctx = context_for(results, 380)
            json_response(self, {"results": results, "woven": woven, "context": ctx})
            return
        if path.startswith("/api/images") or path.startswith("/api/generate"):
            # просто возвращаем статус, сами генерации идут через POST
            json_response(self, {"ok": True, "endpoint": "image"})
            return
        if path.startswith("/generated/"):
            fn = os.path.basename(path)
            p = os.path.join(CORE_DIR, "generated", fn)
            send_file(self, p)
            return
        if path.startswith("/assets/"):
            p = os.path.join(ROOT, path.lstrip("/"))
            send_file(self, p)
            return
        if path.startswith("/core/"):
            p = os.path.join(ROOT, path.lstrip("/"))
            send_file(self, p)
            return
        if path.startswith("/web/"):
            p = os.path.join(ROOT, path.lstrip("/"))
            send_file(self, p)
            return
        if path == "/" or path == "/index.html":
            send_file(self, os.path.join(ROOT, "index.html"))
            return
        if path == "/onyx_studio.html" or path == "/studio":
            send_file(self, os.path.join(ROOT, "onyx_studio.html"))
            return
        if path in ("/llm", "/llm.html"):
            send_file(self, os.path.join(ROOT, "llm.html"))
            return
        # отдаём index.html для любых неизвестных путей (SPA fallback)
        send_file(self, os.path.join(ROOT, "index.html"))
        return

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/api/llm", "/api/llm/"):
            body = read_body(self)
            payload = body.get("messages") or body.get("question") or body.get("text") or ""
            try:
                res = llm_engine.generate(
                    payload,
                    max_tokens=body.get("max_tokens", 128),
                    temperature=body.get("temperature", 0.7),
                )
                res["ok"] = True
                json_response(self, res)
            except Exception as e:  # noqa: BLE001
                json_response(self, {"ok": False, "error": str(e)}, 500)
            return

        if path in ("/v1/chat/completions", "/v1/chat/completions/"):
            self._handle_openai_chat()
            return

        if path in ("/api/chat", "/api/chat/"):
            body = read_body(self)
            question = body.get("question") or body.get("message") or body.get("text") or ""
            temp = float(body.get("temperature", 0.8))
            max_tokens = int(body.get("max_tokens", 220))
            intent = str(body.get("intent", "default"))
            use_web = bool(body.get("use_web", True))
            # Модель ещё грузится: просим клиент повторить запрос через пару секунд,
            # вместо того чтобы держать соединение десятки секунд.
            if _brain is None:
                stub = brain_status_stub()
                stub["retry_after"] = 2
                stub["answer"] = None
                json_response(self, stub, 503)
                return
            brain = get_brain()
            # сначала факты (иногда поиск по сети слишком медленный, но мы его делаем тихо)
            answer = brain.answer(question, temperature=temp, max_tokens=max_tokens)
            woven = ""
            # Поиск в сети нужен только тогда, когда своих знаний не хватило:
            # иначе точный ответ (решатель, база знаний) ждёт 20–30 секунд зря.
            if use_web and answer.get("lowConfidence"):
                try:
                    results = search_sources(question, max_results=4)
                    if results:
                        if answer.get("lowConfidence"):
                            woven = blend_results(results, question)
                            answer["text"] = answer.get("text", "") + ("\n\n" + woven if woven else "")
                            answer["woven"] = woven
                            answer["results"] = results[:3]
                        # учитываем найденный материал в обучении дополнительно
                        try:
                            extra = " ".join((r.get("text") or "") for r in results[:2])[:1200]
                            if extra:
                                brain.learn(question + " " + answer.get("text", ""), extra, score=0.4)
                        except Exception:
                            pass
                except Exception:
                    pass
            answer["learned"] = True
            json_response(self, answer)
            return

        if path in ("/api/debug", "/api/debug/"):
            brain = get_brain()
            q = read_body(self).get("question", "")
            from knowledge.knowledge_base import search_facts
            json_response(self, {"facts": search_facts(q, limit=7), "q": q})
            return

        if path in ("/api/image", "/api/image/"):
            body = read_body(self)
            prompt = body.get("prompt") or body.get("text") or ""
            size = int(body.get("size", 768))
            use_online = bool(body.get("online", True))
            style = str(body.get("style") or "") or None
            try:
                from media.media_image import generate_image
                result = generate_image(prompt, size=size, use_online=use_online, style=style)
                json_response(self, result)
            except Exception as e:
                json_response(self, {"error": str(e)}, 500)
            return

        if path in ("/api/video", "/api/video/"):
            body = read_body(self)
            prompt = body.get("prompt") or body.get("text") or ""
            seconds = int(body.get("seconds", 5))
            width = int(body.get("width", 800))
            height = int(body.get("height", 450))
            use_online = bool(body.get("online", True))
            try:
                from media.media_video import generate_video
                result = generate_video(prompt, seconds=seconds, width=width, height=height, use_online=use_online)
                # автообучение на описании сцены
                try:
                    brain = get_brain()
                    brain.learn(prompt, "видео-сцена: " + prompt, score=0.5)
                except Exception:
                    pass
                json_response(self, result)
            except Exception as e:
                json_response(self, {"error": str(e)}, 500)
            return

        if path in ("/api/train", "/api/train/"):
            body = read_body(self)
            minutes = float(body.get("minutes", 1))
            brain = get_brain()
            result = brain.train_minutes(minutes=minutes)
            json_response(self, result)
            return

        if path in ("/api/learn", "/api/learn/"):
            body = read_body(self)
            question = body.get("question") or ""
            answer = body.get("answer") or ""
            brain = get_brain()
            ok = brain.learn(question, answer)
            json_response(self, {"ok": ok})
            return

        if path in ("/api/save", "/api/save/"):
            brain = get_brain()
            p = brain.save_checkpoint(force=True)
            json_response(self, {"ok": True, "path": p})
            return

        json_response(self, {"error": "unknown endpoint"}, 404)

    def _sse(self, obj):
        data = "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
        self.wfile.write(data.encode("utf-8"))
        self.wfile.flush()

    def _handle_openai_chat(self):
        """OpenAI-совместимый /v1/chat/completions: обычный и потоковый ответ."""
        body = read_body(self)
        payload = body.get("messages") or body.get("question") or ""
        max_tokens = body.get("max_tokens") or 128
        temperature = body.get("temperature", 0.7)
        stream = bool(body.get("stream"))
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        model_id = llm_engine.MODEL_ID

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.close_connection = True
            try:
                for piece in llm_engine.stream(payload, max_tokens, temperature):
                    self._sse({"id": cid, "object": "chat.completion.chunk",
                               "created": created, "model": model_id,
                               "choices": [{"index": 0, "delta": {"content": piece},
                                            "finish_reason": None}]})
                self._sse({"id": cid, "object": "chat.completion.chunk",
                           "created": created, "model": model_id,
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as e:  # noqa: BLE001
                try:
                    self._sse({"error": {"message": str(e), "type": "server_error"}})
                except Exception:
                    pass
            return

        try:
            res = llm_engine.generate(payload, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:  # noqa: BLE001
            json_response(self, {"error": {"message": str(e), "type": "server_error"}}, 500)
            return
        json_response(self, {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": res["text"]},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": res["prompt_tokens"],
                      "completion_tokens": res["tokens"],
                      "total_tokens": res["prompt_tokens"] + res["tokens"]},
        })

    def do_PUT(self):
        self.do_POST()


def main():
    port = 8787
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    server = ThreadingHTTPServer(("127.0.0.1", port), OnyxHandler)
    print(f"OnyX Python server: http://127.0.0.1:{port}")
    print("Сервер уже отвечает; модель загружается в фоне…")

    def boot():
        """Грузим модель в фоне, чтобы сайт мог подключиться сразу."""
        try:
            brain = get_brain()
            print(f"Модель готова: {brain.model.num_params()} параметров.")
            try:
                brain.train_minutes(minutes=0.05)
            except Exception:
                pass
        except Exception as e:
            print("[server] не удалось загрузить модель:", e)

    threading.Thread(target=boot, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")


if __name__ == "__main__":
    main()