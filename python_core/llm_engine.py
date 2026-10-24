# -*- coding: utf-8 -*-
"""
OnyX V0.5 · llm_engine.py
=========================
Базовая языковая модель OnyX: Qwen2.5-1.5B-Instruct (Apache-2.0).

Модуль отвечает только за модель:
  · ленивую загрузку в bfloat16 на CPU;
  · сборку промпта по chat-шаблону;
  · генерацию целиком и потоком (SSE-совместимые куски);
  · статус (загружена/грузится/ошибка).

HTTP-обвязка живёт в server.py, поэтому модуль можно импортировать
из любого приложения: CLI, скрипта, другого сервиса.
"""

import os
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct")
MODEL_NAME = "Qwen2.5-1.5B-Instruct"
MODEL_ID = "onyx-v0.5"

_lock = threading.Lock()          # загрузка модели
_gen_lock = threading.Lock()      # одна генерация за раз: CPU не резиновый
_tokenizer = None
_model = None
_state = {"loading": False, "error": None, "loaded": False, "load_seconds": None,
          "device": "cpu"}


def status():
    """Текущее состояние модели (без загрузки)."""
    return {
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "path": MODEL_DIR,
        "exists": os.path.isdir(MODEL_DIR),
        "loaded": _state["loaded"],
        "loading": _state["loading"],
        "error": _state["error"],
        "load_seconds": _state["load_seconds"],
        "device": _state["device"],
        "dtype": "bfloat16",
    }


def ensure_loaded():
    """Загружает модель при первом обращении. Возвращает (tokenizer, model)."""
    global _tokenizer, _model
    if _state["loaded"]:
        return _tokenizer, _model
    with _lock:
        if _state["loaded"]:
            return _tokenizer, _model
        if _state["error"]:
            raise RuntimeError(_state["error"])
        if not os.path.isdir(MODEL_DIR):
            _state["error"] = "модель не найдена: " + MODEL_DIR
            raise RuntimeError(_state["error"])
        _state["loading"] = True
        t0 = time.time()
        try:
            import torch
            # Сервер выставляет OPENBLAS/OMP в 1 поток ради маленького трансформера.
            # Для 1.5B это вредит: возвращаем потоки явно.
            try:
                torch.set_num_threads(max(1, os.cpu_count() or 4))
            except Exception:
                pass
            try:
                torch.set_num_interop_threads(max(1, min(4, os.cpu_count() or 4)))
            except Exception:
                pass
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(MODEL_DIR)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            model.eval()
            _tokenizer, _model = tok, model
            _state["loaded"] = True
            _state["load_seconds"] = round(time.time() - t0, 2)
            return _tokenizer, _model
        except Exception as e:  # noqa: BLE001
            _state["error"] = str(e)
            raise
        finally:
            _state["loading"] = False


def normalize_messages(payload):
    """Принимает строку или список сообщений OpenAI-формата."""
    if isinstance(payload, str):
        return [{"role": "user", "content": payload}]
    if isinstance(payload, list) and payload:
        out = []
        for m in payload:
            if isinstance(m, dict) and m.get("content"):
                role = m.get("role") or "user"
                if role not in ("system", "user", "assistant"):
                    role = "user"
                out.append({"role": role, "content": str(m["content"])})
        return out or [{"role": "user", "content": ""}]
    return [{"role": "user", "content": ""}]


def build_inputs(messages):
    tok, _ = ensure_loaded()
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt")


def _gen_kwargs(max_tokens, temperature):
    kw = {
        # Верхний лимит убран: длина ответа задаётся только max_tokens из запроса.
        "max_new_tokens": max(1, int(max_tokens or 10000)),
        "do_sample": bool(temperature and temperature > 0.01),
        "repetition_penalty": 1.05,
        "pad_token_id": _tokenizer.eos_token_id,
    }
    if kw["do_sample"]:
        kw["temperature"] = float(min(max(temperature, 0.05), 1.5))
        kw["top_p"] = 0.9
    return kw


def generate(payload, max_tokens=128, temperature=0.7):
    """Один ответ целиком. Возвращает словарь с текстом и метриками."""
    import torch

    messages = normalize_messages(payload)
    tok, model = ensure_loaded()
    inputs = build_inputs(messages)
    kw = _gen_kwargs(max_tokens, temperature)
    t0 = time.time()
    with _gen_lock:
        with torch.no_grad():
            out = model.generate(**inputs, **kw)
    dt = time.time() - t0
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(new_ids, skip_special_tokens=True).strip()
    n = int(new_ids.shape[0])
    return {
        "text": text,
        "model": MODEL_ID,
        "model_name": MODEL_NAME,
        "tokens": n,
        "seconds": round(dt, 2),
        "tok_per_s": round(n / dt, 2) if dt > 0 else 0,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
    }


def stream(payload, max_tokens=128, temperature=0.7):
    """Генератор кусочков текста. Захватывает _gen_lock на всё время ответа."""
    from transformers import TextIteratorStreamer

    messages = normalize_messages(payload)
    tok, model = ensure_loaded()
    inputs = build_inputs(messages)
    kw = _gen_kwargs(max_tokens, temperature)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True,
                                    timeout=600)

    def _run():
        import torch
        with _gen_lock:
            with torch.no_grad():
                model.generate(**inputs, streamer=streamer, **kw)
            streamer.end()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    for piece in streamer:
        if piece:
            yield piece
