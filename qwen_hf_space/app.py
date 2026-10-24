# -*- coding: utf-8 -*-
"""
OnyX Qwen Space · app.py (FastAPI)
==================================
Бесплатный HuggingFace Space: Qwen2.5-1.5B-Instruct как OpenAI-совместимый API.

Эндпоинты:
  POST /v1/chat/completions   — OpenAI-совместимый (основной сайт OnyX использует его)
  POST /api/chat              — простой {question, max_tokens, temperature}
  GET  /api/status            — статус модели
  GET  /                       — информация

Space публичный: ключ не требуется, CORS открыт — работает с GitHub Pages.
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("ONYX_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

print("Загружаю модель:", MODEL_ID)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()
print("Модель готова за %.0f с" % (time.time() - t0))


def generate(messages, max_new_tokens=200, temperature=0.7):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    kw = {
        "max_new_tokens": max(1, min(int(max_new_tokens), 512)),
        "repetition_penalty": 1.05,
        "pad_token_id": tok.eos_token_id,
    }
    if temperature and temperature > 0.01:
        kw["do_sample"] = True
        kw["temperature"] = max(0.1, min(float(temperature), 1.5))
        kw["top_p"] = 0.9
    else:
        kw["do_sample"] = False
    with torch.no_grad():
        out = model.generate(**inputs, **kw)
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip()


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

app = FastAPI(title="Onyx Qwen Space")


@app.middleware("http")
async def cors(request, call_next):
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp


@app.options("/{path:path}")
async def preflight(path: str):
    return JSONResponse({"ok": True})


@app.get("/")
async def root():
    return PlainTextResponse("Onyx Qwen Space OK. POST /v1/chat/completions")


@app.get("/api/status")
async def status():
    return {"model": MODEL_ID, "loaded": True, "device": "cpu", "space": True}


@app.post("/api/chat")
async def api_chat(req: Request):
    body = await req.json()
    text = body.get("question") or body.get("text") or ""
    if not text:
        return JSONResponse({"error": "empty question"}, status_code=400)
    max_new = int(body.get("max_tokens", 200))
    temp = float(body.get("temperature", 0.7))
    t0 = time.time()
    answer = generate([{"role": "user", "content": text}], max_new, temp)
    return {"text": answer, "model": MODEL_ID, "seconds": round(time.time() - t0, 2)}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    messages = body.get("messages") or [{"role": "user", "content": body.get("question", "")}]
    if not messages or not any(m.get("role") == "user" for m in messages):
        return JSONResponse({"error": {"message": "no user message"}}, status_code=400)
    max_new = int(body.get("max_tokens", 200))
    temp = float(body.get("temperature", 0.7))
    stream = bool(body.get("stream"))
    t0 = time.time()
    answer = generate(messages, max_new, temp)
    dt = round(time.time() - t0, 2)
    payload = {
        "id": "onyx-space-" + str(int(t0 * 1000)),
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "seconds": dt,
    }
    if stream:
        from fastapi.responses import StreamingResponse

        def gen():
            yield "data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({"choices": [{"delta": {"content": answer}, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse(payload)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)