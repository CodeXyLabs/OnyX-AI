# -*- coding: utf-8 -*-
"""
Onyx Qwen Space · app.py (Gradio)
=================================
Qwen2.5-1.5B-Instruct на бесплатном HuggingFace Space (ZeroGPU/Gradio).

- Сам интерфейс: Gradio-чат (можно открыть в браузере).
- API для сайта: эндпоинты gradio_api, которые использует основной сайт OnyX:
  POST /gradio_api/call/chat  и  POST /gradio_api/queue/... (стандартный Gradio API)
"""

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
    if temperature and temperature > 0.01:
        kw = dict(do_sample=True, temperature=max(0.1, min(float(temperature), 1.5)), top_p=0.9,
                  max_new_tokens=max(1, min(int(max_new_tokens), 512)), repetition_penalty=1.05,
                  pad_token_id=tok.eos_token_id)
    else:
        kw = dict(do_sample=False, max_new_tokens=max(1, min(int(max_new_tokens), 512)),
                  repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    with torch.no_grad():
        out = model.generate(**inputs, **kw)
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip()


import gradio as gr

def chat_fn(message, history):
    if model is None:
        return "Модель не загрузилась."
    history = history or []
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})
    return generate(messages, max_new_tokens=256, temperature=0.7)


with gr.Blocks(title="Onyx Qwen") as demo:
    gr.Markdown("# 🧠 OnyX · Qwen2.5-1.5B\nБесплатный Space. Чат работает, API доступен.")
    gr.ChatInterface(
        fn=chat_fn,
        title="Чат с Qwen",
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch()