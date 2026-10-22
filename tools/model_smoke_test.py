# -*- coding: utf-8 -*-
"""Проверка, что скачанные модели реально загружаются и работают на этой машине."""
import ctypes
import ctypes.wintypes as wt
import os
import time

import torch

ROOT = r"C:\Users\Admin\Downloads\OnyX Ai WEB — копия"
LLM = os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct")
E5 = os.path.join(ROOT, "models", "multilingual-e5-small")


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def rss_gb():
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize / 1e9


print("RSS до загрузки: %.2f ГБ" % rss_gb())

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

t0 = time.time()
tok = AutoTokenizer.from_pretrained(LLM)
try:
    model = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16, low_cpu_mem_usage=True)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
load_s = time.time() - t0
print("LLM загружена за %.1f с, RSS %.2f ГБ, параметров %s" % (load_s, rss_gb(), f"{model.num_parameters():,}"))

msgs = [{"role": "user", "content": "Назови три языка программирования, кратко."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt")
t1 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                         pad_token_id=tok.eos_token_id)
gen_s = time.time() - t1
new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("Генерация: %d токенов за %.1f с = %.2f ток/с" % (new_tokens, gen_s, new_tokens / max(gen_s, 1e-9)))
print("Ответ:", answer.strip().replace("\n", " ")[:200])
print("RSS после генерации: %.2f ГБ" % rss_gb())

del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

t2 = time.time()
etok = AutoTokenizer.from_pretrained(E5)
emodel = AutoModel.from_pretrained(E5)
emodel.eval()
print("Энкодер e5 загружен за %.1f с, RSS %.2f ГБ" % (time.time() - t2, rss_gb()))


def embed(s):
    b = etok("query: " + s, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        o = emodel(**b)
    v = o.last_hidden_state.mean(dim=1)
    return torch.nn.functional.normalize(v, dim=-1)


pairs = [("нейросеть", "искусственный интеллект"), ("нейросеть", "фотосинтез")]
for a, b in pairs:
    ca = embed(a)
    cb = embed(b)
    print("cos(«%s», «%s») = %.3f" % (a, b, float((ca * cb).sum())))
print("Итоговый RSS: %.2f ГБ" % rss_gb())
