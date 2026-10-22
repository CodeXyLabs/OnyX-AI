# -*- coding: utf-8 -*-
"""
OnyX · model_restore.py
=======================
Скачивает разбитую модель Qwen2.5-1.5B из GitHub-релиза и собирает её обратно.

Почему части: GitHub Release ограничивает один файл 2 ГБ, а веса модели
занимают ~2.9 ГБ. Поэтому в релизе v0.5.1-model лежат части
model.safetensors.part01/02/03 плюс конфигурационные файлы.

Использование:
    python tools/model_restore.py            # скачать и собрать модель
    python tools/model_restore.py --check    # только проверить целостность

После сборки модель оказывается в models/Qwen2.5-1.5B-Instruct/,
где её подхватит server.py — и нейросеть отвечает в чате.
"""

import hashlib
import io
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct")

RELEASE = "https://github.com/CodeXyLabs/OnyX-AI/releases/download/v0.5.1-model"
PARTS = ["model.safetensors.part01", "model.safetensors.part02", "model.safetensors.part03"]
SMALL_FILES = [
    "config.json", "generation_config.json", "merges.txt", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "LICENSE.txt",
]
EXPECTED_SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dest, label):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"[1/1] {label}: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "onyx-restore"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r  {pct}% ({done // 1048576} МБ / {total // 1048576} МБ)", end="", flush=True)
    print()
    os.replace(tmp, dest)
    return dest


def main():
    check_only = "--check" in sys.argv
    os.makedirs(MODEL_DIR, exist_ok=True)

    if check_only:
        assembled = os.path.join(MODEL_DIR, "model.safetensors")
        if not os.path.exists(assembled):
            print("model.safetensors не собран. Запусти: python tools/model_restore.py")
            return 1
        print("SHA256:", sha256_file(assembled))
        print("Ожидалось:", EXPECTED_SHA256)
        print("OK — файл целостный." if sha256_file(assembled) == EXPECTED_SHA256 else "ОШИБКА — файл повреждён!")
        return 0 if sha256_file(assembled) == EXPECTED_SHA256 else 1

    # 1. Скачать части
    parts_local = []
    for p in PARTS:
        dest = os.path.join(ROOT, "python_core", "release_model", p)
        download(f"{RELEASE}/{p}", dest, f"скачиваю {p}")
        parts_local.append(dest)

    # 2. Склеить в model.safetensors
    final = os.path.join(MODEL_DIR, "model.safetensors")
    print("Собираю model.safetensors из частей...")
    with open(final, "wb") as out:
        for p in parts_local:
            with open(p, "rb") as f:
                while True:
                    b = f.read(8 * 1024 * 1024)
                    if not b:
                        break
                    out.write(b)
    print(f"Готово: {final} ({os.path.getsize(final) // 1048576} МБ)")

    # 3. Проверка SHA256
    actual = sha256_file(final)
    print("SHA256:", actual)
    if actual != EXPECTED_SHA256:
        print("ВНИМАНИЕ: контрольная сумма не совпала! Перекачайте части.")
        return 1

    # 4. Скачать конфиги и токенизатор
    for sf in SMALL_FILES:
        dest = os.path.join(MODEL_DIR, sf)
        if not os.path.exists(dest):
            download(f"{RELEASE}/{sf}", dest, f"скачиваю {sf}")

    print("\nМодель готова: models/Qwen2.5-1.5B-Instruct/")
    print("Запусти start_server.bat — нейросеть будет отвечать на http://127.0.0.1:8787")
    return 0


if __name__ == "__main__":
    sys.exit(main())