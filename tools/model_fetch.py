# -*- coding: utf-8 -*-
"""Загрузка базовой модели OnyX с HF: список файлов, докачка, проверка sha256.

Использование:
  python model_fetch.py --list
  python model_fetch.py --download
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

ROOT = r"C:\Users\Admin\Downloads\OnyX Ai WEB — копия"
DEST_ROOT = os.path.join(ROOT, "models")

TARGETS = [
    {"repo": "Qwen/Qwen2.5-1.5B-Instruct", "dir": "Qwen2.5-1.5B-Instruct",
     "license": "apache-2.0", "role": "base-llm",
     "allow": (".safetensors", ".json", ".txt", ".model")},
    {"repo": "intfloat/multilingual-e5-small", "dir": "multilingual-e5-small",
     "license": "mit", "role": "retrieval-encoder",
     "allow": (".safetensors", ".json", ".txt")},
]

SKIP_NAMES = ("README.md", "USE_POLICY.md", "CODE_OF_CONDUCT.md", "tokenizer.json.bak")


def api(repo):
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}",
        headers={"User-Agent": "OnyX-model-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def tree(repo, rev):
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": "OnyX-model-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def select(repo, allow):
    meta = api(repo)
    rev = meta.get("sha") or "main"
    entries = tree(repo, rev)
    files = []
    for e in entries:
        if e.get("type") != "file":
            continue
        path = e["path"]
        name = os.path.basename(path)
        if name in SKIP_NAMES:
            continue
        if not path.endswith(tuple(allow)):
            continue
        # пропускаем альтернативные форматы и лишние веса
        if path.endswith((".bin", ".gguf", ".pth", ".onnx", ".msgpack", ".h5")):
            continue
        lfs = e.get("lfs") or {}
        files.append({
            "path": path,
            "size": lfs.get("size") or e.get("size") or 0,
            "sha256": (lfs.get("oid") or "").replace("sha256:", "") or None,
        })
    return meta, rev, files


def human(n):
    return f"{n / 1e9:.2f} ГБ" if n >= 1e9 else f"{n / 1e6:.1f} МБ"


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(repo, rev, item, dest_dir):
    dest = os.path.join(dest_dir, item["path"].replace("/", os.sep))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    if os.path.exists(dest) and os.path.getsize(dest) == item["size"]:
        if not item["sha256"] or sha256_file(dest) == item["sha256"]:
            print(f"  уже есть: {item['path']}")
            return True
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{item['path']}"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": "OnyX-model-fetch/1.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab") as f:
            if have and r.status != 206:
                f.seek(0)
                f.truncate()
                have = 0
            total = item["size"]
            got = have
            last = time.time()
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if time.time() - last > 5:
                    pct = (got / total * 100) if total else 0
                    print(f"    {item['path']}: {human(got)} / {human(total)} ({pct:.0f}%)")
                    last = time.time()
    except Exception as e:
        print(f"  ОШИБКА {item['path']}: {e}")
        return False
    if item["size"] and os.path.getsize(part) != item["size"]:
        print(f"  РАЗМЕР НЕ СОВПАЛ: {item['path']} "
              f"{os.path.getsize(part)} != {item['size']}")
        return False
    if item["sha256"]:
        got_hash = sha256_file(part)
        if got_hash != item["sha256"]:
            print(f"  SHA256 НЕ СОВПАЛ: {item['path']} ({got_hash[:12]} != {item['sha256'][:12]})")
            return False
    os.replace(part, dest)
    print(f"  готово: {item['path']} ({human(item['size'])})")
    return True


def main():
    mode = "--download" if "--download" in sys.argv else "--list"
    manifest = []
    for t in TARGETS:
        meta, rev, files = select(t["repo"], t["allow"])
        total = sum(f["size"] for f in files)
        print(f"\n{t['repo']}  rev={rev[:12]}  лицензия={t['license']}  "
              f"файлов={len(files)}  объём={human(total)}")
        for f in files:
            print(f"   {f['path']:45s} {human(f['size']):>10s}")
        manifest.append({"repo": t["repo"], "revision": rev, "license": t["license"],
                         "role": t["role"], "files": files, "total_bytes": total})

    total_all = sum(m["total_bytes"] for m in manifest)
    print(f"\nИТОГО к загрузке: {human(total_all)} (лимит 20 ГБ)")
    if mode == "--list":
        return

    ok = True
    for t, m in zip(TARGETS, manifest):
        dest_dir = os.path.join(DEST_ROOT, t["dir"])
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\nскачиваю {t['repo']} -> {dest_dir}")
        for item in m["files"]:
            if not download(m["repo"], m["revision"], item, dest_dir):
                ok = False
        with open(os.path.join(dest_dir, "_onyx_manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"repo": m["repo"], "revision": m["revision"],
                       "license": m["license"], "role": m["role"],
                       "downloaded": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "files": m["files"]}, f, ensure_ascii=False, indent=2)
    print("\nИТОГ:", "все файлы проверены" if ok else "есть ошибки, см. выше")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
