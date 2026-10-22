# OnyX Ai

<div align="center">

**OnyX** — интеллектуальный AI-ассистент с минималистичным красно-чёрным дизайном, самообучаемой нейросетью, генерацией изображений и видео с водяным знаком.

![Версия](https://img.shields.io/badge/версия-4.3%20NOVA-red)
![Лицензия](https://img.shields.io/badge/лицензия-Proprietary-lightgrey)
![Релиз](https://img.shields.io/github/v/release/CodeXyLabs/OnyX-AI)

</div>

---

## 🌟 Возможности

- **Чат с нейросетью**: основной чат отвечает через базовую модель Qwen2.5-1.5B (локально, без интернета)
- **Самообучаемый Python-мозг**: трансформер на numpy (python_core), дообучается на каждом диалоге
- **Режим мышления**: пошаговое рассуждение перед ответом
- **Поиск по сети**: Wikipedia, DuckDuckGo, Stack Overflow — незаметно вплетается в ответ
- **Генерация изображений и видео** с водяным знаком OnyX (перемещается по углам видео)
- **Блоки-цитаты и код**: ответы и код в кавычках показываются в закруглённых прямоугольниках с кнопкой **КОПИРОВАТЬ**
- **Минималистичный красно-чёрный дизайн** в стиле робо-точечной матрицы (Press Start 2P / VT323)

## 📥 Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/CodeXyLabs/OnyX-AI.git
cd OnyX-AI
```

### 2. Установить Python-зависимости

```bash
pip install torch transformers numpy
```

### 3. Скачать базовую модель (Qwen2.5-1.5B-Instruct)

Модель (~2.9 ГБ) не входит в git из-за лимита GitHub на файлы 100 МБ.

**Вариант А — скриптом:**

```bash
python tools/model_fetch.py
```

**Вариант Б — вручную с HuggingFace:**

Скачай [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) и положи содержимое в папку:

```
models/Qwen2.5-1.5B-Instruct/
```

### 4. (Опционально) Чекпойнт самообучаемого мозга

Скачай `onyx_python_brain.json` (9.4 МБ) из [релиза v0.5.0](https://github.com/CodeXyLabs/OnyX-AI/releases/tag/v0.5.0) и положи в:

```
python_core/models/onyx_python_brain.json
```

Это сохранённое состояние трансформера OnyX после обучения.

### 5. Запустить

```bash
start_server.bat
```

Сервер поднимется на `http://127.0.0.1:8787` — открой его в браузере. Отдельный чат с моделью: `http://127.0.0.1:8787/llm`.

## 🛠 Документация API

См. [`docs/API.md`](docs/API.md) — OpenAI-совместимый интерфейс:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="onyx")
r = client.chat.completions.create(model="onyx-v0.5", messages=[{"role":"user","content":"Привет"}])
print(r.choices[0].message.content)
```

## 🦙 Альтернатива: Ollama

Вместо Qwen2.5 можно использовать любую модель через Ollama — см. [`INSTALL_OLLAMA.md`](INSTALL_OLLAMA.md).

## 🎯 Версии OnyX

| Версия | Возможности |
|--------|-------------|
| **2.5** | Базовая база знаний, простой поиск |
| **3.1** | Википедия + DuckDuckGo |
| **4.0 Pro** | Расширенная база знаний |
| **4.3 NOVA** | Максимум: источники + изображения + видео |
| **V0.5** | Python-ядро, генерация видео |

## ⚠️ Безопасность

- Сервер слушает только `127.0.0.1` — наружу не выставлять без прокси и ключа.
- Коды активации (`onyx_activation_codes_*.txt`) в репозиторий не загружаются и не должны публиковаться.

---

© CodeXY Labs ™ 2026 · все права защищены