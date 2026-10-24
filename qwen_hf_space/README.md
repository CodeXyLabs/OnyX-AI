---
title: Onyx Qwen
emoji: 🧠
colorFrom: red
colorTo: black
sdk: docker
pinned: false
app_port: 7860
---

# OnyX Qwen Space

Бесплатный HuggingFace Space: **Qwen2.5-1.5B-Instruct** как OpenAI-совместимый API.

## Эндпоинты

```
POST /v1/chat/completions   — OpenAI-совместимый (stream тоже поддерживается)
POST /api/chat              — простой формат {question, max_tokens, temperature}
GET  /api/status            — статус модели
```

Ключ не требуется. CORS открыт — работает с любых сайтов, включая GitHub Pages.

## Пример

```bash
curl -s https://ВАШ-АККАУНТ-onyx-qwen.hf.space/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Привет!"}],"max_tokens":128}'
```

## Запуск

Space использует Docker (для стабильности CPU-ресурсов). Модель скачивается с
HuggingFace при первом запуске (~3 ГБ), затем кэшируется.