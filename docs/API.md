# OnyX V0.5 · API базовой модели

Локальный сервер отдаёт базовую модель **Qwen2.5-1.5B-Instruct** (Apache-2.0)
через два интерфейса: простой `/api/llm` и совместимый с OpenAI
`/v1/chat/completions`. Оба работают на `http://127.0.0.1:8787`.

## Проверенные характеристики

[ФАКТ] Замеры на этой машине (CPU, bfloat16, 8 потоков):

| Показатель | Значение |
|---|---|
| Загрузка модели при первом запросе | 10.99 с |
| Скорость генерации | 2.76–3.26 ток/с |
| Ответ на 32 токена | 9.9–11.6 с |
| Память процесса сервера с загруженной моделью | 3.37 ГБ |
| Максимум `max_tokens` | до 10000 (интерфейс) / без жёсткого лимита на сервере (ограничен только контекстом модели) |
| Одновременных генераций | 1 (остальные ждут) |

Модель грузится лениво: первый запрос ждёт загрузку, дальше она в памяти.

## Веб-страница `/llm`

Отдельная страница для разговора с моделью: `http://127.0.0.1:8787/llm`.
Что умеет:

- статус модели (готовится / загружается / ошибка) обновляется каждые 4 секунды;
- потоковый ответ по мере генерации, кнопка «Стоп»;
- настройки `max_tokens` (64/128/256) и температуры, сохраняются в браузере;
- панель «Подключить к любому приложению» с базовым URL и готовыми
  примерами для curl, Python и JavaScript с кнопками копирования.

Ссылка на страницу есть в шапке главного интерфейса (кнопка **LLM**).
[ФАКТ] Проверено вживую: первый ответ 56 с (включая загрузку модели и
инициализацию ядра), тёплый — первый токен 2.3 с.

## Минималистичный дизайн

Включён плоский дизайн без стекла: убраны dotField, vignette, noise; отключены blur и backdrop-filter; все поверхности — плоские, без теней и анимаций. Ссылка на LLM и панель подключения работают в этом режиме.

## 1. Простой интерфейс

```bash
curl -s http://127.0.0.1:8787/api/llm/status
curl -s -X POST http://127.0.0.1:8787/api/llm \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Что такое нейросеть?\",\"max_tokens\":64}"
```

Ответ:

```json
{"ok": true, "text": "…", "model": "onyx-v0.5",
 "tokens": 64, "seconds": 21.4, "tok_per_s": 2.99, "prompt_tokens": 41}
```

## 2. OpenAI-совместимый интерфейс

Любая библиотека, умеющая OpenAI API, подключается сменой `base_url`:
`http://127.0.0.1:8787/v1`, ключ — любой непустой (например `onyx`).

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="onyx")
r = client.chat.completions.create(
    model="onyx-v0.5",
    messages=[{"role": "user", "content": "Объясни, что такое нейросеть"}],
    max_tokens=128,
)
print(r.choices[0].message.content)
```

Поток:

```python
stream = client.chat.completions.create(
    model="onyx-v0.5",
    messages=[{"role": "user", "content": "Перечисли три цвета"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

Без библиотек, чистый HTTP:

```python
import json, urllib.request

req = urllib.request.Request(
    "http://127.0.0.1:8787/v1/chat/completions",
    data=json.dumps({
        "model": "onyx-v0.5",
        "messages": [{"role": "user", "content": "Привет"}],
        "max_tokens": 40,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
```

Поток в браузере:

```js
const res = await fetch('http://127.0.0.1:8787/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'onyx-v0.5',
    messages: [{ role: 'user', content: 'Привет' }],
    stream: true,
  }),
});
const reader = res.body.getReader();
const dec = new TextDecoder();
let buf = '';
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += dec.decode(value, { stream: true });
  for (const line of buf.split('\n\n')) {
    if (!line.startsWith('data: ')) continue;
    const data = line.slice(6).trim();
    if (data === '[DONE]') continue;
    const piece = JSON.parse(data).choices?.[0]?.delta?.content || '';
    if (piece) process.stdout.write(piece);
  }
  buf = buf.slice(buf.lastIndexOf('\n\n') + 2);
}
```

## 3. Готовый клиент

`tools/onyx_client.py` — без внешних зависимостей:

```python
from tools.onyx_client import OnyxClient

c = OnyxClient()
print(c.status())
print(c.ask("Что такое фотосинтез?", max_tokens=96))
for piece in c.stream("Перечисли три цвета"):
    print(piece, end="")
```

Или из командной строки:

```bash
python tools/onyx_client.py "Что такое нейросеть?"
python tools/onyx_client.py --stream "Перечисли три цвета"
```

## Ограничения и безопасность

- Сервер слушает только `127.0.0.1` — снаружи недоступен. Аутентификации нет,
  поэтому не выставляйте порт в интернет без обратного прокси и ключа.
- CORS открыт (`*`), это нужно для страницы, открытой с `file://`.
- Одна генерация за раз: при параллельных запросах второй ждёт первый.
- [ОЦЕНКА] На CPU ответ в 200 токенов занимает около 60–70 секунд. Для
  интерактивных сценариев используйте `stream: true` и `max_tokens` 64–128.

## Лицензии

- `Qwen2.5-1.5B-Instruct` — Apache-2.0, текст лицензии: `models/Qwen2.5-1.5B-Instruct/LICENSE.txt`.
- `multilingual-e5-small` — MIT (заявлена в карточке модели).
- Веса не перепродаются и не выдаются за собственные; модель вызывается как
  внешний компонент, а бренд OnyX относится к обвязке, а не к весам.
