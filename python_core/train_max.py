# -*- coding: utf-8 -*-
"""
OnyX Python Core · train_max.py
================================
Максимальное обучение нейросети. Запуск:
    python -X utf8 python_core/train_max.py [минуты] [пресет]

Обучает модель на полном корпусе случайными срезами, сохраняет чекпоинт
в python_core/models/onyx_python_brain.json и выводит прогресс.
"""

import math
import os
import sys
import time

# OpenBLAS: однопоточный режим для скорости на маленьких матрицах.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
CORE = os.path.join(ROOT, "python_core")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from engine.brain import OnyxBrain  # noqa: E402


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    preset = sys.argv[2] if len(sys.argv) > 2 else "nano"
    start_lr = float(sys.argv[3]) if len(sys.argv) > 3 else 8e-5
    print(f"OnyX: максимальное обучение ({minutes} мин, preset={preset}, lr={start_lr})")
    t0 = time.time()
    brain = OnyxBrain(workdir=CORE, preset=preset)
    print(f"Инициализация: {time.time()-t0:.1f} с · шагов уже: {brain.model.steps} · словарь: {brain.tokenizer.vocab_size}")
    t0 = time.time()
    end = time.time() + minutes * 60
    steps = brain.model.steps
    n = 0
    losses = []
    T = 64
    B = 8
    seed = steps
    last_report = time.time()
    total_budget = minutes * 60
    start_lr = 8e-5
    min_lr = 1e-5
    last_losses = []
    high_count = 0
    # ранняя остановка и сохранение ЛУЧШЕГО чекпоинта
    best_avg = None
    best_snapshot = None
    while time.time() < end:
        elapsed = time.time() - t0
        frac = min(1.0, elapsed / total_budget)
        lr = min_lr + (start_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * frac))
        if len(last_losses) >= 10 and sum(last_losses[-10:]) / 10 > 12:
            lr = min(lr, 3e-6)   # экстренный предохранитель от расходимости
        try:
            X = brain._make_batch(brain.corpus_ids, B, T, seed=(seed + n) & 0xFFFF)
            loss = brain.model.train_step(X, lr=lr, clip=1.0, weight_decay=1e-4)
            losses.append(float(loss))
            last_losses.append(float(loss))
            n += 1
            steps += 1
            # скользящее среднее из последних 20 шагов; запоминаем лучший снапшот
            window = losses[-20:]
            avg = sum(window) / len(window)
            if best_avg is None or avg < best_avg:
                best_avg = avg
                best_snapshot = brain.model.export_weights(quantize=False)
            if time.time() - last_report >= 10:
                el = time.time() - t0
                print(f"  шаг {steps} · loss {loss:.4f} · lr {lr:.2e} · {n/el:.1f} шаг/с · best {best_avg:.4f} · прошло {el:.0f} с")
                last_report = time.time()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("  ошибка шага:", e)
            break
    # восстанавливаем лучший снипсет весов
    if best_snapshot is not None:
        brain.model.import_weights(best_snapshot)
    brain.save_checkpoint(force=True)
    el = time.time() - t0
    avg = sum(losses) / len(losses) if losses else 0
    print(f"Готово: +{n} шагов за {el:.0f} с ({n/el:.1f} шаг/с)")
    print(f"Лучший loss (скользящее ср.): {best_avg if best_avg is not None else '—'} · финальный: {losses[-1] if losses else '—'}")
    print(f"Чекпоинт сохранён: {brain.checkpoint_path}")


if __name__ == "__main__":
    main()