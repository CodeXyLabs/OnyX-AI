# -*- coding: utf-8 -*-
"""
OnyX Python Core · tokenizer.py
================================
Лёгкий BPE-токенизатор, совместимый по концепции с JS-движком OnyX:
  - словарь из спецсимволов, символов и объединений (merges);
  - атомарные служебные токены <pad>/<unk>/<bos>/<eos>/<user>/<ai>;
  - добавление новых символов на лету без потери выученных весов;
  - сериализация в JSON (для чекпоинтов).
"""

import json
import re

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<ai>"]


class BPETokenizer:
    def __init__(self):
        self.chars = []
        self.merges = []                 # list of (left_id, right_id)
        self.char_to_id = {}
        self.next_special = len(SPECIALS)

    # ------------------------------------------------------------ utils
    def _normalize(self, text):
        """Приводит текст к нижнему регистру, оставляя кириллицу/латиницу/цифры/знаки."""
        t = text.lower()
        t = t.replace("\r", " ").replace("\n", " ")
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _split_chars(self, text):
        return list(text)

    # ------------------------------------------------------------ training
    def train(self, corpus, max_merges=2500, min_freq=2, verbose=False):
        """
        Обучает BPE на тексте. Возвращает False, если ничего не обучено.
        """
        text = self._normalize(corpus)
        # начальный набор символов
        for ch in text:
            if ch not in self.char_to_id:
                self.char_to_id[ch] = self.next_special + len(self.chars)
                self.chars.append(ch)

        # слова-части (базово: отдельные символы, но для скорости и качества
        # можно сначала разделить на слова по пробелам)
        words = text.split(" ")
        # представим каждое слово как список id символов
        word_ids = []
        for w in words:
            if not w:
                continue
            ids = [self.char_to_id[c] for c in w]
            word_ids.append(ids)

        vocab = set(self.char_to_id.values())

        for step in range(max_merges):
            # подсчёт пар
            pair_freq = {}
            for ids in word_ids:
                for i in range(len(ids) - 1):
                    key = (ids[i], ids[i + 1])
                    pair_freq[key] = pair_freq.get(key, 0) + 1
            if not pair_freq:
                break
            # выбираем пару с частотой >= min_freq и большей суммарной встречаемостью
            best = None
            best_key = None
            for key, freq in pair_freq.items():
                if freq < min_freq:
                    continue
                if best is None or freq > best:
                    best = freq
                    best_key = key
            if best_key is None:
                break
            # создаём новый токен
            new_id = self.next_special + len(self.chars) + len(self.merges)
            self.merges.append(best_key)
            vocab.add(new_id)
            # заменяем пару по всем словам
            new_word_ids = []
            for ids in word_ids:
                new_ids = []
                i = 0
                n = len(ids)
                while i < n:
                    if i + 1 < n and (ids[i], ids[i + 1]) == best_key:
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                new_word_ids.append(new_ids)
            word_ids = new_word_ids
            if verbose:
                print(f"merge {step + 1}: {best_key} -> {new_id} freq={best}")

        return True

    # ------------------------------------------------------------ encode/decode
    def _id_for_char(self, ch):
        return self.char_to_id.get(ch)

    def encode(self, text, with_markers=False):
        """Кодирует текст в список id."""
        t = self._normalize(text)
        chars = []
        ids = []
        for ch in t:
            cid = self.char_to_id.get(ch)
            ids.append(cid if cid is not None else 1)  # <unk>
        # применяем merges
        id_list = ids
        for (a, b) in self.merges:
            new_list = []
            i = 0
            n = len(id_list)
            while i < n:
                if i + 1 < n and id_list[i] == a and id_list[i + 1] == b:
                    new_list.append(self._find_merge_id(a, b))
                    i += 2
                else:
                    new_list.append(id_list[i])
                    i += 1
            id_list = new_list
        if with_markers:
            return id_list
        return id_list

    def _find_merge_id(self, a, b):
        for idx, pair in enumerate(self.merges):
            if pair == (a, b):
                return self.next_special + len(self.chars) + idx
        return 1

    def decode(self, ids):
        """Декодирует список id в текст."""
        out = []
        for i in ids:
            if i < self.next_special:
                if i == 0 or i == 1:
                    continue
                if i == 2:
                    out.append("[bos]")
                elif i == 3:
                    out.append("[eos]")
                elif i == 4:
                    out.append("[user]")
                elif i == 5:
                    out.append("[ai]")
                else:
                    out.append("?")
            elif i < self.next_special + len(self.chars):
                out.append(self.chars[i - self.next_special])
            else:
                # декодируем merge до символов
                seq = self._merge_to_chars(i)
                out.append("".join(seq))
        return "".join(out)

    def _merge_to_chars(self, merge_id):
        idx = merge_id - (self.next_special + len(self.chars))
        if idx < 0 or idx >= len(self.merges):
            return ["\ufffd"]
        left, right = self.merges[idx]
        seq = []
        seq.extend(self._token_to_chars(left))
        seq.extend(self._token_to_chars(right))
        return seq

    def _token_to_chars(self, tid):
        if tid < self.next_special:
            return []
        if tid < self.next_special + len(self.chars):
            return [self.chars[tid - self.next_special]]
        return self._merge_to_chars(tid)

    # ------------------------------------------------------------ vocab
    @property
    def vocab_size(self):
        return self.next_special + len(self.chars) + len(self.merges)

    def vocab_list(self):
        out = list(SPECIALS)
        out.extend(self.chars)
        out.extend(f"<m{len(SPECIALS)+len(self.chars)+i}>" for i in range(len(self.merges)))
        return out

    # ------------------------------------------------------------ save/load
    def to_json(self):
        return json.dumps({
            "chars": self.chars,
            "merges": self.merges,
            "char_to_id": self.char_to_id,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, data_str):
        obj = json.loads(data_str)
        t = cls()
        t.chars = obj["chars"]
        t.merges = [tuple(p) for p in obj["merges"]]
        t.char_to_id = obj["char_to_id"]
        return t