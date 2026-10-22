# -*- coding: utf-8 -*-
"""
OnyX Python Core · transformer.py
==================================
Лёгкий трансформер (decoder-only), написанный с нуля на Python + numpy.
    · token + positional embeddings
    · pre-LayerNorm, multi-head causal self-attention (с маской)
    · GELU MLP, связанная выходная проекция
    · сэмплирование: temperature / top-k / nucleus(top-p) / repetition penalty
    · обучение: cross-entropy + Adam, градиентный клиппинг

Интерфейс рассчитан на то, чтобы модель можно было обучать как в отдельном
Python-процессе (server.py), так и для мелких экспериментов автономно.

Архитектура намеренно небольшая (nano/micro): цель — честная обучаемая языковая
модель на CPU, а не копия ChatGPT. Её «интеллект» усиливается гибридом:
  - факты из knowledge_base.py всегда доступны как контекст;
  - модель учится на реальных диалогах пользователя.
"""

import json
import math
import random
import time

try:
    import numpy as np
except ImportError:
    raise RuntimeError("numpy is required for the OnyX Python transformer")


# -------------------------------------------------------------------------- конфигурация пресетов
PRESETS = {
    "nano":   {"d_model": 80,  "n_layers": 4, "n_heads": 4, "d_ff": 320,  "ctx": 144,  "lr": 1e-3},
    "nano2":  {"d_model": 48,  "n_layers": 3, "n_heads": 3, "d_ff": 192,  "ctx": 128,  "lr": 8e-4},
    "micro":  {"d_model": 128, "n_layers": 6, "n_heads": 4, "d_ff": 512,  "ctx": 192,  "lr": 8e-4},
    "small":  {"d_model": 192, "n_layers": 6, "n_heads": 6, "d_ff": 768,  "ctx": 256,  "lr": 6e-4},
    "medium": {"d_model": 256, "n_layers": 8, "n_heads": 8, "d_ff": 1024, "ctx": 320, "lr": 4e-4},
    "large":  {"d_model": 384, "n_layers": 10, "n_heads": 8, "d_ff": 1536, "ctx": 384, "lr": 3e-4},
    "xl":     {"d_model": 512, "n_layers": 12, "n_heads": 8, "d_ff": 2048, "ctx": 512, "lr": 2.5e-4},
    "xxl":    {"d_model": 768, "n_layers": 12, "n_heads": 12, "d_ff": 3072, "ctx": 512, "lr": 2e-4},
}


def estimate_params(vocab_size, preset_or_cfg):
    """Оценка числа параметров конфигурации без создания модели."""
    cfg = preset_or_cfg
    if isinstance(cfg, str):
        cfg = PRESETS[cfg]
    d, L, ff, ctx = cfg["d_model"], cfg["n_layers"], cfg["d_ff"], cfg["ctx"]
    emb = vocab_size * d + ctx * d
    per_layer = (4 * d * d + 4 * d) + (2 * d * ff + ff) + (4 * d)
    return emb + L * per_layer + 2 * d


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim, dtype=np.float64)
        self.beta = np.zeros(dim, dtype=np.float64)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.norm = (x - self.mean) / self.std
        return self.gamma * self.norm + self.beta

    def backward(self, dout):
        # dout: (B,T,D); стандартная формула LayerNorm backward
        x = self.x
        N = x.shape[-1]
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        invstd = 1.0 / std
        dnorm = dout * self.gamma
        # dx = invstd * (dnorm - mean(dnorm) - norm * mean(dnorm * norm))
        dx = invstd * (dnorm - dnorm.mean(axis=-1, keepdims=True)
                       - self.norm * (dnorm * self.norm).mean(axis=-1, keepdims=True))
        self.dgamma = (dout * self.norm).sum(axis=(0, 1))
        self.dbeta = dout.sum(axis=(0, 1))
        return dx


class CausalSelfAttention:
    def __init__(self, d_model, n_heads, ctx, dtype=np.float64):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ctx = ctx
        self.dtype = dtype
        self.Wq = np.random.randn(d_model, d_model).astype(dtype) * 0.02
        self.Wk = np.random.randn(d_model, d_model).astype(dtype) * 0.02
        self.Wv = np.random.randn(d_model, d_model).astype(dtype) * 0.02
        self.Wo = np.random.randn(d_model, d_model).astype(dtype) * 0.02
        self.bq = np.zeros(d_model, dtype=dtype)
        self.bk = np.zeros(d_model, dtype=dtype)
        self.bv = np.zeros(d_model, dtype=dtype)
        self.bo = np.zeros(d_model, dtype=dtype)
        # causal mask
        mask = np.tril(np.ones((ctx, ctx), dtype=bool))
        self.mask = ~mask

    def forward(self, x):
        B, T, D = x.shape
        q = x @ self.Wq + self.bq
        k = x @ self.Wk + self.bk
        v = x @ self.Wv + self.bv
        H = self.n_heads
        dh = self.d_head
        q = q.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(dh)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        if T <= self.ctx:
            mask = self.mask[:T, :T]
            scores = np.where(mask[None, None, :, :], -1e9, scores)
        attn = softmax(scores, axis=-1)
        out = attn @ v  # (B,H,T,dh)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = out @ self.Wo + self.bo
        # cache for backward
        self.x = x
        self.q, self.k, self.v = q, k, v
        self.attn = attn
        self.scores = scores
        self.scale = scale
        self.B, self.T, self.H, self.dh = B, T, H, dh
        return out

    def backward(self, dout):
        B, T, D = dout.shape
        H, dh = self.H, self.dh
        x_flat = self.x.reshape(-1, D)
        dout_flat = dout.reshape(-1, D)
        dWo = x_flat.T @ dout_flat
        dbo = dout_flat.sum(axis=0)
        dout_out = dout @ self.Wo.T
        dout_out = dout_out.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        # attn backward
        dattn = dout_out @ self.v.transpose(0, 1, 3, 2)
        dv = self.attn.transpose(0, 1, 3, 2) @ dout_out
        # scores backward (softmax jacobian)
        dS = self.attn * (dattn - (dattn * self.attn).sum(axis=-1, keepdims=True))
        dS = dS * self.scale
        # q, k, v
        dq = dS @ self.k
        dk = dS.transpose(0, 1, 3, 2) @ self.q
        dq = dq.transpose(0, 2, 1, 3).reshape(B, T, D)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, T, D)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, T, D)
        dWq = x_flat.T @ dq.reshape(-1, D)
        dWk = x_flat.T @ dk.reshape(-1, D)
        dWv = x_flat.T @ dv.reshape(-1, D)
        dbq = dq.reshape(-1, D).sum(axis=0)
        dbk = dk.reshape(-1, D).sum(axis=0)
        dbv = dv.reshape(-1, D).sum(axis=0)
        dx = dq @ self.Wq.T + dk @ self.Wk.T + dv @ self.Wv.T
        self.dWq, self.dWk, self.dWv, self.dWo = dWq, dWk, dWv, dWo
        self.dbq, self.dbk, self.dbv, self.dbo = dbq, dbk, dbv, dbo
        return dx


class MLP:
    def __init__(self, d_model, d_ff, dtype=np.float64):
        self.Wi = np.random.randn(d_model, d_ff).astype(dtype) * 0.02
        self.bi = np.zeros(d_ff, dtype=dtype)
        self.Wo = np.random.randn(d_ff, d_model).astype(dtype) * 0.02
        self.bo = np.zeros(d_model, dtype=dtype)

    def forward(self, x):
        self.x = x
        self.act = gelu(x @ self.Wi + self.bi)
        return self.act @ self.Wo + self.bo

    def backward(self, dout):
        dact = dout @ self.Wo.T
        self.dWo = self.act.reshape(-1, self.act.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])
        self.dbo = dout.reshape(-1, dout.shape[-1]).sum(axis=0)
        # производная GELU (приближение tanh): точная формула
        x = self.x @ self.Wi + self.bi
        c = math.sqrt(2.0 / math.pi)
        inner = x + 0.044715 * x ** 3
        tanh_v = np.tanh(c * inner)
        der_tanh = 1.0 - tanh_v * tanh_v
        d_gelu = 0.5 * (1.0 + tanh_v) + 0.5 * x * der_tanh * c * (1.0 + 3 * 0.044715 * x * x)
        d_inner = dact * d_gelu
        self.dWi = self.x.reshape(-1, self.x.shape[-1]).T @ d_inner.reshape(-1, d_inner.shape[-1])
        self.dbi = d_inner.reshape(-1, d_inner.shape[-1]).sum(axis=0)
        dx = d_inner @ self.Wi.T
        return dx


class TransformerLM:
    def __init__(self, vocab_size, cfg=None, preset="nano", dtype=np.float64):
        cfg = cfg or PRESETS[preset]
        self.preset = preset
        self.vocab_size = vocab_size
        self.d_model = cfg["d_model"]
        self.n_layers = cfg["n_layers"]
        self.n_heads = cfg["n_heads"]
        self.d_ff = cfg["d_ff"]
        self.ctx = cfg["ctx"]
        self.lr = cfg["lr"]
        self.dtype = dtype
        self.steps = 0
        # embeddings
        self.wte = np.random.randn(vocab_size, self.d_model).astype(dtype) * 0.02
        self.wpe = np.random.randn(self.ctx, self.d_model).astype(dtype) * 0.02
        # layers
        self.ln1s = [LayerNorm(self.d_model) for _ in range(self.n_layers)]
        self.attns = [CausalSelfAttention(self.d_model, self.n_heads, self.ctx, dtype) for _ in range(self.n_layers)]
        self.ln2s = [LayerNorm(self.d_model) for _ in range(self.n_layers)]
        self.mlps = [MLP(self.d_model, self.d_ff, dtype) for _ in range(self.n_layers)]
        self.ln_f = LayerNorm(self.d_model)
        self.opt_m = {}   # adam moments

    # ------------------------------------------------------------ forward
    def forward(self, tokens):
        B, T = tokens.shape
        x = self.wte[tokens] + self.wpe[:T][None, :, :]
        self.x_in = x
        self.cache = []
        for i in range(self.n_layers):
            ln1 = self.ln1s[i].forward(x)
            a = self.attns[i].forward(ln1)
            x = x + a
            ln2 = self.ln2s[i].forward(x)
            m = self.mlps[i].forward(ln2)
            x = x + m
            self.cache.append(x)
        h = self.ln_f.forward(x)
        self.logits_h = h
        logits = h @ self.wte.T  # tied weights
        return logits

    def loss(self, tokens):
        """Возвращает (loss, logits)."""
        B, T = tokens.shape
        logits = self.forward(tokens[:, :-1])
        targets = tokens[:, 1:]
        # cross entropy over vocab
        logits = logits.reshape(-1, self.vocab_size)
        targets = targets.reshape(-1)
        logits_max = logits.max(axis=-1, keepdims=True)
        logsumexp = np.log(np.exp(logits - logits_max).sum(axis=-1, keepdims=True)) + logits_max
        log_probs = logits - logsumexp
        N = targets.size
        ranges = np.arange(N)
        loss = -log_probs[ranges, targets].sum() / N
        # save for backward
        self._log_probs = log_probs
        self._targets = targets
        self._logits_shape = (B, T - 1, self.vocab_size)
        return loss, logits.reshape(B, T - 1, self.vocab_size)

    # ------------------------------------------------------------ backward
    def backward(self):
        """Выполняет обратный проход после loss(). Возвращает словарь градиентов всех параметров."""
        log_probs = self._log_probs
        targets = self._targets
        dlog_probs = np.zeros_like(log_probs)
        ranges = np.arange(targets.size)
        dlog_probs[ranges, targets] = -1.0 / targets.size
        # dlogits = dlog_probs (градиент по logits)
        dlogits = dlog_probs
        B, T, V = self._logits_shape
        dlogits = dlogits.reshape(B, T, V)
        # h
        dh = dlogits @ self.wte
        dwte_out = dlogits.reshape(-1, V).T @ self.logits_h.reshape(-1, self.d_model)
        # ln_f
        dx = self.ln_f.backward(dh)
        # tied: вклад в wte от выходной проекции и от эмбеддингов
        # обратный проход по слоям
        for i in reversed(range(self.n_layers)):
            # x = x + mlp(ln2(x))
            dx_mlp_in = self.mlps[i].backward(dx)
            dx_ln2_in = self.ln2s[i].backward(dx_mlp_in)
            dx = dx + dx_ln2_in
            # x = x + attn(ln1(x))
            dx_attn_in = self.attns[i].backward(dx)
            dx_ln1_in = self.ln1s[i].backward(dx_attn_in)
            dx = dx + dx_ln1_in
        # embedding grads
        B, T = dx.shape[:2]
        x_flat = self.x_in.reshape(-1, self.d_model)
        dx_flat = dx.reshape(-1, self.d_model)
        dwte_in = np.zeros_like(self.wte)
        tokens_flat = self.tokens_flat
        # use bincount-like accumulation: если токены с loose id
        np.add.at(dwte_in, tokens_flat, dx_flat)
        dwte = dwte_in + dwte_out
        dwpe = np.zeros_like(self.wpe)
        # per position
        pos_idx = np.tile(np.arange(T), B)
        np.add.at(dwpe, pos_idx, dx_flat)
        # collect gradients
        grads = {"wte": dwte, "wpe": dwpe}
        grads["ln_f_gamma"] = self.ln_f.dgamma
        grads["ln_f_beta"] = self.ln_f.dbeta
        for i in range(self.n_layers):
            grads[f"ln1_{i}_gamma"] = self.ln1s[i].dgamma
            grads[f"ln1_{i}_beta"] = self.ln1s[i].dbeta
            grads[f"ln2_{i}_gamma"] = self.ln2s[i].dgamma
            grads[f"ln2_{i}_beta"] = self.ln2s[i].dbeta
            grads[f"attn_{i}_Wq"] = self.attns[i].dWq
            grads[f"attn_{i}_Wk"] = self.attns[i].dWk
            grads[f"attn_{i}_Wv"] = self.attns[i].dWv
            grads[f"attn_{i}_Wo"] = self.attns[i].dWo
            grads[f"attn_{i}_bq"] = self.attns[i].dbq
            grads[f"attn_{i}_bk"] = self.attns[i].dbk
            grads[f"attn_{i}_bv"] = self.attns[i].dbv
            grads[f"attn_{i}_bo"] = self.attns[i].dbo
            grads[f"mlp_{i}_Wi"] = self.mlps[i].dWi
            grads[f"mlp_{i}_bi"] = self.mlps[i].dbi
            grads[f"mlp_{i}_Wo"] = self.mlps[i].dWo
            grads[f"mlp_{i}_bo"] = self.mlps[i].dbo
        return grads

    # ------------------------------------------------------------ optimizer
    def apply_grads(self, grads, lr=None, clip=1.0, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        lr = lr or self.lr
        # global norm clip
        total_sq = 0.0
        for k, g in grads.items():
            total_sq += float(np.sum(np.square(g)))
        norm = math.sqrt(total_sq)
        scale = clip / norm if norm > clip else 1.0
        # adam (с weight decay = AdamW)
        params = {}
        params["wte"] = self.wte
        params["wpe"] = self.wpe
        params["ln_f_gamma"] = self.ln_f.gamma
        params["ln_f_beta"] = self.ln_f.beta
        for i in range(self.n_layers):
            params[f"ln1_{i}_gamma"] = self.ln1s[i].gamma
            params[f"ln1_{i}_beta"] = self.ln1s[i].beta
            params[f"ln2_{i}_gamma"] = self.ln2s[i].gamma
            params[f"ln2_{i}_beta"] = self.ln2s[i].beta
            params[f"attn_{i}_Wq"] = self.attns[i].Wq
            params[f"attn_{i}_Wk"] = self.attns[i].Wk
            params[f"attn_{i}_Wv"] = self.attns[i].Wv
            params[f"attn_{i}_Wo"] = self.attns[i].Wo
            params[f"attn_{i}_bq"] = self.attns[i].bq
            params[f"attn_{i}_bk"] = self.attns[i].bk
            params[f"attn_{i}_bv"] = self.attns[i].bv
            params[f"attn_{i}_bo"] = self.attns[i].bo
            params[f"mlp_{i}_Wi"] = self.mlps[i].Wi
            params[f"mlp_{i}_bi"] = self.mlps[i].bi
            params[f"mlp_{i}_Wo"] = self.mlps[i].Wo
            params[f"mlp_{i}_bo"] = self.mlps[i].bo
        self.steps += 1
        t = self.steps
        for k, g in grads.items():
            if k not in params:
                continue
            g = g * scale
            m_key = k + "_m"
            v_key = k + "_v"
            if m_key not in self.opt_m:
                self.opt_m[m_key] = np.zeros_like(g)
                self.opt_m[v_key] = np.zeros_like(g)
            self.opt_m[m_key] = beta1 * self.opt_m[m_key] + (1 - beta1) * g
            self.opt_m[v_key] = beta2 * self.opt_m[v_key] + (1 - beta2) * (g * g)
            m_hat = self.opt_m[m_key] / (1 - beta1 ** t)
            v_hat = self.opt_m[v_key] / (1 - beta2 ** t)
            # AdamW: отдельный weight decay (кроме сдвигов/норм)
            dec = 0.0
            if weight_decay and not k.endswith(("_b", "_beta", "_bo", "_bq", "_bk", "_bv")):
                dec = weight_decay * params[k]
            params[k] -= lr * (m_hat / (np.sqrt(v_hat) + eps) + dec)
        return norm

    # ------------------------------------------------------------ training helpers
    def prep_tokens(self, tokens):
        """Запоминает входные токены для backward."""
        self.tokens_flat = tokens[:, :-1].reshape(-1)

    def train_step(self, tokens, lr=None, clip=1.0, weight_decay=1e-4):
        """Один шаг обучения. tokens: (B,T) np array. Возвращает loss."""
        loss, _ = self.loss(tokens)
        self.tokens_flat = tokens[:, :-1].reshape(-1)
        grads = self.backward()
        self.apply_grads(grads, lr=lr, clip=clip, weight_decay=weight_decay)
        return loss

    # ------------------------------------------------------------ generation
    def generate(self, tokenizer, prompt_ids, max_new=80, temperature=0.8, top_k=40, top_p=0.9,
                 repetition_penalty=1.1, stop_at=None, seed=None):
        """
        Генерирует текст, начиная с prompt_ids. Возвращает список сгенерированных id.
        """
        if seed is not None:
            np.random.seed(seed)
        rng = np.random.default_rng(seed if seed is not None else None)
        ids = list(prompt_ids[-self.ctx:])
        generated = []
        for _ in range(max_new):
            x = np.array(ids[-self.ctx:], dtype=np.int64)[None, :]
            logits = self.forward(x)[0, -1, :]
            # repetition penalty
            if repetition_penalty != 1.0 and len(ids) > 0:
                for t in set(ids[-24:]):
                    if logits[t] > 0:
                        logits[t] /= repetition_penalty
                    else:
                        logits[t] *= repetition_penalty
            # temperature
            logits = logits / max(temperature, 1e-5)
            # top-k
            if top_k and top_k > 0:
                k = min(top_k, len(logits))
                topk_idx = np.argpartition(-logits, k - 1)[:k]
                mask = np.ones_like(logits, dtype=bool)
                mask[topk_idx] = False
                logits[mask] = -1e9
            # top-p
            if top_p and 0 < top_p < 1:
                order = np.argsort(-logits)
                probs = softmax(logits)
                cum = np.cumsum(probs[order])
                keep = order[cum <= top_p]
                if keep.size > 0:
                    mask = np.ones_like(logits, dtype=bool)
                    mask[keep] = False
                    logits[mask] = -1e9
            # sample
            probs = softmax(logits)
            probs = probs / probs.sum()
            try:
                nxt = int(rng.choice(len(probs), p=probs))
            except Exception:
                nxt = int(np.argmax(probs))
            ids.append(nxt)
            generated.append(nxt)
            if stop_at is not None and nxt == stop_at:
                break
            if len(ids) >= self.ctx:
                ids = ids[-self.ctx:]
        return generated

    # ------------------------------------------------------------ export/import
    def _resolve_param(self, name):
        """Возвращает numpy-массив параметра по имени вида 'attn_0_Wq'."""
        if name == "wte":
            return self.wte
        if name == "wpe":
            return self.wpe
        if name == "ln_f_gamma":
            return self.ln_f.gamma
        if name == "ln_f_beta":
            return self.ln_f.beta
        parts = name.split("_")
        kind, idx = parts[0], int(parts[1])
        attr = parts[-1]
        if kind == "ln1":
            obj = self.ln1s[idx]
        elif kind == "ln2":
            obj = self.ln2s[idx]
        elif kind == "attn":
            obj = self.attns[idx]
        elif kind == "mlp":
            obj = self.mlps[idx]
        else:
            return None
        return getattr(obj, attr, None)

    def export_weights(self, quantize=False):
        """Сериализует веса в плоский словарь (для чекпоинта/передачи в браузер)."""
        data = {
            "preset": self.preset,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "ctx": self.ctx,
            "steps": self.steps,
        }
        for name in self.param_names():
            arr = self._resolve_param(name)
            if arr is None:
                continue
            if quantize:
                min_v = float(arr.min())
                max_v = float(arr.max())
                if max_v - min_v < 1e-12:
                    data[name] = {"q": 0, "min": min_v, "max": max_v, "ints": []}
                else:
                    ints = np.round((arr - min_v) / (max_v - min_v) * 255).astype(np.uint8)
                    data[name] = {"q": ints.tolist(), "min": min_v, "max": max_v}
            else:
                data[name] = arr.astype(np.float64).tolist()
        return data

    def import_weights(self, data):
        for name in self.param_names():
            arr = self._resolve_param(name)
            if arr is None or name not in data:
                continue
            val = data[name]
            if isinstance(val, dict) and "q" in val:
                restored = (np.array(val["q"]).astype(np.float64) / 255.0) * (val["max"] - val["min"]) + val["min"]
            else:
                restored = np.array(val, dtype=np.float64)
            if restored.shape == arr.shape:
                arr[...] = restored.astype(arr.dtype)

    def param_names(self):
        names = ["wte", "wpe"]
        for i in range(self.n_layers):
            names += [f"ln1_{i}_gamma", f"ln1_{i}_beta",
                      f"attn_{i}_Wq", f"attn_{i}_Wk", f"attn_{i}_Wv", f"attn_{i}_Wo",
                      f"attn_{i}_bq", f"attn_{i}_bk", f"attn_{i}_bv", f"attn_{i}_bo",
                      f"ln2_{i}_gamma", f"ln2_{i}_beta",
                      f"mlp_{i}_Wi", f"mlp_{i}_bi", f"mlp_{i}_Wo", f"mlp_{i}_bo"]
        names += ["ln_f_gamma", "ln_f_beta"]
        return names

    def num_params(self):
        total = 0
        for name in self.param_names():
            arr = self._resolve_param(name)
            if arr is not None:
                total += arr.size
        return total