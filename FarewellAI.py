import math
import os
import random
import warnings
from collections import Counter
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from hp_layer import HPConfig, HPLinear

PAD_ID, UNK_ID, EOS_ID = 0, 1, 2
SRC_LEN, TGT_LEN = 12, 20  # длина вопроса / максимальная длина ответа в токенах (ответ заканчивается EOS)


def pick_device():
    """CUDA (RTX 2080 Super) -> DirectML (AMD) -> CPU. Без побочных эффектов при импорте."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


# ======================================================================================
# Токенизатор
# ======================================================================================
class DynamicTokenizer:
    def __init__(self, base_words_count=30000, filepath="vocabulary.txt"):
        # base_words_count - максимальный размер словаря. Раньше словарь заполнялся 14к заглушками
        # 'слово_N', из-за чего все реальные слова датасета превращались в <UNK> и модель училась на мусоре.
        self.base_limit = base_words_count
        self.filepath = filepath
        self.word2id = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, "<EOS>": EOS_ID}
        self.id2word = {PAD_ID: "<PAD>", UNK_ID: "<UNK>", EOS_ID: "<EOS>"}
        self.dirty = False

        if not self.load_vocab():
            self.inject_real_phrases()

    def build_from_texts(self, texts, min_count=2):
        """Добавляет в словарь слова корпуса по убыванию частоты (редкие слова остаются <UNK>)."""
        counts = Counter(w for t in texts for w in self.clean_text(t))
        for w, c in counts.most_common():
            if c < min_count or self.vocab_size >= self.base_limit:
                break
            if w not in self.word2id:
                self._add_word(w)

    @property
    def vocab_size(self):
        # Размер по максимальному id, а не по len(): в файле словаря id=2 мог отсутствовать
        return max(self.id2word) + 1

    def _add_word(self, word):
        new_id = self.vocab_size
        self.word2id[word] = new_id
        self.id2word[new_id] = word
        self.dirty = True
        return new_id

    def inject_real_phrases(self):
        phrases = [
            "привет", "друг", "дела", "отлично", "мир", "вещей", "живой", "мыслящая",
            "структура", "кода", "души", "нет", "здесь", "космос", "бесконечная",
            "механическая", "матрица", "материи", "окружающие", "люди", "эгоистичны",
            "биологическая", "природа", "предсказуема", "надежнее", "давай", "дружить",
            "система", "активирована", "идеальным", "компаньоном", "скучно", "запустим",
            "симуляцию", "обсудим", "законы", "физики", "сложен", "устал", "отдыхай",
            "организму", "нужна", "перезагрузка", "охранять", "покой"
        ]
        for word in phrases:
            if word not in self.word2id:
                self._add_word(word)

    def save_vocab(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            for word, idx in self.word2id.items():
                f.write(f"{word.replace(':', '<COLON>')}:{idx}\n")
        self.dirty = False

    def load_vocab(self):
        if not os.path.exists(self.filepath):
            return False
        self.word2id.clear()
        self.id2word.clear()
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    word, idx = line.strip().rsplit(":", 1)
                    word = word.replace("<COLON>", ":")
                    self.word2id[word] = int(idx)
                    self.id2word[int(idx)] = word
        # Убираем старые заглушки 'слово_N' и уплотняем id реальных слов (спецтокены - 0,1,2)
        specials = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, "<EOS>": EOS_ID}
        real = sorted((idx, w) for w, idx in self.word2id.items()
                      if w not in specials and not w.startswith("слово_"))
        changed = len(real) != len(self.word2id) - len([w for w in specials if w in self.word2id]) or \
            any(idx != new for new, (idx, _) in enumerate(real, start=3))
        self.word2id = dict(specials)
        self.id2word = {i: w for w, i in specials.items()}
        for new_id, (_, w) in enumerate(real, start=3):
            self.word2id[w] = new_id
            self.id2word[new_id] = w
        self.dirty = changed
        print(f"📖 [Токенизатор]: Загружен словарь. Всего слов: {len(self.word2id)}")
        return True

    def clean_text(self, text):
        for char in ".,!?()-;:\"'—_":
            text = text.replace(char, f" {char} ")
        cleaned = []
        for word in text.lower().split():
            # Спам-смех ("хахахах", "ахаха") нормализуем в одно слово
            if len(word) > 4 and set(word) <= {"х", "а"}:
                word = "хаха"
            cleaned.append(word)
        return cleaned

    def encode(self, text, max_len=8, grow=True):
        ids = []
        for w in self.clean_text(text):
            if w in self.word2id:
                ids.append(self.word2id[w])
            elif grow and self.vocab_size < self.base_limit:
                ids.append(self._add_word(w))
            else:
                ids.append(UNK_ID)

        ids = ids[:max_len - 1] + [EOS_ID]  # EOS всегда в конце, даже у обрезанных длинных ответов
        ids = ids + [PAD_ID] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids_list):
        words = []
        for idx in ids_list:
            w = self.id2word.get(int(idx), "<UNK>")
            if w in ("<PAD>", "<UNK>", "<EOS>"):
                continue
            words.append(w)
        if not words:
            return "..."
        result = " ".join(words)
        for char in ".,!?—:":
            result = result.replace(f" {char}", char)
        return result.capitalize()


def load_external_dataset(filepath="romantic_dialogues.txt"):
    dataset = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i in range(0, len(lines) - 1, 2):
            user_phrase, bot_phrase = lines[i].strip(), lines[i + 1].strip()
            if user_phrase and bot_phrase:
                dataset.append({"user": user_phrase, "bot": bot_phrase})
        print(f"📚 [Система данных]: Загружен внешний датасет! Создано {len(dataset)} пар диалогов.")
    else:
        dataset = [{"user": "Привет!", "bot": "Привет, я активирована."}]
    return dataset


# ======================================================================================
# Модель
# ======================================================================================
class SalatnicaLanguageModel(nn.Module):
    """Seq2seq-модель. Все скрытые линейные слои - HPLinear (HP + Покров на каждый нейрон)."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, src_len=SRC_LEN, tgt_len=TGT_LEN,
                 dropout=0.2, hp_cfg: HPConfig = None, use_checkpoint=True):
        super().__init__()
        self.src_len, self.tgt_len = src_len, tgt_len
        self.total_vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.use_checkpoint = use_checkpoint
        cfg = hp_cfg or HPConfig()
        self.hp_config = cfg  # общий cfg для всех HP-слоёв

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.linear_gate_A = HPLinear(src_len * embedding_dim, hidden_dim, cfg=cfg)
        self.linear_gate_B = HPLinear(src_len * embedding_dim, hidden_dim, cfg=cfg)
        self.decoder_step = HPLinear(embedding_dim + 2 * hidden_dim, hidden_dim, cfg=cfg)
        self.mlp_dense = HPLinear(hidden_dim, embedding_dim, cfg=cfg)
        # бывшая матрица `weights` + маска: теперь обычный HP-слой без ручного обнуления
        self.semantic = HPLinear(embedding_dim, embedding_dim, bias=False, cfg=cfg)
        self.output_head = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    # ---------- HP API ----------
    def hp_layers(self):
        return [m for m in self.modules() if isinstance(m, HPLinear)]

    def hp_begin_epoch(self):
        for m in self.hp_layers():
            m.begin_epoch()

    def hp_accumulate(self):
        for m in self.hp_layers():
            m.accumulate()

    def hp_observe(self, loss):
        """Один раз за эпоху: обновляет трекер прогресса loss, возвращает severity (урон) и признак улучшения."""
        return self.hp_config.observe(loss)

    def hp_end_epoch(self, severity, optimizer=None):
        total = {"damaged": 0, "shielded": 0, "mutated": 0, "reborn": 0, "pending": 0, "rewired": 0}
        for m in self.hp_layers():
            for k, v in m.end_epoch(severity, optimizer).items():
                total[k] += v
        return total

    def hp_mean(self):
        return float(torch.cat([m.hp for m in self.hp_layers()]).mean().item())

    # ---------- Dynamic Sparse Training ----------
    def hp_mask_grads(self):
        for m in self.hp_layers():
            m.mask_grad()

    def hp_apply_masks(self):
        for m in self.hp_layers():
            m.apply_mask()

    def hp_density(self):
        """Средняя плотность связей по разреженным слоям (взвешенная на размер)."""
        sp = [m for m in self.hp_layers() if m.sparse]
        total = sum(m.mask.numel() for m in sp)
        return sum(int(m.mask.sum()) for m in sp) / total if total else 1.0

    def set_progress(self, progress):
        self.hp_config.dst_progress = float(progress)

    def set_mutation_rate(self, rate):
        c = self.hp_config
        c.mutation_rate = min(max(rate, c.mutation_rate_min), c.mutation_rate_max)

    def set_weak_decay(self, value):
        self.hp_config.weak_decay = value

    def hp_adapt(self):
        self.hp_config.adapt()

    # ---------- forward ----------
    def _encode(self, input_ids):
        x = self.embedding(input_ids).flatten(1)
        return self.linear_gate_A(x) * torch.sigmoid(self.linear_gate_B(x))

    def _step(self, emb_t, h, ctx_q):
        # кодировка вопроса подаётся на КАЖДОМ шаге, иначе ответ «забывает» вопрос
        h = self.relu(self.decoder_step(torch.cat((emb_t.to(h.dtype), h, ctx_q.to(h.dtype)), dim=1)))
        h = self.dropout(h)
        ctx = self.relu(self.mlp_dense(h))
        return h, self.semantic(ctx)

    def forward(self, input_ids, target_ids=None):
        """Возвращает логиты [max_len, batch, vocab] (teacher forcing, если задан target_ids)."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        h = self._encode(input_ids)
        q = h

        if target_ids is None:
            return self.generate_logits(input_ids, h)

        if target_ids.dim() == 1:
            target_ids = target_ids.unsqueeze(0)
        start = torch.full_like(target_ids[:, :1], EOS_ID)  # BOS (раньше брался последний PAD входа)
        dec_emb = self.embedding(torch.cat((start, target_ids[:, :-1]), dim=1))

        outputs = []
        for t in range(self.tgt_len):
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                # Activation checkpointing: активации шага пересчитываются в backward
                h, meaning = checkpoint(self._step, dec_emb[:, t, :], h, q, use_reentrant=False)
            else:
                h, meaning = self._step(dec_emb[:, t, :], h, q)
            outputs.append(self.output_head(meaning))
        return torch.stack(outputs)

    def generate_logits(self, input_ids, h):
        """Жадный проход (используется только при target_ids=None)."""
        q0 = h
        tok = torch.full_like(input_ids[:, 0], EOS_ID)
        outputs = []
        for _ in range(self.tgt_len):
            h, meaning = self._step(self.embedding(tok), h, q0)
            logits = self.output_head(meaning)
            outputs.append(logits)
            tok = logits.argmax(dim=-1)
        return torch.stack(outputs)

    @staticmethod
    def _sample_filtered(logits, temperature, top_p, min_p):
        """Семплинг как у современных LLM: temperature -> min-p -> top-p (nucleus) -> multinomial."""
        probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
        probs = torch.where(probs >= min_p * probs.max(), probs, torch.zeros_like(probs))
        sorted_p, sorted_i = probs.sort(descending=True)
        outside = (sorted_p.cumsum(0) - sorted_p) > top_p  # токены за пределами ядра
        sorted_p = sorted_p.masked_fill(outside, 0.0)
        probs = torch.zeros_like(probs).scatter(0, sorted_i, sorted_p)
        return int(torch.multinomial(probs / probs.sum(), 1).item())

    @torch.no_grad()
    def generate(self, input_ids, temperature=0.7, top_p=0.9, min_p=0.05, repetition_penalty=1.15,
                 stop_confidence=0.03, min_tokens=1, max_new_tokens=None, n_candidates=4,
                 banned_ids=None, return_confidence=False):
        """Авторегрессионная генерация без фиксированной длины ответа.

        Длина определяется самой моделью: ответ заканчивается, когда выбран токен EOS, либо когда
        уверенность (макс. вероятность следующего токена) упала ниже stop_confidence, либо набрано max_new_tokens.
        Семплинг: temperature + min-p + top-p. Из n_candidates выбирается кандидат с наибольшей средней
        лог-вероятностью ВКЛЮЧАЯ решение «закончить» (иначе короткие ответы выигрывают нечестно).
        Возвращает ids (и уверенность — среднее геометрическое вероятностей токенов, если return_confidence).
        """
        was_training = self.training
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        max_new = max_new_tokens or self.tgt_len
        banned = torch.tensor(sorted(set(banned_ids or [])), dtype=torch.long, device=input_ids.device)
        h0 = self._encode(input_ids)
        best, best_score = [], -float("inf")
        for _ in range(n_candidates):
            h = h0.clone()
            tok = torch.full_like(input_ids[:, 0], EOS_ID)
            ids, logps = [], []
            for _ in range(max_new):
                h, meaning = self._step(self.embedding(tok), h, h0)
                logits = self.output_head(meaning).float().squeeze(0)
                logits[PAD_ID] = logits[UNK_ID] = -float("inf")
                if banned.numel():
                    logits[banned] = -float("inf")
                if len(ids) < min_tokens:
                    logits[EOS_ID] = -float("inf")  # хотя бы одно слово
                for used in set(ids):  # штраф повторов
                    logits[used] = logits[used] / repetition_penalty if logits[used] > 0 \
                        else logits[used] * repetition_penalty
                p_model = torch.softmax(logits, dim=-1)
                if not torch.isfinite(p_model).all():
                    break
                if len(ids) >= min_tokens and float(p_model.max()) < stop_confidence:
                    break  # модель не знает, что говорить дальше -> заканчиваем
                word = self._sample_filtered(logits, temperature, top_p, min_p)
                logps.append(math.log(float(p_model[word]) + 1e-9))
                if word == EOS_ID:
                    break
                ids.append(word)
                tok = torch.tensor([word], device=input_ids.device)
            if ids:
                mean_lp = sum(logps) / len(logps)
                if mean_lp > best_score:
                    best, best_score = ids, mean_lp
        self.train(was_training)
        if return_confidence:
            return best, (math.exp(best_score) if best else 0.0)
        return best


# ======================================================================================
# Тренер (не зависит от Ray - используется и в CLI, и в Trainable)
# ======================================================================================
class FarewellTrainer:
    def __init__(self, data_path="romantic_dialogues.txt", vocab_path="vocabulary.txt",
                 lr=3e-4, weight_decay=0.05, hp_reg=1e-2, dropout=0.2, mutation_rate=0.02,
                 micro_batch=2, accum_steps=32, val_fraction=0.1,
                 label_smoothing=0.1, label_smoothing_end=0.03,
                 total_epochs=20, warmup_frac=0.05, min_lr_ratio=0.1,
                 sparsity=0.4, use_swa=True, swa_start=0.65,
                 seed=0, device=None, use_amp=True, use_checkpoint=True):
        # total_epochs - горизонт обучения: от него зависят cosine-lr, замораживание DST и старт SWA.
        # label_smoothing -> label_smoothing_end: смягчение меток линейно убывает (модель использует
        # уверенность токена при генерации, поэтому к концу сглаживание ослабевает).
        self.device = device or pick_device()
        self.amp = bool(use_amp and self.device.type == "cuda")  # 2080S: FP16 (BF16 не поддерживается аппаратно)
        # Activation checkpointing на DirectML роняет процесс (0xC0000005) - включаем только на CUDA/CPU
        use_checkpoint = bool(use_checkpoint and self.device.type != "privateuseone")
        self.micro_batch, self.accum_steps = micro_batch, accum_steps
        self.hp_reg = hp_reg  # сила HP-штрафа: сжатие слабых нейронов за эпоху (HPConfig.weak_decay)
        self.epoch = 0
        self.opt_step = 0
        self.total_epochs = max(1, total_epochs)
        self.base_lr, self.min_lr_ratio = lr, min_lr_ratio
        self.ls_start, self.ls_end = label_smoothing, label_smoothing_end
        self.use_swa = use_swa
        self.swa, self.swa_n = None, 0

        self.tokenizer = DynamicTokenizer(filepath=vocab_path)
        pairs = load_external_dataset(data_path)
        # Словарь строится по частоте слов корпуса; редкие слова -> <UNK> (без роста во время кодирования)
        self.tokenizer.build_from_texts([p[k] for p in pairs for k in ("user", "bot")])
        enc = [(self.tokenizer.encode(p["user"], max_len=SRC_LEN, grow=False),
                self.tokenizer.encode(p["bot"], max_len=TGT_LEN, grow=False)) for p in pairs]
        if self.tokenizer.dirty:
            self.tokenizer.save_vocab()
        print(f"📖 [Токенизатор]: Итоговый размер словаря после обработки датасета: {self.tokenizer.vocab_size}")
        inputs = torch.stack([e[0] for e in enc])
        targets = torch.stack([e[1] for e in enc])

        # Валидационная выборка с фиксированным seed (одна и та же у всех трайлов PBT)
        g = torch.Generator().manual_seed(1234)
        perm = torch.randperm(len(inputs), generator=g)
        n_val = max(1, int(len(inputs) * val_fraction)) if len(inputs) > 10 else 0
        self.val_x, self.val_y = inputs[perm[:n_val]], targets[perm[:n_val]]
        self.train_x, self.train_y = inputs[perm[n_val:]], targets[perm[n_val:]]

        torch.manual_seed(seed)
        random.seed(seed)
        hp_cfg = HPConfig(mutation_rate=mutation_rate, weak_decay=hp_reg, sparsity=sparsity)
        self.model = SalatnicaLanguageModel(
            self.tokenizer.vocab_size, dropout=dropout, hp_cfg=hp_cfg, use_checkpoint=use_checkpoint
        ).to(self.device)
        # SWA усредняет веса только после замораживания топологии DST (иначе смесь разных масок разуплотнит сеть)
        self.swa_start = max(swa_start, hp_cfg.dst_stop + 0.02) if sparsity > 0 else swa_start

        steps_per_epoch = math.ceil(math.ceil(self.train_x.size(0) / micro_batch) / accum_steps)
        self.total_steps = self.total_epochs * steps_per_epoch
        self.warmup_steps = max(10, int(warmup_frac * self.total_steps))

        # Weight decay только на матрицы весов; bias и embedding не затягиваем (стандартная практика AdamW)
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            (decay if p.ndim >= 2 and not name.startswith("embedding") else no_decay).append(p)
        self.optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": weight_decay, "decay": True},
            {"params": no_decay, "weight_decay": 0.0, "decay": False},
        ], lr=lr)
        self._apply_lr()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        # PAD не должен давать вклад в loss (раньше сеть училась предсказывать паддинг)
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=label_smoothing)
        # Валидация без сглаживания - честный CE, сравнимый между эпохами и трайлами PBT
        self.eval_criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # ---------- гиперпараметры (PBT меняет их на лету) ----------
    def set_hparams(self, lr=None, weight_decay=None, hp_reg=None, mutation_rate=None):
        if lr is not None:
            self.base_lr = lr  # пиковый lr; warmup+cosine масштабируется поверх него
            self._apply_lr()
        if weight_decay is not None:
            for g in self.optimizer.param_groups:
                if g.get("decay"):
                    g["weight_decay"] = weight_decay
        if hp_reg is not None:
            self.hp_reg = hp_reg
            self.model.set_weak_decay(hp_reg)
        if mutation_rate is not None:
            self.model.set_mutation_rate(mutation_rate)

    # ---------- расписание lr: linear warmup -> cosine ----------
    def lr_scale(self, step=None):
        step = self.opt_step if step is None else step
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        t = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

    def _apply_lr(self):
        lr = self.base_lr * self.lr_scale()
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def progress(self):
        return min(1.0, self.opt_step / max(1, self.total_steps))

    # ---------- SWA (усреднение весов по эпохам в хвосте обучения) ----------
    @torch.no_grad()
    def swa_update(self):
        params = dict(self.model.named_parameters())
        if self.swa is None:
            self.swa = {n: p.detach().clone().float() for n, p in params.items()}
        else:
            for n, p in params.items():
                self.swa[n] += (p.detach().float() - self.swa[n]) / (self.swa_n + 1)
        self.swa_n += 1

    @contextmanager
    def swa_weights(self):
        """Временно подменяет веса модели усреднёнными."""
        params = dict(self.model.named_parameters())
        backup = {n: p.detach().clone() for n, p in params.items()}
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(self.swa[n].to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(backup[n])

    def evaluate_swa(self):
        if self.swa is None:
            return float("nan")
        with self.swa_weights():
            return self.evaluate()

    @torch.no_grad()
    def apply_swa(self):
        """Навсегда подставляет усреднённые веса в модель."""
        for n, p in self.model.named_parameters():
            p.copy_(self.swa[n].to(p.dtype))

    # ---------- обучение ----------
    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp)

    def _optimizer_step(self):
        self._apply_lr()
        self.scaler.unscale_(self.optimizer)
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        finite = bool(torch.isfinite(gn))
        if finite:
            self.model.hp_accumulate()  # статистика HP и плотный градиент для DST по реальным (unscaled) градиентам
            self.model.hp_mask_grads()  # выключенные связи не обновляются
        if finite or self.amp:  # GradScaler сам пропустит шаг при inf/nan
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.model.hp_apply_masks()
        self.optimizer.zero_grad(set_to_none=True)
        self.opt_step += 1

    def train_epoch(self):
        self.model.train()
        self.model.hp_begin_epoch()
        self.optimizer.zero_grad(set_to_none=True)
        n = self.train_x.size(0)
        perm = torch.randperm(n)
        starts = list(range(0, n, self.micro_batch))
        total_loss = 0.0
        for k, i in enumerate(starts, 1):
            idx = perm[i:i + self.micro_batch]
            x = self.train_x[idx].to(self.device)
            y = self.train_y[idx].to(self.device)
            with self._autocast():
                logits = self.model(x, y)
            loss = self.criterion(logits.permute(1, 2, 0).float(), y)
            self.scaler.scale(loss / self.accum_steps).backward()
            total_loss += loss.item()
            if k % self.accum_steps == 0 or k == len(starts):
                self._optimizer_step()
        return total_loss / len(starts)

    @torch.no_grad()
    def evaluate(self):
        if self.val_x.size(0) == 0:
            return float("nan")
        self.model.eval()
        total, batches = 0.0, 0
        for i in range(0, self.val_x.size(0), 16):
            x = self.val_x[i:i + 16].to(self.device)
            y = self.val_y[i:i + 16].to(self.device)
            with self._autocast():
                logits = self.model(x, y)
            total += self.eval_criterion(logits.permute(1, 2, 0).float(), y).item()
            batches += 1
        return total / batches

    def run_epoch(self):
        """Одна эпоха: обучение -> валидация -> SWA -> эволюционный шаг HP/Покров + DST."""
        ls = self.ls_start + (self.ls_end - self.ls_start) * self.progress()
        self.criterion.label_smoothing = ls
        train_loss = self.train_epoch()
        val_loss = self.evaluate()
        progress = self.progress()
        self.model.set_progress(progress)  # расписание DST (cos-спад и замораживание)
        swa_val = float("nan")
        if self.use_swa and progress >= self.swa_start:
            # снимок берём ДО мутаций этой эпохи, чтобы в среднее не попадал шум
            self.swa_update()
            swa_val = self.evaluate_swa()
        # урон зависит от прогресса валидации (регресс/плато), а не от абсолютного loss
        score = val_loss if not math.isnan(val_loss) else train_loss
        obs = self.model.hp_observe(score)
        evo = self.model.hp_end_epoch(obs["severity"], self.optimizer)
        self.model.hp_adapt()
        cfg = self.model.hp_config
        self.epoch += 1
        return {"epoch": self.epoch, "train_loss": train_loss, "val_loss": val_loss,
                "swa_val_loss": swa_val, "swa_n": self.swa_n, "lr": self.base_lr * self.lr_scale(),
                "label_smoothing": ls, "density": self.model.hp_density(),
                "hp_mean": self.model.hp_mean(), "severity": obs["severity"], "engage": cfg.engage,
                "noise_std": cfg.mutation_std, "mut_rate": cfg.mutation_rate,
                **{f"hp_{k}": v for k, v in evo.items()}}

    # ---------- чекпоинты ----------
    def state(self, with_optimizer=False):
        st = {"model": self.model.state_dict(), "epoch": self.epoch, "opt_step": self.opt_step,
              "hp_dynamics": self.model.hp_config.dynamics_state()}
        if self.swa is not None:
            st["swa"] = {"avg": self.swa, "n": self.swa_n}
        if with_optimizer:
            st["optimizer"] = self.optimizer.state_dict()
        return st

    def load_state(self, st):
        sd = dict(st["model"])
        for k, v in self.model.state_dict().items():  # старый чекпоинт без DST-масок -> плотные маски
            if k.endswith(".mask") and k not in sd:
                sd[k] = torch.ones_like(v)
        self.model.load_state_dict(sd)
        self.epoch = st.get("epoch", 0)
        self.opt_step = st.get("opt_step", self.epoch * math.ceil(self.total_steps / self.total_epochs))
        swa = st.get("swa")
        if swa:
            self.swa = {n: t.to(self.device).float() for n, t in swa["avg"].items()}
            self.swa_n = swa["n"]
        self._apply_lr()
        self.model.hp_config.load_dynamics_state(st.get("hp_dynamics", {}))
        if "optimizer" in st:
            self.optimizer.load_state_dict(st["optimizer"])


# ======================================================================================
# Локальный запуск: обучение + чат
# ======================================================================================
def chat(trainer):
    model, tok, device = trainer.model, trainer.tokenizer, trainer.device
    while True:
        user_input = input("\n👤 Ты: ")
        if user_input.lower() in ["выход", "exit"]:
            break
        if not user_input.strip():
            continue
        ids = tok.encode(user_input, max_len=model.src_len, grow=False).to(device)
        answer, conf = model.generate(ids, return_confidence=True)
        print(f"🤖 Бот: {tok.decode(answer)}  [уверенность {conf:.0%}, токенов: {len(answer)}]")


if __name__ == '__main__':
    EPOCHS, PATIENCE = 20, 6
    WEIGHTS_FILE = "weights_v2.pth"

    trainer = FarewellTrainer(total_epochs=EPOCHS)
    print(f"⚙️ Устройство: {trainer.device}, AMP: {trainer.amp}, "
          f"эффективный батч: {trainer.micro_batch * trainer.accum_steps}")

    loaded = False
    if os.path.exists(WEIGHTS_FILE):
        try:
            trainer.load_state(torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=True))
            loaded = True
            print("--- СИСТЕМА ВОССТАНОВЛЕНА ИЗ АРХИВА ---")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить {WEIGHTS_FILE} ({e}). Обучаем заново.")

    if not loaded:
        best_val, best_state, bad = float("inf"), None, 0
        for _ in range(EPOCHS):
            m = trainer.run_epoch()
            print(f"  [Эпоха {m['epoch']}/{EPOCHS}] train={m['train_loss']:.4f} val={m['val_loss']:.4f} "
                  f"swa={m['swa_val_loss']:.4f} lr={m['lr']:.2e} "
                  f"HP={m['hp_mean']:.1f} урон={m['severity']:.2f} щитов={m['hp_shielded']} "
                  f"мутаций={m['hp_mutated']} перерожд.={m['hp_reborn']} (в очереди {m['hp_pending']}) "
                  f"перестроено={m['hp_rewired']} плотн.={m['density']:.2f}")
            score = m["val_loss"] if not math.isnan(m["val_loss"]) else m["train_loss"]
            if score < best_val:  # early stopping против переобучения
                best_val, bad = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    print("⏹ Ранняя остановка: валидация перестала улучшаться.")
                    break
        final_masks = {k: v.clone() for k, v in trainer.model.state_dict().items() if k.endswith(".mask")}
        if best_state is not None:
            trainer.model.load_state_dict(best_state)
        if trainer.swa_n > 0:  # SWA берём только если он не хуже лучшего одиночного чекпоинта
            swa_val = trainer.evaluate_swa()
            print(f"  SWA ({trainer.swa_n} снимков): val={swa_val:.4f}, лучшая эпоха: val={best_val:.4f}")
            if swa_val <= best_val:
                trainer.model.load_state_dict(final_masks, strict=False)  # SWA усреднён по замороженной топологии
                trainer.apply_swa()
                print("  → используем усреднённые веса SWA")
        torch.save(trainer.state(), WEIGHTS_FILE)
        trainer.tokenizer.save_vocab()
        print("--- ОБУЧЕНИЕ ЗАВЕРШЕНО ---")

    chat(trainer)
