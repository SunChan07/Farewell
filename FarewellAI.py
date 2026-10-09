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
SRC_LEN, TGT_LEN = 128, 128


def pick_device():
    """CUDA -> DirectML (AMD) -> CPU."""
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
    def __init__(self, base_words_count=16000, filepath="vocabulary.txt"):
        self.base_limit = base_words_count
        self.filepath = filepath
        self.word2id = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, "<EOS>": EOS_ID}
        self.id2word = {PAD_ID: "<PAD>", UNK_ID: "<UNK>", EOS_ID: "<EOS>"}
        self.dirty = False

        if not self.load_vocab():
            self.inject_real_phrases()

    def build_from_texts(self, texts, min_count=2):
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


def load_external_dataset(filepath="mixed_dialogues.txt"):
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
from hp_mask import generate_causal_mask, generate_padding_mask, generate_combined_mask
from hp_decoder import HPTransformerDecoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class SalatnicaLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=1024, src_len=SRC_LEN, tgt_len=TGT_LEN,
                 dropout=0.1, hp_cfg: HPConfig = None, use_checkpoint=False):
        super().__init__()
        self.src_len, self.tgt_len = src_len, tgt_len
        self.total_vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.use_checkpoint = use_checkpoint
        cfg = hp_cfg or HPConfig()
        self.hp_config = cfg

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max(src_len, tgt_len))

        self.encoder_proj = HPLinear(embedding_dim, embedding_dim, cfg=cfg)

        self.num_layers = 16
        self.decoder_layers = nn.ModuleList([
            HPTransformerDecoderLayer(
                embed_dim=embedding_dim,
                num_heads=12,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                cfg=cfg
            ) for _ in range(self.num_layers)
        ])

        self.output_head = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def hp_layers(self):
        return [m for m in self.modules() if isinstance(m, HPLinear)]

    def hp_begin_epoch(self):
        for m in self.hp_layers():
            m.begin_epoch()

    def hp_accumulate(self):
        for m in self.hp_layers():
            m.accumulate()

    def hp_observe(self, loss):
        return self.hp_config.observe(loss)

    def hp_end_epoch(self, severity, optimizer=None):
        total = {"damaged": 0, "shielded": 0, "mutated": 0, "reborn": 0, "pending": 0, "rewired": 0}
        for m in self.hp_layers():
            for k, v in m.end_epoch(severity, optimizer).items():
                total[k] += v
        return total

    def hp_mean(self):
        return float(torch.cat([m.hp for m in self.hp_layers()]).mean().item())

    def hp_mask_grads(self):
        for m in self.hp_layers():
            m.mask_grad()

    def hp_apply_masks(self):
        for m in self.hp_layers():
            m.apply_mask()

    def hp_density(self):
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

    def _encode(self, input_ids):
        src_emb = self.embedding(input_ids).transpose(0, 1)
        src_emb = self.pos_encoder(src_emb * math.sqrt(self.embedding_dim))
        seq_len, bsz, emb_dim = src_emb.size()
        src_flat = src_emb.reshape(-1, emb_dim)
        memory = self.encoder_proj(src_flat).reshape(seq_len, bsz, emb_dim)
        return memory

    def forward(self, input_ids, target_ids=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        memory = self._encode(input_ids)
        memory_mask = (input_ids != PAD_ID).to(input_ids.device)

        if target_ids is None:
            return self.generate_fast(input_ids)

        if target_ids.dim() == 1:
            target_ids = target_ids.unsqueeze(0)

        start = torch.full_like(target_ids[:, :1], EOS_ID)
        dec_input = torch.cat((start, target_ids[:, :-1]), dim=1)

        tgt_emb = self.embedding(dec_input).transpose(0, 1)
        tgt_emb = self.pos_encoder(tgt_emb * math.sqrt(self.embedding_dim))
        tgt_mask = generate_combined_mask(dec_input, PAD_ID).to(input_ids.device)

        out = tgt_emb
        if self.training:
            from torch.utils.checkpoint import checkpoint

            for layer in self.decoder_layers:
                def create_custom_forward(l):
                    def custom_forward(x, mem, t_mask, m_mask):
                        return l(x, mem, tgt_mask=t_mask, memory_mask=m_mask)
                    return custom_forward

                out = checkpoint(create_custom_forward(layer), out, memory, tgt_mask, memory_mask, use_reentrant=False)
        else:
            for layer in self.decoder_layers:
                out = layer(out, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        seq_len, bsz, emb_dim = out.size()
        logits = self.output_head(out.reshape(-1, emb_dim)).view(seq_len, bsz, -1)
        return logits.permute(1, 2, 0)

    @torch.no_grad()
    def generate_fast(self, input_ids, max_new_tokens=40, temperature=0.6, top_p=0.90, repetition_penalty=1.2, min_tokens=5):
        was_training = self.training
        self.eval()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        bsz = input_ids.size(0)

        memory = self._encode(input_ids)
        memory_mask = (input_ids != PAD_ID).to(input_ids.device)

        current_tok = torch.full((bsz, 1), EOS_ID, dtype=torch.long, device=input_ids.device)
        ids = []

        past_key_values = [None] * self.num_layers

        for t in range(max_new_tokens):
            tgt_emb = self.embedding(current_tok).transpose(0, 1)

            pe_slice = self.pos_encoder.pe[t:t+1]
            tgt_emb = (tgt_emb * math.sqrt(self.embedding_dim)) + pe_slice

            out = tgt_emb
            new_past_key_values = []

            for idx, layer in enumerate(self.decoder_layers):
                out, next_kv = layer(out, memory, memory_mask=memory_mask, layer_past=past_key_values[idx])
                new_past_key_values.append(next_kv)

            past_key_values = new_past_key_values
            logits = self.output_head(out[-1, :, :]).float()

            logits[:, PAD_ID] = logits[:, UNK_ID] = -float("inf")
            if len(ids) < min_tokens:
                logits[:, EOS_ID] = -float("inf")

            if repetition_penalty != 0 and ids:
                for used in set(ids):
                    logits[:, used] -= repetition_penalty

            word = self._sample_filtered(logits.squeeze(0), temperature, top_p, min_p=0.01)
            if word == EOS_ID:
                break

            ids.append(word)
            current_tok = torch.tensor([[word]], dtype=torch.long, device=input_ids.device)

        self.train(was_training)
        return ids

# ======================================================================================
# Тренер под Windows DirectML
# ======================================================================================
class FarewellTrainer:
    def __init__(self, data_path="mixed_dialogues.txt", vocab_path="vocabulary.txt",
                 lr=6e-4,
                 weight_decay=0.08,
                 hp_reg=0.02,
                 dropout=0.12,
                 mutation_rate=0.015,
                 micro_batch=16, # Полная параллельная загрузка (Подходит для 12GB VRAM систем)
                 accum_steps=16,   # Эффективный батч = 256
                 val_fraction=0.1,
                 label_smoothing=0.15,
                 label_smoothing_end=0.03,
                 total_epochs=5,  # Оптимум под 16 слоев на 100k строках
                 warmup_frac=0.05,
                 min_lr_ratio=0.1,
                 sparsity=0.4,
                 use_swa=True,
                 swa_start=0.80,  # Старт SWA на 4-й эпохе, после заморозки DST
                 seed=0,
                 device=None):

        self.device = device or pick_device()
        print(f"⚙️ [Аппаратная платформа]: Выбрано устройство {self.device}")

        self.amp = False

        self.micro_batch, self.accum_steps = micro_batch, accum_steps
        self.hp_reg = hp_reg
        self.epoch = 0
        self.opt_step = 0
        self.total_epochs = max(1, total_epochs)
        self.base_lr, self.min_lr_ratio = lr, min_lr_ratio
        self.ls_start, self.ls_end = label_smoothing, label_smoothing_end
        self.use_swa = use_swa
        self.swa, self.swa_n = None, 0

        self.tokenizer = DynamicTokenizer(filepath=vocab_path)

        # ПРЯМОЙ ИДЕАЛЬНЫЙ ПАРСИНГ ТАБУЛЯЦИИ (Вопрос \t Ответ)
        pairs = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "\t" in line:
                        user_phrase, bot_phrase = line.strip().split("\t", 1)
                        if user_phrase and bot_phrase:
                            pairs.append({"user": user_phrase, "bot": bot_phrase})
            print(f"📚 [Система данных]: Успешно загружено {len(pairs)} пар диалогов через табуляцию.")
        else:
            pairs = [{"user": "Привет!", "bot": "Система активирована."}]

        self.tokenizer.build_from_texts([p[k] for p in pairs for k in ("user", "bot")])

        enc_inputs = []
        enc_targets = []
        for p in pairs:
            enc_inputs.append(self.tokenizer.encode(p["user"], max_len=SRC_LEN, grow=False))
            enc_targets.append(self.tokenizer.encode(p["bot"], max_len=TGT_LEN, grow=False))

        if self.tokenizer.dirty:
            self.tokenizer.save_vocab()

        inputs = torch.stack(enc_inputs)
        targets = torch.stack(enc_targets)

        g = torch.Generator().manual_seed(1234)
        perm = torch.randperm(len(inputs), generator=g)
        n_val = max(1, int(len(inputs) * val_fraction)) if len(inputs) > 10 else 0
        self.val_x, self.val_y = inputs[perm[:n_val]], targets[perm[:n_val]]
        self.train_x, self.train_y = inputs[perm[n_val:]], targets[perm[n_val:]]

        torch.manual_seed(seed)
        random.seed(seed)

        hp_cfg = HPConfig(mutation_rate=mutation_rate, weak_decay=hp_reg, sparsity=sparsity)

        # Инициализируем обновленную 16-слойную модель
        self.model = SalatnicaLanguageModel(
            self.tokenizer.vocab_size, dropout=dropout, hp_cfg=hp_cfg, use_checkpoint=False
        ).to(self.device)

        self.swa_start = max(swa_start, hp_cfg.dst_stop + 0.02) if sparsity > 0 else swa_start

        steps_per_epoch = math.ceil(math.ceil(self.train_x.size(0) / micro_batch) / accum_steps)
        self.total_steps = self.total_epochs * steps_per_epoch
        self.warmup_steps = max(10, int(warmup_frac * self.total_steps))

        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            (decay if p.ndim >= 2 and not name.startswith("embedding") else no_decay).append(p)

        self.optimizer = torch.optim.SGD([
            {"params": decay, "weight_decay": weight_decay, "decay": True},
            {"params": no_decay, "weight_decay": 0.0, "decay": False},
        ], lr=lr, momentum=0.9, nesterov=True)

        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=label_smoothing)
        self.eval_criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    def set_hparams(self, lr=None, weight_decay=None, hp_reg=None, mutation_rate=None):
        if lr is not None:
            self.base_lr = lr
            self._apply_lr()
        if weight_decay is not None:
            for g in self.optimizer.param_groups:
                if g.get("decay"): g["weight_decay"] = weight_decay
        if hp_reg is not None:
            self.hp_reg = hp_reg
            self.model.set_weak_decay(hp_reg)
        if mutation_rate is not None:
            self.model.set_mutation_rate(mutation_rate)

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
        params = dict(self.model.named_parameters())
        backup = {n: p.detach().clone() for n, p in params.items()}
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(self.swa[n].to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in params.items(): p.copy_(backup[n])

    def evaluate_swa(self):
        if self.swa is None: return float("nan")
        with self.swa_weights(): return self.evaluate()

    @torch.no_grad()
    def apply_swa(self):
        for n, p in self.model.named_parameters():
            p.copy_(self.swa[n].to(p.dtype))

    def _optimizer_step(self):
        self._apply_lr()
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if torch.isfinite(gn):
            self.model.hp_accumulate()
            self.model.hp_mask_grads()
            self.optimizer.step()
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

            logits = self.model(x, y)
            loss = self.criterion(logits, y)

            (loss / self.accum_steps).backward()
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

            logits = self.model(x, y)
            total += self.eval_criterion(logits, y).item()
            batches += 1
        return total / batches

    def run_epoch(self):
        ls = self.ls_start + (self.ls_end - self.ls_start) * self.progress()
        self.criterion.label_smoothing = ls
        train_loss = self.train_epoch()
        val_loss = self.evaluate()
        progress = self.progress()
        self.model.set_progress(progress)
        swa_val = float("nan")
        if self.use_swa and progress >= self.swa_start:
            self.swa_update()
            swa_val = self.evaluate_swa()
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

    def state(self, with_optimizer=False):
        st = {"model": self.model.state_dict(), "epoch": self.epoch, "opt_step": self.opt_step,
              "hp_dynamics": self.model.hp_config.dynamics_state()}
        if self.swa is not None: st["swa"] = {"avg": self.swa, "n": self.swa_n}
        if with_optimizer: st["optimizer"] = self.optimizer.state_dict()
        return st

    def load_state(self, st):
        sd = dict(st["model"])
        for k, v in self.model.state_dict().items():
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
# Потоковый инференс и запуск системы
# ======================================================================================
def chat(trainer):
    model, tok, device = trainer.model, trainer.tokenizer, trainer.device
    print("\n💬 [Когнитивный модуль активен]: Введите 'выход' для завершения.")
    while True:
        user_input = input("\n👤 Ты: ")
        if user_input.lower() in ["выход", "exit"]:
            break
        if not user_input.strip():
            continue
        ids = tok.encode(user_input, max_len=model.src_len, grow=False).to(device)

        answer = model.generate_fast(ids)
        print(f"🤖 Бот: {tok.decode(answer)}  [Размер ответа: {len(answer)} токенов]")

if __name__ == '__main__':
    EPOCHS = 5
    WEIGHTS_FILE = "weights_v2.pth"

    trainer = FarewellTrainer(total_epochs=EPOCHS)
    print(f"⚙️ Эффективный батч: {trainer.micro_batch * trainer.accum_steps} пар диалогов за шаг")

    loaded = False
    if os.path.exists(WEIGHTS_FILE):
        try:
            trainer.load_state(torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=True))
            loaded = True
            print("--- КОГНИТИВНАЯ СИСТЕМА ВОССТАНОВЛЕНА ИЗ АРХИВА ---")
        except Exception as e:
            print(f"⚠️ Чекпоинт {WEIGHTS_FILE} не подошел под структуру ({e}). Запуск чистого обучения.")

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
            if score < best_val:
                best_val = score
                best_state = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}

        final_masks = {k: v.clone() for k, v in trainer.model.state_dict().items() if k.endswith(".mask")}
        if best_state is not None:
            trainer.model.load_state_dict(best_state)
        if trainer.swa_n > 0:
            swa_val = trainer.evaluate_swa()
            print(f"  SWA ({trainer.swa_n} снимков): val={swa_val:.4f}, лучшая эпоха: val={best_val:.4f}")
            if swa_val <= best_val:
                trainer.model.load_state_dict(final_masks, strict=False)
                trainer.apply_swa()
                print("  → Использованы усредненные веса SWA")
        torch.save(trainer.state(), WEIGHTS_FILE)
        trainer.tokenizer.save_vocab()
        print("--- ОБУЧЕНИЕ МОДЕЛИ УСПЕШНО ЗАВЕРШЕНО ---")

    chat(trainer)
