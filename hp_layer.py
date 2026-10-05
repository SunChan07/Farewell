"""
Слой с эволюционной HP-регуляризацией.

Каждый выходной нейрон (строка весовой матрицы) имеет:
  * hp    - здоровье 0..100;
  * cover - "Покров": булев щит, полностью поглощающий урон ОДНОЙ эпохи.

Фазы жизни модели ("дизельный генератор"):
  1. ПРОГРЕВ (warmup_epochs): HP-механика выключена - ни урона, ни шума, ни штрафов. Модель спокойно
     получает хороший старт, и эволюция не ломает ещё не сформированные представления.
  2. РАЗГОН (ramp_epochs): сила механики плавно растёт 0 -> 1 (cfg.engage).
  3. РЕЖИМ: урон зависит от ПРОГРЕССА, а не от абсолютного loss:
       * loss улучшается          -> урона нет (здоровая модель не "лечится" вслепую);
       * loss вырос ПО СРАВНЕНИЮ С ПРОШЛОЙ ЭПОХОЙ -> урон ~ tanh(относительный рост / regress_scale)
         (сравнение именно с прошлой эпохой, а не с рекордом: если переобучение уже случилось и val-loss
         держится на высоком уровне, постоянный максимальный урон только добивал бы модель);
       * застой относительно рекорда -> лёгкое фоновое давление (plateau_floor), растущее с числом эпох.

Эволюционный шаг (HPLinear.end_epoch) выполняется ровно один раз в конце эпохи:
  1. Урон = severity * доля градиентной энергии нейрона. Нейроны под Покровом урон игнорируют,
     после чего Покров сбрасывается.
  2. Эффективность нейрона = -cos(накопленный градиент, фактический сдвиг весов за эпоху).
     Топ-k эффективных лечатся и зарабатывают Покров на СЛЕДУЮЩУЮ эпоху.
  3. Слабые нейроны (низкий HP) слегка сжимаются к нулю (weak_decay) - это и есть "HP-штраф". Он применяется
     напрямую к весам раз в эпоху, а не добавкой к loss: добавка ~mean(w^2)*hp_reg была на 5-6 порядков
     меньше CE-градиента и фактически ничего не делала.
  4. Мутации: 0.2-3.5% нейронов (случайные + "ошибочные") получают гауссов шум и Покров.
  5. HP <= 0 -> нейрон умирает; перерождается не более rebirth_cap (по умолчанию 4%) нейронов слоя за эпоху,
     остальные ждут своей очереди (сначала самые неэффективные). Новорождённый получает Покров.
  6. Dynamic Sparse Training (RigL/SET): крупные слои разрежены маской (cfg.sparsity). Раз в эпоху доля
     dst_alpha*cos-расписание самых слабых связей удаляется (|w| * множитель HP нейрона - у слабых нейронов связи
     уходят охотнее), столько же новых связей включается там, где накопленный градиент больше всего (мёртвые
     позиции, у которых градиент большой, получают шанс). Общая плотность не меняется. После dst_stop
     (доля обучения) топология замораживается - это нужно, чтобы SWA усреднял веса одной и той же структуры.
"""
import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn

# Поля cfg, которые переживают чекпоинт (состояние динамики). Гиперпараметры PBT (mutation_rate, weak_decay) - нет.
_PERSISTENT_FIELDS = ("best_loss", "last_loss", "stagnation", "epochs_seen", "engage", "mutation_std")


@dataclass
class HPConfig:
    # --- урон ---
    max_damage: float = 15.0        # максимальный урон за эпоху (при severity=1)
    regress_scale: float = 0.10     # рост loss на 10% за эпоху -> tanh(1)=0.76 severity
    tolerance: float = 0.005        # относительный шум val-loss, меньше которого улучшением/регрессом не считаем
    plateau_floor: float = 0.25     # максимум фонового severity при плато
    plateau_ramp: int = 4           # за сколько эпох застоя фоновое давление достигает plateau_floor
    rel_clip: float = 3.0           # ограничение относительной энергии при расчёте урона
    # --- фазы ---
    warmup_epochs: int = 3          # эпохи "хорошего старта": HP-механика выключена
    ramp_epochs: int = 3            # эпохи плавного включения
    # --- лечение / Покров ---
    regen: float = 2.0              # лечение эффективных нейронов
    cover_fraction: float = 0.10    # Покров получают лишь топ-10% по эффективности
    # --- HP-штраф ---
    weak_decay: float = 0.01        # сжатие самых слабых нейронов за эпоху (0.01 = -1% весов при HP=0)
    # --- мутации ---
    mutation_rate: float = 0.01     # текущая доля случайных мутантов (адаптивная)
    mutation_rate_min: float = 0.002  # границы доли мутантов: 0.2% ... 3.5%
    mutation_rate_max: float = 0.035
    error_mutation_prob: float = 0.2  # шанс мутации для "ошибочных" нейронов
    error_quantile: float = 0.9     # "ошибочные" - топ-10% по градиентной энергии
    mutation_std: float = 0.02      # текущая сила шума относительно std весов (адаптивная)
    noise_std_min: float = 0.005
    noise_std_max: float = 0.08
    noise_up: float = 1.15          # рост шума и мутаций при ЗАСТОЕ val-loss
    noise_down: float = 0.85        # спад при отсутствии застоя (adaptive parameter noise)
    stagnation_patience: int = 2    # шум растёт только после N эпох подряд без улучшения
    weak_noise_boost: float = 0.5   # слабые (низкий HP) нейроны получают до +50% шума
    # --- Dynamic Sparse Training ---
    sparsity: float = 0.4           # доля обнулённых связей в крупных слоях (0 - выключено)
    dst_alpha: float = 0.3          # стартовая доля перестраиваемых связей за эпоху (затем cos-спад до 0)
    dst_stop: float = 0.6           # доля обучения, после которой топология замораживается
    dst_min_params: int = 20000     # слои меньше этого размера остаются плотными
    dst_hp_bias: float = 0.5        # слабые нейроны теряют связи охотнее (множитель |w| от 1.0 до 1-dst_hp_bias)
    dst_progress: float = 0.0       # прогресс обучения 0..1 (выставляет тренер, в чекпоинт не пишется)
    # --- перерождение ---
    rebirth_cap: float = 0.04       # не более 4% нейронов слоя перерождаются за эпоху
    rebirth_init_scale: float = 0.3  # новорождённый стартует "тихим" (std xavier * 0.3), чтобы не шокировать соседние слои
    eps: float = 1e-8
    # --- состояние динамики (сохраняется в чекпоинт) ---
    best_loss: float = float("inf")
    last_loss: float = float("inf")
    stagnation: int = 0
    epochs_seen: int = 0
    engage: float = 0.0             # 0 (прогрев) ... 1 (полный режим)

    def observe(self, loss: float) -> dict:
        """Вызывать ОДИН раз за эпоху с val-loss (или train-loss). Возвращает severity и признак улучшения."""
        self.epochs_seen += 1
        regress = 0.0
        if math.isfinite(self.last_loss):
            regress = max(0.0, loss / (self.last_loss + self.eps) - 1.0 - self.tolerance)
        self.last_loss = loss
        improved = loss < self.best_loss * (1.0 - self.tolerance)
        self.best_loss = min(self.best_loss, loss)
        self.stagnation = 0 if improved else self.stagnation + 1

        plateau = self.plateau_floor * min(1.0, self.stagnation / max(1, self.plateau_ramp))
        raw = min(1.0, math.tanh(regress / max(self.regress_scale, self.eps)) + plateau)
        self.engage = min(1.0, max(0.0, (self.epochs_seen - self.warmup_epochs) / max(1, self.ramp_epochs)))
        return {"severity": raw * self.engage, "improved": improved}

    def adapt(self):
        """Динамика шума: длительный застой -> больше исследования, иначе шум затухает (в границах).
        Во время прогрева стартовые значения не трогаем."""
        if self.engage <= 0.0:
            return
        k = self.noise_up if self.stagnation >= self.stagnation_patience else self.noise_down
        self.mutation_std = min(max(self.mutation_std * k, self.noise_std_min), self.noise_std_max)
        self.mutation_rate = min(max(self.mutation_rate * k, self.mutation_rate_min), self.mutation_rate_max)

    def dynamics_state(self) -> dict:
        return {k: getattr(self, k) for k in _PERSISTENT_FIELDS}

    def load_dynamics_state(self, st: dict):
        names = {f.name for f in fields(self)}
        for k in _PERSISTENT_FIELDS:
            if k in st and k in names:
                setattr(self, k, st[k])


class HPLinear(nn.Linear):
    """nn.Linear, у которого каждый выходной нейрон имеет HP и Покров."""

    def __init__(self, in_features, out_features, bias=True, cfg: HPConfig = None):
        super().__init__(in_features, out_features, bias=bias)
        self.cfg = cfg or HPConfig()
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # Маска разреженности (Dynamic Sparse Training). Хранится в чекпоинте, чтобы топология переживала PBT
        n_params = in_features * out_features
        self.sparse = self.cfg.sparsity > 0.0 and n_params >= self.cfg.dst_min_params
        mask = torch.ones(n_params, dtype=torch.bool)
        if self.sparse:
            n_active = max(out_features, int(round(n_params * (1.0 - self.cfg.sparsity))))
            mask.zero_()
            mask[torch.randperm(n_params)[:n_active]] = True
            density = n_active / n_params
            with torch.no_grad():
                self.weight.mul_(mask.view(out_features, in_features))
                self.weight.mul_(1.0 / math.sqrt(density))  # сохраняем масштаб активаций
        self.register_buffer("mask", mask.view(out_features, in_features))

        self.register_buffer("hp", torch.full((out_features,), 100.0))
        self.register_buffer("cover", torch.zeros(out_features, dtype=torch.bool))
        self.register_buffer("efficiency", torch.zeros(out_features))
        # Статистика эпохи (не сохраняется в чекпоинт - пересоздаётся в begin_epoch)
        self.register_buffer("_w_start", torch.zeros_like(self.weight), persistent=False)
        self.register_buffer("_g_sum", torch.zeros_like(self.weight), persistent=False)
        self.register_buffer("_g_energy", torch.zeros(out_features), persistent=False)
        self._steps = 0

    # ---------- статистика эпохи ----------
    @torch.no_grad()
    def begin_epoch(self):
        self._w_start.copy_(self.weight)
        self._g_sum.zero_()
        self._g_energy.zero_()
        self._steps = 0

    @torch.no_grad()
    def accumulate(self):
        """Вызывать после backward (и unscale/clip), прямо перед optimizer.step().
        _g_sum хранит ПЛОТНЫЙ градиент (нужен DST для роста связей), энергия - только по активным связям."""
        if self.weight.grad is None:
            return
        g = self.weight.grad.float()
        self._g_sum += g
        ga = g * self.mask.to(g.dtype) if self.sparse else g
        self._g_energy += ga.pow(2).sum(dim=1)
        self._steps += 1

    # ---------- разреженность ----------
    @torch.no_grad()
    def mask_grad(self):
        """Обнуляет градиент выключенных связей (после accumulate, перед optimizer.step())."""
        if self.sparse and self.weight.grad is not None:
            # не in-place и не с bool-маской: на DirectML это портило тензор градиента (access violation)
            self.weight.grad = self.weight.grad * self.mask.to(self.weight.grad.dtype)

    @torch.no_grad()
    def apply_mask(self):
        """Возвращает нули в выключенные связи (после optimizer.step(): импульс Adam мог их сдвинуть)."""
        if self.sparse:
            self.weight.mul_(self.mask.to(self.weight.dtype))  # float-маска: bool-умножение на DirectML ломает dtype

    def density(self):
        return float(self.mask.float().mean().item()) if self.sparse else 1.0

    # ---------- эволюционный шаг ----------
    @torch.no_grad()
    def end_epoch(self, severity: float, optimizer=None) -> dict:
        c = self.cfg
        n = self.out_features
        if self._steps == 0:
            return {"damaged": 0, "shielded": 0, "mutated": 0, "reborn": 0, "pending": 0, "rewired": 0}

        # 1. Урон эпохи (severity уже учитывает прогресс loss и фазу прогрева)
        energy = self._g_energy / self._steps
        rel = (energy / (energy.mean() + c.eps)).clamp(0.0, c.rel_clip)
        damage = c.max_damage * severity * rel / (c.rel_clip * 0.5)  # rel=rel_clip/2 -> полный max_damage*severity
        shielded = self.cover.clone()
        damage[shielded] = 0.0          # Покров поглощает урон
        self.cover.zero_()              # ...и сбрасывается

        # 2. Эффективность: совпадение сдвига весов с направлением антиградиента
        delta = self.weight - self._w_start
        g_act = self._g_sum * self.mask.to(self._g_sum.dtype) if self.sparse else self._g_sum
        num = -(g_act * delta).sum(dim=1)
        den = g_act.norm(dim=1) * delta.norm(dim=1) + c.eps
        eff = (num / den).clamp(-1.0, 1.0)
        self.efficiency.copy_(eff)
        # Покров только топ-k по эффективности (ранжирование, а не квантиль). Кулдаун: кто был под Покровом
        # сейчас, подряд его не получает.
        k = max(1, int(round(c.cover_fraction * n)))
        score = eff.masked_fill(shielded | (eff <= 0), -2.0)
        top = torch.topk(score, min(k, n)).indices
        efficient = torch.zeros_like(shielded)
        efficient[top] = True
        efficient &= score > -2.0

        self.hp -= damage
        self.hp += c.regen * efficient.float()
        self.hp.clamp_(0.0, 100.0)
        self.cover |= efficient         # Покров на следующую эпоху - за высокую эффективность

        # 3. HP-штраф: слабые нейроны (не под Покровом) слегка сжимаются к нулю, сила растёт вместе с engage
        weakness = (1.0 - self.hp / 100.0).clamp(0.0, 1.0).masked_fill(shielded, 0.0)
        shrink = (c.weak_decay * c.engage * weakness).unsqueeze(1)
        self.weight.mul_(1.0 - shrink)

        # 4. Мутации (в прогрев выключены)
        mutants = torch.zeros_like(shielded)
        if c.engage > 0.0:
            rate = min(max(c.mutation_rate, c.mutation_rate_min), c.mutation_rate_max)
            dev = self.hp.device
            random_mut = torch.rand(n, device=dev) < rate
            error_cand = (rel >= torch.quantile(rel, c.error_quantile)) & (damage > 0) & (eff <= eff.median())
            error_mut = error_cand & (torch.rand(n, device=dev) < c.error_mutation_prob)
            mutants = random_mut | error_mut
            cap = max(1, math.ceil(c.mutation_rate_max * n))  # общий потолок мутантов на слой
            if int(mutants.sum()) > cap:
                idx = mutants.nonzero().squeeze(1)
                keep = idx[torch.randperm(idx.numel(), device=dev)[:cap]]
                mutants = torch.zeros_like(mutants)
                mutants[keep] = True
        if mutants.any():
            # шум динамический: общий уровень адаптируется (cfg.adapt), слабые нейроны шумят сильнее
            weak = (1.0 - self.hp / 100.0).clamp(0.0, 1.0)
            scale = (self.weight.std() * c.mutation_std * (1.0 + c.weak_noise_boost * weak)).unsqueeze(1)
            noise = torch.randn_like(self.weight) * scale
            noise = noise[mutants] * self.mask[mutants].to(noise.dtype) if self.sparse else noise[mutants]
            self.weight[mutants] += noise
            self.cover |= mutants

        # 5. Смерть и ограниченное перерождение
        dead = self.hp <= 0
        n_dead = int(dead.sum().item())
        reborn, pending = 0, 0
        if n_dead:
            cap_r = max(1, math.ceil(c.rebirth_cap * n))
            if n_dead > cap_r:
                idx = dead.nonzero().squeeze(1)
                worst = idx[torch.argsort(eff[idx])[:cap_r]]  # первыми возрождаются самые неэффективные
                dead = torch.zeros_like(dead)
                dead[worst] = True
            reborn = int(dead.sum().item())
            pending = n_dead - reborn
            std = math.sqrt(2.0 / (self.in_features + self.out_features))
            new_w = torch.randn(reborn, self.in_features, device=self.weight.device) * (std * c.rebirth_init_scale)
            self.weight[dead] = new_w * self.mask[dead].to(new_w.dtype) if self.sparse else new_w
            if self.bias is not None:
                self.bias[dead] = 0.0
            self.hp[dead] = 100.0
            self.cover[dead] = True     # защита новорождённых на одну эпоху
            self.efficiency[dead] = 0.0
            self._reset_optimizer_rows(optimizer, dead)

        # 6. Dynamic Sparse Training: перестройка топологии
        rewired = self._dst_update(optimizer)

        return {
            "damaged": int((damage > 0).sum().item()),
            "shielded": int(shielded.sum().item()),
            "mutated": int(mutants.sum().item()),
            "reborn": reborn,
            "pending": pending,
            "rewired": rewired,
        }

    @torch.no_grad()
    def _dst_update(self, optimizer=None) -> int:
        """RigL: удаляем k самых слабых активных связей, включаем k неактивных с наибольшим накопленным градиентом."""
        c = self.cfg
        if not self.sparse or c.dst_progress >= c.dst_stop or self._steps == 0:
            return 0
        frac = c.dst_alpha * 0.5 * (1.0 + math.cos(math.pi * c.dst_progress / max(c.dst_stop, c.eps)))
        n_active = int(self.mask.sum().item())
        k = min(int(frac * n_active), self.mask.numel() - n_active)
        if k <= 0:
            return 0
        # слабый нейрон (низкий HP) теряет связи охотнее
        hp_factor = 1.0 - c.dst_hp_bias * (1.0 - self.hp / 100.0).clamp(0.0, 1.0)
        drop_score = (self.weight.abs() * hp_factor.unsqueeze(1)).masked_fill(~self.mask, float("inf")).flatten()
        grow_score = self._g_sum.abs().masked_fill(self.mask, -1.0).flatten()
        drop_idx = torch.topk(drop_score, k, largest=False).indices
        grow_idx = torch.topk(grow_score, k).indices   # кандидаты - только ранее неактивные, т.е. не пересекаются с drop
        flat_mask, flat_w = self.mask.view(-1), self.weight.view(-1)
        flat_mask[drop_idx] = False
        flat_mask[grow_idx] = True
        flat_w[drop_idx] = 0.0
        flat_w[grow_idx] = 0.0          # новые связи стартуют с нуля (RigL) и растут по градиенту
        if optimizer is not None and self.weight in optimizer.state:
            for v in optimizer.state[self.weight].values():
                if torch.is_tensor(v) and v.shape == self.weight.shape:
                    vf = v.view(-1)
                    vf[drop_idx] = 0
                    vf[grow_idx] = 0
        return k

    @staticmethod
    def _reset_rows(state, shape, rows):
        for v in state.values():
            if torch.is_tensor(v) and v.shape == shape:
                v[rows] = 0

    def _reset_optimizer_rows(self, optimizer, rows):
        """Обнуляет моменты Adam для перерожденных строк (иначе старый импульс испортит новые веса)."""
        if optimizer is None:
            return
        for p in (self.weight, self.bias):
            if p is not None and p in optimizer.state:
                self._reset_rows(optimizer.state[p], p.shape, rows)
