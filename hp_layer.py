import math
from dataclasses import dataclass, fields
import torch
import torch.nn as nn

_PERSISTENT_FIELDS = ("best_loss", "last_loss", "stagnation", "epochs_seen", "engage", "mutation_std")

@dataclass
class HPConfig:
    max_damage: float = 8.0
    regress_scale: float = 0.05
    tolerance: float = 0.005
    plateau_floor: float = 0.25
    plateau_ramp: int = 4
    rel_clip: float = 2.5
    warmup_epochs: int = 3
    ramp_epochs: int = 3
    regen: float = 3.0
    cover_fraction: float = 0.10
    weak_decay: float = 0.01
    mutation_rate: float = 0.015
    mutation_rate_min: float = 0.002
    mutation_rate_max: float = 0.035
    error_mutation_prob: float = 0.25
    error_quantile: float = 0.9
    mutation_std: float = 0.025
    noise_std_min: float = 0.005
    noise_std_max: float = 0.04
    noise_up: float = 1.15
    noise_down: float = 0.85
    stagnation_patience: int = 2
    weak_noise_boost: float = 0.5
    sparsity: float = 0.4
    dst_alpha: float = 0.3
    dst_stop: float = 0.7
    dst_min_params: int = 20000
    dst_hp_bias: float = 0.5
    dst_progress: float = 0.0
    rebirth_cap: float = 0.03
    rebirth_init_scale: float = 0.2
    eps: float = 1e-8
    best_loss: float = float("inf")
    last_loss: float = float("inf")
    stagnation: int = 0
    epochs_seen: int = 0
    engage: float = 0.0

    def observe(self, loss: float) -> dict:
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

    def __init__(self, in_features, out_features, bias=True, cfg: HPConfig = None):
        super().__init__(in_features, out_features, bias=bias)
        self.cfg = cfg or HPConfig()
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

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
                self.weight.mul_(1.0 / math.sqrt(density))
        self.register_buffer("mask", mask.view(out_features, in_features))

        self.register_buffer("hp", torch.full((out_features,), 100.0))
        self.register_buffer("cover", torch.zeros(out_features, dtype=torch.bool))
        self.register_buffer("efficiency", torch.zeros(out_features))
        self.register_buffer("_w_start", torch.zeros_like(self.weight), persistent=False)
        self.register_buffer("_g_sum", torch.zeros_like(self.weight), persistent=False)
        self._steps = 0

    @torch.no_grad()
    def begin_epoch(self):
        self._w_start.copy_(self.weight)
        self._g_sum.zero_()
        self._steps = 0

    @torch.no_grad()
    def accumulate(self):
        if self.weight.grad is None:
            return
        self._g_sum += self.weight.grad
        self._steps = 1

    @torch.no_grad()
    def mask_grad(self):
        if self.sparse and self.weight.grad is not None:
            self.weight.grad = self.weight.grad * self.mask.to(self.weight.grad.dtype)

    @torch.no_grad()
    def apply_mask(self):
        if self.sparse:
            self.weight.mul_(self.mask.to(self.weight.dtype))

    def density(self):
        return float(self.mask.float().mean().item()) if self.sparse else 1.0

    @torch.no_grad()
    def end_epoch(self, severity: float, optimizer=None) -> dict:
        c = self.cfg
        n = self.out_features
        if self._steps == 0:
            return {"damaged": 0, "shielded": 0, "mutated": 0, "reborn": 0, "pending": 0, "rewired": 0}

        g_avg = self._g_sum / self._steps
        g_act = g_avg * self.mask.to(g_avg.dtype) if self.sparse else g_avg
        energy = g_act.pow(2).sum(dim=1)

        rel = (energy / (energy.mean() + c.eps)).clamp(0.0, c.rel_clip)
        damage = c.max_damage * severity * rel / (c.rel_clip * 0.5)
        shielded = self.cover.clone()
        damage[shielded] = 0.0
        self.cover.zero_()

        delta = self.weight - self._w_start
        num = -(g_act * delta).sum(dim=1)
        den = g_act.norm(dim=1) * delta.norm(dim=1) + c.eps
        eff = (num / den).clamp(-1.0, 1.0)
        self.efficiency.copy_(eff)

        k = max(1, int(round(c.cover_fraction * n)))
        score = eff.masked_fill(shielded | (eff <= 0), -2.0)
        top = torch.topk(score, min(k, n)).indices
        efficient = torch.zeros_like(shielded)
        efficient[top] = True
        efficient &= score > -2.0

        self.hp -= damage
        self.hp += c.regen * efficient.float()
        self.hp.clamp_(0.0, 100.0)
        self.cover |= efficient

        weakness = (1.0 - self.hp / 100.0).clamp(0.0, 1.0).masked_fill(shielded, 0.0)
        shrink = (c.weak_decay * c.engage * weakness).unsqueeze(1)
        self.weight.mul_(1.0 - shrink)

        mutants = torch.zeros_like(shielded)
        if c.engage > 0.0:
            rate = min(max(c.mutation_rate, c.mutation_rate_min), c.mutation_rate_max)
            dev = self.hp.device
            random_mut = torch.rand(n, device=dev) < rate
            error_cand = (rel >= torch.quantile(rel, c.error_quantile)) & (damage > 0) & (eff <= eff.median())
            error_mut = error_cand & (torch.rand(n, device=dev) < c.error_mutation_prob)
            mutants = random_mut | error_mut
            cap = max(1, math.ceil(c.mutation_rate_max * n))
            if int(mutants.sum()) > cap:
                idx = mutants.nonzero().squeeze(1)
                keep = idx[torch.randperm(idx.numel(), device=dev)[:cap]]
                mutants = torch.zeros_like(mutants)
                mutants[keep] = True
        if mutants.any():
            weak = (1.0 - self.hp / 100.0).clamp(0.0, 1.0)
            scale = (self.weight.std() * c.mutation_std * (1.0 + c.weak_noise_boost * weak)).unsqueeze(1)
            noise = torch.randn_like(self.weight) * scale
            noise = noise[mutants] * self.mask[mutants].to(noise.dtype) if self.sparse else noise[mutants]
            self.weight[mutants] += noise
            self.cover |= mutants

        dead = self.hp <= 0
        n_dead = int(dead.sum().item())
        reborn, pending = 0, 0
        if n_dead:
            cap_r = max(1, math.ceil(c.rebirth_cap * n))
            if n_dead > cap_r:
                idx = dead.nonzero().squeeze(1)
                worst = idx[torch.argsort(eff[idx])[:cap_r]]
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
            self.cover[dead] = True
            self.efficiency[dead] = 0.0
            self._reset_optimizer_rows(optimizer, dead)

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
        c = self.cfg
        if not self.sparse or c.dst_progress >= c.dst_stop or self._steps == 0:
            return 0
        frac = c.dst_alpha * 0.5 * (1.0 + math.cos(math.pi * c.dst_progress / max(c.dst_stop, c.eps)))
        n_active = int(self.mask.sum().item())
        k = min(int(frac * n_active), self.mask.numel() - n_active)
        if k <= 0:
            return 0
        hp_factor = 1.0 - c.dst_hp_bias * (1.0 - self.hp / 100.0).clamp(0.0, 1.0)
        drop_score = (self.weight.abs() * hp_factor.unsqueeze(1)).masked_fill(~self.mask, float("inf")).flatten()
        grow_score = self._g_sum.abs().masked_fill(self.mask, -1.0).flatten()
        drop_idx = torch.topk(drop_score, k, largest=False).indices
        grow_idx = torch.topk(grow_score, k).indices
        flat_mask, flat_w = self.mask.view(-1), self.weight.view(-1)
        flat_mask[drop_idx] = False
        flat_mask[grow_idx] = True
        flat_w[drop_idx] = 0.0
        flat_w[grow_idx] = 0.0
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
        if optimizer is None:
            return
        for p in (self.weight, self.bias):
            if p is not None and p in optimizer.state:
                self._reset_rows(optimizer.state[p], p.shape, rows)

