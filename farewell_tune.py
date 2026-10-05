"""
Ray Tune + Population-Based Training для FarewellAI.

Сеть между ПК (1 Gbps, Windows/Gloo) НЕ нагружается градиентами: DDP не используется, каждый трайл
обучается на своём ПК независимо. Раз в `perturbation_interval` эпох PBT:
  * копирует веса лучших трайлов в худшие через стандартные чекпоинты Ray (save_checkpoint/load_checkpoint,
    только state_dict модели - без состояния оптимизатора, это ~десятки МБ);
  * мутирует гиперпараметры (lr, weight_decay, hp_reg, mutation_rate).

Запуск кластера: на head-ПК `ray start --head`, на остальных `ray start --address=<head_ip>:6379`,
затем `python farewell_tune.py`.
"""
import os
import random

import torch
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

from FarewellAI import FarewellTrainer

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_FILE = "farewell.pt"


class FarewellTrainable(tune.Trainable):
    def setup(self, config):
        self.trainer = FarewellTrainer(
            data_path=config.get("data_path", os.path.join(HERE, "romantic_dialogues.txt")),
            vocab_path=config.get("vocab_path", os.path.join(HERE, "vocabulary.txt")),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            hp_reg=config["hp_reg"],
            mutation_rate=config["mutation_rate"],
            dropout=config.get("dropout", 0.2),
            micro_batch=config.get("micro_batch", 2),
            total_epochs=config.get("total_epochs", 30),  # горизонт cosine-lr / DST / SWA = stop["training_iteration"]
            sparsity=config.get("sparsity", 0.4),
            accum_steps=config.get("accum_steps", 32),
            seed=config.get("seed", random.randrange(10_000)),
        )
        self.ckpt_optimizer = config.get("ckpt_optimizer", False)

    def step(self):
        # Одна эпоха = обучение + валидация + эволюционный шаг HP/Покров
        return self.trainer.run_epoch()

    def save_checkpoint(self, checkpoint_dir):
        torch.save(self.trainer.state(self.ckpt_optimizer), os.path.join(checkpoint_dir, CKPT_FILE))
        return checkpoint_dir

    def load_checkpoint(self, checkpoint):
        path = checkpoint if isinstance(checkpoint, str) else checkpoint.get("path", "")
        if os.path.isdir(path):
            path = os.path.join(path, CKPT_FILE)
        self.trainer.load_state(torch.load(path, map_location="cpu", weights_only=True))

    def reset_config(self, new_config):
        """PBT меняет гиперпараметры без пересоздания актора: сохраняем веса и HP-состояние."""
        self.trainer.set_hparams(
            lr=new_config["lr"], weight_decay=new_config["weight_decay"],
            hp_reg=new_config["hp_reg"], mutation_rate=new_config["mutation_rate"],
        )
        self.config = new_config
        return True


if __name__ == "__main__":
    import ray

    PERTURBATION_INTERVAL = 3  # раз в N эпох синхронизируем лучших
    ray.init(address=os.environ.get("RAY_ADDRESS"), ignore_reinit_error=True)

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=PERTURBATION_INTERVAL,
        quantile_fraction=0.25,
        hyperparam_mutations={
            "lr": lambda: 10 ** random.uniform(-4.5, -2.5),
            "weight_decay": lambda: 10 ** random.uniform(-3, -1),
            "hp_reg": lambda: 10 ** random.uniform(-3, -1),
            "mutation_rate": [0.002, 0.005, 0.01, 0.02, 0.035],
        },
    )

    resources = {"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0}
    tuner = tune.Tuner(
        tune.with_resources(FarewellTrainable, resources),
        tune_config=tune.TuneConfig(metric="val_loss", mode="min", scheduler=pbt, num_samples=4),
        param_space={
            "lr": 3e-4, "weight_decay": 0.05, "hp_reg": 1e-2, "mutation_rate": 0.02,
            "micro_batch": 2, "accum_steps": 32, "total_epochs": 30, "sparsity": 0.4,
        },
        run_config=tune.RunConfig(
            stop={"training_iteration": 30},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=PERTURBATION_INTERVAL, checkpoint_at_end=True, num_to_keep=2),
        ),
    )
    results = tuner.fit()
    best = results.get_best_result()
    print("Лучший val_loss:", best.metrics["val_loss"], "config:", best.config)
