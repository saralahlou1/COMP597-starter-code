import src.config as config
import src.trainer.stats.base as base
import torch
import time
import json
import statistics


trainer_stats_name = "opt"

def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    return OptTrainerStats()


class OptTrainerStats(base.TrainerStats):

    def __init__(self) -> None:
        super().__init__()
        self._opt_start_time = None
        self.train_duration = {
            "optimizer_step": []
        }

    def start_train(self) -> None:
        pass

    def stop_train(self, output_path: str = "opt_timing_stats.json") -> None:
        torch.cuda.synchronize()

        opt_times = self.train_duration["optimizer_step"]

        stats = {
            "optimizer_step": {
                "count": len(opt_times),
                "durations": opt_times,
                "average": sum(opt_times) / len(opt_times) if opt_times else 0.0,
                "std": statistics.stdev(opt_times) if len(opt_times) > 1 else 0.0,
                "min": min(opt_times) if opt_times else 0.0,
                "max": max(opt_times) if opt_times else 0.0,
            }
        }

        with open(output_path, "w") as f:
            json.dump(stats, f, indent=4)

    def start_optimizer_step(self) -> None:
        torch.cuda.synchronize()
        self._opt_start_time = time.perf_counter()

    def stop_optimizer_step(self) -> None:
        torch.cuda.synchronize()
        duration = time.perf_counter() - self._opt_start_time
        self.train_duration["optimizer_step"].append(duration)
        self._opt_start_time = None

    # Unused hooks
    def start_step(self) -> None: pass
    def stop_step(self) -> None: pass
    def start_forward(self) -> None: pass
    def stop_forward(self) -> None: pass
    def start_backward(self) -> None: pass
    def stop_backward(self) -> None: pass
    def start_save_checkpoint(self) -> None: pass
    def stop_save_checkpoint(self) -> None: pass
    def log_step(self) -> None: pass
    def log_stats(self) -> None: pass
    def log_loss(self, loss: torch.Tensor) -> None: pass
