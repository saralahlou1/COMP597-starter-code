import src.config as config
import src.trainer.stats.base as base
import torch
import time
import json
import statistics


trainer_stats_name = "bwd"

def construct_trainer_stats(conf: config.Config, **kwargs) -> base.TrainerStats:
    return BwdTrainerStats()


class BwdTrainerStats(base.TrainerStats):

    def __init__(self) -> None:
        super().__init__()
        self._backward_start_time = None
        self.train_duration = {
            "backward": []
        }

    def start_train(self) -> None:
        pass

    def stop_train(self, output_path: str = "bwd_timing_stats.json") -> None:
        torch.cuda.synchronize()

        backward_times = self.train_duration["backward"]

        stats = {
            "backward": {
                "count": len(backward_times),
                "durations": backward_times,
                "average": sum(backward_times) / len(backward_times) if backward_times else 0.0,
                "std": statistics.stdev(backward_times) if len(backward_times) > 1 else 0.0,
                "min": min(backward_times) if backward_times else 0.0,
                "max": max(backward_times) if backward_times else 0.0,
            }
        }

        with open(output_path, "w") as f:
            json.dump(stats, f, indent=4)

    def start_backward(self) -> None:
        torch.cuda.synchronize()
        self._backward_start_time = time.perf_counter()

    def stop_backward(self) -> None:
        torch.cuda.synchronize()
        duration = time.perf_counter() - self._backward_start_time
        self.train_duration["backward"].append(duration)
        self._backward_start_time = None

    # Unused hooks
    def start_step(self) -> None: pass
    def stop_step(self) -> None: pass
    def start_optimizer_step(self) -> None: pass
    def stop_optimizer_step(self) -> None: pass
    def start_forward(self) -> None: pass
    def stop_forward(self) -> None: pass
    def start_save_checkpoint(self) -> None: pass
    def stop_save_checkpoint(self) -> None: pass
    def log_step(self) -> None: pass
    def log_stats(self) -> None: pass
    def log_loss(self, loss: torch.Tensor) -> None: pass
