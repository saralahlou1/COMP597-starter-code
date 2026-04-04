import src.config as config
import src.trainer.stats.base as base
import torch
import time
import json
import statistics


trainer_stats_name="fwd"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    return FwdTrainerStats()

class FwdTrainerStats(base.TrainerStats):
    """NOOP Trainer stats to ignore data accumulation.

    This class implements the `TrainerStats` interface. All the methods are 
    NOOP so that training can be done with accumulating statistics.

    """

    def __init__(self) -> None:
        super().__init__()
        self._train_start_time = None
        self.train_duration = {
            "forward": []
        }

    def start_train(self) -> None:
        pass

    def stop_train(self, output_path: str = "fwd_timing_stats.json") -> None:
        torch.cuda.synchronize()

        forward_times = self.train_duration["forward"]

        stats = {
            "forward": {
                "count": len(forward_times),
                "durations": forward_times,
                "average": sum(forward_times) / len(forward_times) if forward_times else 0.0,
                "std": statistics.stdev(forward_times) if len(forward_times) > 1 else 0.0,
                "min": min(forward_times) if forward_times else 0.0,
                "max": max(forward_times) if forward_times else 0.0,
            }
        }

        with open(output_path, "w") as f:
            json.dump(stats, f, indent=4)

    def start_step(self) -> None:
        pass

    def stop_step(self) -> None:
        pass

    def start_optimizer_step(self) -> None:
        pass

    def stop_optimizer_step(self) -> None:
        pass

    def start_forward(self) -> None:
        torch.cuda.synchronize()
        self._forward_start_time = time.perf_counter()

    def stop_forward(self) -> None:
        torch.cuda.synchronize()
        duration = time.perf_counter() - self._forward_start_time
        self.train_duration["forward"].append(duration)
        self._forward_start_time = None

    def start_backward(self) -> None:
        pass

    def stop_backward(self) -> None:
        pass
    
    def start_save_checkpoint(self) -> None:
        pass

    def stop_save_checkpoint(self) -> None:
        pass

    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        pass

    def log_loss(self, loss: torch.Tensor) -> None:
        pass
