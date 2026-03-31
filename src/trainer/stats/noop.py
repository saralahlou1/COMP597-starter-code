import src.config as config
import src.trainer.stats.base as base
import torch
import time


trainer_stats_name="noop"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    return NOOPTrainerStats()

class NOOPTrainerStats(base.TrainerStats):
    """NOOP Trainer stats to ignore data accumulation.

    This class implements the `TrainerStats` interface. All the methods are 
    NOOP so that training can be done with accumulating statistics.

    """

    def __init__(self) -> None:
        super().__init__()
        self._train_start_time = None
        self.train_duration = None

    def start_train(self) -> None:
        self._train_start_time = time.time()

    def stop_train(self) -> None:
        self.train_duration = time.time() - self._train_start_time
        print(f"Training took {self.train_duration:.3f} seconds")

    def start_step(self) -> None:
        pass

    def stop_step(self) -> None:
        pass

    def start_optimizer_step(self) -> None:
        pass

    def stop_optimizer_step(self) -> None:
        pass

    def start_forward(self) -> None:
        pass

    def stop_forward(self) -> None:
        pass

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
