# src/trainer/simple2.py

import torch
from typing import Any
from torch.utils.data._utils.collate import default_collate
from src.trainer.simple import SimpleTrainer


class SimpleTrainer2(SimpleTrainer):
    """
    Converts everything into a dict expected by Trainer.base.
    """

    def process_batch(self, i: int, batch: Any) -> Any:
        # If it's a list/tuple
        if isinstance(batch, (list, tuple)):
            # batch is a tuple (images, labels)
            if len(batch) == 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]):
                images, labels = batch
                return {"image": images.to(self.device), "label": labels.to(self.device)}

            # Singleton list: try to recurse
            if len(batch) == 1:
                return self.process_batch(i, batch[0])

