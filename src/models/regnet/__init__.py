# === import necessary modules ===
import src.models.regnet.model as model_impl
import src.config as config # Configurations
import src.trainer as trainer # Trainer base class

# === import necessary external modules ===
from typing import Any, Dict, Optional, Tuple
import torch.utils.data as data

model_name = "regnet"

def init_model(conf : config.Config, dataset : data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    model = model_impl.build_model(conf)

    tr = model_impl.build_trainer(
        conf=conf,
        model=model,
        dataset=dataset,
    )

    return tr
