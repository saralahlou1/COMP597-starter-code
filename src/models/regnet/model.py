import torch
import src.trainer.stats as trainer_stats
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.models import regnet_y_128gf, RegNet_Y_128GF_Weights

from src.trainer.simple2 import SimpleTrainer2
import src.config as config

from types import SimpleNamespace
import torch
import torch.nn as nn


class ModelWithLoss(nn.Module):
    """
    Adapter around a torchvision model so that:
      - forward(**batch) works (extracts image from common keys)
      - returns an object with .loss and .logits
    """
    def __init__(self, base_model: nn.Module, device: torch.device):
        super().__init__()
        self.base = base_model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        images = None
        labels = None
        
        # keyword-image keys
        images = kwargs.pop("image")

        # label keys
        labels = kwargs.pop("label")

        # Move images to device and call base
        if images is not None:
            images = images.to(self.device)
            logits = self.base(images)

        # compute loss if labels provided
        if labels is not None:
            labels = labels.to(self.device)
            loss = self.criterion(logits, labels)

        return SimpleNamespace(loss=loss, logits=logits)


def build_model(conf: config.Config) -> nn.Module:
    weights = RegNet_Y_128GF_Weights.DEFAULT
    model = regnet_y_128gf(weights=weights)

    return model


def build_trainer(conf, model, dataset):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    wrapped_model = ModelWithLoss(model, device)

    # DataLoader settings (use defaults from conf if present)
    batch_size = getattr(conf, "batch_size", 32)
    print(f"Batch size: {batch_size}")
    num_workers = getattr(conf, "num_workers", 2)
    shuffle = True
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    #loader = data.DataLoader(dataset, batch_size=conf.batch_size)

    optimizer = optim.SGD(
        model.parameters(),
        lr=conf.learning_rate,
        momentum=0.9,
    )

    stats=trainer_stats.init_from_conf(conf=conf, device=wrapped_model.device, num_train_steps=len(loader))

    step_size = 1
    gamma = 0.1
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    trainer = SimpleTrainer2(
        loader=loader,
        model=wrapped_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        stats=stats,
        conf=conf,
    )

    metadata = {"model": model}
    return trainer, metadata


