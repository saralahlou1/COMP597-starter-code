import src.config as config
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

data_load_name="FakeImageNet"

def load_data(conf: config.Config) -> torch.utils.data.Dataset:
    print("loading FakeImageNet")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
# /home/slurm/comp597/students/slahlo4 "/home/2023/slahlo4/COMP597-starter-code/src/data/regnet/fakeimage/stylegan3/stylegan3-80K"
    trainset = torchvision.datasets.ImageFolder(
        root="/home/slurm/comp597/students/slahlo4/data/fakeimage/stylegan3/stylegan3-80K",  # path to FakeImageNet/train
        transform=transform,
    )

    return trainset

