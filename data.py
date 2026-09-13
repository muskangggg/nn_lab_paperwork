"""
CIFAR-10 loading.

Matches the original repo's preprocessing: it only applies
transforms.ToTensor() (no normalization, no augmentation) in
load_vision_dataset.py / run_conv.py. We keep that for a faithful
"standard conditions" replication.

If you want to push accuracy up as a secondary experiment (NOT part of the
core replication claim), the commented-out `train_transform` below adds the
standard CIFAR-10 mean/std normalization + random crop/flip augmentation --
mention clearly in your paper if you switch this on, since it's a deviation
from the original's setup.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(data_dir: str = "./data", batch_size: int = 128, num_workers: int = 2):
    transform = transforms.Compose([transforms.ToTensor()])

    # Optional augmented version (kept off by default -- see docstring above):
    # transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    # ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
