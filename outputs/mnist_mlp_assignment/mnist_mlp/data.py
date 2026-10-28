import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import BATCH_SIZE, DATA_DIR, SEED, TRAIN_SIZE, VALIDATION_SIZE


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data_loaders():
    """MNIST'i eğitim/doğrulama/test olarak ayırıp DataLoader'ları döndürür."""
    transform = transforms.ToTensor()  # [0, 255] -> [0, 1]
    full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_data, validation_data = random_split(
        full_train,
        [TRAIN_SIZE, VALIDATION_SIZE],
        generator=torch.Generator().manual_seed(SEED),
    )
    loaders = {
        "train": DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True),
        "train_eval": DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False),
        "validation": DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False),
        "test": DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False),
    }
    sizes = {"train": len(train_data), "validation": len(validation_data), "test": len(test_data)}
    return loaders, sizes
