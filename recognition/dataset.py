import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import datasets


class Dataset:
    @staticmethod
    def load_datasets(validation_split: float = 0.1, transform_to_tensors: bool = True):
        transform = ToTensor() if transform_to_tensors else None
        train_dataset = datasets.LFWPairs(
            root="../data",
            split="train",
            download=False,
            image_set='deepfunneled',
            transform=transform
        )
        test_data = datasets.LFWPairs(
            root="../data",
            split="test",
            download=False,
            image_set='deepfunneled',
            transform=transform
        )
        train_data, val_data = random_split(train_dataset, (1-validation_split, validation_split))
        return train_data, val_data, test_data

    @staticmethod
    def get_dataloaders(train_data, val_data, test_data, batch_size: int):
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
        return train_dataloader, val_dataloader, test_dataloader
