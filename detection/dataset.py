import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

from datasets import FDDBDataset


class Dataset:
    @staticmethod
    def load(validation_split: float = 0.1, transform_to_tensors: bool = True, target_transform_to_tensors: bool = True, data_dir: str = None):
        transform = ToTensor() if transform_to_tensors else None
        target_transform = torch.tensor if target_transform_to_tensors else None
        train_dataset = FDDBDataset(data_dir, transform=transform, target_transform=target_transform)
        test_data = FDDBDataset(data_dir, split='test', transform=transform, target_transform=target_transform)
        train_data, val_data = random_split(train_dataset, (1 - validation_split, validation_split))
        return train_data, val_data, test_data

    @staticmethod
    def load_test(transform_to_tensors: bool = False, target_transform_to_tensors: bool = True, data_dir: str = None):
        transform = ToTensor() if transform_to_tensors else None
        target_transform = torch.tensor if target_transform_to_tensors else None
        test_data = FDDBDataset(data_dir, split='', transform=transform, target_transform=target_transform)
        return test_data

    @staticmethod
    def get_dataloaders(train_data, val_data, test_data, batch_size: int):
        train_dataloader = DataLoader(train_data, batch_size=batch_size)
        val_dataloader = DataLoader(val_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
        return train_dataloader, val_dataloader, test_dataloader
