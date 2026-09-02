"""
Dataset loading utilities for distributed training
Supports various data formats and distributed sampling
"""

import os
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from ..trainer import DistributedEvalSampler


class BaseDataset(Dataset, ABC):
    """Abstract base class for datasets"""

    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class ImageClassificationDataset(BaseDataset):
    """Dataset for image classification tasks"""

    def __init__(
        self, image_paths: List[str], labels: List[int], transform: Optional[Callable] = None
    ):
        super().__init__(transform)
        self.image_paths = image_paths
        self.labels = labels

        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        # Apply transforms, or provide a collatable tensor by default.
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.functional.to_tensor(image)

        # Get label
        label = self.labels[idx]

        return image, label


class DefectDetectionDataset(ImageClassificationDataset):
    """Dataset specifically for defect detection in manufacturing"""

    def __init__(self, data_dir: str, split: str = "train", transform: Optional[Callable] = None):
        """
        Expected directory structure:
        data_dir/
        ├── train/
        │   ├── good/  # label 0
        │   └── defective/  # label 1
        └── val/
            ├── good/
            └── defective/
        """

        self.data_dir = data_dir
        self.split = split

        # Collect image paths and labels
        image_paths = []
        labels = []

        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")

        # Good samples (label 0)
        good_dir = os.path.join(split_dir, "good")
        if os.path.exists(good_dir):
            for img_file in sorted(os.listdir(good_dir)):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(good_dir, img_file))
                    labels.append(0)

        # Defective samples (label 1)
        defective_dir = os.path.join(split_dir, "defective")
        if os.path.exists(defective_dir):
            for img_file in sorted(os.listdir(defective_dir)):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(defective_dir, img_file))
                    labels.append(1)

        super().__init__(image_paths, labels, transform)


class TimeSeriesDataset(BaseDataset):
    """Dataset for time series data"""

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
        stride: int = 1,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            data: Time series data of shape (num_samples, num_features)
            targets: Target values of shape (num_samples,) or (num_samples, num_targets)
            sequence_length: Length of sequences to extract
            stride: Stride for sequence extraction
        """
        super().__init__(transform)
        if len(data) != len(targets):
            raise ValueError("data and targets must have the same length")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero")
        if sequence_length >= len(data):
            raise ValueError("sequence_length must be smaller than the number of samples")
        if stride <= 0:
            raise ValueError("stride must be greater than zero")
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
        self.stride = stride

        # Extract sequences
        self.sequences = []
        self.sequence_targets = []

        for i in range(0, len(data) - sequence_length, stride):
            seq = data[i : i + sequence_length]
            target = targets[i + sequence_length]
            self.sequences.append(seq)
            self.sequence_targets.append(target)

        self.sequences = np.array(self.sequences)
        self.sequence_targets = np.array(self.sequence_targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.sequence_targets[idx]

        # Convert to torch tensors
        sequence = torch.FloatTensor(sequence)
        target = torch.tensor([target] if np.isscalar(target) else target, dtype=torch.float32)

        if self.transform:
            sequence, target = self.transform(sequence, target)

        return sequence, target


class CSVDataset(BaseDataset):
    """Generic dataset from CSV files"""

    def __init__(
        self,
        csv_path: str,
        feature_cols: List[str],
        target_col: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__(transform)

        # Load data
        self.df = pd.read_csv(csv_path)
        self.feature_cols = feature_cols
        self.target_col = target_col

        # Extract features and targets
        self.features = self.df[feature_cols].values
        self.targets = self.df[target_col].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.targets[idx]

        # Keep integer targets suitable for classification and normalize floating data to float32.
        features = torch.as_tensor(features, dtype=torch.float32)
        target = torch.as_tensor(target)
        if target.is_floating_point():
            target = target.float()
            if target.ndim == 0:
                target = target.unsqueeze(0)
        else:
            target = target.long()

        if self.transform:
            features, target = self.transform(features, target)

        return features, target


def get_image_transforms(img_size: int = 224, augment: bool = True) -> transforms.Compose:
    """Get image transforms for training/augmentation"""

    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        # Validation/test transforms
        transform = transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.14)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transform


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False,
    distributed_evaluation: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    drop_last: bool = False,
) -> DataLoader:
    """Create data loader with optional distributed sampling"""

    if distributed:
        if (rank is None) != (world_size is None):
            raise ValueError("rank and world_size must be provided together")
        if rank is None and world_size is None:
            if not dist.is_initialized():
                raise RuntimeError("rank and world_size are required before DDP is initialized")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        if distributed_evaluation:
            sampler = DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
        else:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
            )
        shuffle = False  # Shuffle is handled by DistributedSampler
    else:
        sampler = None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )

    return data_loader


def create_defect_detection_loaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    distributed: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation loaders for defect detection"""

    # Training dataset with augmentation
    train_transform = get_image_transforms(img_size, augment=True)
    train_dataset = DefectDetectionDataset(data_dir, "train", train_transform)

    # Validation dataset without augmentation
    val_transform = get_image_transforms(img_size, augment=False)
    val_dataset = DefectDetectionDataset(data_dir, "val", val_transform)

    # Create loaders
    train_loader = create_data_loader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    val_loader = create_data_loader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        distributed=distributed,
        distributed_evaluation=True,
        rank=rank,
        world_size=world_size,
    )

    return train_loader, val_loader


def create_time_series_loaders(
    csv_path: str,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 4,
    distributed: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation loaders for time series data"""

    # Load data
    df = pd.read_csv(csv_path)
    data = df[feature_cols].values
    targets = df[target_col].values

    # Split data
    if not 0 < train_split < 1:
        raise ValueError("train_split must be between zero and one")
    split_idx = int(len(data) * train_split)
    if split_idx <= sequence_length:
        raise ValueError("the training split must contain more samples than sequence_length")
    train_data = data[:split_idx]
    train_targets = targets[:split_idx]
    val_start = split_idx - sequence_length
    val_data = data[val_start:]
    val_targets = targets[val_start:]

    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, train_targets, sequence_length)
    val_dataset = TimeSeriesDataset(val_data, val_targets, sequence_length)

    # Create loaders
    train_loader = create_data_loader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    val_loader = create_data_loader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        distributed=distributed,
        distributed_evaluation=True,
        rank=rank,
        world_size=world_size,
    )

    return train_loader, val_loader
