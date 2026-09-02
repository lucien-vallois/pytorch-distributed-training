from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

pd = pytest.importorskip("pandas")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("torchvision")

from pytorch_distributed_training import TrainingConfig, create_trainer
from pytorch_distributed_training.data import (
    CSVDataset,
    ImageClassificationDataset,
    create_data_loader,
    create_time_series_loaders,
)


def test_image_dataset_without_transform_is_collatable(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (3, 2), color=(10, 20, 30)).save(image_path)
    dataset = ImageClassificationDataset([str(image_path)], [1])

    images, labels = next(iter(DataLoader(dataset, batch_size=1)))

    assert images.shape == (1, 3, 2, 3)
    assert images.dtype == torch.float32
    assert labels.tolist() == [1]


def test_csv_integer_target_works_with_default_classification_loss(tmp_path) -> None:
    csv_path = tmp_path / "classification.csv"
    pd.DataFrame(
        {
            "first": [1.0, -1.0, 0.5, -0.5],
            "second": [0.0, 1.0, -0.5, 0.5],
            "label": [1, 0, 1, 0],
        }
    ).to_csv(csv_path, index=False)
    loader = DataLoader(CSVDataset(str(csv_path), ["first", "second"], "label"), batch_size=2)
    trainer = create_trainer("local", TrainingConfig(), nn.Linear(2, 2))

    metrics = trainer.train_epoch(loader)

    assert metrics["samples"] == 4
    assert trainer.global_step == 2


def test_csv_float_target_keeps_regression_column_shape(tmp_path) -> None:
    csv_path = tmp_path / "regression.csv"
    pd.DataFrame(
        {
            "first": [1.0, -1.0],
            "second": [0.0, 1.0],
            "target": [0.25, -0.5],
        }
    ).to_csv(csv_path, index=False)
    loader = DataLoader(CSVDataset(str(csv_path), ["first", "second"], "target"), batch_size=2)
    _, targets = next(iter(loader))
    trainer = create_trainer(
        "local",
        TrainingConfig(),
        nn.Linear(2, 1),
        loss_fn=nn.MSELoss(),
    )

    metrics = trainer.train_epoch(loader)

    assert targets.shape == (2, 1)
    assert metrics["samples"] == 2


def test_time_series_validation_uses_training_tail_as_context(tmp_path) -> None:
    csv_path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "feature": np.arange(8, dtype=np.float32),
            "target": np.arange(8, dtype=np.float32),
        }
    ).to_csv(csv_path, index=False)

    train_loader, val_loader = create_time_series_loaders(
        str(csv_path),
        ["feature"],
        "target",
        sequence_length=2,
        batch_size=2,
        train_split=0.75,
        num_workers=0,
    )

    assert len(train_loader.dataset) == 4
    assert len(val_loader.dataset) == 2
    assert val_loader.dataset.sequence_targets.tolist() == [6.0, 7.0]


def test_data_loader_selects_no_padding_evaluation_sampler() -> None:
    dataset = TensorDataset(torch.arange(5))

    loader = create_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        distributed=True,
        distributed_evaluation=True,
        rank=1,
        world_size=2,
    )

    assert list(loader.sampler) == [1, 3]
