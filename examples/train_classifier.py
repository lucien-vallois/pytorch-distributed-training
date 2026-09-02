"""Train a small classifier on synthetic data.

Run locally:
    python examples/train_classifier.py

Run with two DDP workers:
    torchrun --standalone --nproc-per-node=2 examples/train_classifier.py --backend ddp --device cpu

Windows builds without libuv require the file-store fallback documented in README.md.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from pytorch_distributed_training import TrainingConfig, create_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("local", "ddp"), default="local")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def make_dataset(sample_count: int) -> TensorDataset:
    if sample_count <= 0:
        raise ValueError("sample_count must be greater than zero")
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(sample_count, 4, generator=generator)
    targets = ((features[:, 0] + features[:, 1] * 0.5) > 0).long()
    return TensorDataset(features, targets)


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)

    config = TrainingConfig(
        backend=args.backend,
        learning_rate=0.03,
        max_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    trainer = create_trainer(args.backend, config, model)

    try:
        trainer.setup_distributed()
        dataset = make_dataset(args.samples)
        sampler = None
        if trainer.get_world_size() > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=trainer.get_world_size(),
                rank=trainer.get_rank(),
                shuffle=True,
            )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
        )
        history = trainer.fit(loader)
        if trainer.is_main_process():
            print(json.dumps(history[-1], sort_keys=True))
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
