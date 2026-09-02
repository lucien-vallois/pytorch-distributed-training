from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from pytorch_distributed_training import (
    DistributedEvalSampler,
    TrainingConfig,
    create_trainer,
)


def _ddp_worker(rank: int, world_size: int, rendezvous_uri: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=rendezvous_uri,
        rank=rank,
        world_size=world_size,
    )
    try:
        generator = torch.Generator().manual_seed(21)
        features = torch.randn(16, 4, generator=generator)
        targets = (features[:, 0] > 0).long()
        dataset = TensorDataset(features, targets)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(dataset, batch_size=4, sampler=sampler)

        torch.manual_seed(21)
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
        config = TrainingConfig(backend="ddp", max_epochs=1, device="cpu")
        trainer = create_trainer("ddp", config, model)
        history = trainer.fit(loader)

        eval_dataset = TensorDataset(features[:5], targets[:5])
        eval_sampler = DistributedEvalSampler(eval_dataset, num_replicas=world_size, rank=rank)
        eval_loader = DataLoader(eval_dataset, batch_size=2, sampler=eval_sampler)
        eval_metrics = trainer.evaluate(eval_loader)

        assert trainer.get_world_size() == world_size
        assert history[0]["train_loss"] > 0
        assert eval_metrics["samples"] == 5

        torch.manual_seed(31)
        uneven_count = 2 if rank == 0 else 1
        uneven_start = rank * 2
        uneven_dataset = TensorDataset(
            features[uneven_start : uneven_start + uneven_count],
            targets[uneven_start : uneven_start + uneven_count],
        )
        uneven_loader = DataLoader(uneven_dataset, batch_size=2)
        uneven_trainer = create_trainer(
            "ddp",
            config,
            nn.Linear(4, 2),
            optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
        )
        uneven_trainer.train_epoch(uneven_loader)
        parameters = torch.cat(
            [parameter.detach().flatten() for parameter in uneven_trainer.model.parameters()]
        )
        gathered_parameters = [torch.zeros_like(parameters) for _ in range(world_size)]
        dist.all_gather(gathered_parameters, parameters)
        assert torch.allclose(gathered_parameters[0], gathered_parameters[1])

        torch.manual_seed(41)
        ignored_targets = torch.tensor([-100] if rank == 0 else [1])
        ignored_loader = DataLoader(
            TensorDataset(features[rank : rank + 1], ignored_targets),
            batch_size=1,
        )
        ignored_trainer = create_trainer(
            "ddp",
            config,
            nn.Linear(4, 2),
            optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
        )
        ignored_metrics = ignored_trainer.train_epoch(ignored_loader)
        assert ignored_metrics["samples"] == 2
        assert torch.isfinite(torch.tensor(ignored_metrics["loss"]))
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.skipif(
    os.environ.get("RUN_DDP_TESTS") != "1",
    reason="set RUN_DDP_TESTS=1 to run the two-process DDP smoke test",
)
def test_two_process_cpu_ddp(tmp_path: Path) -> None:
    rendezvous_uri = (tmp_path / "ddp-rendezvous").as_uri()
    mp.spawn(_ddp_worker, args=(2, rendezvous_uri), nprocs=2, join=True)
