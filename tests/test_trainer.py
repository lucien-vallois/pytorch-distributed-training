from __future__ import annotations

import math
from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from pytorch_distributed_training import (
    DistributedEvalSampler,
    Trainer,
    TrainingConfig,
    create_trainer,
)


def make_loader(sample_count: int = 20, batch_size: int = 5) -> DataLoader:
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(sample_count, 4, generator=generator)
    targets = (features[:, 0] > 0).long()
    return DataLoader(TensorDataset(features, targets), batch_size=batch_size)


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def assert_nested_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_nested_equal(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


class StatefulSchedule:
    def __call__(self, epoch: int) -> float:
        if epoch >= 1:
            self.late_epoch = epoch
        return 1.0


def test_public_api_and_config_defaults() -> None:
    config = TrainingConfig()
    trainer = create_trainer("local", config, make_model())

    assert isinstance(trainer, Trainer)
    assert config.backend == "local"
    assert config.max_epochs == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "ray"),
        ("learning_rate", 0),
        ("learning_rate", float("nan")),
        ("weight_decay", -1),
        ("weight_decay", float("inf")),
        ("max_epochs", 0),
        ("max_epochs", 1.5),
        ("gradient_accumulation_steps", 0),
        ("gradient_accumulation_steps", True),
        ("checkpoint_every", 1.5),
        ("scheduler_interval", "batch"),
        ("log_interval", 1.5),
    ],
)
def test_config_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**{field: value})


def test_local_training_and_checkpoint_round_trip(tmp_path) -> None:
    config = TrainingConfig(
        max_epochs=2,
        gradient_accumulation_steps=3,
        checkpoint_dir=str(tmp_path),
    )
    trainer = create_trainer("local", config, make_model())
    history = trainer.fit(make_loader())

    assert len(history) == 2
    assert trainer.global_step == 4
    assert all(math.isfinite(record["train_loss"]) for record in history)

    checkpoint_path = trainer.save_checkpoint()
    restored = create_trainer("local", config, make_model())
    restored.load_checkpoint(checkpoint_path)

    assert restored.epoch == trainer.epoch
    assert restored.global_step == trainer.global_step
    for expected, actual in zip(
        trainer._unwrapped_model().parameters(), restored._unwrapped_model().parameters()
    ):
        assert torch.equal(expected, actual)


def test_checkpoint_load_rolls_back_and_rejects_invalid_scheduler_state(tmp_path) -> None:
    config = TrainingConfig(checkpoint_dir=str(tmp_path))
    trainer = create_trainer(
        "local",
        config,
        make_model(),
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
    )
    trainer.setup_distributed()
    checkpoint = torch.load(trainer.save_checkpoint(), weights_only=True)
    initial_model = deepcopy(trainer.model.state_dict())
    initial_optimizer = deepcopy(trainer.optimizer.state_dict())
    initial_scheduler = deepcopy(trainer.scheduler.state_dict())

    checkpoint["model_state_dict"] = {
        key: value + 1 for key, value in checkpoint["model_state_dict"].items()
    }
    checkpoint["model_state_dict"].pop(next(reversed(checkpoint["model_state_dict"])))
    invalid_model_path = tmp_path / "invalid-model.pt"
    torch.save(checkpoint, invalid_model_path)

    with pytest.raises(RuntimeError):
        trainer.load_checkpoint(invalid_model_path)

    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, initial_model[key])
    assert_nested_equal(trainer.optimizer.state_dict(), initial_optimizer)
    assert trainer.scheduler.state_dict() == initial_scheduler
    assert (trainer.epoch, trainer.global_step, trainer.best_loss) == (0, 0, float("inf"))

    checkpoint = torch.load(trainer.save_checkpoint(), weights_only=True)
    checkpoint["optimizer_state_dict"]["param_groups"] = []
    invalid_optimizer_path = tmp_path / "invalid-optimizer.pt"
    torch.save(checkpoint, invalid_optimizer_path)

    with pytest.raises(ValueError, match="parameter groups"):
        trainer.load_checkpoint(invalid_optimizer_path)

    checkpoint = torch.load(trainer.save_checkpoint(), weights_only=True)
    checkpoint["scheduler_state_dict"]["last_epoch"] = "broken"
    invalid_scheduler_path = tmp_path / "invalid-scheduler.pt"
    torch.save(checkpoint, invalid_scheduler_path)

    with pytest.raises(ValueError, match="scheduler_state_dict"):
        trainer.load_checkpoint(invalid_scheduler_path)

    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, initial_model[key])


def test_checkpoint_accepts_state_added_by_lambda_scheduler(tmp_path) -> None:
    config = TrainingConfig(checkpoint_dir=str(tmp_path))
    scheduler_factory = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
        optimizer, StatefulSchedule()
    )
    trainer = create_trainer(
        "local",
        config,
        make_model(),
        scheduler_factory=scheduler_factory,
    )
    trainer.fit(make_loader())
    checkpoint_path = trainer.save_checkpoint()

    restored = create_trainer(
        "local",
        config,
        make_model(),
        scheduler_factory=scheduler_factory,
    )
    restored.load_checkpoint(checkpoint_path)

    assert restored.epoch == trainer.epoch
    assert restored.scheduler.state_dict() == trainer.scheduler.state_dict()


def test_multistep_scheduler_checkpoint_round_trip(tmp_path) -> None:
    config = TrainingConfig(checkpoint_dir=str(tmp_path))
    scheduler_factory = lambda optimizer: torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[1, 3]
    )
    trainer = create_trainer(
        "local",
        config,
        make_model(),
        scheduler_factory=scheduler_factory,
    )
    trainer.fit(make_loader())

    restored = create_trainer(
        "local",
        config,
        make_model(),
        scheduler_factory=scheduler_factory,
    )
    restored.load_checkpoint(trainer.save_checkpoint())
    trainer.scheduler.step()
    restored.scheduler.step()

    assert restored.scheduler.state_dict() == trainer.scheduler.state_dict()
    assert restored.optimizer.param_groups[0]["lr"] == trainer.optimizer.param_groups[0]["lr"]


def test_checkpoint_rejects_incompatible_optimizer_tensor_before_mutation(tmp_path) -> None:
    config = TrainingConfig(checkpoint_dir=str(tmp_path))
    trainer = create_trainer("local", config, make_model())
    trainer.fit(make_loader())
    checkpoint = torch.load(trainer.save_checkpoint(), weights_only=True)
    initial_model = deepcopy(trainer.model.state_dict())
    initial_optimizer = deepcopy(trainer.optimizer.state_dict())
    initial_progress = (trainer.epoch, trainer.global_step, trainer.best_loss)

    checkpoint["model_state_dict"] = {
        key: value + 1 for key, value in checkpoint["model_state_dict"].items()
    }
    checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] = 9.0
    parameter_state = next(
        state
        for state in checkpoint["optimizer_state_dict"]["state"].values()
        if "exp_avg" in state
    )
    exp_avg = parameter_state["exp_avg"]
    parameter_state["exp_avg"] = torch.zeros(
        exp_avg.numel() + 1, dtype=exp_avg.dtype, device=exp_avg.device
    )
    invalid_path = tmp_path / "invalid-optimizer-tensor.pt"
    torch.save(checkpoint, invalid_path)

    with pytest.raises(ValueError, match="exp_avg.*shape"):
        trainer.load_checkpoint(invalid_path)

    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, initial_model[key])
    assert_nested_equal(trainer.optimizer.state_dict(), initial_optimizer)
    assert (trainer.epoch, trainer.global_step, trainer.best_loss) == initial_progress


def test_factory_rejects_backend_mismatch() -> None:
    with pytest.raises(ValueError, match="must match"):
        create_trainer("ddp", TrainingConfig(backend="local"), make_model())


def test_empty_loader_is_rejected() -> None:
    empty = DataLoader(
        TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)),
        batch_size=4,
    )
    trainer = create_trainer("local", TrainingConfig(), make_model())
    with pytest.raises(ValueError, match="at least one batch"):
        trainer.train_epoch(empty)


def test_iterable_dataset_supports_partial_accumulation() -> None:
    class Stream(IterableDataset):
        def __iter__(self):
            for value in range(5):
                features = torch.tensor([float(value), 0.0, 0.0, 0.0])
                yield features, torch.tensor(value % 2)

    loader = DataLoader(Stream(), batch_size=2)
    trainer = create_trainer(
        "local",
        TrainingConfig(gradient_accumulation_steps=2),
        make_model(),
    )

    metrics = trainer.train_epoch(loader)

    assert metrics["samples"] == 5
    assert trainer.global_step == 2


def test_module_loss_is_moved_to_the_training_device(monkeypatch) -> None:
    loss_fn = nn.CrossEntropyLoss(weight=torch.ones(2))
    original_to = loss_fn.to
    calls = []

    def record_to(*args, **kwargs):
        calls.append((args, kwargs))
        return original_to(*args, **kwargs)

    monkeypatch.setattr(loss_fn, "to", record_to)
    trainer = create_trainer("local", TrainingConfig(device="cpu"), make_model(), loss_fn=loss_fn)
    trainer.setup_distributed()

    assert calls == [((torch.device("cpu"),), {})]
    assert loss_fn.weight.device == trainer.device


def test_module_loss_tracks_train_and_evaluation_mode() -> None:
    class DropoutLoss(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = nn.Dropout(p=1)

        def forward(self, outputs, targets):
            return self.dropout((outputs - targets) ** 2).mean()

    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    loss_fn = DropoutLoss()
    loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.ones(2, 1)),
        batch_size=2,
    )
    trainer = create_trainer("local", TrainingConfig(), model, loss_fn=loss_fn)

    metrics = trainer.evaluate(loader)
    assert metrics["loss"] == 1
    assert not loss_fn.training

    trainer.train_epoch(loader)
    assert loss_fn.training


def test_batch_size_comes_from_targets_not_auxiliary_inputs() -> None:
    class ContextModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, context, features):
            return self.linear(features) + context.sum() * 0

    batch = (
        {"context": torch.eye(3), "features": torch.randn(2, 4)},
        torch.tensor([0, 1]),
    )
    trainer = create_trainer("local", TrainingConfig(), ContextModel())

    assert trainer.evaluate([batch])["samples"] == 2


def test_bare_cuda_selects_device_zero_for_local_training(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer = create_trainer("local", TrainingConfig(device="cuda"), make_model())

    assert trainer._select_device() == torch.device("cuda", 0)


def test_unsupported_optimizer_and_accumulation_loss_fail_clearly() -> None:
    lbfgs = create_trainer(
        "local",
        TrainingConfig(),
        make_model(),
        optimizer_factory=lambda parameters: torch.optim.LBFGS(parameters),
    )
    with pytest.raises(ValueError, match="LBFGS"):
        lbfgs.setup_distributed()

    summed_loss = create_trainer(
        "local",
        TrainingConfig(),
        nn.Linear(1, 1),
        loss_fn=nn.MSELoss(reduction="sum"),
    )
    regression_loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.ones(2, 1)),
        batch_size=2,
    )
    with pytest.raises(ValueError, match="mean or batchmean"):
        summed_loss.evaluate(regression_loader)


def test_non_finite_loss_fails_before_parameters_are_updated() -> None:
    model = nn.Linear(1, 1)
    trainer = create_trainer(
        "local",
        TrainingConfig(),
        model,
        loss_fn=lambda outputs, targets: outputs.mean() * float("nan"),
    )
    loader = DataLoader(
        TensorDataset(torch.ones(1, 1), torch.ones(1, 1)),
        batch_size=1,
    )
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    with pytest.raises(ValueError, match="non-finite"):
        trainer.evaluate(loader)
    with pytest.raises(ValueError, match="non-finite"):
        trainer.train_epoch(loader)

    assert trainer.global_step == 0
    for initial, current in zip(initial_parameters, model.parameters()):
        assert torch.equal(initial, current)


def test_step_scheduler_advances_with_optimizer_steps() -> None:
    trainer = create_trainer(
        "local",
        TrainingConfig(scheduler_interval="step"),
        make_model(),
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            total_steps=4,
        ),
    )

    trainer.fit(make_loader())

    assert trainer.global_step == 4
    assert trainer.scheduler.last_epoch == 4


def test_gradient_accumulation_weights_partial_batches_by_sample() -> None:
    features = torch.arange(10, dtype=torch.float32).reshape(-1, 1) / 10
    targets = features * 2
    dataset = TensorDataset(features, targets)
    initial_model = nn.Linear(1, 1)

    def optimizer_factory(parameters):
        return torch.optim.SGD(parameters, lr=0.1)

    def scheduler_factory(optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    def plateau_factory(optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    full_batch = create_trainer(
        "local",
        TrainingConfig(max_epochs=1),
        deepcopy(initial_model),
        loss_fn=nn.MSELoss(),
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )
    accumulated = create_trainer(
        "local",
        TrainingConfig(max_epochs=1, gradient_accumulation_steps=3),
        deepcopy(initial_model),
        loss_fn=nn.MSELoss(),
        optimizer_factory=optimizer_factory,
        scheduler_factory=plateau_factory,
    )

    full_batch.fit(DataLoader(dataset, batch_size=10))
    accumulated.fit(DataLoader(dataset, batch_size=4))

    assert isinstance(accumulated.optimizer, torch.optim.SGD)
    assert isinstance(accumulated.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    for expected, actual in zip(full_batch.model.parameters(), accumulated.model.parameters()):
        assert torch.allclose(expected, actual, atol=1e-7)


def test_weighted_cross_entropy_is_exact_across_batches_and_accumulation() -> None:
    features = torch.tensor([[2.0, -1.0], [-0.5, 1.0], [1.0, 0.5], [-1.0, 2.0], [0.5, 1.5]])
    targets = torch.tensor([0, 1, 1, 0, 1])
    dataset = TensorDataset(features, targets)
    initial_model = nn.Linear(2, 2)

    def optimizer_factory(parameters):
        return torch.optim.SGD(parameters, lr=0.1)

    full_batch = create_trainer(
        "local",
        TrainingConfig(),
        deepcopy(initial_model),
        loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0])),
        optimizer_factory=optimizer_factory,
    )
    accumulated = create_trainer(
        "local",
        TrainingConfig(gradient_accumulation_steps=3),
        deepcopy(initial_model),
        loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0])),
        optimizer_factory=optimizer_factory,
    )

    full_batch.fit(DataLoader(dataset, batch_size=5))
    accumulated.fit(DataLoader(dataset, batch_size=2))

    for expected, actual in zip(full_batch.model.parameters(), accumulated.model.parameters()):
        assert torch.allclose(expected, actual, atol=1e-7)


def test_weighted_ignored_cross_entropy_evaluation_matches_full_batch() -> None:
    features = torch.tensor([[2.0, -1.0], [-0.5, 1.0], [1.0, 0.5], [-1.0, 2.0]])
    targets = torch.tensor([0, 1, -100, 1])
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]))
    trainer = create_trainer("local", TrainingConfig(), model, loss_fn=loss_fn)

    metrics = trainer.evaluate(DataLoader(TensorDataset(features, targets), batch_size=2))
    expected = loss_fn(model(features), targets).item()

    assert metrics["loss"] == pytest.approx(expected)
    assert metrics["samples"] == 4


def test_fully_ignored_batch_is_skipped_before_a_valid_optimizer_step() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([-100, 1])
    trainer = create_trainer(
        "local",
        TrainingConfig(),
        nn.Linear(2, 2),
        loss_fn=nn.CrossEntropyLoss(),
    )

    metrics = trainer.train_epoch(DataLoader(TensorDataset(features, targets), batch_size=1))

    assert math.isfinite(metrics["loss"])
    assert trainer.global_step == 1


def test_fully_ignored_train_loader_fails_clearly() -> None:
    loader = DataLoader(
        TensorDataset(torch.ones(2, 2), torch.full((2,), -100)),
        batch_size=1,
    )
    trainer = create_trainer(
        "local",
        TrainingConfig(),
        nn.Linear(2, 2),
        loss_fn=nn.CrossEntropyLoss(),
    )

    with pytest.raises(ValueError, match="train loader"):
        trainer.train_epoch(loader)


def test_batchmean_loss_is_exact_with_accumulation() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]])
    targets = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.3, 0.7]])
    dataset = TensorDataset(features, targets)
    initial_model = nn.Sequential(nn.Linear(2, 2), nn.LogSoftmax(dim=1))

    def optimizer_factory(parameters):
        return torch.optim.SGD(parameters, lr=0.1)

    full_batch = create_trainer(
        "local",
        TrainingConfig(),
        deepcopy(initial_model),
        loss_fn=nn.KLDivLoss(reduction="batchmean"),
        optimizer_factory=optimizer_factory,
    )
    accumulated = create_trainer(
        "local",
        TrainingConfig(gradient_accumulation_steps=2),
        deepcopy(initial_model),
        loss_fn=nn.KLDivLoss(reduction="batchmean"),
        optimizer_factory=optimizer_factory,
    )

    full_batch.fit(DataLoader(dataset, batch_size=4))
    accumulated.fit(DataLoader(dataset, batch_size=2))

    for expected, actual in zip(full_batch.model.parameters(), accumulated.model.parameters()):
        assert torch.allclose(expected, actual, atol=1e-7)


def test_distributed_eval_sampler_does_not_duplicate_padding() -> None:
    dataset = TensorDataset(torch.arange(5))
    rank_zero = list(DistributedEvalSampler(dataset, num_replicas=2, rank=0))
    rank_one = list(DistributedEvalSampler(dataset, num_replicas=2, rank=1))

    assert rank_zero == [0, 2, 4]
    assert rank_one == [1, 3]
    assert sorted(rank_zero + rank_one) == list(range(len(dataset)))
