from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pytorch_distributed_training.utils import (
    CheckpointManager,
    MetricsTracker,
    TrainingMonitor,
)


def test_checkpoint_manager_round_trip(tmp_path, monkeypatch) -> None:
    model = nn.Linear(3, 1)
    checkpoint = {"model_state_dict": model.state_dict(), "epoch": 2, "loss": 0.25}
    manager = CheckpointManager(str(tmp_path))

    saved_path = manager.save_checkpoint(checkpoint, step=2)
    loaded = manager.load_checkpoint(saved_path)

    assert loaded is not None
    assert loaded["epoch"] == 2
    assert loaded["loss"] == 0.25

    export_path = tmp_path / "inference.pt"
    manager.export_for_inference(model, {"inputs": 3}, str(export_path), step=2)
    exported = torch.load(export_path, weights_only=True)
    assert exported["config"] == {"inputs": 3}
    assert type(exported["framework_version"]) is str

    def interrupted_save(value, path) -> None:
        Path(path).write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted"):
        manager.save_checkpoint({"epoch": 3}, step=3)

    assert manager.load_checkpoint()["epoch"] == 2
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_prefix_with_underscores_keeps_step(tmp_path) -> None:
    manager = CheckpointManager(str(tmp_path))
    manager.save_best_model(nn.Linear(2, 1), step=12)

    latest = manager.get_latest_checkpoint_info(prefix="best_model")
    assert latest is not None
    assert latest["step"] == 12


def test_custom_checkpoint_prefix_respects_retention(tmp_path) -> None:
    manager = CheckpointManager(str(tmp_path), max_checkpoints=2)
    for step in (1, 2, 3):
        manager.save_checkpoint({"step": step}, step=step, prefix="model_state")

    assert len(list(tmp_path.glob("model_state_*.pt"))) == 2


def test_checkpoint_retention_keeps_the_most_recent_save(tmp_path) -> None:
    manager = CheckpointManager(str(tmp_path), max_checkpoints=1)
    old_path = manager.save_checkpoint({"version": "old"}, step=10)
    new_path = manager.save_checkpoint({"version": "new"}, step=5)

    assert not Path(old_path).exists()
    assert Path(new_path).exists()


def test_latest_checkpoint_breaks_step_ties_by_timestamp(tmp_path) -> None:
    manager = CheckpointManager(str(tmp_path))
    first = manager.save_checkpoint({"version": "old"}, step=5)
    second = manager.save_checkpoint({"version": "new"}, step=5)

    assert first != second
    assert manager.load_checkpoint()["version"] == "new"
    assert [item["path"] for item in manager.list_checkpoints()] == [second, first]


def test_checkpoint_rejects_unsafe_payload_and_negative_step(tmp_path) -> None:
    manager = CheckpointManager(str(tmp_path))

    class CustomDict(dict):
        pass

    class Mode(IntEnum):
        TRAIN = 1

    with pytest.raises(TypeError, match="unsupported type"):
        manager.save_checkpoint({"model": nn.Linear(2, 1)}, step=0)
    with pytest.raises(TypeError, match="unsupported type"):
        manager.save_checkpoint(CustomDict(value=1), step=0)
    with pytest.raises(TypeError, match="unsupported type"):
        manager.save_checkpoint({"mode": Mode.TRAIN}, step=0)
    with pytest.raises(ValueError, match="non-negative integer"):
        manager.save_checkpoint({"value": 1}, step=-1)
    with pytest.raises(ValueError, match="prefix"):
        manager.save_checkpoint({"value": 1}, step=0, prefix="../outside")

    assert list(tmp_path.iterdir()) == []


def test_metrics_export_is_json_serializable(tmp_path) -> None:
    tracker = MetricsTracker()
    tracker.update_batch("train", loss=0.5, accuracy=0.75, step=1)
    output_path = tmp_path / "metrics.json"

    tracker.export_metrics(str(output_path))
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert exported["metrics"]["train/loss"] == 0.5
    assert exported["metrics"]["train/accuracy_stats"]["mean"] == 0.75


def test_metrics_rejects_empty_window() -> None:
    with pytest.raises(ValueError, match="window_size"):
        MetricsTracker(window_size=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_metrics_reject_non_finite_values_before_recording(value: float) -> None:
    tracker = MetricsTracker(use_prometheus=True)

    with pytest.raises(ValueError, match="finite"):
        tracker.update("train/loss", value)

    assert tracker.get_all_metrics() == {}
    assert (
        tracker.get_prometheus_registry().get_sample_value(
            "training_loss_count", {"phase": "train"}
        )
        is None
    )


def test_prometheus_epoch_and_reset() -> None:
    tracker = MetricsTracker(use_prometheus=True)
    registry = tracker.get_prometheus_registry()
    tracker.update_epoch(3, {"loss": 0.5})
    tracker.update_batch("train", loss=0.4, step=1)

    assert registry.get_sample_value("current_epoch") == 3
    assert registry.get_sample_value("training_steps_total") == 1

    tracker.reset()
    reset_registry = tracker.get_prometheus_registry()
    assert reset_registry is registry
    assert reset_registry.get_sample_value("current_epoch") == 0
    assert tracker.get_all_metrics() == {}


def test_training_monitor_alerts_are_descriptive() -> None:
    tracker = MetricsTracker()
    for loss in [0.1] * 5 + [0.5] * 5:
        tracker.update("train/loss", loss)

    alerts = TrainingMonitor(tracker, {"memory_threshold_mb": 1024}).check_alerts()
    assert alerts and "Loss changed" in alerts[0]
