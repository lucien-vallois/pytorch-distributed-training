"""Training loop for local PyTorch and native DistributedDataParallel runs."""

from __future__ import annotations

import logging
import math
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sized, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from .utils.checkpointing import CheckpointManager

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
SchedulerFactory = Callable[[torch.optim.Optimizer], Any]


_PARAMETER_SHAPED_OPTIMIZER_STATES = frozenset(
    {
        "acc_delta",
        "ax",
        "exp_avg",
        "exp_avg_sq",
        "exp_inf",
        "grad_avg",
        "max_exp_avg_sq",
        "momentum_buffer",
        "prev",
        "square_avg",
        "step_size",
        "sum",
        "variance",
    }
)


def _validate_state_structure(
    expected: Any,
    candidate: Any,
    path: str,
    *,
    strict_dict_keys: bool = True,
) -> None:
    """Reject state that would only fail during a later optimizer or scheduler step."""

    if isinstance(expected, dict):
        if not isinstance(candidate, dict):
            raise ValueError(f"{path} has incompatible keys")
        expected_keys = set(expected)
        candidate_keys = set(candidate)
        keys_match = (
            candidate_keys == expected_keys if strict_dict_keys else expected_keys <= candidate_keys
        )
        if not keys_match:
            raise ValueError(f"{path} has incompatible keys")
        for key, value in expected.items():
            _validate_state_structure(
                value,
                candidate[key],
                f"{path}[{key!r}]",
                strict_dict_keys=False,
            )
        return
    if isinstance(expected, (list, tuple)):
        if type(candidate) is not type(expected) or len(candidate) != len(expected):
            raise ValueError(f"{path} has incompatible sequence structure")
        for index, value in enumerate(expected):
            _validate_state_structure(
                value,
                candidate[index],
                f"{path}[{index}]",
                strict_dict_keys=strict_dict_keys,
            )
        return
    if isinstance(expected, torch.Tensor):
        if (
            not isinstance(candidate, torch.Tensor)
            or candidate.shape != expected.shape
            or candidate.dtype != expected.dtype
        ):
            raise ValueError(f"{path} has incompatible tensor metadata")
        return
    if type(candidate) is not type(expected):
        raise ValueError(f"{path} has incompatible type")


def _validate_loaded_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Reject malformed standard tensor slots after PyTorch maps saved parameter IDs."""

    for parameter, parameter_state in optimizer.state.items():
        if not isinstance(parameter, torch.Tensor):
            continue
        if not isinstance(parameter_state, dict):
            raise ValueError("optimizer state has invalid structure")
        for state_name in _PARAMETER_SHAPED_OPTIMIZER_STATES & parameter_state.keys():
            value = parameter_state[state_name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise ValueError(f"optimizer state {state_name!r} has incompatible parameter shape")


@dataclass
class TrainingConfig:
    """Configuration shared by local and DDP training."""

    backend: str = "local"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 1
    gradient_accumulation_steps: int = 1
    checkpoint_dir: Optional[str] = None
    checkpoint_every: int = 1
    scheduler_interval: str = "epoch"
    device: str = "auto"
    log_interval: int = 0

    def __post_init__(self) -> None:
        if self.backend not in {"local", "ddp"}:
            raise ValueError("backend must be 'local' or 'ddp'")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and greater than zero")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if type(self.max_epochs) is not int or self.max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer")
        if (
            type(self.gradient_accumulation_steps) is not int
            or self.gradient_accumulation_steps <= 0
        ):
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        if type(self.checkpoint_every) is not int or self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be a positive integer")
        if self.scheduler_interval not in {"epoch", "step"}:
            raise ValueError("scheduler_interval must be 'epoch' or 'step'")
        if type(self.log_interval) is not int or self.log_interval < 0:
            raise ValueError("log_interval must be a non-negative integer")


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data across ranks without padding or duplication."""

    def __init__(
        self,
        dataset: Sized,
        *,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        if num_replicas is None:
            if not dist.is_initialized():
                raise RuntimeError("num_replicas is required before DDP is initialized")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_initialized():
                raise RuntimeError("rank is required before DDP is initialized")
            rank = dist.get_rank()
        if num_replicas <= 0:
            raise ValueError("num_replicas must be greater than zero")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be between zero and num_replicas - 1")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas


class Trainer:
    """Train a model from ``(inputs, targets)`` batches.

    Use ``backend="local"`` for one process. Use ``backend="ddp"`` under
    ``torchrun`` for native PyTorch DistributedDataParallel execution. A custom
    loss function must return a scalar mean over its batch so gradient
    accumulation can weight partial batches correctly.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        *,
        loss_fn: Optional[LossFunction] = None,
        optimizer_factory: Optional[OptimizerFactory] = None,
        scheduler_factory: Optional[SchedulerFactory] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self._optimizer_factory = optimizer_factory
        self._scheduler_factory = scheduler_factory
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        self._device: Optional[torch.device] = None
        self._rank = 0
        self._world_size = 1
        self._local_rank = 0
        self._setup_complete = False
        self._owns_process_group = False
        self._checkpoint_manager: Optional[CheckpointManager] = None
        self.logger = logging.getLogger(__name__)

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("trainer is not set up; call setup_distributed() first")
        return self._device

    def setup_distributed(self) -> None:
        """Initialize the requested execution backend and move the model."""

        if self._setup_complete:
            return

        if self.config.backend == "ddp":
            self._configure_ddp()

        self._device = self._select_device()
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)

        if self.config.backend == "ddp" and self._world_size > 1 and not dist.is_initialized():
            if self._device.type == "cuda" and dist.is_nccl_available():
                process_backend = "nccl"
            elif dist.is_gloo_available():
                process_backend = "gloo"
            else:
                raise RuntimeError("this PyTorch build has no usable DDP process backend")
            dist.init_process_group(backend=process_backend, init_method="env://")
            self._owns_process_group = True

        self.model.to(self._device)
        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.to(self._device)
        if self.config.backend == "ddp" and self._world_size > 1:
            device_ids = [self._local_rank] if self._device.type == "cuda" else None
            self.model = DistributedDataParallel(self.model, device_ids=device_ids)

        if self._optimizer_factory is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = self._optimizer_factory(self.model.parameters())
        if isinstance(self.optimizer, torch.optim.LBFGS):
            raise ValueError("LBFGS is not supported because its step requires a closure")
        if self._scheduler_factory is not None:
            self.scheduler = self._scheduler_factory(self.optimizer)
        if self.config.scheduler_interval == "step" and isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            raise ValueError("ReduceLROnPlateau requires scheduler_interval='epoch'")

        self._setup_complete = True

    def _configure_ddp(self) -> None:
        if dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
            self._local_rank = int(os.environ.get("LOCAL_RANK", self._rank))
            return

        self._rank = int(os.environ.get("RANK", "0"))
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self._local_rank = int(os.environ.get("LOCAL_RANK", self._rank))

    def _select_device(self) -> torch.device:
        if self.config.device != "auto":
            requested = torch.device(self.config.device)
            if requested.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            if (
                requested.type == "cuda"
                and requested.index is not None
                and self.config.backend == "ddp"
                and self._world_size > 1
                and requested.index != self._local_rank
            ):
                raise ValueError(
                    "a fixed CUDA index must match LOCAL_RANK in DDP; use 'auto' or 'cuda'"
                )
            if requested.type == "cuda" and requested.index is None:
                index = self._local_rank if self.config.backend == "ddp" else 0
                return torch.device("cuda", index)
            return requested
        if torch.cuda.is_available():
            index = self._local_rank if self.config.backend == "ddp" else 0
            return torch.device("cuda", index)
        return torch.device("cpu")

    def get_world_size(self) -> int:
        return self._world_size

    def get_rank(self) -> int:
        return self._rank

    def is_main_process(self) -> bool:
        return self._rank == 0

    def train_epoch(self, train_loader: DataLoader[Any]) -> Dict[str, float]:
        """Train for one epoch and return a globally reduced mean loss."""

        self.setup_distributed()
        sampler = getattr(train_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(self.epoch)

        assert self.optimizer is not None
        self.model.train()
        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        loss_weight = 0.0
        sample_count = 0
        accumulated_weight = 0.0
        accumulated_batches = 0
        last_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            inputs, targets = self._prepare_batch(batch)
            outputs = self._forward(inputs)
            loss = self.loss_fn(outputs, targets)
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")

            batch_size = self._batch_size(targets)
            batch_weight = self._loss_normalizer(outputs, targets, batch_size)
            if batch_weight == 0:
                loss = outputs.sum() * 0
            last_loss = float(loss.detach().item())
            if not math.isfinite(last_loss):
                raise ValueError("loss_fn returned a non-finite value")
            loss_sum += last_loss * batch_weight
            loss_weight += batch_weight
            sample_count += batch_size
            accumulated_weight += batch_weight
            accumulated_batches += 1
            (loss * batch_weight).backward()

            if (batch_index + 1) % self.config.gradient_accumulation_steps == 0:
                self._optimizer_step(accumulated_weight, last_loss)
                accumulated_weight = 0.0
                accumulated_batches = 0

        if sample_count == 0:
            raise ValueError("train_loader must contain at least one batch")
        if accumulated_batches:
            self._optimizer_step(accumulated_weight, last_loss)

        loss_sum, loss_weight, sample_count = self._reduce_totals(
            loss_sum, loss_weight, sample_count
        )
        if loss_weight == 0:
            raise ValueError("loss_fn has no positive normalization weight in the train loader")
        return {"loss": loss_sum / loss_weight, "samples": float(sample_count)}

    def evaluate(self, data_loader: DataLoader[Any]) -> Dict[str, float]:
        """Evaluate without gradients and return a globally reduced mean loss."""

        self.setup_distributed()
        self.model.eval()
        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.eval()
        evaluation_model = self._unwrapped_model()
        if self.config.backend == "ddp" and self._world_size > 1:
            for buffer in evaluation_model.buffers():
                dist.broadcast(buffer, src=0)
        loss_sum = 0.0
        loss_weight = 0.0
        sample_count = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = self._prepare_batch(batch)
                outputs = self._forward(inputs, model=evaluation_model)
                loss = self.loss_fn(outputs, targets)
                if loss.ndim != 0:
                    raise ValueError("loss_fn must return a scalar tensor")
                batch_size = self._batch_size(targets)
                batch_weight = self._loss_normalizer(outputs, targets, batch_size)
                if batch_weight:
                    loss_value = float(loss.item())
                    if not math.isfinite(loss_value):
                        raise ValueError("loss_fn returned a non-finite value")
                    loss_sum += loss_value * batch_weight
                loss_weight += batch_weight
                sample_count += batch_size

        loss_sum, loss_weight, sample_count = self._reduce_totals(
            loss_sum, loss_weight, sample_count
        )
        if sample_count == 0:
            raise ValueError("data_loader must contain at least one sample")
        if loss_weight == 0:
            raise ValueError("loss_fn has no positive normalization weight in the data loader")
        return {"loss": loss_sum / loss_weight, "samples": float(sample_count)}

    def fit(
        self,
        train_loader: DataLoader[Any],
        eval_loader: Optional[DataLoader[Any]] = None,
        *,
        max_epochs: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Run training and return one metrics record per completed epoch."""

        self.setup_distributed()
        final_epoch = max_epochs if max_epochs is not None else self.config.max_epochs
        if final_epoch <= self.epoch:
            raise ValueError("max_epochs must be greater than the current epoch")

        history: List[Dict[str, float]] = []
        for epoch_index in range(self.epoch, final_epoch):
            self.epoch = epoch_index
            train_metrics = self.train_epoch(train_loader)
            record = {
                "epoch": float(epoch_index + 1),
                "train_loss": train_metrics["loss"],
            }

            monitored_loss = train_metrics["loss"]
            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                record["eval_loss"] = eval_metrics["loss"]
                monitored_loss = eval_metrics["loss"]

            if self.scheduler is not None and self.config.scheduler_interval == "epoch":
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(monitored_loss)
                else:
                    self.scheduler.step()

            self.epoch = epoch_index + 1
            self.best_loss = min(self.best_loss, monitored_loss)
            if (
                self.config.checkpoint_dir
                and self.epoch % self.config.checkpoint_every == 0
                and self.is_main_process()
            ):
                self.save_checkpoint()

            history.append(record)

        return history

    def save_checkpoint(self) -> str:
        """Save model and optimizer state on the main process."""

        if not self.is_main_process():
            raise RuntimeError("only rank 0 can save checkpoints")
        if not self.config.checkpoint_dir:
            raise ValueError("checkpoint_dir is not configured")
        self.setup_distributed()
        assert self.optimizer is not None

        if self._checkpoint_manager is None:
            self._checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self._unwrapped_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_loss": self.best_loss,
            "config": asdict(self.config),
        }
        return self._checkpoint_manager.save_checkpoint(checkpoint, self.global_step)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Restore a checkpoint created by :meth:`save_checkpoint`."""

        self.setup_distributed()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError("checkpoint must contain a dictionary")
        required = {
            "epoch",
            "global_step",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "best_loss",
        }
        missing = required - set(checkpoint)
        if missing:
            raise ValueError(f"checkpoint is missing required fields: {', '.join(sorted(missing))}")
        for field in ("epoch", "global_step"):
            if type(checkpoint[field]) is not int or checkpoint[field] < 0:
                raise ValueError(f"checkpoint {field} must be a non-negative integer")
        if isinstance(checkpoint["best_loss"], bool) or not isinstance(
            checkpoint["best_loss"], (int, float)
        ):
            raise ValueError("checkpoint best_loss must be numeric")
        best_loss = float(checkpoint["best_loss"])
        if math.isnan(best_loss) or best_loss == float("-inf"):
            raise ValueError("checkpoint best_loss is invalid")
        if not isinstance(checkpoint["model_state_dict"], dict):
            raise ValueError("checkpoint model_state_dict must be a dictionary")
        if not isinstance(checkpoint["optimizer_state_dict"], dict):
            raise ValueError("checkpoint optimizer_state_dict must be a dictionary")

        assert self.optimizer is not None
        scheduler_state = checkpoint["scheduler_state_dict"]
        if (self.scheduler is None) != (scheduler_state is None):
            raise ValueError("checkpoint scheduler state does not match the configured trainer")
        if self.scheduler is not None:
            if not isinstance(scheduler_state, dict):
                raise ValueError("checkpoint scheduler_state_dict must be a dictionary")
            _validate_state_structure(
                self.scheduler.state_dict(), scheduler_state, "scheduler_state_dict"
            )

        model = self._unwrapped_model()
        model_snapshot = deepcopy(model.state_dict())
        optimizer_snapshot = deepcopy(self.optimizer.state_dict())
        scheduler_snapshot = deepcopy(self.scheduler.state_dict()) if self.scheduler else None
        progress_snapshot = (self.epoch, self.global_step, self.best_loss)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _validate_loaded_optimizer_state(self.optimizer)
            if self.scheduler is not None:
                self.scheduler.load_state_dict(scheduler_state)
            self.epoch = checkpoint["epoch"]
            self.global_step = checkpoint["global_step"]
            self.best_loss = best_loss
        except BaseException:
            model.load_state_dict(model_snapshot)
            self.optimizer.load_state_dict(optimizer_snapshot)
            if self.scheduler is not None and scheduler_snapshot is not None:
                self.scheduler.load_state_dict(scheduler_snapshot)
            self.epoch, self.global_step, self.best_loss = progress_snapshot
            raise

    def cleanup(self) -> None:
        """Release a process group created by this trainer."""

        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
            self._owns_process_group = False

    def _optimizer_step(self, accumulated_weight: float, loss_value: float) -> None:
        assert self.optimizer is not None
        gradient_denominator = accumulated_weight
        if self.config.backend == "ddp" and self._world_size > 1:
            global_samples = torch.tensor(
                gradient_denominator,
                dtype=torch.float64,
                device=self.device,
            )
            dist.all_reduce(global_samples, op=dist.ReduceOp.SUM)
            gradient_denominator = global_samples.item() / self._world_size
        if not math.isfinite(gradient_denominator) or gradient_denominator < 0:
            raise ValueError("loss_fn has no positive finite normalization weight for this step")
        if gradient_denominator == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(gradient_denominator)
        self.optimizer.step()
        if self.scheduler is not None and self.config.scheduler_interval == "step":
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        if self.config.log_interval and self.global_step % self.config.log_interval == 0:
            self.logger.info("step=%d loss=%.6f", self.global_step, loss_value)

    def _loss_normalizer(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int,
    ) -> float:
        reduction = getattr(self.loss_fn, "reduction", "mean")
        if reduction not in {"mean", "batchmean"}:
            raise ValueError("loss_fn must use mean or batchmean reduction")

        normalizer = float(batch_size)
        if isinstance(self.loss_fn, (nn.CrossEntropyLoss, nn.NLLLoss)):
            if isinstance(self.loss_fn, nn.CrossEntropyLoss) and targets.is_floating_point():
                normalizer = float(targets.numel() // outputs.shape[1])
            else:
                valid_targets = targets[targets != self.loss_fn.ignore_index]
                if self.loss_fn.weight is None:
                    normalizer = float(valid_targets.numel())
                else:
                    if (
                        not torch.isfinite(self.loss_fn.weight).all().item()
                        or torch.any(self.loss_fn.weight < 0).item()
                    ):
                        raise ValueError("loss weights must be finite and non-negative")
                    normalizer = float(self.loss_fn.weight[valid_targets].sum().item())

        if not math.isfinite(normalizer) or normalizer < 0:
            raise ValueError("loss_fn has an invalid normalization weight for this batch")
        return normalizer

    def _prepare_batch(self, batch: Any) -> Tuple[Any, torch.Tensor]:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ValueError("each batch must contain exactly (inputs, targets)")
        inputs = self._to_device(batch[0])
        targets = self._to_device(batch[1])
        if not isinstance(targets, torch.Tensor):
            raise TypeError("targets must be a torch.Tensor")
        return inputs, targets

    def _to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, tuple):
            return tuple(self._to_device(item) for item in value)
        if isinstance(value, list):
            return [self._to_device(item) for item in value]
        if isinstance(value, dict):
            return {key: self._to_device(item) for key, item in value.items()}
        return value

    def _forward(self, inputs: Any, *, model: Optional[nn.Module] = None) -> torch.Tensor:
        active_model = self.model if model is None else model
        if isinstance(inputs, dict):
            return active_model(**inputs)
        if isinstance(inputs, (tuple, list)):
            return active_model(*inputs)
        return active_model(inputs)

    @staticmethod
    def _batch_size(targets: torch.Tensor) -> int:
        if targets.ndim == 0:
            raise ValueError("batched targets must have at least one dimension")
        return targets.shape[0]

    def _reduce_totals(
        self,
        loss_sum: float,
        loss_weight: float,
        sample_count: int,
    ) -> Tuple[float, float, int]:
        if self.config.backend != "ddp" or self._world_size == 1:
            return loss_sum, loss_weight, sample_count
        totals = torch.tensor(
            [loss_sum, loss_weight, sample_count],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        return float(totals[0].item()), float(totals[1].item()), int(totals[2].item())

    def _unwrapped_model(self) -> nn.Module:
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model


def create_trainer(
    backend: str,
    config: TrainingConfig,
    model: nn.Module,
    *,
    loss_fn: Optional[LossFunction] = None,
    optimizer_factory: Optional[OptimizerFactory] = None,
    scheduler_factory: Optional[SchedulerFactory] = None,
) -> Trainer:
    """Create a trainer after checking that backend and config agree."""

    if backend not in {"local", "ddp"}:
        raise ValueError("backend must be 'local' or 'ddp'")
    if config.backend != backend:
        raise ValueError("backend argument must match config.backend")
    return Trainer(
        config,
        model,
        loss_fn=loss_fn,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )
