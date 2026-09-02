"""
Checkpoint management utilities for distributed training
"""

import logging
import re
import tempfile
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_CHECKPOINT_SUFFIX = re.compile(r"_(\d+)_(\d{8}_\d{6}(?:_\d{6}){0,2})\.pt$")
_SAFE_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _atomic_torch_save(value: Any, destination: str | Path) -> None:
    """Write beside the destination and expose it only after a complete save."""

    path = Path(destination)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        torch.save(value, temporary_path)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _validate_checkpoint_payload(value: Any, path: str = "checkpoint") -> None:
    """Reject objects that cannot be loaded with ``weights_only=True``."""

    if value is None or type(value) in {
        bool,
        int,
        float,
        complex,
        str,
        bytes,
        torch.Tensor,
        torch.nn.Parameter,
        torch.device,
        torch.dtype,
    }:
        return
    if type(value) in {dict, OrderedDict, Counter}:
        for key, item in value.items():
            _validate_checkpoint_payload(key, f"{path} key")
            _validate_checkpoint_payload(item, f"{path}[{key!r}]")
        return
    if type(value) in {list, tuple, torch.Size}:
        for index, item in enumerate(value):
            _validate_checkpoint_payload(item, f"{path}[{index}]")
        return
    raise TypeError(
        f"{path} contains unsupported type {type(value).__name__}; "
        "use tensors, primitive values, lists, tuples, and dictionaries"
    )


class CheckpointManager:
    """Manages model checkpoints with automatic saving and loading"""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        if max_checkpoints <= 0:
            raise ValueError("max_checkpoints must be greater than zero")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], step: int, prefix: str = "checkpoint"
    ) -> str:
        """Save a weights-only-safe checkpoint to disk.

        Payloads may contain tensors, primitive values, and built-in containers.
        Custom classes are rejected so the result can be loaded safely.
        """

        self._validate_step(step)
        self._validate_prefix(prefix)
        _validate_checkpoint_payload(checkpoint)
        filepath = self._next_checkpoint_path(prefix, step)

        _atomic_torch_save(checkpoint, filepath)

        self.logger.info(f"Saved checkpoint: {filepath}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints(prefix)

        return str(filepath)

    def load_checkpoint(
        self, checkpoint_path: Optional[str] = None, prefix: str = "checkpoint"
    ) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint"""

        if checkpoint_path is None:
            self._validate_prefix(prefix)
            # Find latest checkpoint
            checkpoint_files = list(self.checkpoint_dir.glob(f"{prefix}_*.pt"))
            if not checkpoint_files:
                self.logger.warning("No checkpoints found")
                return None

            # Sort by step number (extract from filename)
            checkpoint_files.sort(key=self._checkpoint_sort_key, reverse=True)
            checkpoint_path = checkpoint_files[0]

        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        return checkpoint

    def save_best_model(self, model: torch.nn.Module, step: int, prefix: str = "best_model") -> str:
        """Save best performing model"""

        self._validate_step(step)
        self._validate_prefix(prefix)
        filepath = self._next_checkpoint_path(prefix, step)
        timestamp = self._extract_timestamp(filepath.name)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "step": step,
            "timestamp": timestamp,
        }
        _validate_checkpoint_payload(checkpoint)
        _atomic_torch_save(checkpoint, filepath)

        self.logger.info(f"Saved best model: {filepath}")
        self._cleanup_old_checkpoints(prefix)
        return str(filepath)

    def load_best_model(
        self,
        model: torch.nn.Module,
        checkpoint_path: Optional[str] = None,
        prefix: str = "best_model",
    ) -> int:
        """Load best model for inference"""

        if checkpoint_path is None:
            self._validate_prefix(prefix)
            # Find best model
            best_model_files = list(self.checkpoint_dir.glob(f"{prefix}_*.pt"))
            if not best_model_files:
                raise FileNotFoundError("No best model checkpoints found")

            # Sort by step number
            best_model_files.sort(key=self._checkpoint_sort_key, reverse=True)
            checkpoint_path = best_model_files[0]

        self.logger.info(f"Loading best model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        return checkpoint["step"]

    def list_checkpoints(self, prefix: str = "checkpoint") -> list:
        """List all available checkpoints"""

        self._validate_prefix(prefix)
        checkpoint_files = list(self.checkpoint_dir.glob(f"{prefix}_*.pt"))
        checkpoints = []

        for filepath in checkpoint_files:
            step = self._extract_step(filepath.name)
            timestamp = self._extract_timestamp(filepath.name)
            checkpoints.append(
                {
                    "path": str(filepath),
                    "step": step,
                    "timestamp": timestamp,
                    "filename": filepath.name,
                }
            )

        # Sort by step
        checkpoints.sort(key=lambda item: (item["step"], item["timestamp"]), reverse=True)
        return checkpoints

    def get_latest_checkpoint_info(self, prefix: str = "checkpoint") -> Optional[Dict]:
        """Get information about the latest checkpoint"""

        checkpoints = self.list_checkpoints(prefix)
        return checkpoints[0] if checkpoints else None

    def _extract_step(self, filename: str) -> int:
        """Extract step number from checkpoint filename"""
        match = _CHECKPOINT_SUFFIX.search(filename)
        return int(match.group(1)) if match else 0

    def _extract_timestamp(self, filename: str) -> str:
        """Extract timestamp from checkpoint filename"""
        match = _CHECKPOINT_SUFFIX.search(filename)
        return match.group(2) if match else ""

    def _checkpoint_sort_key(self, path: Path) -> tuple[int, str]:
        return self._extract_step(path.name), self._extract_timestamp(path.name)

    def _next_checkpoint_path(self, prefix: str, step: int) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.checkpoint_dir / f"{prefix}_{step}_{timestamp}.pt"
        collision = 0
        while path.exists():
            collision += 1
            path = self.checkpoint_dir / f"{prefix}_{step}_{timestamp}_{collision:06d}.pt"
        return path

    @staticmethod
    def _validate_step(step: int) -> None:
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a non-negative integer")

    @staticmethod
    def _validate_prefix(prefix: str) -> None:
        if not isinstance(prefix, str) or not _SAFE_PREFIX.fullmatch(prefix):
            raise ValueError(
                "prefix must contain only letters, numbers, dots, dashes, or underscores"
            )

    def _cleanup_old_checkpoints(self, prefix: str = "checkpoint"):
        """Remove old checkpoints to save disk space"""

        checkpoint_files = list(self.checkpoint_dir.glob(f"{prefix}_*.pt"))

        if len(checkpoint_files) <= self.max_checkpoints:
            return

        checkpoint_files.sort(
            key=lambda path: (self._extract_timestamp(path.name), path.name),
            reverse=True,
        )
        checkpoints_to_remove = checkpoint_files[self.max_checkpoints :]
        for checkpoint_file in checkpoints_to_remove:
            checkpoint_file.unlink()
            self.logger.info(f"Removed old checkpoint: {checkpoint_file}")

    def export_for_inference(
        self, model: torch.nn.Module, config: Dict[str, Any], output_path: str, step: int
    ):
        """Export a model with weights-only-safe configuration metadata."""

        self._validate_step(step)
        export_data = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "framework_version": str(torch.__version__),
        }

        _validate_checkpoint_payload(export_data)
        _atomic_torch_save(export_data, output_path)
        self.logger.info(f"Exported model for inference: {output_path}")
