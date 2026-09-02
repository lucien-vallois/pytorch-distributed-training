"""
Metrics tracking and logging utilities for distributed training
"""

import logging
import math
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class MetricsTracker:
    """Tracks training metrics with support for distributed training"""

    def __init__(
        self, use_prometheus: bool = False, use_wandb: bool = False, window_size: int = 100
    ):
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")
        if use_prometheus and not PROMETHEUS_AVAILABLE:
            raise ImportError("Prometheus support requires the 'monitoring' extra")
        if use_wandb and not WANDB_AVAILABLE:
            raise ImportError("Weights & Biases support requires the 'monitoring' extra")

        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.scalar_metrics = {}
        self.start_time = time.time()

        self.use_prometheus = use_prometheus and PROMETHEUS_AVAILABLE
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        if self.use_prometheus:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()

        if self.use_wandb:
            self._setup_wandb()

        self.logger = logging.getLogger(self.__class__.__name__)

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""

        self.prometheus_metrics = {
            "loss": Histogram("training_loss", "Training loss", ["phase"], registry=self.registry),
            "accuracy": Gauge(
                "training_accuracy", "Training accuracy", ["phase"], registry=self.registry
            ),
            "learning_rate": Gauge("learning_rate", "Learning rate", registry=self.registry),
            "throughput": Gauge(
                "training_throughput", "Training throughput (samples/sec)", registry=self.registry
            ),
            "gpu_memory": Gauge(
                "gpu_memory_usage", "GPU memory usage (MB)", ["gpu_id"], registry=self.registry
            ),
            "epoch": Gauge("current_epoch", "Current training epoch", registry=self.registry),
            "step": Counter("training_steps", "Number of training steps", registry=self.registry),
        }

    def _setup_wandb(self):
        """Setup Weights & Biases logging"""

        if not wandb.run:
            wandb.init(project="distributed-training", config={})
        self.wandb_run = wandb.run

    def update(self, key: str, value: float, step: Optional[int] = None):
        """Update a metric value"""

        value = float(value)
        if not math.isfinite(value):
            raise ValueError("metric value must be finite")

        # Store in deque for rolling statistics
        self.metrics[key].append(value)

        # Update scalar metrics
        self.scalar_metrics[key] = value

        # Update Prometheus if enabled
        if self.use_prometheus:
            self._update_prometheus(key, value, step)

        # Update W&B if enabled
        if self.use_wandb and step is not None:
            wandb.log({key: value}, step=step)

    def update_batch(
        self,
        phase: str,
        loss: float,
        accuracy: Optional[float] = None,
        lr: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """Update batch-level metrics"""

        self.update(f"{phase}/loss", loss, step)

        if accuracy is not None:
            self.update(f"{phase}/accuracy", accuracy, step)

        if lr is not None:
            self.update("learning_rate", lr, step)

        if step is not None:
            self.update("global_step", step, step)

    def update_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """Update epoch-level metrics"""

        self.update("epoch", epoch)

        for key, value in train_metrics.items():
            self.update(f"train/{key}", value)

        if val_metrics:
            for key, value in val_metrics.items():
                self.update(f"val/{key}", value)

        # Log to console
        self._log_epoch_metrics(epoch, train_metrics, val_metrics)

    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log epoch metrics to console"""

        log_msg = f"Epoch {epoch}: "

        # Training metrics
        train_parts = []
        for key, value in train_metrics.items():
            if isinstance(value, float):
                train_parts.append(f"{key}={value:.4f}")
        log_msg += " | ".join(train_parts)

        # Validation metrics
        if val_metrics:
            val_parts = []
            for key, value in val_metrics.items():
                if isinstance(value, float):
                    val_parts.append(f"{key}={value:.4f}")
            log_msg += " | Val: " + " | ".join(val_parts)

        self.logger.info(log_msg)

    def get_rolling_average(self, key: str, window: Optional[int] = None) -> float:
        """Get rolling average for a metric"""

        values = list(self.metrics[key])
        if window:
            values = values[-window:]

        return float(np.mean(values)) if values else 0.0

    def get_statistics(self, key: str) -> Dict[str, float]:
        """Get statistics for a metric"""

        values = list(self.metrics[key])
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""

        result = dict(self.scalar_metrics)

        # Add rolling statistics
        for key in self.metrics:
            stats = self.get_statistics(key)
            result[f"{key}_stats"] = stats

        return result

    def _update_prometheus(self, key: str, value: float, step: Optional[int]):
        """Update Prometheus metrics"""

        # Map metric keys to Prometheus metrics
        if "loss" in key.lower():
            phase = key.split("/")[0] if "/" in key else "train"
            self.prometheus_metrics["loss"].labels(phase=phase).observe(value)
        elif "accuracy" in key.lower():
            phase = key.split("/")[0] if "/" in key else "train"
            self.prometheus_metrics["accuracy"].labels(phase=phase).set(value)
        elif key == "learning_rate":
            self.prometheus_metrics["learning_rate"].set(value)
        elif key == "epoch":
            self.prometheus_metrics["epoch"].set(value)
        elif key == "global_step":
            self.prometheus_metrics["step"].inc()

    def get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU memory usage metrics"""

        gpu_metrics = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**2  # MB

                gpu_metrics[f"gpu_{i}_allocated_mb"] = memory_allocated
                gpu_metrics[f"gpu_{i}_reserved_mb"] = memory_reserved

                # Update Prometheus
                if self.use_prometheus:
                    self.prometheus_metrics["gpu_memory"].labels(gpu_id=str(i)).set(
                        memory_allocated
                    )

        return gpu_metrics

    def get_throughput_metrics(self, samples_processed: int) -> Dict[str, float]:
        """Calculate throughput metrics"""

        elapsed_time = time.time() - self.start_time
        throughput = samples_processed / elapsed_time if elapsed_time > 0 else 0

        metrics = {
            "throughput_samples_per_sec": throughput,
            "elapsed_time_sec": elapsed_time,
            "total_samples_processed": samples_processed,
        }

        if self.use_prometheus:
            self.prometheus_metrics["throughput"].set(throughput)

        return metrics

    def reset(self):
        """Reset in-memory and Prometheus metrics.

        External Weights & Biases history is not modified.
        """

        self.metrics.clear()
        self.scalar_metrics.clear()
        self.start_time = time.time()
        if self.use_prometheus:
            for collector in self.prometheus_metrics.values():
                self.registry.unregister(collector)
            self._setup_prometheus_metrics()

    def export_metrics(self, filepath: str):
        """Export metrics to file"""

        import json

        metrics_data = {
            "timestamp": time.time(),
            "metrics": self.get_all_metrics(),
            "gpu_metrics": self.get_gpu_metrics(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

        self.logger.info(f"Metrics exported to {filepath}")

    def get_prometheus_registry(self):
        """Get Prometheus registry for external exposure"""

        return self.registry if self.use_prometheus else None


class TrainingMonitor:
    """Monitors training progress and triggers alerts"""

    def __init__(
        self, metrics_tracker: MetricsTracker, alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.metrics_tracker = metrics_tracker
        default_thresholds = {
            "loss_gradient_threshold": 0.1,  # Alert if loss gradient is too steep
            "accuracy_plateau_threshold": 0.001,  # Alert if accuracy improvement stalls
            "memory_threshold_mb": 8000,  # Alert if GPU memory usage is high
        }
        self.alert_thresholds = {**default_thresholds, **(alert_thresholds or {})}

        self.logger = logging.getLogger(self.__class__.__name__)

    def check_alerts(self) -> List[str]:
        """Check for training alerts"""

        alerts = []

        # Check loss gradient
        loss_values = list(self.metrics_tracker.metrics.get("train/loss", []))
        if len(loss_values) >= 10:
            recent_loss = np.mean(loss_values[-5:])
            older_loss = np.mean(loss_values[-10:-5])
            loss_gradient = abs(recent_loss - older_loss)

            if loss_gradient > self.alert_thresholds["loss_gradient_threshold"]:
                alerts.append(f"Loss changed by {loss_gradient:.4f} across recent windows")

        # Check accuracy plateau
        acc_values = list(self.metrics_tracker.metrics.get("train/accuracy", []))
        if len(acc_values) >= 20:
            recent_acc = np.mean(acc_values[-10:])
            older_acc = np.mean(acc_values[-20:-10])
            acc_improvement = recent_acc - older_acc

            if abs(acc_improvement) < self.alert_thresholds["accuracy_plateau_threshold"]:
                alerts.append(f"Accuracy improvement is only {acc_improvement:.4f}")

        # Check GPU memory
        gpu_metrics = self.metrics_tracker.get_gpu_metrics()
        for key, value in gpu_metrics.items():
            if "allocated" in key and value > self.alert_thresholds["memory_threshold_mb"]:
                gpu_id = key.split("_")[1]
                alerts.append(f"GPU {gpu_id} is using {value:.0f} MB")

        return alerts

    def log_system_status(self):
        """Log current system status"""

        gpu_metrics = self.metrics_tracker.get_gpu_metrics()

        status_msg = "System Status: "
        if gpu_metrics:
            memory_usage = []
            for key, value in gpu_metrics.items():
                if "allocated" in key:
                    gpu_id = key.split("_")[1]
                    memory_usage.append(f"GPU{gpu_id}: {value:.0f}MB")
            status_msg += " | ".join(memory_usage)
        else:
            status_msg += "No GPU available"

        self.logger.info(status_msg)
