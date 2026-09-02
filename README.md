# PyTorch Distributed Training

A small Python library for training PyTorch models in one process or with native
`DistributedDataParallel` (DDP). The repository includes a reusable training loop,
checkpoint helpers, model implementations, dataset utilities, and a synthetic example.

The project is currently an alpha source distribution. Install it from a clone; no package
registry, hosted documentation, benchmark result, release pipeline, or deployment target is
claimed by this repository.

## Implemented scope

- Local CPU or CUDA training for `(inputs, targets)` batches.
- Native multi-process DDP support with a standard `torchrun` launch path.
- Custom loss functions plus optimizer and scheduler factories; gradient accumulation; and checkpoints.
- Vision Transformer and temporal network building blocks.
- Optional image and CSV dataset helpers.
- Optional Prometheus and Weights & Biases metric adapters.

## Verification status

The current checkout has been exercised with Python 3.11.9 and PyTorch 2.12.0+cpu: editable
install, wheel and source-distribution builds, the local example, 57 unit tests including the
optional data utilities, the Prometheus adapter, a two-process DDP smoke test, and the Windows
file-store `torchrun` fallback. CUDA, multi-node rendezvous, the declared PyTorch 2.0 lower bound,
and Weights & Biases have not been exercised in this environment and are not presented as
validated results. This Windows PyTorch build lacks libuv, so the standard `--standalone` launcher
fails before workers start; CI is configured to exercise that standard path on Ubuntu.
The 52-test core suite and DDP smoke also pass with Python 3.13.5 and PyTorch 2.14.0+cpu in an
isolated install; Python 3.13 is included in the package metadata and CI matrix.

## Requirements

- Python 3.10 or newer.
- PyTorch 2.0 or newer.
- A CUDA-enabled PyTorch build only when GPU execution is required.

Ray, Horovod, Kubernetes, Docker, and cloud provisioning are not required or configured.

## Install from source

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

POSIX shells:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For development and the complete test suite, include the data dependencies:

```bash
python -m pip install -e ".[dev,data]"
```

Then run the tests and the local example:

```bash
python -m pytest
python examples/train_classifier.py --epochs 2
```

The example prints one JSON object containing the final epoch and loss.

## Library usage

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_distributed_training import TrainingConfig, create_trainer

features = torch.randn(64, 4)
targets = (features[:, 0] > 0).long()
loader = DataLoader(TensorDataset(features, targets), batch_size=16, shuffle=True)

config = TrainingConfig(backend="local", max_epochs=2)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
trainer = create_trainer("local", config, model)

try:
    history = trainer.fit(loader)
    print(history[-1])
finally:
    trainer.cleanup()
```

`CrossEntropyLoss` is the default. For regression, the model output and floating-point targets
must have matching shapes. Custom loss functions must return a scalar batch mean; summed losses
are rejected because they would break metric and gradient normalization. Weighted and
`ignore_index` variants of the standard `CrossEntropyLoss` and `NLLLoss` are normalized across
microbatches and DDP ranks.

```python
regression_model = nn.Linear(4, 1)
regression_targets = torch.randn(64, 1)
regression_loader = DataLoader(TensorDataset(features, regression_targets), batch_size=16)
trainer = create_trainer("local", config, regression_model, loss_fn=nn.MSELoss())
history = trainer.fit(regression_loader)
```

Optimizer and scheduler factories run after the model is moved to its configured device:

```python
trainer = create_trainer(
    "local",
    config,
    model,
    optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
    scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
)
```

Optimizer factories must return an optimizer whose `step()` does not require a closure; LBFGS is
not supported by this batch-oriented loop. Schedulers advance once per epoch by default. Set
`scheduler_interval="step"` for schedulers such as `OneCycleLR`; the interval follows optimizer
steps, including when gradient accumulation is enabled.

## Distributed execution

On POSIX systems and PyTorch builds with libuv, launch the included example with two workers:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 examples/train_classifier.py --backend ddp --device cpu
```

Some Windows CPU builds do not include libuv. If `--standalone` reports that libuv was
requested but unavailable, use the file-store fallback exercised by this repository:

```powershell
$env:USE_LIBUV = "0"
$env:TORCH_DISABLE_SHARE_RDZV_TCP_STORE = "1"
python -m torch.distributed.run --nnodes=1 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-id=local-libuvless --rdzv-conf=store_type=file examples/train_classifier.py --backend ddp --device cpu
```

The trainer reads the standard `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` variables created by
`torchrun`. It uses NCCL for CUDA when that backend is available and falls back to Gloo otherwise.
Application data loaders
must use a `DistributedSampler` for training and `DistributedEvalSampler` for evaluation without
padding duplicates; the included code and data-loader helper show that setup. For CUDA, omit
`--device cpu`, use `auto` or bare `cuda` rather than a fixed index, and keep the worker count at
or below the number of available GPUs.

Multi-node behavior depends on the network and `torchrun` rendezvous configuration. This
repository does not provision machines or cluster infrastructure.

The repository also contains a launcher-independent, two-process CPU smoke test. It uses a
temporary file rendezvous and is opt-in locally:

```powershell
$env:RUN_DDP_TESTS = "1"
python -m pytest tests/test_ddp.py
```

```bash
RUN_DDP_TESTS=1 python -m pytest tests/test_ddp.py
```

## Optional data utilities

Install the data extra before importing `pytorch_distributed_training.data`:

```bash
python -m pip install ".[data]"
```

It provides image classification datasets, CSV/time-series datasets, transforms, and a helper
that can attach a `DistributedSampler` to a `DataLoader`. Images without an explicit transform
are returned as float tensors. CSV targets retain classification dtypes and scalar regression
shapes, and time-series validation windows use the end of the training split only as input context.

## Optional monitoring

Install the monitoring extra only when Prometheus or Weights & Biases integration is needed:

```bash
python -m pip install ".[monitoring]"
```

```python
from pytorch_distributed_training.utils import MetricsTracker

tracker = MetricsTracker(use_prometheus=True)
tracker.update("train/loss", 0.5, step=1)
registry = tracker.get_prometheus_registry()
```

Enabling Weights & Biases may initialize its client and use its configured external service.

## Public modules

The package exposes common entry points without `src.*` imports:

```python
from pytorch_distributed_training import DistributedEvalSampler, TrainingConfig
from pytorch_distributed_training.data import TimeSeriesDataset  # requires the data extra
from pytorch_distributed_training.models import TCNClassifier, VisionTransformer
from pytorch_distributed_training.utils import CheckpointManager, MetricsTracker
```

## Checkpoints

Set `checkpoint_dir` to save state after each configured interval:

```python
config = TrainingConfig(
    backend="local",
    max_epochs=5,
    checkpoint_dir="checkpoints",
    checkpoint_every=1,
)
```

Load checkpoints only from a trusted source. Checkpoint files contain model, optimizer, scheduler,
and training state. They do not capture Python, data-loader, or random-number-generator state, so
stochastic runs are not guaranteed to resume bit-for-bit. Direct `CheckpointManager` payloads and
inference metadata are limited to tensors/state dictionaries, primitive values, and built-in
containers; custom classes are rejected to preserve `weights_only=True` loading. Saves become
visible atomically, and a failed restore rolls the model, optimizer, scheduler, and counters back.

## Project layout

```text
src/        installable package source
examples/   runnable synthetic example
tests/      local unit and smoke tests
```

Package metadata lives only in `pyproject.toml`. CI is limited to testing and building and
checking wheel and source distributions; it does not publish or deploy artifacts.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). For security reports, see
[SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE).
