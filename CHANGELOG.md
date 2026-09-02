# Changelog

Notable user-visible changes are recorded here.

## Unreleased

### Changed

- Replaced the non-functional Ray and Horovod wrappers with local and native PyTorch DDP paths.
- Made `pyproject.toml` the single package metadata source.
- Added a canonical `pytorch_distributed_training` import namespace.
- Added locally exercised Python 3.13 support to package metadata and CI.
- Replaced broken examples with a CPU-safe synthetic training example.
- Added optimizer/scheduler factories, explicit device selection, and a no-padding
  `DistributedEvalSampler` for exact distributed evaluation counts.
- Added epoch/step scheduler intervals and streaming `IterableDataset` support.
- Reduced default dependencies; data and monitoring integrations are optional extras.
- Rewrote usage, contribution, and security documentation around verifiable repository behavior.
- Added complete wheel/source-distribution checks without publishing artifacts.
- Added a verified file-store `torchrun` fallback for Windows builds without libuv.

### Fixed

- Added the missing `TrainingConfig` fields and validation.
- Added checkpoint, training, model, metrics, and package smoke tests.
- Corrected weighted/ignored loss normalization across microbatches and DDP ranks.
- Made checkpoint names unique, retention deterministic, and payloads safe for
  `weights_only=True` loading.
- Made checkpoint and inference writes atomic so interrupted saves never become latest.
- Made trainer checkpoint restore validate optimizer/scheduler state and roll back every mutable
  state on failure while accepting legitimate dynamic and `MultiStepLR` scheduler state.
- Fixed model initialization/dropout contracts and Prometheus reset/epoch behavior.
- Rejected truncated Vision Transformer inputs and unknown temporal-factory options.
- Preserved CSV classification dtypes, scalar regression shapes, and time-series validation context.
- Made image datasets without an explicit transform return collatable tensors.
- Rejected non-finite configuration and loss values before optimizer state is corrupted.
- Rejected non-finite metrics before they reach JSON, Prometheus, or monitoring history.
- Removed automatic publishing and documentation deployment from CI.

### Removed

- Unverified production, performance, coverage, registry, and hosted-documentation claims.
- Incomplete Kubernetes manifests and duplicated packaging files.
