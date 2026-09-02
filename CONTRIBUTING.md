# Contributing

Contributions should keep the package small, testable, and usable from a fresh clone.

## Setup

```bash
git clone https://github.com/lucien-vallois/pytorch-distributed-training.git
cd pytorch-distributed-training
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,data]"
python -m pytest
```

POSIX shells:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,data]"
python -m pytest
```

## Changes

1. Create a focused branch.
2. Change only the behavior needed for the issue.
3. Add or update the smallest test that proves the behavior.
4. Run `python -m pytest` and `python -m black --check src tests examples`.
5. Update the README or changelog when the public contract changes.

Do not add publishing credentials, automatic deployment, unverifiable performance numbers, or
personal contact details. New dependencies need a concrete runtime or test use.

## Distributed changes

Local unit tests must remain CPU-safe. If a change requires multiple processes or GPUs, document
the exact `torchrun` command and keep that test opt-in with the `distributed` or `gpu` marker.

## Pull requests

Describe the problem, the chosen fix, and the commands used to verify it. Do not claim GPU,
multi-node, performance, or platform support that was not exercised.

Report vulnerabilities using [SECURITY.md](SECURITY.md), not a public issue containing exploit
details.
