# Contributing to pySTAMPS

## Repository setup

```bash
git clone git@github.com:sirbastiano/pystamps.git
cd pystamps
uv sync
python -m pip install -e ".[dev]"
```

## Code standards

- Type hints and dataclasses are used across public modules.
- Prefer explicit exceptions derived from module-specific classes.
- Keep behavior deterministic where feasible and document inferred behavior in implementation.

## PR workflow

1. Implement a small, scoped change.
2. Run local static checks and project tests.
3. Include reproducible commands or examples in the PR description.
4. Run parity-related checks when touching stage math.

## Branching model

Default model is short-lived feature branches with one purpose per branch.
Merge after passing quality checks and audit gates.
