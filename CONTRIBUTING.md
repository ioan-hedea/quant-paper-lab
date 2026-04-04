# Contributing

Thanks for contributing to Quant Paper Lab.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you need end-to-end RL baselines:

```bash
python -m pip install -e ".[dev,e2e]"
```

## Development Checklist

1. Create a focused branch from `main`.
2. Keep config or methodological changes explicit in commit messages.
3. Run lint + tests locally before opening a PR.
4. Regenerate plots/tables only when the PR intentionally changes results.

## Quality Gates

```bash
flake8 .
pytest -q
```

## Research Reproducibility Expectations

- Preserve the shared empirical contract when comparing controllers.
- Avoid mixing unrelated experimental changes in one PR.
- Prefer checkpoint-compatible edits when iterating on reporting/visualization.
- Document any data, universe, or cost-model contract changes in the PR description.

## Pull Request Notes

For result-changing PRs, include:

- universe(s) used (`A`, `B`, or both)
- run command(s)
- output bundle path(s)
- a short summary of metric movement
