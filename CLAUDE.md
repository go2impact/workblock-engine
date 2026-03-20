# CLAUDE.md

## Quick Start

```bash
uv pip install -e ".[dev]"
pytest
```

## Key Principles

1. **Pure calculation library** — no I/O, no cloud, no AI
2. **Use Polars** exclusively for data processing
3. **Do not rewrite ported logic** — the calculation code was ported from productivity-insights as-is
4. **Single public API** — `process(activities_df, keystrokes_df) -> WorkblockResult`

## Testing

```bash
pytest                    # all tests
pytest tests/test_engine.py  # end-to-end
pytest -v                 # verbose
```

## Git Commit Guidelines

- Author: Rustan Corpuz <rustanacexd@gmail.com>
- No "Generated with Claude Code" signatures
- No "Co-Authored-By: Claude" lines
