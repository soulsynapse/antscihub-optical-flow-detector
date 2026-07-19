"""No-GUI entry points. Nothing here may import PyQt, directly or transitively:
these run on compute nodes with no display, and an import-time Qt dependency
turns a headless job into a crash at startup. ``tests/test_batch_cli.py`` asserts
it stays that way."""
