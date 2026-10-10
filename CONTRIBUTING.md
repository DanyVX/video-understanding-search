# Contributing

Use one milestone per branch (e.g. `m1-ingestion`), open a pull request, and do not
merge unless lint, typing, tests, audit and CI have passed. Commit in small logical
units with Conventional Commits. Reproduce bugs with a deterministic regression test.

For local checks run `python scripts/tasks.py lint`, `typecheck`, and `test`.
Do not commit personal videos, audio, secrets, model weights or generated data.
