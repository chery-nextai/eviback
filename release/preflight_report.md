# Release preflight report

Date: 2026-07-28
Version: 0.1.0rc1

## Passed

- `PYTHONPATH=src:. python -m unittest discover -s tests -v`: 24 tests passed.
- `python scripts/run_reward_smoke.py`: 8-response all-zero group, 16 mock Teacher calls, score `0.2`, scale `0.1`.
- `python scripts/prepare_data.py --check-only --manifest datasets-examples/manifests/train_5100.json`: schema and quota check passed.
- Training `--dry-run`: generated the VERL command with model/data/service/group/scale arguments.
- `pip wheel --no-deps --no-build-isolation ./EviBack`: wheel build passed (`eviback-0.1.0rc1`).
- `detect-secrets scan EviBack`: no findings.
- Relative-path scanner: no internal workspace/home prefixes, credential literals, or private-key blocks.
- Large-file and forbidden-artifact scan: no weights, checkpoints, caches, `*.pyc`, or files over 1 MiB.
- `git diff --check`: passed for tracked files; the new package has no trailing whitespace.

## Blocked

- `reuse lint --root EviBack`: 90 files lack copyright/license metadata because the institutional EviBack license is not approved. `LICENSE` intentionally says so; it must be replaced before publication.
- Formal GPU training and 3,500-question inference were not run in this CPU-only verification environment.
- Dataset/corpus/APE benchmark distribution remains disabled. The APE current-vs-historical hash mismatch is recorded in `blockers.md`.
- The exact upstream VERL revision corresponding to the internal fork is unresolved.

This report is a release-candidate audit, not a publication approval.