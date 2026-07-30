# Third-party integration

EviBack does not vendor Search-R1 or the full VERL fork. Install a compatible
VERL revision and apply `verl/group_postnorm.patch`. The patch adds one optional
per-response metadata array to GRPO. It validates that the value is positive,
finite, and constant within each group, then multiplies the advantage only
after mean/std normalization.

The original internal fork is identified in `release/source_snapshot.json`.
The exact upstream VERL base revision remains a release blocker.