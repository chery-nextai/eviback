# E2E-APE prompt artifact

E2E-APE is EviBack's development-time prompt-engineering process. The public
implementation artifacts are the [English prompt](controller_instructions.md),
the [Chinese prompt](controller_instructions_cn.md), and the [English contract](contract_en.md).
They are intended for an agent/model that performs sample construction, labeling,
prompt ablation, evaluation, and strategy selection.

The repository does not distribute the model-generated runner, scorer, ablation,
or selection code. `prompts/frozen_policy.json` records the selected prompt
identifiers. `benchmark/manifest.json` and `benchmark/schema.json` describe the
private benchmark without redistributing its cases.