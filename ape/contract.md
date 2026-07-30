# E2E-APE contract

E2E-APE is the development-time prompt-engineering process that produced
EviBack's frozen Teacher strategy. The public artifact is a controller prompt,
not an executable controller or evaluation framework. The prompt covers grouped
sampling, evidence-only labeling, question-level dev/holdout splitting,
prompt-variable ablation, cache-free evaluation, selection on dev, and
evaluation of the frozen policy on holdout.

The prompt is intended to be given to an agent/model that can inspect the local
experiment workspace and generate the temporary scripts needed for one study.
Those generated scripts and raw responses are not part of this repository.
GPT-5.5 acted as the development-time controller and assistant; the repository
does not claim to provide an independent GPT-5.5 controller, automatic budget
manager, or candidate lifecycle service. GLM-4.7-Flash is the frozen training
Teacher. Neither the Teacher nor E2E-APE is used during Actor inference.

The reference answer used in Stage B is a hypothesis to calibrate answer form.
It is never evidence. Benchmarks must group splits by normalized Original
question. Once a holdout is inspected for policy selection, it must be renamed
as diagnostic and cannot be called untouched.