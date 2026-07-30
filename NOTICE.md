# Notices

EviBack is a clean public-facing implementation derived from an internal
research prototype. The following upstream projects informed or support the
training and retrieval paths; they are not vendored in this source snapshot.

| Component | Upstream | Role | License |
| --- | --- | --- | --- |
| Search-R1 | https://github.com/PeterGriffinJin/Search-R1 | Search-agent grammar and training baseline | Apache-2.0 |
| VERL | https://github.com/volcengine/verl | GRPO training framework | Apache-2.0 |
| vLLM | https://github.com/vllm-project/vllm | Actor and Teacher serving | Apache-2.0 |
| Transformers | https://github.com/huggingface/transformers | Model/tokenizer interfaces | Apache-2.0 |
| Ray | https://github.com/ray-project/ray | Distributed execution used by VERL | Apache-2.0 |
| FAISS | https://github.com/facebookresearch/faiss | Dense retrieval index | MIT |

The exact internal source snapshot and local modifications are recorded under
`release/`. The `third_party/verl/group_postnorm.patch` file is a minimal patch
description and must retain the upstream VERL license when applied.