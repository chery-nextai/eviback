# Reward and prompt freeze

- Public method name: EviBack
- Reward: `spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2`
- Stage A prompt: `spad_teacher_evidence_status_answer_v2`
- Stage B prompt: `gold_support_evidence_only_v3`
- Strategy: `spad_teacher_hard_gate_r5_literal_canonical_v2`
- Base partial reward (`beta`): `0.1`
- Gold token-F1 bonus weight (`gamma`): `0.1`
- Default post-normalization fallback scale (`lambda`): `0.1`
- Formal lambda sweep: `0.1`, `0.3`, `0.5`, `1.0`
- Rollouts per question: `8`
- Train examples / steps / seed: `5,100 / 79 / 42`
- Evaluation questions / bootstrap samples: `3,500 / 10,000`

The Actor reward is authoritative whenever any rollout in its question group
has exact match. The Evidence-Constrained Teacher is called only for all-zero
groups. Stage B is conditional on a parsed Stage A non-insufficient result;
it cannot cross the Stage A insufficiency boundary. The reference answer is a
calibration candidate, never evidence.