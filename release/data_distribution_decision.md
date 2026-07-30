# Data distribution decision

Status: metadata-only release pending dataset-owner approval.

The repository does not redistribute raw QA datasets, the retrieval corpus,
the retrieval index, model weights, rollout logs, or the 512-case E2E-APE
benchmark. It provides preparation code, schemas, counts, split rules, and
SHA-256 identifiers. Users must obtain every upstream dataset under its own
license and create the derived files locally.

The local frozen artifacts had these identifiers:

- train 5,100: `6e9307a8b3a866ecd045170bc0e92048e7e00fba0a0098b4ced5dd227ba9b09c`
- evaluation 3,500: `bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283`
- E2E-APE 512 current file: `ae1e3494d9dc9a189086f70ddc89c94f331cb115a30bdd4b827fb524c72de072`

The historical E2E-APE manifest records a different benchmark hash. This must
be resolved before the benchmark itself can be published.