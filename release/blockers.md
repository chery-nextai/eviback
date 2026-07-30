# Release blockers

| Blocker | Evidence | Required decision | Status |
| --- | --- | --- | --- |
| Project license | Internal plan section 12 | Institution-approved license | Open |
| Exact PyTorch/Ray versions | `environment_snapshot.txt` | Recover formal-run environment | Open |
| Dataset redistribution | `data_distribution_decision.md` | Dataset-owner/legal approval | Open |
| E2E-APE benchmark hash mismatch | Historical manifest vs current file | Freeze and regenerate manifest | Open |
| VERL upstream revision | Internal fork is embedded in the main repository | Identify upstream commit and validate patch | Open |
| Teacher public availability | GLM-4.7-Flash was served internally | Document a reproducible public endpoint/model revision | Open |

All software tests can pass while these release-governance blockers remain.
Do not publish a release archive until every row is resolved and dated.