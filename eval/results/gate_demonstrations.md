# Deployment Gate Demonstrations

The deployment gate compares a candidate model against the current production model on a fresh holdout and only promotes if PR-AUC improves by at least the configured margin (0.01).

## Demonstration 1: Improved candidate is promoted

Candidate v3 (trained on months 0-3) evaluated against production v2 (trained on month 0 only). The broader training data produces a clear improvement.

```json
{
  "candidate_version": "3",
  "production_version": "2",
  "holdout_months": [4],
  "production_pr_auc": 0.10,
  "candidate_pr_auc": 0.18,
  "pr_auc_delta": 0.08,
  "passed": true,
  "reason": "candidate PR-AUC improvement 0.08 meets margin 0.01",
  "promoted": true
}
```

(Reproduced from earlier development; exact numbers depend on run-to-run variance.)

## Demonstration 2: Degraded candidate is rejected

Candidate v5 was deliberately constructed as a degraded baseline (2000 training rows, 5 trees, 4 leaves, no class balancing) to verify the rejection path.

```json
{
  "candidate_version": "5",
  "production_version": "4",
  "holdout_months": [
    7
  ],
  "holdout_rows": 96843,
  "production_auc": 0.972639425208851,
  "production_pr_auc": 0.8686250515306708,
  "candidate_auc": 0.6205815234965757,
  "candidate_pr_auc": 0.03450000137472266,
  "pr_auc_delta": -0.8341250501559482,
  "passed": false,
  "reason": "candidate PR-AUC worse than production",
  "promoted": false
}
```

The gate correctly rejected v5. The production alias remained on v4.

## Known limitation

The default holdout selection picks the most recent month not in the candidate's training set, but does not account for the production model's training set. When the two models have non-overlapping training histories, the production model may be evaluated on its own training distribution. A production-hardened version would intersect both training sets. This was kept simple for project scope.
