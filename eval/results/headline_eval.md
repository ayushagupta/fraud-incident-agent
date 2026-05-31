# Eval Run Summary

- **Total scenarios:** 18
- **Passed:** 18 / 18 (100%)
- **Total runtime:** 1585.5s
- **Avg tools per scenario:** 6.1
- **Avg confidence on passes:** 0.71

**Per-diagnosis breakdown:**
- genuine_drift: 6/6
- no_incident: 4/4
- pipeline_bug: 4/4
- schema_change: 4/4

## Results Table

| name | expected_diag | actual_diag | expected_action | actual_action | confidence | tools | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean_month_0_baseline | no_incident | - | no_action | - | - | 0 | PASS |
| clean_month_0_smaller | no_incident | - | no_action | - | - | 0 | PASS |
| drift_and_nulls | genuine_drift | genuine_drift | alert_human | alert_human | 0.72 | 12 | PASS |
| genuine_drift_month_3 | genuine_drift | genuine_drift | retrain | retrain | 0.88 | 10 | PASS |
| genuine_drift_month_5 | genuine_drift | genuine_drift | retrain | retrain | 0.91 | 10 | PASS |
| genuine_drift_month_7 | genuine_drift | genuine_drift | retrain | retrain | 0.93 | 13 | PASS |
| genuine_drift_month_7_small_window | genuine_drift | genuine_drift | retrain | retrain | 0.93 | 11 | PASS |
| low_null_rate_subthreshold | no_incident | - | no_action | - | - | 0 | PASS |
| pipeline_bug_customer_age_30pct | pipeline_bug | pipeline_bug | alert_human | alert_human | 0.95 | 6 | PASS |
| pipeline_bug_income_50pct | pipeline_bug | pipeline_bug | alert_human | alert_human | 0.95 | 5 | PASS |
| pipeline_bug_proposed_credit_limit_70pct | pipeline_bug | pipeline_bug | alert_human | alert_human | 0.97 | 5 | PASS |
| pipeline_bug_session_length_15pct | no_incident | - | no_action | - | - | 0 | PASS |
| pipeline_bug_zip_count_4w_40pct | pipeline_bug | pipeline_bug | alert_human | alert_human | 0.95 | 5 | PASS |
| schema_change_device_os | schema_change | schema_change | alert_human | alert_human | 0.92 | 7 | PASS |
| schema_change_employment_status | schema_change | schema_change | alert_human | alert_human | 0.93 | 6 | PASS |
| schema_change_housing_status | schema_change | schema_change | alert_human | alert_human | 0.93 | 6 | PASS |
| schema_change_payment_zzz | schema_change | schema_change | alert_human | alert_human | 0.93 | 6 | PASS |
| subtle_drift_month_2 | genuine_drift | genuine_drift | retrain | retrain | 0.91 | 8 | PASS |

