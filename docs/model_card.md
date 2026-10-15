# Model card: ThaliTwin spike predictor and glucose forecaster

## Overview
| | |
|---|---|
| Models | LightGBM binary classifier (spike) + three LightGBM regressors (30/60/120-min glucose change), fed by a personalised physiological digital twin |
| Version | 1.0 (Digital Twin Challenge 2026 proof of concept) |
| Files | `artifacts/models/synthetic/` (deployed in the dashboard and API), `artifacts/models/shanghai/` (real-data fold-1 model, for reference) |
| Owner | Team Gym Bro, IIIT Delhi |
| Licence | MIT |

## Intended use
- **Intended:** decision support for care teams (endocrinologists, physicians, diabetes educators) remotely monitoring adults with type-2 diabetes who wear a CGM and, ideally, a smartwatch. It prioritises patients, gives early warning of post-meal hyperglycaemia, and helps with counselling through what-if simulation.
- **Not intended:** insulin or medication dosing, type-1 diabetes, children, pregnancy, hospital inpatient glycaemic control, or any autonomous clinical action. It is not a medical device.

## Inputs (at time *t*, using only data up to *t*)
CGM history; EHR (age, sex, BMI, HbA1c, fasting glucose and insulin, HOMA-IR, eGFR, lipids, BP, medicines, genotypes); wearables (steps, heart rate, HRV, sleep stages); app logs (meals with estimated carbs, medicine doses); twin outputs (simulated glucose at 30/60/120 min, simulated peak, learned insulin sensitivity, disturbance).

## Outputs
- P(glucose crosses 180 mg/dL within the next 2 h | currently < 180). Alerts are raised at a threshold set on validation patients for 80% sensitivity, with a 60-min snooze on repeat notifications.
- Glucose forecasts at 30/60/120 min with split-conformal 80% intervals.
- The top drivers of each alert, in plain language (TreeSHAP).

## Training and evaluation data
- **Synthetic Indian cohort.** 200 patients × 14 days at 5-min resolution. Patient-level split of 140 train / 30 validation / 30 test. The first 2 days are twin warm-up and are excluded from evaluation.
- **ShanghaiT2DM.** 100 real people (109 recordings) at 15-min resolution, 5-fold cross-validation grouped by person.

See [`data_card.md`](data_card.md).

## Performance (held-out patients)
| | Synthetic | ShanghaiT2DM (real) |
|---|---|---|
| AUROC, new spike within 2 h | 0.957 | 0.878 |
| AUPRC (prevalence) | 0.916 (0.33) | 0.601 (0.15) |
| Sensitivity / specificity at the deployed threshold | 0.84 / 0.91 | 0.80 / 0.80 |
| Spike events flagged ≥ 15 min ahead (median warning) | 98.8% (110 min) | 99.0% (120 min) |
| False alerts per patient-day: default / quiet threshold | 2.7 / 1.0 | 4.4 / 0.9 |
| Glucose RMSE 30 / 60 / 120 min (mg/dL) | 8.4 / 13.9 / 21.9 | 12.6 / 22.2 / 31.6 |
| No-change baseline RMSE | 16.0 / 27.0 / 39.8 | 17.6 / 29.4 / 43.9 |
| Clarke zone A (60 min) | 96.7% | 83.9% |
| 80% interval coverage | 80.1–80.5% | 79.6–80.5% |

Full tables, including the ablation and the deep-learning comparison, are in [`results.md`](results.md).

## Ethical considerations and limitations
- **Synthetic optimism.** The synthetic cohort and the twin share the same physiological structure, and the wearable effects on glucose come from literature effect sizes. Treat synthetic metrics as an upper bound and real-data metrics as the conservative estimate.
- **Population shift.** The real-data validation is a Chinese hospital cohort. Indian outpatients, diets and devices may differ, so the system needs local prospective validation and recalibration.
- **Alarm fatigue.** The default threshold favours sensitivity. Programmes should choose an operating point from the trade-off table.
- **Fairness.** Performance has not yet been audited by sex, age, region or insulin use. This is required before any deployment.
- **Data quality.** Missed or mis-estimated meal logs degrade forecasts. The disturbance observer partly compensates.
- **Privacy.** Only synthetic or published anonymised data is used, and no identifiers are needed at inference.
