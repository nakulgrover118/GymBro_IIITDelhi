# Data card: GlucoTwin

GlucoTwin follows the challenge's sandbox rules. It uses **synthetic** data (for the full EHR + wearable fusion) and **anonymised open data** (for real-world validation). No identifiable patient data is collected, stored or needed.

## 1. Synthetic Indian type-2 diabetes cohort (main development set)

| | |
|---|---|
| Patients | 200 virtual adults with T2D, 14 days each |
| Generator | `glucotwin/data/ehr_generator.py` (EHR) + `glucotwin/data/simulator.py` (daily life, physiology, sensors). Seed `2026`, fully reproducible |
| Static stream (EHR) | Demographics, region, diet pattern, occupation; ICD-10 diagnoses; LOINC-coded labs (HbA1c, FPG, fasting insulin, lipids, creatinine/eGFR, UACR, ALT, BP, BMI, waist); 2-year quarterly lab history; medicines; genetic markers (TCF7L2 rs7903146, KCNQ1 rs2237892 risk-allele counts, polygenic risk z-score). Exported as CSV and as FHIR R4 bundles (`data/synthetic/fhir/`) |
| Dynamic stream (wearables) | Every 5 min: CGM glucose, steps, heart rate, HRV (RMSSD), sleep stage (wake/light/deep/REM) |
| App logs | Meals from a 35-dish Indian menu (carbs, GI), logged with ~80–95% adherence and ±18% carb-estimation error; medicine doses (taken/missed) |
| Splits | Patient-level 70/15/15 (140/30/30). No patient appears in two splits |

**How it was generated**
- Distributions are informed by ICMR-INDIAB (Anjana et al., *Lancet Diabetes Endocrinol* 2023): age about 52 ± 10, BMI about 26 (Asian cut-offs), HbA1c about 7.9 ± 1.2%, hypertension 35–60% depending on age/BMI, and the "thin-fat" phenotype (high insulin resistance at a modest BMI). Prescribing follows common Indian practice: metformin about 92%, sulfonylurea about 30–60%, DPP-4 inhibitors about 38%, premixed insulin for long-duration, poorly controlled patients.
- Daily routines: wake/bed times, regional breakfasts (idli, poha, paratha...), lunch around 13:00, evening chai or snack, late dinner around 20:45, weekend shifts, post-meal walks, morning walks, desk-job stress days, festival sweets and missed doses.
- Physiology: an extended Bergman minimal model with insulin secretion, a two-compartment gut model driven by glycaemic index, exercise-driven glucose uptake, raised insulin sensitivity after exercise, lower insulin sensitivity after short sleep (Donga 2010; Spiegel 1999), the dawn phenomenon, stress-driven hepatic output, missed-metformin effects and slow unmodelled drift.
- Each patient's *true* physiology is a hidden perturbation of what their EHR implies, so the twin has to learn it.
- Sensors: CGM with interstitial lag (about 8 min), calibration bias, AR(1) noise and dropouts; heart rate driven by activity, stress, sleep and meals; HRV reduced by stress, activity and poor control.

**Known limitations.** The coupling between wearables and glucose is built from published effect sizes, not from measured data. The ML results on this cohort therefore show that the *pipeline* can exploit such effects; they are not clinical evidence that the effects have this size.

## 2. ShanghaiT2DM (real-world validation set)

| | |
|---|---|
| Source | Zhao Q. *et al.* "Chinese diabetes datasets for data-driven machine learning", *Scientific Data* 10, 35 (2023). figshare doi:10.6084/m9.figshare.20444397 |
| Licence | CC BY 4.0 (attribution above) |
| Content used | 109 CGM recordings (100 people with T2D), 15-min CGM for 3–14 days; free-text diet logs; insulin doses (s.c., pump bolus, i.v.); clinical summary (age, sex, BMI, HbA1c, fasting glucose/insulin, lipids, eGFR, comorbidities, drugs) |
| Not available | Wearable steps/HR/HRV/sleep, genetics |
| Processing | `glucotwin/data/shanghai.py`: timestamps rounded to a 15-min grid, carbs estimated from free text with a food-to-carbohydrate table, insulin types mapped to absorption speeds, units converted (HbA1c mmol/mol to %, pmol/L to µU/mL, mmol/L to mg/dL) |
| Splits | Person-level 70/15/15. Repeat recordings from the same person stay in one split |

## 3. Privacy and governance
- No real identifiers anywhere. Synthetic names are random combinations of common Indian names, and every record carries a `synthetic` tag.
- The ShanghaiT2DM data was anonymised by its publishers; we only use their numeric patient codes.
- The design follows DPDP Act 2023 principles (purpose limitation, data minimisation, storage limitation). The API exposes model outputs, not raw data dumps.
- **GlucoTwin is a research prototype, not a medical device.** It is decision support for clinicians and is not intended for dosing.
