"""Synthetic Electronic Health Records for an Indian type-2 diabetes cohort.

Synthea-style generator written in Python (no Java needed). Distributions
are loosely calibrated to Indian epidemiology: ICMR-INDIAB (Anjana et al.,
Lancet Diabetes Endocrinol 2023), the "thin-fat" Asian Indian phenotype
(diabetes at lower BMI, high insulin resistance) and common prescribing
patterns in India (metformin + sulfonylurea / DPP-4 inhibitors, premixed
insulin).

Every record is fully synthetic - no real person is represented.

Outputs
-------
* a flat table (one row per patient) used by the models and the dashboard
* a FHIR R4-style JSON bundle per patient (Patient, Condition, Observation,
  MedicationStatement) to show how the twin would read a real EHR
* a 2-year lab history (HbA1c, FPG, lipids) for trend charts
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

FIRST_M = ["Rahul", "Amit", "Suresh", "Ramesh", "Vikram", "Arjun", "Sanjay", "Rajesh", "Anil", "Manoj",
           "Karthik", "Venkatesh", "Srinivas", "Prakash", "Deepak", "Imran", "Farhan", "Harpreet", "Gurpreet",
           "Subhash", "Mahesh", "Naveen", "Ravi", "Arun", "Sunil", "Joseph", "Thomas", "Abhishek", "Rohit", "Pranav"]
FIRST_F = ["Priya", "Sunita", "Lakshmi", "Anita", "Kavita", "Meena", "Geeta", "Pooja", "Divya", "Radha",
           "Shanthi", "Revathi", "Fatima", "Ayesha", "Simran", "Harleen", "Rekha", "Usha", "Neha", "Anjali",
           "Bhavna", "Deepa", "Savita", "Mary", "Asha", "Swati", "Nandini", "Padma", "Jyoti", "Smita"]
LAST = {"north": ["Sharma", "Verma", "Gupta", "Singh", "Malhotra", "Kapoor", "Chauhan", "Yadav", "Khan", "Gill"],
        "south": ["Reddy", "Iyer", "Nair", "Rao", "Pillai", "Krishnan", "Menon", "Naidu", "Subramanian", "Kumar"],
        "west": ["Patel", "Shah", "Desai", "Joshi", "Kulkarni", "Patil", "Mehta", "Pawar", "Shinde", "Bhatt"],
        "east": ["Das", "Banerjee", "Chatterjee", "Mukherjee", "Bose", "Ghosh", "Sen", "Mohanty", "Roy", "Dutta"]}
CITIES = {"north": ["New Delhi", "Lucknow", "Chandigarh", "Jaipur", "Kanpur"],
          "south": ["Bengaluru", "Chennai", "Hyderabad", "Kochi", "Mysuru"],
          "west": ["Mumbai", "Pune", "Ahmedabad", "Surat", "Nagpur"],
          "east": ["Kolkata", "Bhubaneswar", "Patna", "Guwahati", "Ranchi"]}

LOINC = {
    "hba1c": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    "fpg": ("1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma", "mg/dL"),
    "fasting_insulin": ("20448-7", "Insulin [Units/volume] in Serum or Plasma", "uU/mL"),
    "total_chol": ("2093-3", "Cholesterol [Mass/volume] in Serum or Plasma", "mg/dL"),
    "ldl": ("13457-7", "LDL Cholesterol (calculated)", "mg/dL"),
    "hdl": ("2085-9", "HDL Cholesterol [Mass/volume] in Serum or Plasma", "mg/dL"),
    "triglycerides": ("2571-8", "Triglyceride [Mass/volume] in Serum or Plasma", "mg/dL"),
    "creatinine": ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", "mg/dL"),
    "egfr": ("62238-1", "eGFR (CKD-EPI)", "mL/min/1.73m2"),
    "alt": ("1742-6", "Alanine aminotransferase [Enzymatic activity/volume]", "U/L"),
    "uacr": ("9318-7", "Albumin/Creatinine [Mass Ratio] in Urine", "mg/g"),
    "bmi": ("39156-5", "Body mass index (BMI)", "kg/m2"),
    "weight_kg": ("29463-7", "Body weight", "kg"),
    "height_cm": ("8302-2", "Body height", "cm"),
    "waist_cm": ("8280-0", "Waist circumference", "cm"),
    "sbp": ("8480-6", "Systolic blood pressure", "mm[Hg]"),
    "dbp": ("8462-4", "Diastolic blood pressure", "mm[Hg]"),
}

CONDITIONS = {
    "t2d": ("E11.9", "Type 2 diabetes mellitus"),
    "hypertension": ("I10", "Essential (primary) hypertension"),
    "dyslipidemia": ("E78.5", "Hyperlipidaemia, unspecified"),
    "obesity": ("E66.9", "Obesity, unspecified"),
    "nafld": ("K76.0", "Fatty (change of) liver, not elsewhere classified"),
    "ckd": ("N18.3", "Chronic kidney disease, stage 3"),
    "neuropathy": ("E11.40", "Type 2 diabetes with diabetic neuropathy"),
    "retinopathy": ("E11.319", "Type 2 diabetes with diabetic retinopathy"),
}


def _clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def generate_patient(rng: np.random.Generator, idx: int) -> dict:
    sex = "M" if rng.random() < 0.55 else "F"
    region = rng.choice(["north", "south", "west", "east"], p=[0.32, 0.30, 0.22, 0.16])
    first = rng.choice(FIRST_M if sex == "M" else FIRST_F)
    name = f"{first} {rng.choice(LAST[region])}"
    age = int(np.clip(rng.normal(52, 10), 30, 78))
    vegetarian = bool(rng.random() < {"north": 0.45, "south": 0.30, "west": 0.55, "east": 0.10}[region])

    height = rng.normal(166 if sex == "M" else 153, 6.5)
    bmi = _clip(rng.normal(26.3, 3.8), 19.0, 40.0)       # Asian cut-offs: >=23 overweight, >=25 obese
    weight = bmi * (height / 100) ** 2
    waist = _clip((88 if sex == "M" else 84) + 2.1 * (bmi - 25) + rng.normal(0, 5), 68, 130)

    duration = _clip(rng.gamma(2.0, 3.8), 0.3, 30.0)
    family_history = bool(rng.random() < 0.58)
    # Genetics: TCF7L2 rs7903146 (risk allele T, freq ~0.30 in South Asians),
    # KCNQ1 rs2237892 (risk allele C, freq ~0.60), plus a polygenic risk score.
    tcf7l2 = int(rng.binomial(2, 0.30))
    kcnq1 = int(rng.binomial(2, 0.60))
    prs = float(rng.normal(0.25 * family_history, 1.0))

    occupation = rng.choice(["desk job", "field work", "homemaker", "retired", "shopkeeper"],
                            p=[0.34, 0.12, 0.24, 0.18, 0.12] if age < 60 else [0.05, 0.03, 0.30, 0.55, 0.07])

    # Glycaemic control: poorer with longer duration and higher genetic risk
    hba1c = _clip(rng.normal(7.9 + 0.05 * (duration - 7) + 0.15 * prs + 0.1 * tcf7l2, 1.2), 6.3, 12.5)
    eag = 28.7 * hba1c - 46.7                              # ADAG estimated average glucose
    fpg = _clip(18 + 0.72 * eag + rng.normal(0, 12), 95, 280)
    fasting_insulin = float(np.exp(rng.normal(np.log(11) + 0.04 * (bmi - 25) - 0.02 * (duration - 7)
                                               + 0.05 * kcnq1, 0.33)))
    fasting_insulin = _clip(fasting_insulin, 3, 40)
    ppg = _clip(fpg + rng.normal(75, 25), fpg + 20, 420)

    hypertension = bool(rng.random() < 0.35 + 0.006 * (age - 45) + 0.01 * (bmi - 25))
    sbp = _clip(rng.normal(138 if hypertension else 124, 12), 100, 190)
    dbp = _clip(sbp * 0.62 + rng.normal(0, 6), 60, 115)
    tg = _clip(np.exp(rng.normal(np.log(165) + 0.05 * (hba1c - 8), 0.35)), 60, 600)
    hdl = _clip(rng.normal(42 if sex == "M" else 48, 8), 22, 85)
    ldl = _clip(rng.normal(112, 30), 45, 220)
    total_chol = ldl + hdl + tg / 5
    dyslipidemia = bool(tg > 150 or ldl > 100 or hdl < (40 if sex == "M" else 50))
    egfr = _clip(rng.normal(98 - 0.8 * (age - 40) - 0.6 * duration, 13), 25, 125)
    creat = _clip((1.0 if sex == "M" else 0.8) * (90 / egfr) ** 0.9, 0.5, 3.5)
    uacr = _clip(np.exp(rng.normal(np.log(18) + 0.05 * duration + 0.1 * (hba1c - 8), 0.9)), 2, 1500)
    alt = _clip(np.exp(rng.normal(np.log(30) + 0.03 * (bmi - 25), 0.4)), 8, 160)
    nafld = bool(alt > 40 or (bmi > 27 and rng.random() < 0.5))
    ckd = egfr < 60
    neuropathy = bool(rng.random() < 0.05 + 0.015 * duration)
    retinopathy = bool(rng.random() < 0.03 + 0.012 * duration + 0.02 * (hba1c - 8))

    # Medications (typical Indian prescribing)
    metformin = bool(egfr >= 30 and rng.random() < 0.92)
    sulfonylurea = bool(rng.random() < 0.30 + 0.02 * duration)
    dpp4 = bool(rng.random() < 0.38)
    sglt2 = bool(rng.random() < 0.18)
    insulin = bool(hba1c > 9.0 and duration > 6 and rng.random() < 0.6) or bool(duration > 15 and rng.random() < 0.4)
    insulin_regimen = (rng.choice(["premix_30_70", "basal_glargine"], p=[0.65, 0.35]) if insulin else "none")
    statin = bool(dyslipidemia and rng.random() < 0.7)
    arb = bool(hypertension and rng.random() < 0.75)

    # Lifestyle persona (drives the wearable/lifestyle simulator)
    active = {"field work": 0.75, "homemaker": 0.5, "shopkeeper": 0.45, "retired": 0.45, "desk job": 0.25}[occupation]
    persona = {
        "wake_min": int(rng.normal(390, 35)),                    # ~06:30
        "bedtime_min": int(rng.normal(1380, 40)),                # ~23:00
        "morning_walk_prob": _clip(rng.beta(2, 2) * (0.4 + active), 0, 0.95),
        "post_meal_walk_prob": _clip(rng.beta(1.5, 3), 0, 0.9),
        "activity_level": active,
        "bad_sleep_prob": _clip(rng.beta(2, 7), 0.02, 0.6),
        "stress_prob": _clip(rng.beta(2, 6) * (1.4 if occupation == "desk job" else 1.0), 0.02, 0.6),
        "sweets_prob": _clip(rng.beta(2, 5), 0.02, 0.7),
        "portion": _clip(rng.normal(1.0 + 0.012 * (bmi - 25), 0.12), 0.7, 1.45),
        "snack_prob": _clip(rng.beta(4, 2), 0.2, 0.98),
        "med_adherence": _clip(rng.beta(9, 1.6), 0.5, 1.0),
        "logging_adherence": _clip(rng.beta(8, 1.5), 0.55, 1.0),
        "resting_hr": _clip(rng.normal(74 - 6 * active + 0.3 * (bmi - 25), 6), 55, 95),
        "rmssd": _clip(rng.normal(36 - 0.35 * (age - 40) - 0.4 * duration - 1.2 * (hba1c - 7), 7), 10, 70),
    }

    pid = f"GT-{idx:04d}"
    return dict(
        patient_id=pid, name=name, synthetic=True, sex=sex, age=age, region=region,
        city=str(rng.choice(CITIES[region])), vegetarian=vegetarian, occupation=str(occupation),
        height_cm=round(height, 1), weight_kg=round(weight, 1), bmi=round(bmi, 1), waist_cm=round(waist, 1),
        diabetes_duration_y=round(duration, 1), family_history_t2d=family_history,
        tcf7l2_rs7903146_risk_alleles=tcf7l2, kcnq1_rs2237892_risk_alleles=kcnq1, polygenic_risk_z=round(prs, 2),
        hba1c=round(hba1c, 1), fpg=round(fpg), fasting_insulin=round(fasting_insulin, 1), ppg_2h=round(ppg),
        homa_ir=round(fpg * fasting_insulin / 405, 2),
        sbp=round(sbp), dbp=round(dbp), total_chol=round(total_chol), ldl=round(ldl), hdl=round(hdl),
        triglycerides=round(tg), creatinine=round(creat, 2), egfr=round(egfr), uacr=round(uacr, 1), alt=round(alt),
        hypertension=hypertension, dyslipidemia=dyslipidemia, obesity=bmi >= 27.5, nafld=nafld, ckd=ckd,
        neuropathy=neuropathy, retinopathy=retinopathy,
        metformin=metformin, sulfonylurea=sulfonylurea, dpp4=dpp4, sglt2=sglt2,
        insulin=insulin, insulin_regimen=str(insulin_regimen), statin=statin, arb=arb,
        smoker=bool(sex == "M" and rng.random() < 0.18), alcohol=bool(sex == "M" and rng.random() < 0.25),
        **{f"persona_{k}": v for k, v in persona.items()},
    )


def generate_cohort(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame([generate_patient(rng, i + 1) for i in range(n)])


def lab_history(ehr: pd.DataFrame, seed: int, end: date) -> pd.DataFrame:
    """Quarterly labs for the 2 years before the monitoring period."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    for r in ehr.itertuples():
        start_a1c = r.hba1c + rng.normal(0.4, 0.8)
        for q in range(8, 0, -1):
            d = end - timedelta(days=91 * q - int(rng.integers(-10, 10)))
            frac = (8 - q) / 8
            a1c = start_a1c + (r.hba1c - start_a1c) * frac + rng.normal(0, 0.25)
            rows.append(dict(patient_id=r.patient_id, date=d.isoformat(), hba1c=round(a1c, 1),
                             fpg=round(r.fpg + 25 * (a1c - r.hba1c) + rng.normal(0, 8)),
                             ldl=round(r.ldl + rng.normal(0, 8)), egfr=round(r.egfr + rng.normal(0, 3) + 0.5 * q),
                             weight_kg=round(r.weight_kg + rng.normal(0.3 * q, 0.8), 1)))
        rows.append(dict(patient_id=r.patient_id, date=end.isoformat(), hba1c=r.hba1c, fpg=r.fpg,
                         ldl=r.ldl, egfr=r.egfr, weight_kg=r.weight_kg))
    return pd.DataFrame(rows)


def to_fhir_bundle(r: dict, obs_date: date) -> dict:
    """Minimal FHIR R4 collection bundle for one patient."""
    pid = r["patient_id"]
    entries = [{"resource": {
        "resourceType": "Patient", "id": pid,
        "meta": {"tag": [{"system": "urn:glucotwin", "code": "synthetic"}]},
        "name": [{"text": r["name"]}], "gender": "male" if r["sex"] == "M" else "female",
        "birthDate": str(obs_date.year - r["age"]),
        "address": [{"city": r["city"], "country": "IN"}],
    }}]
    for key, (code, text) in CONDITIONS.items():
        if key == "t2d" or r.get(key):
            entries.append({"resource": {
                "resourceType": "Condition", "subject": {"reference": f"Patient/{pid}"},
                "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": code, "display": text}]},
                "clinicalStatus": {"coding": [{"code": "active"}]}}})
    for key, (code, text, unit) in LOINC.items():
        entries.append({"resource": {
            "resourceType": "Observation", "status": "final", "subject": {"reference": f"Patient/{pid}"},
            "effectiveDateTime": obs_date.isoformat(),
            "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": text}]},
            "valueQuantity": {"value": r[key], "unit": unit}}})
    entries.append({"resource": {
        "resourceType": "Observation", "status": "final", "subject": {"reference": f"Patient/{pid}"},
        "code": {"text": "Genotype TCF7L2 rs7903146 (risk allele count)"},
        "valueInteger": r["tcf7l2_rs7903146_risk_alleles"]}})
    meds = {"metformin": "Metformin 500 mg BD", "sulfonylurea": "Glimepiride 2 mg OD",
            "dpp4": "Teneligliptin 20 mg OD", "sglt2": "Dapagliflozin 10 mg OD",
            "statin": "Atorvastatin 20 mg OD", "arb": "Telmisartan 40 mg OD"}
    if r["insulin"]:
        meds["insulin"] = ("Premixed insulin 30/70 BD" if r["insulin_regimen"] == "premix_30_70"
                           else "Insulin glargine OD at bedtime")
    for key, text in meds.items():
        if r.get(key):
            entries.append({"resource": {"resourceType": "MedicationStatement", "status": "active",
                                         "subject": {"reference": f"Patient/{pid}"},
                                         "medicationCodeableConcept": {"text": text}}})
    return {"resourceType": "Bundle", "type": "collection", "id": f"bundle-{pid}", "entry": entries}


def write_fhir(ehr: pd.DataFrame, out_dir, obs_date: date) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in ehr.to_dict("records"):
        r = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        with open(out_dir / f"{r['patient_id']}.json", "w", encoding="utf-8") as f:
            json.dump(to_fhir_bundle(r, obs_date), f, indent=1)
