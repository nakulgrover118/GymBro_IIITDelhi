# Video script: ThaliTwin (target 22 minutes)

**Format:** screen recording. Present `docs/ThaliTwin_Presentation.pptx` (speaker notes are in the deck), then switch to the live dashboard and a terminal. Upload as an **unlisted YouTube video** and paste the link into the README.

**Before recording**
1. Run `streamlit run dashboard/app.py` and open http://localhost:8501. Set the sidebar clock to **day 5, 13:00**.
2. In a second terminal, run `uvicorn api.main:app` and open http://127.0.0.1:8000/docs.
3. Open the deck in slideshow mode.
4. Close notifications and use 1080p.

Every number below comes from `docs/results.md`. If you re-run the pipeline, re-check them there.

---

## Part 1: Why (0:00–3:00), slides 1–3

| Time | Slide | Say (key points) |
|---|---|---|
| 0:00 | 1 Title | Introduce Team Gym Bro (IIIT Delhi). One line: *"ThaliTwin is a digital twin for type-2 diabetes that sees glucose spikes coming up to two hours ahead, explains why, and lets doctors test what would prevent them."* Explain the name: a *thali* is the everyday Indian plate, where most post-meal spikes begin; ThaliTwin knows what is on the plate and simulates what it will do. |
| 0:45 | 2 Problem | 101 M adults with diabetes, 136 M with prediabetes (ICMR-INDIAB 2023). Care is built on HbA1c about 4 times a year, while a CGM gives 288 readings a day. Post-meal spikes are common with Indian diets and are invisible until much later. State the challenge question on the slide. |
| 2:00 | 3 Use case | Predict, explain, act. Users are care teams running remote monitoring. Stress that it is decision support only, never automated dosing. |

## Part 2: How it works (3:00–9:30), slides 4–8

| Time | Slide | Say |
|---|---|---|
| 3:00 | 4 Architecture | Walk left to right through the five layers: data sources, ingestion and features, twin engine, ML layer, delivery. Point out the governance box. |
| 4:30 | 5 Data | Static stream: synthetic EHR as FHIR R4, ICD-10, LOINC labs with 2-year history, medicines, genetics. Dynamic stream: 5-min CGM, steps, HR, HRV, sleep stages and logs. Real-world track: ShanghaiT2DM, 100 people, CC BY 4.0. Everything is synthetic or anonymised, as the sandbox rules require. |
| 5:45 | 6 Built for India | Regional menus with GI, Asian BMI cut-offs and the thin-fat phenotype, Indian prescribing, chai and late dinners, festival sweets and missed doses, TCF7L2/KCNQ1. |
| 6:45 | 7 Twin engine | The four steps. **Explain why this is a twin, not just a predictor:** it has a personal physiology that is initialised from the EHR, re-learned every night, kept in sync with every CGM reading, and can simulate *what if*. Show the learning chart: correlation with the hidden true insulin sensitivity rises from about 0.80 to about 0.88. |
| 8:15 | 8 AI layer | 94 features across 5 streams. LightGBM classifier and forecasters, conformal 80% intervals, exact SHAP, a GRU for comparison, and patient-level splits. |

## Part 3: Does it work? (9:30–14:00), slides 9–12

| Time | Slide | Say |
|---|---|---|
| 9:30 | 9 Alert results | Synthetic held-out patients: **AUROC 0.957** for a new spike within 2 h. **99% of 804 spike events flagged ≥15 min ahead**, median warning **110 min**. Walk through the ablation: CGM only 0.931, plus wearables and logs 0.953, plus twin 0.957. A trend rule gets 0.706. |
| 10:45 | 10 Forecasting | Error roughly halves vs a "no change" forecast (1-h RMSE 13.9 vs 27.0 mg/dL). Clarke zone A coverage. Real data: about 25% lower error. Conformal intervals hit their 80% target. |
| 11:45 | 11 Real-world | ShanghaiT2DM with 5-fold CV by person. Read the AUROC and RMSE off the slide. **Say the honest part:** with no wearables and free-text meals, the extra streams add little here; prospective validation in India is the next step. |
| 13:00 | 12 Explainability | Share of SHAP by stream (the twin's simulation is the largest). Read the example alert's reasons aloud. |

## Part 4: Live demo (14:00–20:00), dashboard + API

1. **Ward overview (14:00).** "Thirty monitored patients ranked by risk." Point out the KPI tiles, the spike-risk bars and the *main driver* column. Pick the top amber patient and click **Open twin →**.
2. **Patient twin (15:00).**
   - Read the alert banner and its reasons.
   - Explain the chart: blue is CGM, orange is the ThaliTwin forecast with its 80% band, the dashed line is the twin's physiology-only simulation. Point out meal markers (🍽 carbs logged), medicine markers (red × = missed dose), the spike-risk strip, steps, and heart rate with HRV and the asleep band.
   - Right side: the *Why this prediction* bars and the twin's learned insulin sensitivity across days.
   - Scroll down: EHR with ICD-10 chips, medicines, genetics, LOINC labs, the HbA1c trend, the 7-day glucose profile and time in ranges.
3. **Live replay (17:00).** Toggle **▶ Live replay** in the sidebar and let the clock advance about an hour. Show the risk and forecast updating as new CGM readings arrive. Toggle it off.
4. **What-if simulator (17:45).** For the same patient, choose *Rice + dal + sabzi* in 15 min. Compare the alternatives the twin proposes: walk after the meal, half portion, millet swap. Then move *Last night's sleep* down to 4.5 h and show the peak rise. Say: "personal, because it uses this patient's calibrated twin."
5. **Model performance page (19:00).** Show the ablation, the Clarke grid, the alert-burden curve (sensitive vs quiet mode) and the real-data tab.
6. **API (19:30).** In the Swagger UI, run `GET /patients/{id}/state` and `POST /patients/{id}/what-if`. "Any EHR, CGM app or WhatsApp bot can plug in."

## Part 5: Responsible AI and close (20:00–22:00), slides 15–18

| Time | Slide | Say |
|---|---|---|
| 20:00 | 15 Alert setting | Sensitive vs quiet mode, and how alarm fatigue is handled with a 60-min snooze. |
| 20:40 | 16 Privacy and limits | Synthetic or anonymised data only, DPDP principles, decision support. Read the limitations aloud; judges value candour. |
| 21:15 | 17 Roadmap | IEC-approved pilot, ABDM/FHIR integration, federated learning, hypertension and CVD twins. |
| 21:45 | 18 Thanks | Repo link and the one-command rebuild (`python run_all.py`). |

**Tip:** if you go long, trim Part 2 (slides 5–6), not the demo.
