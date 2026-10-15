// Builds docs/GlucoTwin_Presentation.pptx from artifacts/results/summary.json + docs/figures/*.png
// Usage:  cd docs/deck && npm install && node build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = path.join(ROOT, "docs", "figures");
const S_ALL = JSON.parse(fs.readFileSync(path.join(ROOT, "artifacts", "results", "summary.json"), "utf8"));
const S = S_ALL.synthetic, R = S_ALL.shanghai, W = S_ALL.whatif, TL = S_ALL.twin_learning, EX = S_ALL.example_alert;
const OUT = path.join(ROOT, "docs", "GlucoTwin_Presentation.pptx");

// ---------------------------------------------------------------- palette & type
const INK = "0F172A", DARK = "0B2E33", TEAL = "0F766E", MINT = "CCEDE8", TINT = "F0F7F6", AMBER = "D97706",
  RED = "DC2626", BLUE = "2563EB", PURPLE = "7C3AED", MUTED = "64748B", WHITE = "FFFFFF", LINE = "D7E3E1";
const HEAD = "Cambria", BODY = "Calibri";
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const f1 = (x) => Number(x).toFixed(1);
const SHORT = (m) => m.replace("Trend rule (glucose + 2x 30-min change)", "Trend rule")
  .replace("Digital twin alone (simulated peak)", "Twin alone (physiology)")
  .replace("LightGBM: GlucoTwin (fusion + digital twin)", "GlucoTwin: all streams + twin")
  .replace(/LightGBM: Fusion \(.*\)/, "ML: all streams, no twin")
  .replace("LightGBM: CGM + wearables & logs", "ML: CGM + wearables & logs")
  .replace("LightGBM: CGM + meal & insulin logs", "ML: CGM + meal/insulin logs")
  .replace("LightGBM: CGM + EHR", "ML: CGM + EHR").replace("LightGBM: CGM only", "ML: CGM only")
  .replace("Deep learning: GRU fusion (sequence + EHR + twin)", "Deep GRU: all streams + twin");

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}
async function fit(file, x, y, w, h) {
  const m = await sharp(file).metadata();
  const r = Math.min(w / m.width, h / m.height);
  const ww = m.width * r, hh = m.height * r;
  return { path: file, x: x + (w - ww) / 2, y: y + (h - hh) / 2, w: ww, h: hh };
}
const shadow = () => ({ type: "outer", color: "94A3B8", blur: 6, offset: 1.5, angle: 90, opacity: 0.25 });

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
  pres.author = "Team Gym Bro (IIIT Delhi)";
  pres.company = "IIIT Delhi";
  pres.title = "GlucoTwin - Digital Twin for Type-2 Diabetes";
  pres.subject = "Digital Twin Challenge 2026";
  pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: DARK },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 7.6, h: 1.5, fontFace: HEAD, fontSize: 60, bold: true, color: WHITE, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.65, w: 7.4, h: 1.4, fontFace: BODY, fontSize: 22, color: MINT, valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: WHITE },
    margin: [0.5, 0.6, 0.6, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontFace: HEAD, fontSize: 34, bold: true, color: INK, valign: "middle", align: "left", margin: 0 }, text: "" } },
      { text: { text: "GlucoTwin  ·  Team Gym Bro  ·  IIIT Delhi  ·  Digital Twin Challenge 2026", options: { x: 0.6, y: 7.05, w: 9, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED, margin: 0 } } },
    ],
    slideNumber: { x: 12.25, y: 7.05, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED, align: "right" },
  });

  const I = {};
  for (const [k, C, col] of [["food", fa.FaUtensils, TEAL], ["bmi", fa.FaWeight, TEAL], ["pills", fa.FaPills, TEAL],
    ["chai", fa.FaMugHot, TEAL], ["sweet", fa.FaCandyCane, TEAL], ["dna", fa.FaDna, TEAL], ["bell", fa.FaBell, WHITE],
    ["brain", fa.FaBrain, WHITE], ["flask", fa.FaFlask, WHITE], ["user", fa.FaUserMd, WHITE], ["shield", fa.FaShieldAlt, TEAL],
    ["warn", fa.FaExclamationTriangle, TEAL], ["db", fa.FaDatabase, TEAL], ["road", fa.FaRoute, WHITE],
    ["heart", fa.FaHeartbeat, TEAL], ["heartW", fa.FaHeartbeat, WHITE], ["walk", fa.FaWalking, TEAL], ["sync", fa.FaSyncAlt, WHITE], ["cogs", fa.FaCogs, WHITE],
    ["chart", fa.FaChartLine, WHITE], ["hosp", fa.FaHospital, WHITE], ["lock", fa.FaLock, TEAL], ["mobile", fa.FaMobileAlt, WHITE]]) {
    I[k] = await icon(C, col);
  }
  const iconDot = (slide, key, x, y, d = 0.62, fill = TEAL) => {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: `icon-bg-${key}` });
    slide.addImage({ data: I[key], x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56, objectName: `icon-${key}` });
  };
  const card = (slide, x, y, w, h, fill = TINT, name = "card") =>
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, shadow: shadow(), objectName: name });
  const stat = (slide, x, y, w, value, label, color = TEAL, name = "stat") => {
    card(slide, x, y, w, 1.55, TINT, `${name}-card`);
    slide.addText(value, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.8, fontFace: HEAD, fontSize: 36, bold: true, color, margin: 0, isTextBox: true, objectName: `${name}-value` });
    slide.addText(label, { x: x + 0.2, y: y + 0.9, w: w - 0.4, h: 0.55, fontFace: BODY, fontSize: 12, color: MUTED, margin: 0, valign: "top", isTextBox: true, objectName: `${name}-label` });
  };
  const bullets = (items, size = 15) => items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6, fontSize: size, color: INK } }));
  const chartText = { catAxisLabelColor: "334155", valAxisLabelColor: "334155", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 10, dataLabelColor: "334155", titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: INK, legendFontFace: "+mn-lt", legendFontSize: 10, valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" } };
  const content = (title, section) => { const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section }); s.addText(title, { placeholder: "title" }); return s; };

  // ================================================================ 1. Title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("GlucoTwin", { placeholder: "title" });
  s.addText("A digital twin for type-2 diabetes that sees glucose spikes coming up to 2 hours ahead, explains why, and lets doctors test what would prevent them", { placeholder: "body" });
  s.addText([{ text: "Team Gym Bro  ·  IIIT Delhi", options: { bold: true, breakLine: true } }, { text: "Digital Twin Challenge 2026  ·  Happiest Health Summit, Bengaluru" }],
    { x: 0.8, y: 5.6, w: 7.5, h: 0.8, fontFace: BODY, fontSize: 15, color: WHITE, margin: 0, isTextBox: true, objectName: "team" });
  const tiles = [["2 h", "early warning of spikes > 180 mg/dL"], ["300", "patients: 200 synthetic Indian + 100 real (ShanghaiT2DM)"], [S.auroc.toFixed(2), "AUROC on held-out synthetic patients"]];
  tiles.forEach(([v, l], i) => {
    const y = 1.45 + i * 1.6;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.0, y, w: 3.6, h: 1.35, rectRadius: 0.12, fill: { color: "123F45" }, line: { color: "1D5A60" }, objectName: `title-tile-${i}` });
    s.addText(v, { x: 9.25, y: y + 0.1, w: 3.1, h: 0.7, fontFace: HEAD, fontSize: 32, bold: true, color: AMBER, margin: 0, isTextBox: true, objectName: `title-tile-v-${i}` });
    s.addText(l, { x: 9.25, y: y + 0.78, w: 3.2, h: 0.5, fontFace: BODY, fontSize: 12, color: MINT, margin: 0, isTextBox: true, objectName: `title-tile-l-${i}` });
  });
  s.addNotes("Hi, we are Team Gym Bro from IIIT Delhi. GlucoTwin is a proof-of-concept digital twin for people with type-2 diabetes. It fuses a patient's health record with live wearable data, predicts glucose spikes up to two hours before they happen, explains the reasons, and lets a doctor test alternatives on the virtual patient.");

  // ================================================================ 2. Problem
  s = content("India's diabetes burden is daily, but care is quarterly", "Introduction");
  stat(s, 0.6, 1.5, 3.0, "101 M", "adults in India live with diabetes (ICMR-INDIAB, 2023)", RED, "p1");
  stat(s, 3.85, 1.5, 3.0, "136 M", "more have prediabetes - the pipeline keeps growing", AMBER, "p2");
  stat(s, 7.1, 1.5, 2.6, "~4 / yr", "HbA1c checks: a 3-month average that hides daily spikes", BLUE, "p3");
  stat(s, 9.95, 1.5, 2.75, "288 / day", "readings from a CGM sensor - signal that is rarely acted on", TEAL, "p4");
  card(s, 0.6, 3.35, 12.1, 3.4, TINT, "problem-card");
  s.addText("Why spikes matter", { x: 0.95, y: 3.55, w: 5.5, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: INK, margin: 0, isTextBox: true });
  s.addText(bullets(["Post-meal glucose spikes are common in Indian diets (high-GI rice, roti, chai with sugar, late dinners)",
    "Repeated hyperglycaemia is linked to vascular complications, yet clinicians see it only weeks later",
    "Doctors cannot watch hundreds of CGM traces - they need early, explained, prioritised alerts"]), { x: 0.95, y: 4.1, w: 5.6, h: 2.5, fontFace: BODY, margin: 0, valign: "top" });
  s.addText("Our challenge question", { x: 7.0, y: 3.55, w: 5.4, h: 0.45, fontFace: HEAD, fontSize: 20, bold: true, color: INK, margin: 0, isTextBox: true });
  s.addText("Can a virtual replica of each patient - built from their EHR and continuously updated from wearables - predict a new glucose spike above 180 mg/dL up to 2 hours ahead, early and accurately enough for a care team to act?",
    { x: 7.0, y: 4.1, w: 5.4, h: 2.4, fontFace: BODY, fontSize: 16, italic: true, color: TEAL, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("India has about 101 million adults with diabetes and 136 million with prediabetes according to ICMR-INDIAB. Care revolves around HbA1c every few months, which averages away the daily post-meal spikes. CGMs produce 288 readings a day, but nobody can watch them all. Our question: can a digital twin predict new spikes two hours ahead, early enough to act?");

  // ================================================================ 3. Use case
  s = content("The use case: predict, explain, act", "Introduction");
  const steps = [["bell", "Predict", "Every 5 minutes the twin estimates the probability of a new spike above 180 mg/dL within 2 hours, plus 30/60/120-min glucose forecasts with 80% intervals."],
    ["brain", "Explain", "Each alert lists its drivers in plain language - e.g. a rice-heavy lunch, 4.5 h of sleep, no walk, high insulin resistance - from exact SHAP contributions."],
    ["flask", "Act", "The doctor or diabetes educator tests options on the virtual patient - smaller portion, millet swap, 15-min walk - and counsels with evidence."]];
  steps.forEach(([ic, h, t], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.55, 3.85, 3.6, TINT, `step-${i}`);
    iconDot(s, ic, x + 0.3, 1.8, 0.75, TEAL);
    s.addText(h, { x: x + 1.25, y: 1.9, w: 2.4, h: 0.55, fontFace: HEAD, fontSize: 24, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.3, y: 2.8, w: 3.3, h: 2.2, fontFace: BODY, fontSize: 15, color: INK, margin: 0, valign: "top", isTextBox: true });
  });
  iconDot(s, "user", 0.6, 5.55, 0.62, AMBER);
  s.addText([{ text: "Who uses it: ", options: { bold: true } }, { text: "endocrinologists, physicians and diabetes educators running remote-monitoring programmes for patients who wear a CGM and a smartwatch. Decision support only - not automated dosing." }],
    { x: 1.4, y: 5.45, w: 11.3, h: 0.85, fontFace: BODY, fontSize: 15, color: INK, margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("The use case has three verbs. Predict: every five minutes, the probability of a new spike in the next two hours. Explain: the top reasons in plain language. Act: the clinician tests alternatives on the twin before counselling. It is decision support for a care team, never automated dosing.");

  // ================================================================ 4. Architecture
  pres.addSection({ title: "Solution" });
  s = content("Architecture: five layers, from raw data to clinician", "Solution");
  s.addImage(await fit(path.join(FIG, "architecture.png"), 0.6, 1.3, 12.1, 5.6));
  s.addNotes("Five layers. Data sources: a synthetic EHR in FHIR format, wearable and CGM streams, app logs, and the real ShanghaiT2DM dataset. Ingestion aligns everything on a 5-minute grid. The twin engine is a personalised physiological model that calibrates nightly and synchronises live. The ML layer fuses features and twin simulations. Delivery is a doctor dashboard and a REST API.");

  // ================================================================ 5. Data
  s = content("Two data streams, fused on one timeline", "Solution");
  card(s, 0.6, 1.4, 4.3, 5.4, TINT, "ehr-card");
  iconDot(s, "db", 0.85, 1.6, 0.6, MINT);
  s.addText("Static: EHR (synthetic)", { x: 1.6, y: 1.65, w: 3.2, h: 0.5, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0, isTextBox: true });
  s.addText(bullets(["200 virtual Indian patients, FHIR R4 bundles", "ICD-10 diagnoses, LOINC-coded labs + 2-year history", "Medicines: metformin, sulfonylurea, DPP-4, premixed insulin", "Genetic markers: TCF7L2, KCNQ1, polygenic risk", "Real-world track: ShanghaiT2DM clinical summaries (100 people)"], 13),
    { x: 0.85, y: 2.35, w: 3.9, h: 4.3, fontFace: BODY, margin: 0, valign: "top" });
  s.addImage(await fit(path.join(FIG, "data_streams.png"), 5.15, 1.4, 7.55, 4.95));
  s.addText("Dynamic stream (5-min): CGM · steps · heart rate · HRV · sleep stages + meal & medicine logs", { x: 5.15, y: 6.4, w: 7.55, h: 0.4, fontFace: BODY, fontSize: 12, color: MUTED, margin: 0, align: "center", isTextBox: true });
  s.addNotes("The static stream is the EHR: demographics, ICD-10 diagnoses, LOINC labs with history, medicines and genetic markers, exported as FHIR bundles. The dynamic stream is five-minute CGM, steps, heart rate, HRV and sleep stages, plus meal and medicine logs. Everything is synthetic or anonymised, as the sandbox rules require.");

  // ================================================================ 6. India
  s = content("Built for Indian patients, not adapted from elsewhere", "Solution");
  const india = [["food", "Regional menus", "35 dishes - idli, poha, paratha, rajma chawal, ragi mudde, biryani - with carbs and glycaemic index"],
    ["bmi", "Asian phenotype", "Asian BMI cut-offs and the 'thin-fat' profile: high insulin resistance at modest BMI"],
    ["pills", "Indian prescribing", "Metformin, glimepiride, teneligliptin, premixed 30/70 insulin, as commonly prescribed"],
    ["chai", "Daily rhythm", "Sweet chai at 5 pm, dinner after 8.30 pm, weekend shifts, desk-job stress days"],
    ["sweet", "Festivals & adherence", "Mithai days, missed metformin doses and imperfect meal logging"],
    ["dna", "Genetics", "TCF7L2 and KCNQ1 variants relevant in South Asians lower beta-cell function"]];
  india.forEach(([ic, h, t], i) => {
    const x = 0.6 + (i % 3) * 4.1, y = 1.5 + Math.floor(i / 3) * 2.65;
    card(s, x, y, 3.85, 2.4, TINT, `india-${i}`);
    iconDot(s, ic, x + 0.25, y + 0.25, 0.62, MINT);
    s.addText(h, { x: x + 1.05, y: y + 0.3, w: 2.65, h: 0.55, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0, valign: "middle", isTextBox: true });
    s.addText(t, { x: x + 0.25, y: y + 1.0, w: 3.4, h: 1.3, fontFace: BODY, fontSize: 14, color: INK, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("We localised the problem. Meals come from a 35-dish regional Indian menu with carbs and glycaemic index. Patients follow the Asian thin-fat phenotype and Indian prescribing patterns. Routines include evening chai, late dinners, festival sweets and missed doses. Genetic markers relevant to South Asians shape beta-cell function.");

  // ================================================================ 7. Twin engine
  s = content("The twin: a personal physiology that keeps learning", "Solution");
  const tw = [["cogs", "1  Prior from the EHR", "Fasting glucose and insulin (HOMA-IR, HOMA-B), weight, HbA1c, medicines and genotype initialise a glucose-insulin model"],
    ["sync", "2  Nightly calibration", "A cross-entropy search re-learns insulin sensitivity, meal response, basal glucose and dawn effect from yesterday's CGM"],
    ["heartW", "3  Live synchronisation", "An observer nudges the twin to every CGM reading and learns unexplained inputs such as unlogged snacks or stress"],
    ["chart", "4  Forecast & what-if", "From the synced state the twin simulates the next hours using its learned routine - or any change a doctor proposes"]];
  tw.forEach(([ic, h, t], i) => {
    const y = 1.45 + i * 1.33;
    iconDot(s, ic, 0.6, y + 0.1, 0.62, TEAL);
    s.addText(h, { x: 1.4, y, w: 5.0, h: 0.42, fontFace: HEAD, fontSize: 17, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(t, { x: 1.4, y: y + 0.42, w: 5.1, h: 0.85, fontFace: BODY, fontSize: 13, color: INK, margin: 0, valign: "top", isTextBox: true });
  });
  s.addImage(await fit(path.join(FIG, "twin_learning.png"), 6.8, 1.45, 5.9, 2.6));
  card(s, 6.8, 4.35, 5.9, 2.35, TINT, "twin-proof");
  s.addText([{ text: "Does it learn the hidden patient? ", options: { bold: true, breakLine: true } },
    { text: `Each virtual patient has a hidden true insulin sensitivity. The EHR alone predicts it with correlation ${TL.si_corr_day0.toFixed(2)}; with nightly calibration the twin averages ${TL.si_corr_last.toFixed(2)} and its median error falls from ${f1(TL.si_err_day0)}% to ${f1(TL.si_err_last)}% (days 1-13). On real patients the synchronised twin alone already beats a "no change" forecast at every horizon.` }],
    { x: 7.05, y: 4.5, w: 5.45, h: 2.1, fontFace: BODY, fontSize: 14, color: INK, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("The twin is an extended Bergman minimal model with gut absorption by glycaemic index, exercise uptake, sleep-dependent insulin sensitivity and the dawn phenomenon. It starts from the EHR, re-calibrates every night, synchronises to each CGM reading with a disturbance observer, and simulates forward. The charts show it learning each patient's hidden insulin sensitivity.");

  // ================================================================ 8. AI layer
  s = content("AI layer: fusion, honest uncertainty, explanations", "Solution");
  s.addChart(pres.charts.BAR, [{ name: "features", labels: ["Twin simulation", "EHR", "CGM history", "Wearables", "Meal & medicine logs"], values: [18, 24, 23, 17, 12] }],
    { x: 0.6, y: 1.4, w: 5.6, h: 4.9, barDir: "bar", chartColors: [TEAL, BLUE, "334155", PURPLE, AMBER], showValue: true, dataLabelPosition: "outEnd", showLegend: false, showTitle: true, title: "94 features per time point, by data stream", valAxisHidden: true, ...chartText, objectName: "feature-chart" });
  const ai = [["LightGBM spike classifier", "P(new spike > 180 mg/dL within 2 h); threshold tuned on validation patients for 80% sensitivity; 60-min snooze against alarm fatigue"],
    ["LightGBM glucose forecasters", "30 / 60 / 120 min ahead with split-conformal 80% prediction intervals"],
    ["Exact TreeSHAP explanations", "per-prediction contributions grouped and translated into clinician language"],
    ["Fair comparisons", "patient-level splits; baselines (persistence, trend, twin-only), an ablation per data stream and a deep-learning GRU fusion model"]];
  ai.forEach(([h, t], i) => {
    const y = 1.45 + i * 1.3;
    card(s, 6.6, y, 6.1, 1.15, TINT, `ai-${i}`);
    s.addText([{ text: h, options: { bold: true, breakLine: true, fontSize: 15 } }, { text: t, options: { fontSize: 13 } }],
      { x: 6.85, y: y + 0.08, w: 5.7, h: 1.0, fontFace: BODY, color: INK, margin: 0, valign: "middle", isTextBox: true });
  });
  s.addNotes("The ML layer fuses 94 features: CGM history, EHR, wearables, logs and the twin's own simulations. LightGBM produces the spike probability and the glucose forecasts, conformal prediction gives honest 80% intervals, and TreeSHAP gives exact explanations. Everything is evaluated on patients the model never saw, against baselines and a deep-learning GRU.");

  // ================================================================ 9. Results: alerts
  pres.addSection({ title: "Results" });
  s = content("Results: spikes flagged early on unseen patients", "Results");
  stat(s, 0.6, 1.45, 2.85, S.auroc.toFixed(3), "AUROC - new spike > 180 mg/dL within 2 h", TEAL, "r1");
  stat(s, 0.6, 3.15, 2.85, pct(S.event_recall), `of ${S.events} spike events flagged >= 15 min ahead`, TEAL, "r2");
  stat(s, 3.65, 1.45, 2.85, `${Math.round(S.median_lead)} min`, "median warning before glucose crosses 180", AMBER, "r3");
  stat(s, 3.65, 3.15, 2.85, `${f1(S.false_per_day)} / day`, `false alerts per patient-day (${f1(S.events_per_day)} real spikes / day)`, MUTED, "r4");
  const abl = S.ablation.filter(([m]) => !m.startsWith("Trend"));
  s.addChart(pres.charts.BAR, [{ name: "AUROC", labels: abl.map(([m]) => SHORT(m)), values: abl.map(([, a]) => +a.toFixed(3)) }],
    { x: 6.75, y: 1.4, w: 5.95, h: 3.4, barDir: "bar", chartColors: abl.map(([m]) => m.includes("GlucoTwin") ? TEAL : m.includes("Deep") ? PURPLE : m.includes("Twin alone") ? AMBER : "94A3B8"), showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0.8, valAxisMaxVal: 1.0, valAxisHidden: true, showLegend: false, showTitle: true, title: "Ablation (AUROC): every stream adds signal", catAxisOrientation: "maxMin", ...chartText, objectName: "ablation-chart" });
  s.addText(`Synthetic Indian cohort: ${S.patients.test} held-out patients, ${S.rows.toLocaleString("en-US")} time points. Trend-rule baseline AUROC ${S.ablation.find(([m]) => m.startsWith("Trend"))[1].toFixed(3)}. Sensitivity ${pct(S.sensitivity)} at specificity ${pct(S.specificity)}.`,
    { x: 0.6, y: 4.95, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 13, color: MUTED, margin: 0, isTextBox: true });
  card(s, 0.6, 5.55, 12.1, 1.25, TINT, "r-takeaway");
  s.addText([{ text: "Takeaway: ", options: { bold: true, color: TEAL } }, { text: "CGM alone is good, but adding wearables, logs and the twin's simulation lifts discrimination and cuts false alarms. The twin's simulated peak is the single most important feature." }],
    { x: 0.85, y: 5.6, w: 11.6, h: 1.15, fontFace: BODY, fontSize: 15, color: INK, margin: 0, valign: "middle", isTextBox: true });
  s.addNotes(`On thirty held-out synthetic patients, GlucoTwin reaches an AUROC of ${S.auroc.toFixed(3)} for a new spike within two hours. It flags ${pct(S.event_recall)} of spike events at least fifteen minutes ahead, with a median warning of ${Math.round(S.median_lead)} minutes, at ${f1(S.false_per_day)} false alerts per patient-day. The ablation shows each data stream adding signal.`);

  // ================================================================ 10. Results: forecasting
  s = content("Forecasts: about half the error of a naive guess", "Results");
  const fc = (X) => [{ name: "No-change forecast", labels: ["30 min", "60 min", "120 min"], values: [30, 60, 120].map((h) => +X.rmse_persistence[h].toFixed(1)) },
    { name: "Twin alone", labels: ["30 min", "60 min", "120 min"], values: [30, 60, 120].map((h) => +X.rmse_twin[h].toFixed(1)) },
    { name: "GlucoTwin", labels: ["30 min", "60 min", "120 min"], values: [30, 60, 120].map((h) => +X.rmse[h].toFixed(1)) }];
  const fcOpts = (title, x) => ({ x, y: 1.4, w: 5.95, h: 3.9, barDir: "col", barGrouping: "clustered", chartColors: ["94A3B8", AMBER, TEAL], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisHidden: true, showLegend: true, legendPos: "b", showTitle: true, title, ...chartText });
  s.addChart(pres.charts.BAR, fc(S), { ...fcOpts("RMSE (mg/dL) - synthetic Indian cohort", 0.6), objectName: "rmse-syn" });
  s.addChart(pres.charts.BAR, fc(R), { ...fcOpts("RMSE (mg/dL) - real ShanghaiT2DM, 5-fold CV", 6.75), objectName: "rmse-real" });
  const cut = (X, h) => Math.round((1 - X.rmse[h] / X.rmse_persistence[h]) * 100);
  [[0.6, `-${cut(S, 60)}%`, "1-h forecast error vs no-change (synthetic)"], [3.65, `${f1(S.clarkeA60)}%`, "Clarke zone A at 60 min (synthetic)"],
   [6.75, `-${cut(R, 60)}%`, "1-h forecast error vs no-change (real patients)"], [9.85, pct(R.coverage80["60"]), "real coverage of the 80% interval (60 min)"]].forEach(([x, v, l], i) => stat(s, x, 5.4, 2.85, v, l, i < 2 ? TEAL : PURPLE, `fc-${i}`));
  s.addNotes(`Forecast error, measured as RMSE, is roughly halved on the synthetic cohort and about a quarter lower on real ShanghaiT2DM patients, compared with a no-change forecast. ${f1(S.clarkeA60)} percent of one-hour forecasts fall in Clarke zone A, and our conformal intervals cover the stated eighty percent on real data.`);

  // ================================================================ 11. Real-world validation
  s = content("Real-world check: 100 people with T2D", "Results");
  stat(s, 0.6, 1.45, 2.85, R.auroc.toFixed(3), "AUROC, 5-fold CV grouped by person", PURPLE, "rw1");
  stat(s, 3.65, 1.45, 2.85, `${f1(R.rmse[60])}`, `mg/dL 1-h RMSE (no-change: ${f1(R.rmse_persistence[60])})`, PURPLE, "rw2");
  stat(s, 0.6, 3.15, 2.85, pct(R.event_recall), `of ${R.events} real spike events flagged >= 15 min ahead`, PURPLE, "rw3");
  stat(s, 3.65, 3.15, 2.85, `${f1(R.clarkeA60)}%`, "Clarke zone A at 60 min", PURPLE, "rw4");
  const rab = R.ablation;
  s.addChart(pres.charts.BAR, [{ name: "AUROC", labels: rab.map(([m]) => SHORT(m)), values: rab.map(([, a]) => +a.toFixed(3)) }],
    { x: 6.75, y: 1.4, w: 5.95, h: 3.4, barDir: "bar", chartColors: rab.map(([m]) => m.includes("GlucoTwin") ? PURPLE : m.startsWith("Trend") || m.includes("Twin alone") ? "CBD5E1" : "A78BFA"), showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0.6, valAxisMaxVal: 1.0, valAxisHidden: true, showLegend: false, showTitle: true, title: "AUROC on real patients (out-of-fold)", catAxisOrientation: "maxMin", ...chartText, objectName: "real-ablation" });
  card(s, 0.6, 4.95, 12.1, 1.85, TINT, "rw-honest");
  s.addText([{ text: "What the real data tells us (honestly): ", options: { bold: true, color: PURPLE } },
    { text: "the same pipeline transfers to real, noisy, hospital CGM and beats rule-based and no-change baselines by a wide margin. This dataset has no wearables and free-text diet logs, so EHR and twin features add little beyond CGM here - the fusion gains we see on synthetic data still need prospective validation with Indian patients wearing CGM + smartwatch." }],
    { x: 0.85, y: 5.0, w: 11.6, h: 1.75, fontFace: BODY, fontSize: 14, color: INK, margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("We validated on ShanghaiT2DM: one hundred real people with type-2 diabetes, fifteen-minute CGM, and a clinical summary. With five-fold cross-validation grouped by person, the model clearly beats rule-based and no-change baselines. Because this dataset has no wearables and only free-text meals, the extra streams add little here, and we say so plainly. That is exactly what a prospective Indian pilot should test.");

  // ================================================================ 12. Explainability
  s = content("Explainability: every alert comes with its reasons", "Results");
  const st1 = Object.entries(S.streams);
  s.addChart(pres.charts.BAR, [{ name: "share", labels: st1.map(([k]) => k), values: st1.map(([, v]) => +(v * 100).toFixed(1)) }],
    { x: 0.6, y: 1.4, w: 5.8, h: 3.6, barDir: "bar", chartColors: [TEAL, PURPLE, "334155", AMBER, BLUE], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"", valAxisHidden: true, showLegend: false, showTitle: true, title: "Share of SHAP importance by data stream (synthetic)", catAxisOrientation: "maxMin", ...chartText, objectName: "shap-streams" });
  s.addText("The twin's simulation carries most of the signal, followed by wearables and CGM history - the fusion is what the model actually uses.",
    { x: 0.6, y: 5.15, w: 5.8, h: 0.9, fontFace: BODY, fontSize: 14, color: INK, margin: 0, isTextBox: true });
  card(s, 6.75, 1.45, 5.95, 5.3, "FFF7ED", "alert-example");
  iconDot(s, "warn", 7.0, 1.7, 0.62, "FDE7C7");
  s.addText(EX ? `Example alert · ${EX.patient} · ${EX.time}` : "Example alert", { x: 7.8, y: 1.72, w: 4.7, h: 0.6, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0, valign: "middle", isTextBox: true });
  if (EX) {
    s.addText(`Glucose ${Math.round(EX.glucose)} mg/dL · spike risk ${pct(EX.prob)} · forecast in 2 h ${Math.round(EX.forecast_2h)} mg/dL`, { x: 7.0, y: 2.5, w: 5.5, h: 0.5, fontFace: BODY, fontSize: 14, color: AMBER, bold: true, margin: 0, isTextBox: true });
    s.addText(EX.reasons.map((r, i) => ({ text: `${r.text}  (${r.direction})`, options: { bullet: true, breakLine: i < EX.reasons.length - 1, paraSpaceAfter: 8 } })),
      { x: 7.0, y: 3.15, w: 5.5, h: 3.4, fontFace: BODY, fontSize: 15, color: INK, margin: 0, valign: "top" });
  }
  s.addNotes("Explanations are exact TreeSHAP contributions, grouped and translated into plain language. Across the cohort, the twin's simulation carries most of the importance, followed by wearables and CGM history. On the right is a real alert from our dashboard for a synthetic patient, with its reasons.");

  // ================================================================ 13. Dashboard
  pres.addSection({ title: "Prototype" });
  s = content("The doctor's view: a ward of digital twins", "Prototype");
  const shots = [["screen_ward.png", "Ward overview: every patient ranked by 2-h spike risk"], ["screen_patient.png", "Patient twin: CGM, forecast band, alert reasons, wearables, EHR"]];
  for (let i = 0; i < 2; i++) {
    const f = path.join(FIG, shots[i][0]);
    let bottom = 5.6;
    if (fs.existsSync(f)) {
      const im = await fit(f, 0.6 + i * 6.15, 1.5, 5.95, 4.6);
      im.y = 1.6;
      s.addImage({ ...im, shadow: shadow(), objectName: `screen-${i}` });
      bottom = im.y + im.h;
    }
    s.addText(shots[i][1], { x: 0.6 + i * 6.15, y: bottom + 0.2, w: 5.95, h: 0.45, fontFace: BODY, fontSize: 14, color: MUTED, align: "center", margin: 0, isTextBox: true });
  }
  s.addText("Also: what-if simulator, model-performance page, live replay mode and a FastAPI REST API (/patients, /state, /twin, /what-if).",
    { x: 0.6, y: 6.3, w: 12.1, h: 0.45, fontFace: BODY, fontSize: 14, color: INK, margin: 0, align: "center", isTextBox: true });
  s.addNotes("This is the Streamlit dashboard. The ward view ranks every monitored patient by spike risk, with a one-line main driver. Opening a patient shows the twin: the CGM trace, the forecast with its eighty percent band, the alert and its reasons, wearables, the EHR, labs with LOINC codes, and the seven-day profile. A live-replay mode streams the data as if from real devices.");

  // ================================================================ 14. What-if
  s = content("What-if: test the plan on the twin before the patient eats", "Prototype");
  s.addImage(await fit(path.join(FIG, "whatif_example.png"), 0.6, 1.4, 7.4, 4.3));
  if (W) {
    const base = W.scenarios[0].peak;
    const rows = [[{ text: "Scenario", options: { bold: true, color: WHITE, fill: { color: TEAL } } }, { text: "Peak", options: { bold: true, color: WHITE, fill: { color: TEAL } } }, { text: "vs usual", options: { bold: true, color: WHITE, fill: { color: TEAL } } }]]
      .concat(W.scenarios.map((r) => [r.scenario, `${r.peak} mg/dL`, r.peak === base ? "-" : `${r.peak - base > 0 ? "+" : ""}${r.peak - base}`]));
    s.addTable(rows, { x: 8.25, y: 1.45, w: 4.45, colW: [2.55, 1.0, 0.9], fontFace: BODY, fontSize: 12, color: INK, border: { type: "solid", pt: 0.5, color: LINE }, rowH: 0.42, objectName: "whatif-table" });
  }
  s.addText("Simulated with the patient's own calibrated twin - so the advice is personal: a walk helps some patients more than a swap, and vice versa.",
    { x: 8.25, y: 4.6, w: 4.45, h: 1.2, fontFace: BODY, fontSize: 13, italic: true, color: TEAL, margin: 0, isTextBox: true });
  s.addNotes("The what-if simulator asks the patient's own twin about alternatives: a smaller portion, swapping rice for millet roti, a fifteen-minute walk after eating, or the effect of a short night's sleep. Because the twin is calibrated to this patient, the comparison is personal. This is for counselling, not dosing.");

  // ================================================================ 15. Alert burden
  s = content("Choosing the alert setting: sensitive vs quiet", "Prototype");
  s.addImage(await fit(path.join(FIG, "alert_tradeoff.png"), 0.6, 1.4, 6.6, 4.4));
  const modes = [["Sensitive (default)", `${pct(S.event_recall)} of spikes flagged`, `median ${Math.round(S.median_lead)} min warning`, `${f1(S.false_per_day)} false alerts / day`, TEAL],
    ["Quiet", `${pct(S.quiet_mode.event_recall)} of spikes flagged`, `median ${Math.round(S.quiet_mode.median_lead_min)} min warning`, `${f1(S.quiet_mode.false_alerts_per_day)} false alerts / day`, AMBER]];
  modes.forEach(([h, a, b, c, col], i) => {
    const y = 1.45 + i * 2.25;
    card(s, 7.55, y, 5.15, 2.0, TINT, `mode-${i}`);
    s.addText([{ text: h, options: { bold: true, fontSize: 18, color: col, breakLine: true } }, { text: a, options: { breakLine: true } }, { text: b, options: { breakLine: true } }, { text: c }],
      { x: 7.8, y: y + 0.12, w: 4.7, h: 1.8, fontFace: BODY, fontSize: 15, color: INK, margin: 0, valign: "middle", isTextBox: true });
  });
  s.addText("Synthetic cohort; repeat notifications snoozed for 60 min. Care teams can pick the operating point that fits their capacity.",
    { x: 0.6, y: 5.95, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 13, color: MUTED, margin: 0, isTextBox: true });
  s.addNotes("Alarm fatigue is real, so we report the trade-off explicitly. The default setting flags almost every spike with a long warning. A quieter setting cuts false alerts to about one per patient-day while still catching most spikes. A programme can choose its operating point.");

  // ================================================================ 16. Privacy & limits
  pres.addSection({ title: "Responsible AI & next steps" });
  s = content("Privacy, safety and honest limitations", "Responsible AI & next steps");
  const cols3 = [["lock", "Privacy by design", ["Only synthetic or published anonymised data (sandbox rules)", "No identifiers in features; FHIR bundles tagged synthetic", "DPDP Act 2023 principles: purpose limitation, data minimisation"]],
    ["shield", "Clinical safety", ["Decision support for a care team, never automated dosing", "Calibrated 80% intervals shown with every forecast", "Alerts carry reasons so clinicians can disagree"]],
    ["warn", "Limitations", ["Wearable-glucose effects in the synthetic cohort come from literature effect sizes", "Real-data track lacks wearables; carbs estimated from free text", "Needs prospective validation in Indian clinics before any clinical use"]]];
  cols3.forEach(([ic, h, items], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.45, 3.85, 3.95, TINT, `resp-${i}`);
    iconDot(s, ic, x + 0.25, 1.7, 0.62, MINT);
    s.addText(h, { x: x + 1.05, y: 1.72, w: 2.7, h: 0.6, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0, valign: "middle", isTextBox: true });
    s.addText(bullets(items, 14), { x: x + 0.25, y: 2.6, w: 3.4, h: 2.7, fontFace: BODY, margin: 0, valign: "top" });
  });
  card(s, 0.6, 5.65, 12.1, 1.1, "FFF7ED", "resp-banner");
  s.addText([{ text: "0 identifiable records used. ", options: { bold: true, color: AMBER } }, { text: "200 synthetic patients + 100 published anonymised patients (CC BY 4.0). GlucoTwin is a research prototype, not a medical device." }],
    { x: 0.85, y: 5.7, w: 11.6, h: 1.0, fontFace: BODY, fontSize: 15, color: INK, margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("We used only synthetic or published anonymised data, with no identifiers, following the DPDP Act's principles. GlucoTwin is decision support with calibrated uncertainty and visible reasons. Its limitations: synthetic wearable effects come from the literature, the real dataset lacks wearables, and it needs prospective validation before any clinical use.");

  // ================================================================ 17. Roadmap
  s = content("From prototype to pilot", "Responsible AI & next steps");
  const road = [["hosp", "Pilot", "3-month IEC-approved pilot with CGM + smartwatch users at an Indian diabetes clinic"],
    ["mobile", "Integrate", "ABDM / FHIR connectors, Libre & Health Connect ingestion, WhatsApp alerts for care teams"],
    ["road", "Scale", "Federated learning across hospitals so data never leaves the site"],
    ["heartW", "Extend", "Add hypertension and cardiovascular twins on the same engine and dashboard"]];
  road.forEach(([ic, h, t], i) => {
    const x = 0.6 + i * 3.08;
    s.addShape(pres.shapes.OVAL, { x: x + 1.05, y: 1.65, w: 0.8, h: 0.8, fill: { color: i === 0 ? AMBER : TEAL }, line: { color: i === 0 ? AMBER : TEAL }, objectName: `road-dot-${i}` });
    s.addImage({ data: I[ic], x: x + 1.23, y: 1.83, w: 0.44, h: 0.44, objectName: `road-icon-${i}` });
    if (i < 3) s.addShape(pres.shapes.LINE, { x: x + 1.95, y: 2.05, w: 2.9, h: 0, line: { color: LINE, width: 2, dashType: "dash" }, objectName: `road-line-${i}` });
    card(s, x, 2.75, 2.85, 2.6, TINT, `road-card-${i}`);
    s.addText(h, { x: x + 0.2, y: 2.9, w: 2.45, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.2, y: 3.45, w: 2.45, h: 1.8, fontFace: BODY, fontSize: 14, color: INK, margin: 0, valign: "top", isTextBox: true });
  });
  card(s, 0.6, 5.65, 12.1, 1.1, "FFF7ED", "impact");
  s.addText([{ text: "Impact goal: ", options: { bold: true, color: AMBER } }, { text: "turn the CGM data India's patients already generate into early, explained, personal action - fewer hours above range, without more work for clinicians." }],
    { x: 0.85, y: 5.7, w: 11.6, h: 1.0, fontFace: BODY, fontSize: 15, color: INK, margin: 0, valign: "middle", isTextBox: true });
  s.addNotes("Next we want an ethics-approved pilot with Indian CGM and smartwatch users, integration with ABDM and FHIR, federated learning across hospitals, and new twins for hypertension and cardiovascular risk on the same engine.");

  // ================================================================ 18. Thanks
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Responsible AI & next steps" });
  s.addText("Thank you", { placeholder: "title" });
  s.addText("GlucoTwin - Team Gym Bro, IIIT Delhi", { placeholder: "body" });
  s.addText([{ text: "Code, data card, model card and video: see the GitHub repository README", options: { breakLine: true } },
    { text: "Run it:  python run_all.py  ·  streamlit run dashboard/app.py", options: { breakLine: true } },
    { text: "MIT licence · ShanghaiT2DM data CC BY 4.0 (Zhao et al., Sci Data 2023)" }],
    { x: 0.8, y: 5.2, w: 11.5, h: 1.3, fontFace: BODY, fontSize: 15, color: MINT, margin: 0, isTextBox: true, objectName: "links" });
  s.addNotes("Thank you. Everything - code, data card, model card and this video - is in our public GitHub repository, and the whole pipeline rebuilds with one command.");

  await pres.writeFile({ fileName: OUT });
  if (process.env.PPTX_THEME_SCRIPT && fs.existsSync(process.env.PPTX_THEME_SCRIPT)) {
    const { applyTheme } = require(process.env.PPTX_THEME_SCRIPT);
    await applyTheme(OUT, { name: "GlucoTwin", headFontFace: HEAD, bodyFontFace: BODY, colors: { dk1: INK, lt1: WHITE, dk2: DARK, lt2: TINT, accent1: TEAL, accent2: AMBER, accent3: BLUE, accent4: RED, accent5: PURPLE, accent6: MUTED, hlink: TEAL, folHlink: PURPLE } });
  }
  console.log("wrote", OUT);
})();
