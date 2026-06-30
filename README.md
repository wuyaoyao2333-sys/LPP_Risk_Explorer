---
title: LPP Risk Explorer
emoji: ""
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

# LPP Risk Explorer

LPP Risk Explorer is a Flask web calculator for overall survival risk estimation in nutritionally at-risk solid tumor patients. It uses only LCR, PG-SGA, and prealbumin for prediction; tumor type, TNM stage, and BMI group are used only for contextual percentile comparison.

## Install

```bash
cd LPP_Risk_Explorer
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open http://127.0.0.1:5057.

## Hugging Face Spaces Docker Deployment

Create a new Hugging Face Space, choose **Docker** as the SDK, and upload the contents of this directory. The app listens on port `7860` in Docker.

The Space uses:

```text
Dockerfile
requirements.txt
app.py
templates/
static/
config/
data/
models/final_LPP_model.joblib
```

Hugging Face will build the image automatically after upload.

## Model Files

The app expects:

```text
models/final_LPP_model.joblib
config/cutoff.json
data/reference_distribution.csv
data/survival_probability_lookup.csv
```

The current scaffold already includes `models/final_LPP_model.joblib` and reference data generated from the local project outputs. If you replace the model, keep the same bundle interface:

```text
model
imputer
scaler
base_features = ["risk_LCR", "PGSGA_patient", "risk_prealbumin"]
triad_weights
add_triad_score
```

## Expected Inputs

Single-patient mode supports two formats.

Format A calculates LCR:

```text
lymphocyte, CRP, PGSGA_patient, prealbumin, prealbumin_unit
```

Format B uses LCR directly:

```text
LCR, PGSGA_patient, prealbumin, prealbumin_unit
```

Optional context fields:

```text
tumor_type, stage_group, bmi_group
```

These context fields do not enter the prediction.

## Unit Rules

Prealbumin is converted to mg/L:

```text
mg/L  -> value
g/L   -> value * 1000
mg/dL -> value * 10
```

The app warns, but does not silently change user inputs, when converted prealbumin is outside 20-600 mg/L.

## Single-Patient Mode

Open `/single-patient`, enter the LPP variables, and run calculation. The result includes:

- LPP score and normalized LPP score
- Low-risk or high-risk group
- 12-, 24-, 36-, and 60-month OS probabilities
- Patient-specific survival curve
- Variable contribution bars
- Reference cohort percentile
- What-if simulation
- Print-ready report page

## Batch Mode

Open `/batch-prediction` and upload a CSV. The output file includes:

```text
patient_id,LCR,prealbumin_mg_L,LPP_score,normalized_LPP,risk_group,OS_12m,OS_24m,OS_36m,OS_60m,warnings
```

The page also shows risk-group distribution and LPP score histogram.

## Output Interpretation

Higher LPP score indicates higher model-estimated risk. The risk-group cutoff is stored in `config/cutoff.json`. Survival probabilities are model-based estimates and should be interpreted in clinical context.

## Customize Reference Distributions

Replace `data/reference_distribution.csv` with a CSV containing:

```text
record_id,cohort,tumor_type,stage_group,bmi_group,LCR,PGSGA_patient,prealbumin_mg_L,LPP_score,normalized_LPP,risk_group
```

Replace `data/survival_probability_lookup.csv` to update low-risk and high-risk reference curves.

## Tests

```bash
pip install pytest
pytest
```

## Disclaimer

The LPP Risk Explorer is intended for research use and clinical decision support only. It does not provide medical advice and should not replace clinician judgment. Predicted survival probabilities are model-based estimates and should be interpreted in the context of clinical evaluation.
