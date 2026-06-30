from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
)
from werkzeug.utils import secure_filename


APP_DIR = Path(__file__).resolve().parent
CONFIG_DIR = APP_DIR / "config"
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"
OUTPUT_DIR = APP_DIR / "outputs"
BATCH_DIR = OUTPUT_DIR / "batch_results"
MODEL_PATH = MODEL_DIR / "final_LPP_model.joblib"
REFERENCE_PATH = DATA_DIR / "reference_distribution.csv"
SURVIVAL_LOOKUP_PATH = DATA_DIR / "survival_probability_lookup.csv"
METADATA_PATH = CONFIG_DIR / "model_metadata.json"
CUTOFF_PATH = CONFIG_DIR / "cutoff.json"

TIME_POINT_DAYS = {
    12: 365.25,
    24: 730.5,
    36: 1095.75,
    60: 1826.25,
}
DISCLAIMER = (
    "The LPP Risk Explorer is intended for research use and clinical decision "
    "support only. It does not provide medical advice and should not replace "
    "clinician judgment. Predicted survival probabilities are model-based "
    "estimates and should be interpreted in the context of clinical evaluation."
)

MODEL_CACHE: dict[str, Any] | None = None
REFERENCE_CACHE: pd.DataFrame | None = None
SURVIVAL_LOOKUP_CACHE: pd.DataFrame | None = None


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


MODEL_METADATA = load_json(METADATA_PATH, {})
CUTOFF_CONFIG = load_json(CUTOFF_PATH, {"cutoff": None})


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    app.secret_key = "lpp-risk-explorer-local"

    @app.context_processor
    def inject_common() -> dict[str, Any]:
        return {
            "metadata": MODEL_METADATA,
            "disclaimer": DISCLAIMER,
            "model_status": model_status(),
        }

    @app.route("/")
    def index():
        ref = get_reference_distribution()
        stats = {
            "n": int(len(ref)),
            "cutoff": get_cutoff(),
            "high_risk_pct": float((ref["risk_group"] == "High risk").mean() * 100)
            if len(ref)
            else None,
        }
        return render_template("index.html", stats=stats)

    @app.route("/single-patient", methods=["GET", "POST"])
    def single_patient():
        result = None
        payload = default_single_payload()
        if request.method == "POST":
            payload = dict(request.form)
            result = predict_from_payload(payload)
        return render_template(
            "single_patient.html",
            result=result,
            result_json=json.dumps(result) if result else "",
            payload=payload,
            options=reference_options(),
        )

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        payload = request.get_json(silent=True) or request.form.to_dict()
        result = predict_from_payload(payload)
        return jsonify(result), 200 if result.get("ok") else 422

    @app.route("/batch-prediction", methods=["GET", "POST"])
    def batch_prediction():
        batch_result = None
        if request.method == "POST":
            uploaded = request.files.get("batch_file")
            if not uploaded or not uploaded.filename:
                batch_result = {"ok": False, "message": "Please upload a CSV file."}
            else:
                frame = pd.read_csv(uploaded)
                predictions = predict_batch_frame(frame)
                filename = f"lpp_batch_results_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.csv"
                output_path = BATCH_DIR / secure_filename(filename)
                predictions.to_csv(output_path, index=False)
                batch_result = {
                    "ok": True,
                    "filename": output_path.name,
                    "records": predictions.head(100).to_dict(orient="records"),
                    "all_records": predictions.to_dict(orient="records"),
                    "summary": batch_summary(predictions),
                }
        return render_template("batch_prediction.html", batch_result=batch_result)

    @app.route("/download-batch/<filename>")
    def download_batch(filename: str):
        safe = secure_filename(filename)
        path = BATCH_DIR / safe
        if not path.exists() or not path.resolve().is_relative_to(BATCH_DIR.resolve()):
            abort(404)
        return send_file(path, as_attachment=True, download_name=safe)

    @app.route("/example-batch")
    def example_batch():
        return send_file(DATA_DIR / "example_batch_input.csv", as_attachment=True)

    @app.route("/cohort-explorer")
    def cohort_explorer():
        return render_template("cohort_explorer.html", options=reference_options())

    @app.route("/api/reference")
    def api_reference():
        ref = get_reference_distribution()
        records = ref.fillna("").to_dict(orient="records")
        return jsonify({"records": records, "cutoff": get_cutoff()})

    @app.route("/documentation")
    def documentation():
        ref = get_reference_distribution()
        summary = {
            "reference_n": int(len(ref)),
            "tumor_types": sorted(ref["tumor_type"].dropna().unique().tolist()),
            "cutoff": get_cutoff(),
        }
        return render_template("documentation.html", summary=summary)

    @app.route("/report", methods=["POST"])
    def report():
        raw = request.form.get("result_json", "")
        result = json.loads(raw) if raw else predict_from_payload(request.form)
        return render_template(
            "report_template.html",
            result=result,
            result_json=json.dumps(result),
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

    return app


def model_status(path: Path = MODEL_PATH) -> dict[str, Any]:
    if not path.exists():
        return {
            "available": False,
            "label": "Model file missing",
            "detail": "Place final_LPP_model.joblib in the models directory.",
        }
    return {"available": True, "label": "Final LPP model loaded", "detail": path.name}


def load_model_bundle(path: Path = MODEL_PATH) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    global MODEL_CACHE
    status = model_status(path)
    if not status["available"]:
        return None, status
    if MODEL_CACHE is None:
        try:
            MODEL_CACHE = joblib.load(path)
        except Exception as exc:  # pragma: no cover - defensive startup path
            return None, {
                "available": False,
                "label": "Model could not be loaded",
                "detail": str(exc),
            }
    return MODEL_CACHE, status


def get_reference_distribution() -> pd.DataFrame:
    global REFERENCE_CACHE
    if REFERENCE_CACHE is None:
        if REFERENCE_PATH.exists():
            REFERENCE_CACHE = pd.read_csv(REFERENCE_PATH)
        else:
            REFERENCE_CACHE = pd.DataFrame(
                columns=[
                    "record_id",
                    "cohort",
                    "tumor_type",
                    "stage_group",
                    "bmi_group",
                    "LCR",
                    "PGSGA_patient",
                    "prealbumin_mg_L",
                    "LPP_score",
                    "normalized_LPP",
                    "risk_group",
                ]
            )
    return REFERENCE_CACHE.copy()


def get_survival_lookup() -> pd.DataFrame:
    global SURVIVAL_LOOKUP_CACHE
    if SURVIVAL_LOOKUP_CACHE is None:
        if SURVIVAL_LOOKUP_PATH.exists():
            SURVIVAL_LOOKUP_CACHE = pd.read_csv(SURVIVAL_LOOKUP_PATH)
        else:
            SURVIVAL_LOOKUP_CACHE = pd.DataFrame(columns=["curve", "month", "survival_probability"])
    return SURVIVAL_LOOKUP_CACHE.copy()


def get_cutoff() -> float:
    value = CUTOFF_CONFIG.get("cutoff", MODEL_METADATA.get("cutoff"))
    return float(value) if value is not None else 0.5


def reference_options() -> dict[str, list[str]]:
    ref = get_reference_distribution()
    return {
        "tumor_types": sorted(x for x in ref["tumor_type"].dropna().unique().tolist() if x),
        "stage_groups": ["I", "II", "III", "IV"],
        "bmi_groups": sorted(x for x in ref["bmi_group"].dropna().unique().tolist() if x),
    }


def default_single_payload() -> dict[str, Any]:
    path = DATA_DIR / "example_single_patient.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "input_mode": "auto_lcr",
        "lymphocyte": 0.8,
        "CRP": 35,
        "PGSGA_patient": 12,
        "prealbumin": 150,
        "prealbumin_unit": "mg/L",
    }


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def warning_or_error_range(
    value: float | None,
    label: str,
    low: float,
    high: float,
    errors: list[str],
    warnings: list[str],
    missing_error: bool = True,
) -> None:
    if value is None:
        if missing_error:
            errors.append(f"{label} is required.")
        return
    if value < 0:
        errors.append(f"{label} cannot be negative.")
    elif value < low or value > high:
        warnings.append(f"{label} is outside the usual range ({low:g}-{high:g}).")


def convert_prealbumin(value: Any, unit: str = "mg/L") -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    numeric = parse_float(value)
    unit = (unit or "mg/L").strip()
    factors = {"mg/L": 1.0, "g/L": 1000.0, "mg/dL": 10.0}
    if numeric is None:
        errors.append("Prealbumin is required.")
        return {"value": None, "unit": unit, "mg_L": None, "errors": errors, "warnings": warnings}
    if unit not in factors:
        errors.append("Prealbumin unit must be mg/L, g/L, or mg/dL.")
        return {"value": numeric, "unit": unit, "mg_L": None, "errors": errors, "warnings": warnings}
    if numeric < 0:
        errors.append("Prealbumin cannot be negative.")
    if unit == "mg/L" and 0 < numeric < 1:
        warnings.append("This prealbumin value is unusually low for mg/L and may be entered in g/L.")
    if unit == "g/L":
        warnings.append(f"{numeric:g} g/L is equivalent to {numeric * 1000:g} mg/L.")
    if unit == "mg/dL":
        warnings.append(f"{numeric:g} mg/dL is equivalent to {numeric * 10:g} mg/L.")
    mg_l = numeric * factors[unit]
    if mg_l < 20 or mg_l > 600:
        warnings.append("Converted prealbumin is outside the usual 20-600 mg/L range.")
    return {
        "value": numeric,
        "unit": unit,
        "mg_L": mg_l,
        "errors": errors,
        "warnings": warnings,
        "conversion_record": f"{numeric:g} {unit} -> {mg_l:g} mg/L",
    }


def calculate_lcr(payload: dict[str, Any], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    mode = payload.get("input_mode", "auto_lcr")
    if mode == "direct_lcr":
        lcr = parse_float(payload.get("LCR"))
        warning_or_error_range(lcr, "LCR", 0, 100, errors, warnings)
        return {"input_mode": mode, "LCR": lcr, "CRP_adj": None, "lymphocyte": None, "CRP": None}

    lymphocyte = parse_float(payload.get("lymphocyte"))
    crp = parse_float(payload.get("CRP", payload.get("crp")))
    warning_or_error_range(lymphocyte, "Lymphocyte count", 0.1, 20, errors, warnings)
    warning_or_error_range(crp, "CRP", 0, 300, errors, warnings)
    if crp is None or lymphocyte is None:
        return {"input_mode": mode, "LCR": None, "CRP_adj": None, "lymphocyte": lymphocyte, "CRP": crp}
    crp_adj = max(crp, 0.1)
    if crp == 0:
        warnings.append("CRP=0 was adjusted to CRP_adj=0.1 for LCR calculation.")
    lcr = lymphocyte / crp_adj
    if lcr > 100:
        warnings.append("Calculated LCR is outside the usual 0-100 range.")
    return {"input_mode": mode, "LCR": lcr, "CRP_adj": crp_adj, "lymphocyte": lymphocyte, "CRP": crp}


def model_matrix(bundle: dict[str, Any], lcr: float, pgsga: float, prealbumin_mg_l: float) -> np.ndarray:
    base = pd.DataFrame(
        [
            {
                "risk_LCR": -float(lcr),
                "PGSGA_patient": float(pgsga),
                "risk_prealbumin": -float(prealbumin_mg_l),
            }
        ],
        columns=bundle["base_features"],
    )
    x_base = bundle["scaler"].transform(bundle["imputer"].transform(base))
    if bundle.get("add_triad_score"):
        triad = x_base @ np.asarray(bundle["triad_weights"], dtype=float)
        return np.column_stack([triad, x_base])
    return x_base


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def predict_model_values(
    lcr: float,
    pgsga: float,
    prealbumin_mg_l: float,
    curve_months: list[int] | None = None,
) -> dict[str, Any]:
    bundle, status = load_model_bundle()
    curve_months = curve_months or list(range(0, 61, 3))
    if bundle is None:
        score = 180 + pgsga * 12 - min(lcr, 100) * 1.5 - prealbumin_mg_l * 0.25
        survival = {
            month: clamp_probability(math.exp(-max(score, 1) / 450 * (month / 60)))
            for month in TIME_POINT_DAYS
        }
        curve = [
            {"month": month, "survival": clamp_probability(math.exp(-max(score, 1) / 450 * (month / 60)))}
            for month in curve_months
        ]
        return {"score": float(score), "survival": survival, "curve": curve, "model_status": status}

    x = model_matrix(bundle, lcr, pgsga, prealbumin_mg_l)
    model = bundle["model"]
    score = float(model.predict(x)[0])
    fn = model.predict_survival_function(x)[0]
    survival = {
        month: clamp_probability(float(fn(days)))
        for month, days in TIME_POINT_DAYS.items()
    }
    curve = []
    for month in curve_months:
        value = 1.0 if month == 0 else clamp_probability(float(fn(month * 30.4375)))
        curve.append({"month": month, "survival": value})
    return {"score": score, "survival": survival, "curve": curve, "model_status": status}


def normalize_score(score: float) -> float:
    low = float(MODEL_METADATA.get("score_min_reference", 0.0))
    high = float(MODEL_METADATA.get("score_max_reference", 1.0))
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (score - low) / (high - low)))


def percentile_for(score: float, frame: pd.DataFrame) -> float | None:
    values = pd.to_numeric(frame.get("LPP_score"), errors="coerce").dropna()
    if values.empty:
        return None
    return float((values <= score).mean() * 100)


def context_percentiles(score: float, payload: dict[str, Any]) -> dict[str, Any]:
    ref = get_reference_distribution()
    result = {"overall": percentile_for(score, ref)}
    groups = [
        ("tumor_type", payload.get("tumor_type"), "same_tumor_type"),
        ("stage_group", payload.get("stage_group"), "same_stage"),
        ("bmi_group", payload.get("bmi_group"), "same_bmi_group"),
    ]
    for column, value, key in groups:
        value = str(value or "").strip()
        result[key] = percentile_for(score, ref[ref[column] == value]) if value else None
    return result


def reference_curves() -> dict[str, list[dict[str, float]]]:
    lookup = get_survival_lookup()
    curves: dict[str, list[dict[str, float]]] = {}
    for curve_name, frame in lookup.groupby("curve"):
        curves[curve_name] = [
            {"month": int(row.month), "survival": float(row.survival_probability)}
            for row in frame.itertuples(index=False)
        ]
    return curves


def calculate_contributions(lcr: float, pgsga: float, prealbumin_mg_l: float) -> list[dict[str, Any]]:
    ref = get_reference_distribution()
    medians = {
        "LCR": float(pd.to_numeric(ref["LCR"], errors="coerce").median()),
        "PG-SGA": float(pd.to_numeric(ref["PGSGA_patient"], errors="coerce").median()),
        "Prealbumin": float(pd.to_numeric(ref["prealbumin_mg_L"], errors="coerce").median()),
    }
    actual = {"LCR": lcr, "PG-SGA": pgsga, "Prealbumin": prealbumin_mg_l}
    baseline = predict_model_values(medians["LCR"], medians["PG-SGA"], medians["Prealbumin"])["score"]
    rows = []
    for name in ["LCR", "PG-SGA", "Prealbumin"]:
        point = medians.copy()
        point[name] = actual[name]
        changed = predict_model_values(point["LCR"], point["PG-SGA"], point["Prealbumin"])["score"]
        delta = float(changed - baseline)
        rows.append(
            {
                "variable": name,
                "value": actual[name],
                "reference_median": medians[name],
                "delta_score": delta,
                "direction": "increases risk" if delta > 0 else "decreases risk" if delta < 0 else "neutral",
            }
        )
    return sorted(rows, key=lambda item: abs(item["delta_score"]), reverse=True)


def predict_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    lcr_info = calculate_lcr(payload, errors, warnings)
    pgsga = parse_float(payload.get("PGSGA_patient", payload.get("pgsga")))
    warning_or_error_range(pgsga, "PG-SGA score", 0, 40, errors, warnings)
    prealbumin = convert_prealbumin(payload.get("prealbumin"), payload.get("prealbumin_unit", "mg/L"))
    errors.extend(prealbumin["errors"])
    warnings.extend(prealbumin["warnings"])

    if errors:
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "inputs": {"raw": payload},
            "model_status": model_status(),
        }

    lcr = float(lcr_info["LCR"])
    pgsga = float(pgsga)
    prealbumin_mg_l = float(prealbumin["mg_L"])
    prediction = predict_model_values(lcr, pgsga, prealbumin_mg_l)
    score = float(prediction["score"])
    cutoff = get_cutoff()
    risk_group = "High risk" if score >= cutoff else "Low risk"
    result = {
        "ok": True,
        "patient_id": str(payload.get("patient_id") or "").strip(),
        "risk_group": risk_group,
        "LPP_score": score,
        "normalized_LPP": normalize_score(score),
        "cutoff": cutoff,
        "survival": {str(k): v for k, v in prediction["survival"].items()},
        "curve": prediction["curve"],
        "reference_curves": reference_curves(),
        "percentiles": context_percentiles(score, payload),
        "contributions": calculate_contributions(lcr, pgsga, prealbumin_mg_l),
        "warnings": sorted(dict.fromkeys(warnings)),
        "errors": [],
        "inputs": {
            "input_mode": lcr_info["input_mode"],
            "lymphocyte": lcr_info["lymphocyte"],
            "CRP": lcr_info["CRP"],
            "CRP_adj": lcr_info["CRP_adj"],
            "LCR": lcr,
            "PGSGA_patient": pgsga,
            "prealbumin": prealbumin["value"],
            "prealbumin_unit": prealbumin["unit"],
            "prealbumin_mg_L": prealbumin_mg_l,
            "prealbumin_conversion": prealbumin.get("conversion_record"),
            "tumor_type": str(payload.get("tumor_type") or "").strip(),
            "stage_group": str(payload.get("stage_group") or "").strip(),
            "bmi_group": str(payload.get("bmi_group") or "").strip(),
        },
        "model_status": prediction["model_status"],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "disclaimer": DISCLAIMER,
    }
    return result


def first_present(row: pd.Series, names: list[str]) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]).strip():
            return row[name]
    return None


def row_payload(row: pd.Series) -> dict[str, Any]:
    lcr = first_present(row, ["LCR", "lcr"])
    lymphocyte = first_present(row, ["lymphocyte", "lymphocyte_count"])
    crp = first_present(row, ["CRP", "crp"])
    mode = "direct_lcr" if lcr is not None and (lymphocyte is None or crp is None) else "auto_lcr"
    return {
        "patient_id": first_present(row, ["patient_id", "record_id", "id"]) or "",
        "input_mode": mode,
        "lymphocyte": lymphocyte,
        "CRP": crp,
        "LCR": lcr,
        "PGSGA_patient": first_present(row, ["PGSGA_patient", "PG_SGA", "pgsga"]),
        "prealbumin": first_present(row, ["prealbumin", "prealbumin_mg_L"]),
        "prealbumin_unit": first_present(row, ["prealbumin_unit", "unit"]) or "mg/L",
        "tumor_type": first_present(row, ["tumor_type"]) or "",
        "stage_group": first_present(row, ["stage_group", "stage", "TNM_stage"]) or "",
        "bmi_group": first_present(row, ["bmi_group", "BMI_group"]) or "",
    }


def predict_batch_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in frame.iterrows():
        payload = row_payload(row)
        result = predict_from_payload(payload)
        rows.append(
            {
                "patient_id": payload["patient_id"] or f"row_{idx + 1}",
                "LCR": result.get("inputs", {}).get("LCR"),
                "prealbumin_mg_L": result.get("inputs", {}).get("prealbumin_mg_L"),
                "LPP_score": result.get("LPP_score"),
                "normalized_LPP": result.get("normalized_LPP"),
                "risk_group": result.get("risk_group"),
                "OS_12m": result.get("survival", {}).get("12"),
                "OS_24m": result.get("survival", {}).get("24"),
                "OS_36m": result.get("survival", {}).get("36"),
                "OS_60m": result.get("survival", {}).get("60"),
                "warnings": "; ".join(result.get("warnings", []) + result.get("errors", [])),
                "tumor_type": payload.get("tumor_type"),
                "stage_group": payload.get("stage_group"),
                "bmi_group": payload.get("bmi_group"),
            }
        )
    return pd.DataFrame(rows)


def batch_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    valid = predictions.dropna(subset=["LPP_score"])
    return {
        "n": int(len(predictions)),
        "valid_n": int(len(valid)),
        "high_risk_n": int((valid["risk_group"] == "High risk").sum()) if len(valid) else 0,
        "median_score": float(valid["LPP_score"].median()) if len(valid) else None,
    }


app = create_app()


if __name__ == "__main__":
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5057")), debug=False)
