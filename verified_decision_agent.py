"""Deterministic tools and validation for Rewaa's Verified Decision Agent.

This module deliberately contains no Streamlit state and no random data. OpenAI
may orchestrate these tools, but every number displayed by the feature must be
present in a deterministic tool result and pass ``validate_decision_result``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATE_COLUMN = "التاريخ"
COUNTRY_COLUMN = "الدولة"
AREA_COLUMN = "الحي"
TEMPERATURE_COLUMN = "درجة_الحرارة"
HOUSEHOLD_COLUMN = "عدد_الأفراد"
CONSUMPTION_COLUMN = "الاستهلاك_اللتر"

REQUIRED_TOOLS = [
    "get_area_observations",
    "calculate_area_statistics",
    "detect_consumption_anomalies",
    "analyze_temperature_relationship",
    "calculate_household_normalized_usage",
    "compare_intervention_scenarios",
]

UI_TEXT = {
    "العربية": {
        "title": "وكيل رواء للتحقق من القرارات",
        "question": "لماذا يتغير استهلاك هذه المنطقة، وما الإجراء الذي ينبغي اعتماده هذا الأسبوع؟",
        "run": "تشغيل التحليل الموثّق",
        "summary": "ملخص القرار",
        "risk_score": "درجة المخاطر النسبية",
        "findings": "النتائج الرئيسية",
        "actions": "الإجراءات الموصى بها للمراجعة",
        "impact": "الأثر المتوقع (تقديري)",
        "assumptions": "الافتراضات",
        "uncertainty": "عدم اليقين",
        "evidence": "جدول الأدلة الرقمية",
        "trace": "مسار الأدوات الحتمية",
        "source_openai": "تم التحقق من نتيجة GPT-5.6 مقابل مخرجات الأدوات",
        "source_local": "نتيجة محلية موثّقة — تم استخدام البديل الآمن",
        "risk_low": "منخفض",
        "risk_moderate": "متوسط",
        "risk_high": "مرتفع",
        "risk_unknown": "بيانات غير كافية",
    },
    "English": {
        "title": "Rewaa Verified Decision Agent",
        "question": "Why is consumption changing in this area, and what action should be approved this week?",
        "run": "Run verified analysis",
        "summary": "Decision Summary",
        "risk_score": "Relative Risk Score",
        "findings": "Key Findings",
        "actions": "Recommended Actions for Review",
        "impact": "Expected Impact (Estimate)",
        "assumptions": "Assumptions",
        "uncertainty": "Uncertainty",
        "evidence": "Numerical Evidence Table",
        "trace": "Deterministic Tool Trace",
        "source_openai": "GPT-5.6 result validated against deterministic tool outputs",
        "source_local": "Validated local result — safe fallback used",
        "risk_low": "Low",
        "risk_moderate": "Moderate",
        "risk_high": "High",
        "risk_unknown": "Insufficient data",
    },
}


def get_ui_text(language: str) -> dict[str, str]:
    """Return bilingual labels without coupling tests to Streamlit."""
    return UI_TEXT["العربية" if language == "العربية" else "English"].copy()


def _round(value: Any, digits: int = 2) -> float | None:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _prepare_data(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, list):
        frame = pd.DataFrame(data)
    elif data is None:
        frame = pd.DataFrame()
    else:
        raise TypeError("data must be a pandas DataFrame, a list of records, or None")

    required = [DATE_COLUMN, CONSUMPTION_COLUMN]
    if frame.empty or any(column not in frame.columns for column in required):
        return pd.DataFrame(columns=[
            DATE_COLUMN, COUNTRY_COLUMN, AREA_COLUMN, TEMPERATURE_COLUMN,
            HOUSEHOLD_COLUMN, CONSUMPTION_COLUMN,
        ])

    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    for column in [CONSUMPTION_COLUMN, TEMPERATURE_COLUMN, HOUSEHOLD_COLUMN]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[DATE_COLUMN, CONSUMPTION_COLUMN])
    return frame.sort_values(DATE_COLUMN).reset_index(drop=True)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        result.append({
            "date": row[DATE_COLUMN].strftime("%Y-%m-%d"),
            "country": str(row.get(COUNTRY_COLUMN, "")),
            "neighborhood": str(row.get(AREA_COLUMN, "")),
            "temperature_c": _round(row.get(TEMPERATURE_COLUMN)),
            "household_members": _round(row.get(HOUSEHOLD_COLUMN)),
            "consumption_liters": _round(row[CONSUMPTION_COLUMN]),
        })
    return result


def _records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return _prepare_data(pd.DataFrame([
        {
            DATE_COLUMN: item.get("date"),
            COUNTRY_COLUMN: item.get("country"),
            AREA_COLUMN: item.get("neighborhood"),
            TEMPERATURE_COLUMN: item.get("temperature_c"),
            HOUSEHOLD_COLUMN: item.get("household_members"),
            CONSUMPTION_COLUMN: item.get("consumption_liters"),
        }
        for item in records
    ]))


def get_area_observations(
    country: str,
    neighborhood: str,
    dataset: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return chronological raw observations for one selected area.

    ``dataset`` is injectable for tests and for the app's pre-cleaning snapshot.
    If omitted, the bundled CSV is read relative to this module.
    """
    if dataset is None:
        csv_path = Path(__file__).with_name("rewaa_gcc_data.csv")
        try:
            dataset = pd.read_csv(csv_path)
        except Exception:
            dataset = pd.DataFrame()

    frame = _prepare_data(dataset)
    if not frame.empty and COUNTRY_COLUMN in frame and AREA_COLUMN in frame:
        frame = frame[
            (frame[COUNTRY_COLUMN] == country) & (frame[AREA_COLUMN] == neighborhood)
        ].copy()
    return {
        "country": country,
        "neighborhood": neighborhood,
        "observation_count": int(len(frame)),
        "observations": _records(frame),
    }


def calculate_area_statistics(data: Any) -> dict[str, Any]:
    """Calculate distribution-based risk from the selected area's own history."""
    frame = _records_to_frame(data) if isinstance(data, list) and (
        not data or "consumption_liters" in data[0]
    ) else _prepare_data(data)
    if frame.empty:
        return {
            "status": "insufficient_data", "sample_size": 0,
            "risk_level": "unknown", "risk_score": None,
            "risk_method": "Neighborhood percentile rank; at least 4 observations required.",
        }

    values = frame[CONSUMPTION_COLUMN].astype(float)
    latest = float(values.iloc[-1])
    percentile_rank = float((values <= latest).mean() * 100)
    if len(values) < 4:
        risk_level = "unknown"
        risk_score = None
        status = "insufficient_data"
    else:
        risk_score = round(percentile_rank, 1)
        risk_level = "high" if risk_score >= 90 else "moderate" if risk_score >= 75 else "low"
        status = "ok"

    trend = None
    if len(values) >= 14:
        previous = float(values.iloc[-14:-7].mean())
        recent = float(values.iloc[-7:].mean())
        trend = ((recent - previous) / previous * 100) if previous else None

    return {
        "status": status,
        "sample_size": int(len(values)),
        "latest_date": frame[DATE_COLUMN].iloc[-1].strftime("%Y-%m-%d"),
        "latest_consumption_liters": _round(latest),
        "mean_consumption_liters": _round(values.mean()),
        "median_consumption_liters": _round(values.median()),
        "standard_deviation_liters": _round(values.std(ddof=0)),
        "minimum_consumption_liters": _round(values.min()),
        "maximum_consumption_liters": _round(values.max()),
        "p25_liters": _round(values.quantile(0.25)),
        "p75_liters": _round(values.quantile(0.75)),
        "p90_liters": _round(values.quantile(0.90)),
        "p95_liters": _round(values.quantile(0.95)),
        "latest_14_day_comparison_percent": _round(trend, 1),
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_method": "Latest observation percentile within this neighborhood: low <75, moderate 75-<90, high >=90.",
    }


def detect_consumption_anomalies(data: Any) -> dict[str, Any]:
    """Detect unusually low/high readings using transparent IQR bounds and z-score."""
    frame = _records_to_frame(data) if isinstance(data, list) and (
        not data or "consumption_liters" in data[0]
    ) else _prepare_data(data)
    if len(frame) < 4:
        return {
            "status": "insufficient_data", "sample_size": int(len(frame)),
            "method": "IQR bounds and population z-score; at least 4 observations required.",
            "anomaly_count": 0, "latest_is_anomaly": False, "anomalies": [],
        }

    values = frame[CONSUMPTION_COLUMN].astype(float)
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    lower = float(q1 - 1.5 * iqr)
    upper = float(q3 + 1.5 * iqr)
    std = float(values.std(ddof=0))
    zscores = (values - float(values.mean())) / std if std else pd.Series(0.0, index=values.index)
    mask = (values < lower) | (values > upper) | (zscores.abs() >= 3.0)
    anomalies = []
    for index in frame.index[mask]:
        anomalies.append({
            "date": frame.loc[index, DATE_COLUMN].strftime("%Y-%m-%d"),
            "consumption_liters": _round(values.loc[index]),
            "z_score": _round(zscores.loc[index]),
        })
    return {
        "status": "ok",
        "sample_size": int(len(frame)),
        "method": "IQR outside Q1-1.5×IQR/Q3+1.5×IQR or absolute population z-score >=3.",
        "lower_iqr_bound_liters": _round(lower),
        "upper_iqr_bound_liters": _round(upper),
        "latest_z_score": _round(zscores.iloc[-1]),
        "anomaly_count": int(mask.sum()),
        "latest_is_anomaly": bool(mask.iloc[-1]),
        "anomalies": anomalies,
    }


def analyze_temperature_relationship(data: Any) -> dict[str, Any]:
    """Measure, without causal claims, the observed temperature relationship."""
    frame = _records_to_frame(data) if isinstance(data, list) and (
        not data or "consumption_liters" in data[0]
    ) else _prepare_data(data)
    paired = frame.dropna(subset=[TEMPERATURE_COLUMN, CONSUMPTION_COLUMN])
    if len(paired) < 4 or paired[TEMPERATURE_COLUMN].nunique() < 2:
        return {
            "status": "insufficient_data", "paired_observations": int(len(paired)),
            "correlation": None, "slope_liters_per_c": None,
            "interpretation": "Insufficient variation to assess a relationship.",
        }
    correlation = float(paired[TEMPERATURE_COLUMN].corr(paired[CONSUMPTION_COLUMN]))
    slope = float(np.polyfit(paired[TEMPERATURE_COLUMN], paired[CONSUMPTION_COLUMN], 1)[0])
    strength = "weak" if abs(correlation) < 0.3 else "moderate" if abs(correlation) < 0.7 else "strong"
    direction = "positive" if correlation > 0 else "negative" if correlation < 0 else "neutral"
    return {
        "status": "ok",
        "paired_observations": int(len(paired)),
        "average_temperature_c": _round(paired[TEMPERATURE_COLUMN].mean()),
        "correlation": _round(correlation, 3),
        "slope_liters_per_c": _round(slope),
        "relationship_strength": strength,
        "relationship_direction": direction,
        "interpretation": "Observed association only; it does not establish causation.",
    }


def calculate_household_normalized_usage(data: Any) -> dict[str, Any]:
    """Normalize area consumption by household members in each observation."""
    frame = _records_to_frame(data) if isinstance(data, list) and (
        not data or "consumption_liters" in data[0]
    ) else _prepare_data(data)
    paired = frame.dropna(subset=[HOUSEHOLD_COLUMN, CONSUMPTION_COLUMN])
    paired = paired[paired[HOUSEHOLD_COLUMN] > 0].copy()
    if paired.empty:
        return {
            "status": "insufficient_data", "paired_observations": 0,
            "average_liters_per_person": None, "latest_liters_per_person": None,
        }
    normalized = paired[CONSUMPTION_COLUMN] / paired[HOUSEHOLD_COLUMN]
    correlation = None
    if len(paired) >= 4 and paired[HOUSEHOLD_COLUMN].nunique() >= 2:
        correlation = paired[HOUSEHOLD_COLUMN].corr(paired[CONSUMPTION_COLUMN])
    return {
        "status": "ok",
        "paired_observations": int(len(paired)),
        "average_household_members": _round(paired[HOUSEHOLD_COLUMN].mean()),
        "average_liters_per_person": _round(normalized.mean()),
        "median_liters_per_person": _round(normalized.median()),
        "latest_liters_per_person": _round(normalized.iloc[-1]),
        "household_consumption_correlation": _round(correlation, 3),
    }


def compare_intervention_scenarios(context: dict[str, Any]) -> dict[str, Any]:
    """Compare simple efficiency estimates; never present them as forecasts."""
    current = float(context.get("current_liters") or 0)
    selected_gain = max(0.0, min(50.0, float(context.get("efficiency_gain_percent") or 0)))
    enhanced_gain = min(50.0, selected_gain + 10.0)

    def scenario(name: str, efficiency: float) -> dict[str, Any]:
        estimated = current * (1 - efficiency / 100)
        return {
            "scenario": name,
            "efficiency_gain_percent": _round(efficiency, 1),
            "estimated_consumption_liters": _round(estimated),
            "estimated_reduction_liters": _round(current - estimated),
            "value_type": "estimate",
        }

    return {
        "status": "ok" if current > 0 else "insufficient_data",
        "method": "Arithmetic what-if estimate from the latest observed consumption; not a forecast or approved action.",
        "scenarios": [
            scenario("baseline", 0.0),
            scenario("selected_efficiency", selected_gain),
            scenario("enhanced_efficiency", enhanced_gain),
        ],
    }


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision_summary": {"type": "string"},
        "risk_level": {"type": "string", "enum": ["low", "moderate", "high", "unknown"]},
        "risk_score": {"type": ["number", "null"]},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["statement", "evidence_refs"],
                "additionalProperties": False,
            },
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "metric": {"type": "string"},
                    "value": {"type": ["number", "null"]},
                    "unit": {"type": "string"},
                    "source_tool": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["id", "metric", "value", "unit", "source_tool", "description"],
                "additionalProperties": False,
            },
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "uncertainty": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "expected_impact": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scenario": {"type": "string"},
                    "estimated_value": {"type": ["number", "null"]},
                    "unit": {"type": "string"},
                    "label": {"type": "string", "enum": ["estimate"]},
                },
                "required": ["scenario", "estimated_value", "unit", "label"],
                "additionalProperties": False,
            },
        },
        "tools_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "decision_summary", "risk_level", "risk_score", "findings", "evidence",
        "assumptions", "uncertainty", "recommended_actions", "expected_impact", "tools_used",
    ],
    "additionalProperties": False,
}


def _flatten_numbers(value: Any) -> list[float]:
    numbers: list[float] = []
    if isinstance(value, bool) or value is None:
        return numbers
    if isinstance(value, (int, float, np.number)):
        if math.isfinite(float(value)):
            numbers.append(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            numbers.extend(_flatten_numbers(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            numbers.extend(_flatten_numbers(item))
    return numbers


def _number_supported(number: float, allowed: Iterable[float]) -> bool:
    return any(math.isclose(number, candidate, rel_tol=1e-6, abs_tol=0.011) for candidate in allowed)


def _textual_numbers(result: dict[str, Any]) -> list[float]:
    numeric_pattern = re.compile(r"(?<![\w-])-?\d+(?:\.\d+)?(?![\w-])")
    text_values: list[str] = [result.get("decision_summary", "")]
    text_values.extend(item.get("statement", "") for item in result.get("findings", []))
    text_values.extend(result.get("assumptions", []))
    text_values.extend(result.get("uncertainty", []))
    text_values.extend(result.get("recommended_actions", []))
    text_values.extend(item.get("description", "") for item in result.get("evidence", []))
    return [float(match.group()) for text in text_values for match in numeric_pattern.finditer(str(text))]


def validate_decision_result(
    result: Any,
    tool_results: dict[str, Any],
    tool_trace: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Validate schema essentials, provenance, and every numerical claim."""
    errors: list[str] = []
    required_fields = set(DECISION_SCHEMA["required"])
    if not isinstance(result, dict):
        return False, ["Result is not an object."]
    missing = sorted(required_fields - set(result))
    extra = sorted(set(result) - required_fields)
    if missing:
        errors.append(f"Missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"Unexpected fields: {', '.join(extra)}")

    if result.get("risk_level") not in {"low", "moderate", "high", "unknown"}:
        errors.append("Invalid risk_level.")
    stats = tool_results.get("calculate_area_statistics", {})
    if result.get("risk_level") != stats.get("risk_level"):
        errors.append("risk_level does not match calculate_area_statistics.")
    expected_score = stats.get("risk_score")
    actual_score = result.get("risk_score")
    if expected_score is None:
        if actual_score is not None:
            errors.append("risk_score must be null when data is insufficient.")
    elif not isinstance(actual_score, (int, float)) or not math.isclose(
        float(actual_score), float(expected_score), abs_tol=0.011
    ):
        errors.append("risk_score does not match calculate_area_statistics.")

    called_tools = [item.get("tool") for item in tool_trace]
    if sorted(set(result.get("tools_used", []))) != sorted(set(called_tools)):
        errors.append("tools_used does not match the executed tool trace.")
    if not all(tool in called_tools for tool in REQUIRED_TOOLS):
        errors.append("The complete deterministic tool chain was not executed.")

    allowed_numbers = _flatten_numbers(tool_results)
    for evidence in result.get("evidence", []):
        if not isinstance(evidence, dict):
            errors.append("Evidence entry is not an object.")
            continue
        if evidence.get("source_tool") not in called_tools:
            errors.append(f"Unknown evidence source tool: {evidence.get('source_tool')}")
        value = evidence.get("value")
        if value is not None and (
            not isinstance(value, (int, float)) or not _number_supported(float(value), allowed_numbers)
        ):
            errors.append(f"Unsupported evidence number: {value}")
    for impact in result.get("expected_impact", []):
        if impact.get("label") != "estimate":
            errors.append("Expected impact is not explicitly labeled as an estimate.")
        value = impact.get("estimated_value")
        if value is not None and (
            not isinstance(value, (int, float)) or not _number_supported(float(value), allowed_numbers)
        ):
            errors.append(f"Unsupported expected-impact number: {value}")
    for number in _textual_numbers(result):
        if not _number_supported(number, allowed_numbers):
            errors.append(f"Unsupported numerical claim in text: {number}")
    return not errors, errors


def build_local_decision_result(
    tool_results: dict[str, Any],
    language: str,
) -> dict[str, Any]:
    """Build a conservative, schema-compatible fallback from tool outputs only."""
    is_ar = language == "العربية"
    stats = tool_results.get("calculate_area_statistics", {})
    anomalies = tool_results.get("detect_consumption_anomalies", {})
    temperature = tool_results.get("analyze_temperature_relationship", {})
    normalized = tool_results.get("calculate_household_normalized_usage", {})
    scenarios = tool_results.get("compare_intervention_scenarios", {}).get("scenarios", [])
    risk_level = stats.get("risk_level", "unknown")
    risk_score = stats.get("risk_score")

    evidence = [
        {
            "id": "latest_usage", "metric": "الاستهلاك الأحدث" if is_ar else "Latest consumption",
            "value": stats.get("latest_consumption_liters"), "unit": "لتر" if is_ar else "L",
            "source_tool": "calculate_area_statistics",
            "description": "أحدث قراءة زمنية في بيانات المنطقة." if is_ar else "Latest dated observation in the area data.",
        },
        {
            "id": "area_average", "metric": "متوسط المنطقة" if is_ar else "Area average",
            "value": stats.get("mean_consumption_liters"), "unit": "لتر" if is_ar else "L",
            "source_tool": "calculate_area_statistics",
            "description": "متوسط جميع ملاحظات المنطقة المتاحة." if is_ar else "Mean of all available area observations.",
        },
        {
            "id": "risk_percentile", "metric": "الرتبة المئينية للمخاطر" if is_ar else "Risk percentile rank",
            "value": risk_score, "unit": "%",
            "source_tool": "calculate_area_statistics",
            "description": "مقارنة القراءة الأحدث بتوزيع المنطقة نفسها." if is_ar else "Latest reading compared with this area's own distribution.",
        },
        {
            "id": "anomaly_count", "metric": "عدد القيم الشاذة" if is_ar else "Detected anomalies",
            "value": anomalies.get("anomaly_count"), "unit": "قراءة" if is_ar else "observations",
            "source_tool": "detect_consumption_anomalies",
            "description": "نتيجة حدود المدى الربيعي واختبار الانحراف المعياري." if is_ar else "Result of the IQR bounds and standardized-deviation check.",
        },
    ]
    if temperature.get("correlation") is not None:
        evidence.append({
            "id": "temperature_correlation",
            "metric": "ارتباط الحرارة بالاستهلاك" if is_ar else "Temperature correlation",
            "value": temperature.get("correlation"), "unit": "معامل ارتباط" if is_ar else "correlation",
            "source_tool": "analyze_temperature_relationship",
            "description": "ارتباط ملحوظ في العينة ولا يثبت السببية." if is_ar else "Observed sample association; it does not establish causation.",
        })
    if normalized.get("latest_liters_per_person") is not None:
        evidence.append({
            "id": "per_person_usage",
            "metric": "الاستهلاك الأحدث للفرد" if is_ar else "Latest per-person usage",
            "value": normalized.get("latest_liters_per_person"),
            "unit": "لتر/فرد" if is_ar else "L/person",
            "source_tool": "calculate_household_normalized_usage",
            "description": "تطبيع القراءة الأحدث حسب حجم الأسرة المسجل." if is_ar else "Latest observation normalized by its recorded household size.",
        })

    findings = [
        {
            "statement": (
                "تعتمد درجة المخاطر على موقع القراءة الأحدث داخل توزيع هذه المنطقة، وليس على حد ثابت."
                if is_ar else
                "Risk is based on where the latest reading sits within this area's distribution, not a fixed threshold."
            ),
            "evidence_refs": ["latest_usage", "area_average", "risk_percentile"],
        },
        {
            "statement": (
                "تم فحص القيم الشاذة بطريقة المدى الربيعي والانحراف المعياري."
                if is_ar else
                "Unusual observations were checked using IQR bounds and standardized deviation."
            ),
            "evidence_refs": ["anomaly_count"],
        },
    ]
    if temperature.get("correlation") is not None:
        findings.append({
            "statement": (
                "توجد علاقة مرصودة بين الحرارة والاستهلاك في العينة، لكنها لا تثبت أن الحرارة هي السبب."
                if is_ar else
                "The sample shows an observed temperature relationship, but it does not prove temperature is the cause."
            ),
            "evidence_refs": ["temperature_correlation"],
        })

    expected_impact = []
    for item in scenarios:
        if item.get("scenario") == "baseline":
            continue
        expected_impact.append({
            "scenario": item.get("scenario", "scenario"),
            "estimated_value": item.get("estimated_consumption_liters"),
            "unit": "لتر تقديري" if is_ar else "estimated L",
            "label": "estimate",
        })

    return {
        "decision_summary": (
            "يوصى بمراجعة اتجاه الاستهلاك والأدلة النسبية قبل اعتماد أي تدخل، مع إعطاء الأولوية للتحقق عندما تكون المخاطر مرتفعة."
            if is_ar else
            "Review the consumption trend and relative evidence before approving an intervention, prioritizing verification when risk is high."
        ),
        "risk_level": risk_level,
        "risk_score": risk_score,
        "findings": findings,
        "evidence": evidence,
        "assumptions": [
            "القيم تخص بيانات العرض المتاحة للمنطقة المختارة." if is_ar else "Values use the available demonstration data for the selected area.",
            "مقارنة السيناريو حساب تقديري وليست توقعاً أو قراراً معتمداً." if is_ar else "Scenario comparison is an arithmetic estimate, not a forecast or approved decision.",
        ],
        "uncertainty": [
            "الفترة الزمنية والعينة محدودتان ولا تتضمنان بيانات تشغيلية حية." if is_ar else "The time window and sample are limited and contain no live operational feed.",
            "الارتباطات المرصودة لا تثبت السببية." if is_ar else "Observed relationships do not establish causation.",
        ],
        "recommended_actions": [
            "مراجعة القراءات الأحدث والتحقق من سياقها قبل اعتماد إجراء تشغيلي." if is_ar else "Review recent readings and verify their context before approving an operational action.",
            "متابعة التوزيع النسبي للمنطقة عند إضافة بيانات جديدة." if is_ar else "Recalculate the area's relative distribution when new observations are added.",
        ],
        "expected_impact": expected_impact,
        "tools_used": REQUIRED_TOOLS.copy(),
    }


FUNCTION_TOOLS = [
    {
        "type": "function", "name": "get_area_observations",
        "description": "Retrieve chronological observations for the selected country and neighborhood. Call this first.",
        "parameters": {
            "type": "object",
            "properties": {"country": {"type": "string"}, "neighborhood": {"type": "string"}},
            "required": ["country", "neighborhood"], "additionalProperties": False,
        },
        "strict": True,
    },
    *[
        {
            "type": "function", "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            "strict": True,
        }
        for name, description in [
            ("calculate_area_statistics", "Calculate area statistics and percentile-based risk from retrieved observations."),
            ("detect_consumption_anomalies", "Detect consumption anomalies with IQR bounds and z-score."),
            ("analyze_temperature_relationship", "Measure the observed temperature-consumption relationship."),
            ("calculate_household_normalized_usage", "Calculate usage normalized by recorded household size."),
            ("compare_intervention_scenarios", "Compare explicitly estimated efficiency scenarios using the selected app setting."),
        ]
    ],
]


def _execute_tool(
    name: str,
    arguments: dict[str, Any],
    dataset: pd.DataFrame,
    country: str,
    neighborhood: str,
    efficiency_gain: float,
    tool_results: dict[str, Any],
) -> dict[str, Any]:
    if name == "get_area_observations":
        return get_area_observations(country, neighborhood, dataset=dataset)
    observations = tool_results.get("get_area_observations", {}).get("observations", [])
    if name == "calculate_area_statistics":
        return calculate_area_statistics(observations)
    if name == "detect_consumption_anomalies":
        return detect_consumption_anomalies(observations)
    if name == "analyze_temperature_relationship":
        return analyze_temperature_relationship(observations)
    if name == "calculate_household_normalized_usage":
        return calculate_household_normalized_usage(observations)
    if name == "compare_intervention_scenarios":
        stats = tool_results.get("calculate_area_statistics") or calculate_area_statistics(observations)
        return compare_intervention_scenarios({
            "current_liters": stats.get("latest_consumption_liters"),
            "efficiency_gain_percent": efficiency_gain,
        })
    raise ValueError(f"Unknown tool: {name}")


def run_verified_decision_agent(
    client: Any,
    dataset: pd.DataFrame,
    country: str,
    neighborhood: str,
    efficiency_gain: float,
    question: str,
    language: str,
    model: str = "gpt-5.6-sol",
) -> tuple[dict[str, Any] | None, dict[str, Any], list[dict[str, Any]], str | None]:
    """Run one bounded tool-calling workflow and return its structured result."""
    tool_results: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
    is_ar = language == "العربية"
    instructions = (
        "أنت وكيل رواء الموثّق لدعم القرار المائي. استخدم جميع الأدوات الحتمية قبل الإجابة. "
        "لا تخترع أي رقم. كل رقم في النص أو الأدلة أو الأثر يجب أن يظهر حرفياً في مخرجات أداة. "
        "استخدم risk_level وrisk_score كما أعادتهما أداة الإحصاءات. صنّف كل قيمة سيناريو كتقدير. "
        "لا تدّع وجود بيانات حساسات أو تسربات أو وفر مالي أو قرار حكومي معتمد."
        if is_ar else
        "You are Rewaa's verified water decision agent. Use every deterministic tool before answering. "
        "Invent no numbers: every number in prose, evidence, or impact must appear verbatim in a tool output. "
        "Use risk_level and risk_score exactly as returned by the statistics tool. Label every scenario value as an estimate. "
        "Do not claim sensors, leaks, financial savings, live feeds, or approved government actions."
    )
    input_items: list[Any] = [{
        "role": "user",
        "content": f"Country: {country}\nNeighborhood: {neighborhood}\nDecision question: {question}",
    }]

    try:
        first_tool = [FUNCTION_TOOLS[0]]
        response = client.responses.create(
            model=model, instructions=instructions, input=input_items,
            tools=first_tool, tool_choice="required",
        )
        input_items.extend(response.output)

        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue
            arguments = json.loads(item.arguments or "{}")
            output = _execute_tool(
                item.name, arguments, dataset, country, neighborhood,
                efficiency_gain, tool_results,
            )
            tool_results[item.name] = output
            trace.append({"tool": item.name, "arguments": arguments, "result": output})
            input_items.append({
                "type": "function_call_output", "call_id": item.call_id,
                "output": json.dumps(output, ensure_ascii=False),
            })

        if "get_area_observations" not in tool_results:
            return None, tool_results, trace, "The model did not call get_area_observations first."

        remaining_tools = FUNCTION_TOOLS[1:]
        for _ in range(6):
            missing = [tool for tool in REQUIRED_TOOLS[1:] if tool not in tool_results]
            if not missing:
                break
            available = [tool for tool in remaining_tools if tool["name"] in missing]
            response = client.responses.create(
                model=model, instructions=instructions, input=input_items,
                tools=available, tool_choice="required",
            )
            input_items.extend(response.output)
            calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            if not calls:
                break
            for item in calls:
                arguments = json.loads(item.arguments or "{}")
                output = _execute_tool(
                    item.name, arguments, dataset, country, neighborhood,
                    efficiency_gain, tool_results,
                )
                tool_results[item.name] = output
                trace.append({"tool": item.name, "arguments": arguments, "result": output})
                input_items.append({
                    "type": "function_call_output", "call_id": item.call_id,
                    "output": json.dumps(output, ensure_ascii=False),
                })

        missing = [tool for tool in REQUIRED_TOOLS if tool not in tool_results]
        if missing:
            return None, tool_results, trace, f"Incomplete tool chain: {', '.join(missing)}"

        final_response = client.responses.create(
            model=model,
            instructions=instructions + (
                " أعد النتيجة النهائية بالعربية وفق المخطط المطلوب. تجنب كتابة أرقام داخل النص؛ ضع الأرقام في حقول الأدلة والأثر فقط."
                if is_ar else
                " Return the final result in English using the required schema. Avoid numbers in prose; put numbers only in evidence and impact value fields."
            ),
            input=input_items,
            tools=FUNCTION_TOOLS,
            tool_choice="none",
            text={"format": {
                "type": "json_schema", "name": "rewaa_verified_decision",
                "strict": True, "schema": DECISION_SCHEMA,
            }},
        )
        result = json.loads(final_response.output_text)
        valid, errors = validate_decision_result(result, tool_results, trace)
        if not valid:
            return None, tool_results, trace, "; ".join(errors)
        return result, tool_results, trace, None
    except Exception as exc:
        return None, tool_results, trace, str(exc)


def run_local_tool_chain(
    dataset: pd.DataFrame,
    country: str,
    neighborhood: str,
    efficiency_gain: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute the exact same tool chain without OpenAI for safe fallback."""
    results: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
    for name in REQUIRED_TOOLS:
        arguments = {"country": country, "neighborhood": neighborhood} if name == "get_area_observations" else {}
        output = _execute_tool(
            name, arguments, dataset, country, neighborhood, efficiency_gain, results,
        )
        results[name] = output
        trace.append({"tool": name, "arguments": arguments, "result": output})
    return results, trace
