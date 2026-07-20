import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from verified_decision_agent import (  # noqa: E402
    AREA_COLUMN,
    CONSUMPTION_COLUMN,
    COUNTRY_COLUMN,
    DATE_COLUMN,
    HOUSEHOLD_COLUMN,
    REQUIRED_TOOLS,
    TEMPERATURE_COLUMN,
    build_local_decision_result,
    calculate_area_statistics,
    compare_intervention_scenarios,
    detect_consumption_anomalies,
    get_ui_text,
    run_local_tool_chain,
    validate_decision_result,
)


def make_data(consumption):
    size = len(consumption)
    return pd.DataFrame({
        DATE_COLUMN: pd.date_range("2026-01-01", periods=size, freq="D"),
        COUNTRY_COLUMN: ["قطر"] * size,
        AREA_COLUMN: ["لوسيل"] * size,
        TEMPERATURE_COLUMN: list(range(30, 30 + size)),
        HOUSEHOLD_COLUMN: [4 + (index % 3) for index in range(size)],
        CONSUMPTION_COLUMN: consumption,
    })


class VerifiedDecisionAgentTests(unittest.TestCase):
    def test_low_risk_data_uses_area_percentile(self):
        data = make_data([100, 110, 120, 130, 140, 150, 160, 170, 180, 105])
        result = calculate_area_statistics(data)
        self.assertEqual(result["risk_level"], "low")
        self.assertLess(result["risk_score"], 75)
        self.assertIn("percentile", result["risk_method"].lower())

    def test_anomalous_latest_value_is_high_risk(self):
        data = make_data([100] * 9 + [500])
        statistics = calculate_area_statistics(data)
        anomalies = detect_consumption_anomalies(data)
        self.assertEqual(statistics["risk_level"], "high")
        self.assertTrue(anomalies["latest_is_anomaly"])
        self.assertEqual(anomalies["anomaly_count"], 1)

    def test_empty_data_is_safe(self):
        statistics = calculate_area_statistics(pd.DataFrame())
        anomalies = detect_consumption_anomalies(pd.DataFrame())
        self.assertEqual(statistics["status"], "insufficient_data")
        self.assertEqual(statistics["risk_level"], "unknown")
        self.assertIsNone(statistics["risk_score"])
        self.assertEqual(anomalies["anomaly_count"], 0)

    def test_bilingual_labels_and_demo_questions(self):
        arabic = get_ui_text("العربية")
        english = get_ui_text("English")
        self.assertEqual(arabic["title"], "وكيل رواء للتحقق من القرارات")
        self.assertEqual(
            arabic["question"],
            "لماذا يتغير استهلاك هذه المنطقة، وما الإجراء الذي ينبغي اعتماده هذا الأسبوع؟",
        )
        self.assertEqual(english["title"], "Rewaa Verified Decision Agent")
        self.assertEqual(
            english["question"],
            "Why is consumption changing in this area, and what action should be approved this week?",
        )

    def test_numerical_evidence_matches_tool_output(self):
        data = make_data([100, 110, 120, 130, 140, 150, 160, 170, 180, 105])
        results, trace = run_local_tool_chain(data, "قطر", "لوسيل", 20)
        decision = build_local_decision_result(results, "English")
        valid, errors = validate_decision_result(decision, results, trace)
        self.assertTrue(valid, errors)
        self.assertEqual(sorted(decision["tools_used"]), sorted(REQUIRED_TOOLS))

    def test_invented_numerical_evidence_is_rejected(self):
        data = make_data([100, 110, 120, 130, 140, 150, 160, 170, 180, 105])
        results, trace = run_local_tool_chain(data, "قطر", "لوسيل", 20)
        decision = build_local_decision_result(results, "English")
        decision["evidence"][0]["value"] = 999999.0
        valid, errors = validate_decision_result(decision, results, trace)
        self.assertFalse(valid)
        self.assertTrue(any("Unsupported evidence number" in error for error in errors))

    def test_scenarios_are_deterministic_and_labeled_estimates(self):
        scenarios = compare_intervention_scenarios({
            "current_liters": 500,
            "efficiency_gain_percent": 20,
        })["scenarios"]
        self.assertEqual(scenarios[1]["estimated_consumption_liters"], 400.0)
        self.assertEqual(scenarios[1]["estimated_reduction_liters"], 100.0)
        self.assertTrue(all(item["value_type"] == "estimate" for item in scenarios))


if __name__ == "__main__":
    unittest.main()
