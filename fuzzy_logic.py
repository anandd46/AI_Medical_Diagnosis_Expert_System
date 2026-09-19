
"""
fuzzy_logic.py
===============
Fuzzy-logic reasoning for symptom severity, built on scikit-fuzzy.

Rather than hard thresholds (e.g. "fever if temperature > 100.4F"), this
module represents symptoms with fuzzy membership functions - a
temperature of 100.5F is simultaneously "moderately" and "highly"
feverish to different degrees - and uses a small Mamdani-style fuzzy
inference system to turn numeric readings (temperature, pain score,
symptom duration) into a fuzzy severity label (mild / moderate / high /
very_high) plus a numeric severity score in [0, 1] that feeds directly
into the certainty-factor calculation in reasoning.py.

Why fuzzy logic here specifically
----------------------------------
Medical severity is inherently a matter of degree, not a crisp category.
Two patients at 99.9F and 104F are both "feverish" but should not
contribute equally to a diagnosis's confidence. Fuzzy sets let the engine
express that gradient explicitly and combine multiple fuzzy readings
(temperature + pain + duration) using standard fuzzy operators (AND = min,
OR = max) instead of ad-hoc if/elif ladders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from config import DURATION_RANGE, FEVER_RANGE, PAIN_RANGE, get_logger

logger = get_logger(__name__)


@dataclass
class FuzzySeverityResult:
    """Result of fuzzifying a single numeric symptom reading."""
    crisp_value: float
    memberships: Dict[str, float]   # e.g. {"mild": 0.0, "moderate": 0.4, "high": 0.6, "very_high": 0.0}
    dominant_label: str
    severity_score: float           # defuzzified score in [0, 1]


def _labelled_trapmf(universe: np.ndarray, low: float, mid_low: float, mid_high: float, high: float) -> np.ndarray:
    """Thin wrapper around skfuzzy's trapezoidal membership function for readability."""
    return fuzz.trapmf(universe, [low, mid_low, mid_high, high])


class FuzzySymptomEngine:
    """
    Builds fuzzy membership functions for temperature, pain, and symptom
    duration, and fuzzifies raw numeric readings into severity labels and
    scores. A separate Mamdani control system combines temperature + pain
    into a single overall "severity" fuzzy inference, demonstrating true
    fuzzy rule-based inference (not just membership lookup).
    """

    def __init__(self) -> None:
        self._build_temperature_sets()
        self._build_pain_sets()
        self._build_duration_sets()
        self._build_control_system()

    # ------------------------------------------------------------- sets
    def _build_temperature_sets(self) -> None:
        lo, hi = FEVER_RANGE
        self.temp_universe = np.linspace(lo, hi, int((hi - lo) * 10) + 1)
        self.temp_mf = {
            "mild": _labelled_trapmf(self.temp_universe, lo, lo, 99.5, 100.5),
            "moderate": _labelled_trapmf(self.temp_universe, 99.5, 100.5, 101.5, 102.5),
            "high": _labelled_trapmf(self.temp_universe, 101.5, 102.5, 103.5, 104.5),
            "very_high": _labelled_trapmf(self.temp_universe, 103.5, 104.5, hi, hi),
        }

    def _build_pain_sets(self) -> None:
        lo, hi = PAIN_RANGE
        self.pain_universe = np.linspace(lo, hi, (hi - lo) * 10 + 1)
        self.pain_mf = {
            "mild": _labelled_trapmf(self.pain_universe, lo, lo, 2, 4),
            "moderate": _labelled_trapmf(self.pain_universe, 2, 4, 5, 7),
            "high": _labelled_trapmf(self.pain_universe, 5, 7, 8, 9),
            "very_high": _labelled_trapmf(self.pain_universe, 8, 9, hi, hi),
        }

    def _build_duration_sets(self) -> None:
        lo, hi = DURATION_RANGE
        self.duration_universe = np.linspace(lo, hi, (hi - lo) * 2 + 1)
        self.duration_mf = {
            "mild": _labelled_trapmf(self.duration_universe, lo, lo, 2, 4),
            "moderate": _labelled_trapmf(self.duration_universe, 2, 4, 7, 10),
            "high": _labelled_trapmf(self.duration_universe, 7, 10, 14, 18),
            "very_high": _labelled_trapmf(self.duration_universe, 14, 18, hi, hi),
        }

    # -------------------------------------------------------- fuzzify
    def fuzzify_temperature(self, temperature_f: float) -> FuzzySeverityResult:
        """
        Purpose: Convert a raw temperature reading (°F) into fuzzy
                 membership degrees across mild/moderate/high/very_high,
                 plus a single defuzzified severity score.
        Time Complexity : O(U) where U = size of the temperature universe
                           (fixed, ~110 points) - effectively O(1).
        """
        temperature_f = float(np.clip(temperature_f, *FEVER_RANGE))
        memberships = {
            label: float(fuzz.interp_membership(self.temp_universe, mf, temperature_f))
            for label, mf in self.temp_mf.items()
        }
        return self._package_result(temperature_f, memberships)

    def fuzzify_pain(self, pain_score: float) -> FuzzySeverityResult:
        """Same as fuzzify_temperature but for a 0-10 pain scale reading."""
        pain_score = float(np.clip(pain_score, *PAIN_RANGE))
        memberships = {
            label: float(fuzz.interp_membership(self.pain_universe, mf, pain_score))
            for label, mf in self.pain_mf.items()
        }
        return self._package_result(pain_score, memberships)

    def fuzzify_duration(self, days: float) -> FuzzySeverityResult:
        """Same as fuzzify_temperature but for symptom duration in days."""
        days = float(np.clip(days, *DURATION_RANGE))
        memberships = {
            label: float(fuzz.interp_membership(self.duration_universe, mf, days))
            for label, mf in self.duration_mf.items()
        }
        return self._package_result(days, memberships)

    @staticmethod
    def _package_result(crisp_value: float, memberships: Dict[str, float]) -> FuzzySeverityResult:
        """
        Purpose: Turn a memberships dict into a FuzzySeverityResult,
                 picking the dominant (highest-membership) label and
                 computing a defuzzified severity_score as the
                 membership-weighted average of ordinal severity ranks
                 (mild=0.0, moderate=0.33, high=0.67, very_high=1.0).
        Time Complexity : O(1) - fixed 4 labels.
        """
        rank = {"mild": 0.0, "moderate": 1 / 3, "high": 2 / 3, "very_high": 1.0}
        total_weight = sum(memberships.values()) or 1.0
        severity_score = sum(memberships[label] * rank[label] for label in memberships) / total_weight
        dominant = max(memberships.items(), key=lambda kv: kv[1])[0]
        return FuzzySeverityResult(
            crisp_value=crisp_value,
            memberships=memberships,
            dominant_label=dominant,
            severity_score=round(severity_score, 3),
        )

    # ------------------------------------------------ Mamdani control system
    def _build_control_system(self) -> None:
        """
        Purpose: Build a genuine Mamdani fuzzy inference system that
                 combines temperature AND pain fuzzy inputs into a single
                 "overall_severity" fuzzy output using explicit IF-THEN
                 fuzzy rules (not just averaging), demonstrating classical
                 fuzzy rule-based inference as required by the spec.
        Logic  : 9 rules covering the cross-product of {mild,moderate,high}
                 temperature x {mild,moderate,high} pain (very_high inputs
                 fold into "high" antecedents to keep the rule base compact
                 while still covering the full input space).
        Time Complexity : O(1) to build (fixed rule count); simulate() is
                           O(U) per call, U = universe resolution.
        """
        temperature = ctrl.Antecedent(self.temp_universe, "temperature")
        pain = ctrl.Antecedent(self.pain_universe, "pain")
        severity = ctrl.Consequent(np.linspace(0, 1, 101), "severity")

        for label, mf in self.temp_mf.items():
            temperature[label] = mf
        for label, mf in self.pain_mf.items():
            pain[label] = mf

        severity["mild"] = fuzz.trimf(severity.universe, [0, 0, 0.4])
        severity["moderate"] = fuzz.trimf(severity.universe, [0.2, 0.45, 0.7])
        severity["high"] = fuzz.trimf(severity.universe, [0.5, 0.75, 0.9])
        severity["very_high"] = fuzz.trimf(severity.universe, [0.75, 1.0, 1.0])

        def sev(label: str):
            return severity[label]

        rules = [
            ctrl.Rule(temperature["mild"] & pain["mild"], sev("mild")),
            ctrl.Rule(temperature["mild"] & pain["moderate"], sev("mild")),
            ctrl.Rule(temperature["mild"] & (pain["high"] | pain["very_high"]), sev("moderate")),
            ctrl.Rule(temperature["moderate"] & pain["mild"], sev("mild")),
            ctrl.Rule(temperature["moderate"] & pain["moderate"], sev("moderate")),
            ctrl.Rule(temperature["moderate"] & (pain["high"] | pain["very_high"]), sev("high")),
            ctrl.Rule((temperature["high"] | temperature["very_high"]) & pain["mild"], sev("moderate")),
            ctrl.Rule((temperature["high"] | temperature["very_high"]) & pain["moderate"], sev("high")),
            ctrl.Rule(
                (temperature["high"] | temperature["very_high"]) & (pain["high"] | pain["very_high"]),
                sev("very_high"),
            ),
        ]
        self._control_system = ctrl.ControlSystem(rules)

    def combined_severity(self, temperature_f: float, pain_score: float) -> float:
        """
        Purpose: Run the Mamdani fuzzy inference system end-to-end
                 (fuzzify -> apply rules -> aggregate -> defuzzify) to get
                 one overall severity score in [0, 1] from temperature and
                 pain readings together.
        Output : float severity in [0, 1] (0 = mild, 1 = very severe)
        Time Complexity : O(U) - dominated by skfuzzy's defuzzification
                           over the output universe (101 points, effectively O(1)).
        """
        simulation = ctrl.ControlSystemSimulation(self._control_system)
        simulation.input["temperature"] = float(np.clip(temperature_f, *FEVER_RANGE))
        simulation.input["pain"] = float(np.clip(pain_score, *PAIN_RANGE))
        try:
            simulation.compute()
            return round(float(simulation.output["severity"]), 3)
        except Exception:
            logger.warning("Fuzzy control system could not compute an output; falling back to a simple average.")
            t = self.fuzzify_temperature(temperature_f).severity_score
            p = self.fuzzify_pain(pain_score).severity_score
            return round((t + p) / 2, 3)


# Module-level singleton so callers don't rebuild the (cheap but non-trivial)
# membership functions and control system on every diagnosis.
_engine_instance: FuzzySymptomEngine | None = None


def get_fuzzy_engine() -> FuzzySymptomEngine:
    """Return a process-wide singleton FuzzySymptomEngine (lazy-initialised)."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FuzzySymptomEngine()
    return _engine_instance
