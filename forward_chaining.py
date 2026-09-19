"""
forward_chaining.py
====================
Data-driven ("bottom-up") forward chaining inference engine.

Given a set of reported patient symptoms, this engine starts from those
facts and repeatedly applies medical rules to derive every disease that
is at least partially supported by the evidence, ranking results by how
many of each disease's required symptoms were matched.

This is a genuine from-scratch forward chaining implementation (built on
top of predicate_logic.KnowledgeBase.forward_chain), specialised with
medical scoring so results are useful for ranking, not just true/false
membership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from config import get_logger
from knowledge_base import MedicalKnowledgeBase
from predicate_logic import Predicate

logger = get_logger(__name__)


@dataclass
class ForwardChainResult:
    """One candidate diagnosis produced by forward chaining."""
    disease_name: str
    matched_required: List[str]
    missing_required: List[str]
    matched_supporting: List[str]
    match_ratio: float          # matched_required / total_required
    rule_fired: bool            # True if ALL required symptoms were present (rule fully fired)


class ForwardChainingEngine:
    """
    Purpose: Implement classic data-driven inference - "given these facts,
             what can we conclude?" - specialised for the medical domain.
    """

    def __init__(self, kb: MedicalKnowledgeBase) -> None:
        self.kb = kb

    def run(self, patient_id: str, reported_symptoms: Set[str]) -> List[ForwardChainResult]:
        """
        Purpose: Run forward chaining starting from the patient's reported
                 symptoms and return every disease with at least one
                 matching symptom, ranked by match_ratio.
        Input  : patient_id - an identifier used only for predicate naming
                 reported_symptoms - set of symptom names the patient has
        Output : List[ForwardChainResult] sorted best-match first
        Logic  : 1) Clone the static logic KB and assert HasSymptom facts.
                 2) Run predicate_logic forward_chain() to derive
                    Diagnosis(patient, disease) facts for diseases whose
                    FULL required-symptom rule body fired.
                 3) Separately score every disease that shares >=1 symptom
                    with the patient (even if the rule didn't fully fire)
                    so partial matches are still visible to the user -
                    this mirrors how a real clinician forward-chains
                    partial evidence rather than requiring 100% certainty
                    before considering a hypothesis.
        Time Complexity : O(D * k) for the scoring pass (D diseases, k
                           symptoms per disease) + the forward_chain() cost
                           documented in predicate_logic.py.
        Space Complexity: O(D + F) for results and the cloned fact base.
        """
        session_kb = self.kb.clone_logic_kb()
        for sym in reported_symptoms:
            session_kb.add_fact(Predicate("HasSymptom", [patient_id, sym]))

        derived = session_kb.forward_chain()
        fired_diseases = {
            d.args[1] for d in derived if d.name == "Diagnosis" and d.args[0] == patient_id
        }
        logger.info("Forward chaining fired rules for: %s", sorted(fired_diseases))

        candidate_names = self.kb.diseases_matching_any(reported_symptoms) | fired_diseases
        results: List[ForwardChainResult] = []

        for name in candidate_names:
            disease = self.kb.get_disease(name)
            if disease is None:
                continue
            required = set(disease.required_symptoms) or set(disease.all_symptoms[:1])
            supporting = set(disease.all_symptoms) - required

            matched_required = sorted(required & reported_symptoms)
            missing_required = sorted(required - reported_symptoms)
            matched_supporting = sorted(supporting & reported_symptoms)

            match_ratio = len(matched_required) / len(required) if required else 0.0

            results.append(
                ForwardChainResult(
                    disease_name=name,
                    matched_required=matched_required,
                    missing_required=missing_required,
                    matched_supporting=matched_supporting,
                    match_ratio=round(match_ratio, 3),
                    rule_fired=name in fired_diseases,
                )
            )

        results.sort(key=lambda r: (r.rule_fired, r.match_ratio, len(r.matched_supporting)), reverse=True)
        return results

    def explain(self, patient_id: str, disease_name: str, reported_symptoms: Set[str]) -> str:
        """
        Purpose: Produce a short natural-language explanation of why a
                 given disease was (or wasn't) concluded by forward chaining.
        Time Complexity : O(k), k = symptoms of the disease.
        """
        disease = self.kb.get_disease(disease_name)
        if disease is None:
            return f"No knowledge available for '{disease_name}'."

        required = set(disease.required_symptoms)
        matched = sorted(required & reported_symptoms)
        missing = sorted(required - reported_symptoms)

        lines = [f"Rule: IF patient has {', '.join(sorted(required))} THEN consider {disease_name}."]
        if matched:
            lines.append(f"Matched: {', '.join(matched)}")
        if missing:
            lines.append(f"Missing: {', '.join(missing)}")
        lines.append(
            "Rule fully FIRED (all required symptoms present)."
            if not missing
            else f"Rule PARTIALLY matched ({len(matched)}/{len(required)} required symptoms)."
        )
        return " ".join(lines)
