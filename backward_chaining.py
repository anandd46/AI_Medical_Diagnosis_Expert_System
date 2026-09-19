
"""
backward_chaining.py
=====================
Goal-driven ("top-down") backward chaining inference engine.

Given a hypothesised disease (the "goal"), this engine walks the
disease's required-symptom rule backward: for each required symptom it
checks whether the fact is already known (asserted or previously
answered); if not, it is added to a list of clarifying questions to ask
the patient. This mirrors how a clinician confirms a suspected diagnosis
by asking targeted follow-up questions instead of asking everything at
once (the dynamic-questioning requirement).

Typical flow (see expert_system.py):
    1. forward_chaining suggests a shortlist of likely diseases.
    2. backward_chaining is used per hypothesis to find out exactly which
       *additional* questions would confirm/deny it.
    3. The highest-value question (the one shared by the most remaining
       hypotheses) is asked first - this is the "means-end analysis"
       piece implemented in reasoning.py, which calls into this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from config import get_logger, MAX_QUESTIONS_PER_SESSION
from knowledge_base import MedicalKnowledgeBase

logger = get_logger(__name__)


@dataclass
class BackwardChainResult:
    """Outcome of trying to prove one disease-goal via backward chaining."""
    disease_name: str
    proved: bool                 # True if every required symptom is confirmed present
    confirmed_symptoms: List[str]
    denied_symptoms: List[str]   # required symptoms explicitly answered "no"
    unknown_symptoms: List[str]  # required symptoms not yet asked about
    proof_trace: List[str]


class BackwardChainingEngine:
    """
    Purpose: Implement goal-driven inference - "could the patient have
             disease X? What do we still need to know to be sure?"
    """

    def __init__(self, kb: MedicalKnowledgeBase) -> None:
        self.kb = kb

    def prove(
        self,
        disease_name: str,
        known_present: Set[str],
        known_absent: Set[str],
    ) -> BackwardChainResult:
        """
        Purpose: Attempt to prove Diagnosis(patient, disease_name) by
                 backward-chaining over its required-symptom rule.
        Input  : disease_name - the goal disease
                 known_present - symptoms already confirmed present
                 known_absent  - symptoms already confirmed absent
        Output : BackwardChainResult describing what's confirmed, denied,
                 or still unknown, plus a step-by-step proof_trace.
        Logic  : Standard backward-chaining over a single Horn clause:
                     RequiresSymptom(D, s1) ^ ... ^ RequiresSymptom(D, sn) -> Diagnosis(P, D)
                 For each required symptom (subgoal), check the fact base
                 (known_present / known_absent); if unresolved, it becomes
                 an open subgoal (a question the patient must answer).
                 The goal is PROVED only if every subgoal resolves to True.
        Time Complexity : O(k) where k = number of required symptoms for
                           the disease.
        Space Complexity: O(k)
        """
        disease = self.kb.get_disease(disease_name)
        if disease is None:
            return BackwardChainResult(disease_name, False, [], [], [], [f"Unknown disease '{disease_name}'."])

        required = disease.required_symptoms or disease.all_symptoms[:1]
        confirmed, denied, unknown, trace = [], [], [], []

        trace.append(f"GOAL: Diagnosis(patient, {disease_name})")
        trace.append(f"Subgoals (required symptoms): {', '.join(required)}")

        for symptom in required:
            if symptom in known_present:
                confirmed.append(symptom)
                trace.append(f"  ✔ RequiresSymptom({disease_name}, {symptom}) satisfied - patient HAS {symptom}")
            elif symptom in known_absent:
                denied.append(symptom)
                trace.append(f"  ✘ RequiresSymptom({disease_name}, {symptom}) FAILED - patient does NOT have {symptom}")
            else:
                unknown.append(symptom)
                trace.append(f"  ? RequiresSymptom({disease_name}, {symptom}) UNKNOWN - need to ask patient")

        proved = len(denied) == 0 and len(unknown) == 0
        trace.append(
            f"CONCLUSION: {'PROVED' if proved else 'NOT YET PROVED'} "
            f"({len(confirmed)}/{len(required)} confirmed, {len(denied)} denied, {len(unknown)} unknown)"
        )

        return BackwardChainResult(
            disease_name=disease_name,
            proved=proved,
            confirmed_symptoms=confirmed,
            denied_symptoms=denied,
            unknown_symptoms=unknown,
            proof_trace=trace,
        )

    def next_question(
        self,
        candidate_diseases: List[str],
        known_present: Set[str],
        known_absent: Set[str],
        already_asked: Set[str],
    ) -> str | None:
        """
        Purpose: Pick the single most informative next question to ask -
                 the unknown required symptom that appears across the
                 largest number of remaining candidate diseases. This
                 keeps the question count low (MAX_QUESTIONS_PER_SESSION)
                 by maximising information gain per question, similar in
                 spirit to a decision-tree split criterion.
        Input  : candidate_diseases - diseases still "in play"
                 known_present / known_absent - facts already known
                 already_asked - symptoms already asked about (skip)
        Output : the symptom name to ask about next, or None if nothing left to ask
        Time Complexity : O(C * k) where C = candidate diseases, k = required
                           symptoms per disease.
        Space Complexity: O(U) where U = number of distinct unknown symptoms.
        """
        frequency: Dict[str, int] = {}
        for disease_name in candidate_diseases:
            disease = self.kb.get_disease(disease_name)
            if disease is None:
                continue
            for symptom in disease.required_symptoms:
                if symptom in known_present or symptom in known_absent or symptom in already_asked:
                    continue
                frequency[symptom] = frequency.get(symptom, 0) + 1

        if not frequency:
            return None
        return max(frequency.items(), key=lambda kv: kv[1])[0]

    def run_session(
        self,
        candidate_diseases: List[str],
        known_present: Set[str],
        answer_fn,
        max_questions: int = MAX_QUESTIONS_PER_SESSION,
    ) -> Dict[str, BackwardChainResult]:
        """
        Purpose: Drive a full interactive backward-chaining session:
                 repeatedly pick the best next question, get an answer via
                 `answer_fn(symptom) -> bool`, and update the fact base,
                 until every candidate is proved/disproved or the question
                 budget is exhausted.
        Input  : candidate_diseases - shortlist to investigate
                 known_present - symptoms already confirmed
                 answer_fn - callable(symptom_name) -> bool, supplies answers
                             (in the Streamlit UI this reads from session state;
                             in tests/CLI it can be a scripted function)
                 max_questions - question budget (config.MAX_QUESTIONS_PER_SESSION)
        Output : dict mapping disease_name -> BackwardChainResult
        Time Complexity : O(Q * C * k) where Q = questions asked (bounded
                           by max_questions), C/k as above.
        """
        present = set(known_present)
        absent: Set[str] = set()
        asked: Set[str] = set()

        remaining = list(candidate_diseases)
        for _ in range(max_questions):
            question = self.next_question(remaining, present, absent, asked)
            if question is None:
                break
            asked.add(question)
            if answer_fn(question):
                present.add(question)
            else:
                absent.add(question)

            # Drop diseases that are now definitively disproved to focus remaining questions
            remaining = [
                d for d in remaining
                if not (set(self.kb.get_disease(d).required_symptoms) & absent)
            ]
            if not remaining:
                break

        return {d: self.prove(d, present, absent) for d in candidate_diseases}
