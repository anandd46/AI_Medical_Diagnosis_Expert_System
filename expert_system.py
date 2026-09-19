
"""
expert_system.py
==================
The top-level orchestrator (Facade) that wires together every reasoning
engine in this project into one coherent, easy-to-use API for app.py.

Responsibilities
----------------
- Own a single shared MedicalKnowledgeBase instance.
- Run a full diagnosis session (forward chaining -> certainty factors ->
  explainable report -> CSP safety validation) from a set of reported
  symptoms plus optional patient attributes and vitals.
- Provide access to the individual search/reasoning techniques
  (best-first search, generate-and-test, hill climbing, problem
  reduction, means-end analysis, backward chaining) for the dedicated
  Streamlit pages that showcase them individually.
- Persist patients, visits and diagnosis logs to the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from backward_chaining import BackwardChainingEngine, BackwardChainResult
from config import get_logger
from constraints import ConstraintSolver, ConstraintViolation, PatientProfile
from database import get_session
from forward_chaining import ForwardChainingEngine
from knowledge_base import DiseaseProfile, MedicalKnowledgeBase
from models import DiagnosisLog, MedicalHistory, Patient, Visit
from reasoning import (
    BestFirstSearchEngine,
    CertaintyFactorEngine,
    CertaintyResult,
    ExplainableAIEngine,
    ExplanationReport,
    GenerateAndTestEngine,
    HillClimbingOptimizer,
    MeansEndAnalyzer,
    MeansEndSuggestion,
    ProblemReducer,
    ReasoningTreeBuilder,
)

logger = get_logger(__name__)


@dataclass
class DiagnosisSessionResult:
    """Everything produced by a single run_diagnosis() call, bundled for the UI."""
    reported_symptoms: Set[str]
    ranked_candidates: List[CertaintyResult]
    explanation: ExplanationReport
    reasoning_graph: nx.DiGraph
    constraint_violations: List[ConstraintViolation]
    generate_and_test_results: List[Tuple[str, float]]
    best_first_results: List[Tuple[str, float]]
    hill_climb_weights: Dict[str, float]
    problem_reduction: Dict[str, List[str]]
    means_end_suggestion: MeansEndSuggestion


class ExpertSystem:
    """
    Facade class - the single entry point app.py talks to. Internally
    delegates to the specialised engine classes so each AI technique
    remains independently testable and swappable.
    """

    def __init__(self) -> None:
        self.kb = MedicalKnowledgeBase()
        self.forward_engine = ForwardChainingEngine(self.kb)
        self.backward_engine = BackwardChainingEngine(self.kb)
        self.certainty_engine = CertaintyFactorEngine(self.kb)
        self.xai_engine = ExplainableAIEngine(self.kb)
        self.tree_builder = ReasoningTreeBuilder(self.kb)
        self.means_end = MeansEndAnalyzer(self.kb)
        self.generate_and_test = GenerateAndTestEngine(self.kb)
        self.best_first = BestFirstSearchEngine(self.kb)
        self.hill_climber = HillClimbingOptimizer()
        self.problem_reducer = ProblemReducer(self.kb)
        self.constraint_solver = ConstraintSolver(self.kb)
        logger.info("ExpertSystem initialised and ready.")

    # ---------------------------------------------------------- diagnosis
    def run_diagnosis(
        self,
        reported_symptoms: Set[str],
        patient: Optional[PatientProfile] = None,
        temperature_f: Optional[float] = None,
        pain_score: Optional[float] = None,
        patient_id: str = "patient",
    ) -> DiagnosisSessionResult:
        """
        Purpose: Run the full diagnosis pipeline end-to-end for a set of
                 reported symptoms, combining every reasoning technique
                 in the project into one coherent result object.
        Input  : reported_symptoms - symptom names the patient reports
                 patient - optional PatientProfile for CSP safety checks
                 temperature_f / pain_score - optional vitals for fuzzy severity
                 patient_id - identifier used only for predicate naming
        Output : DiagnosisSessionResult bundling every engine's output
        Logic  : 1) Forward chain to find candidate diseases + matches.
                 2) Score each with the Certainty Factor engine (fuzzy-boosted).
                 3) Build the explainable-AI report from the ranking.
                 4) Build the reasoning tree graph.
                 5) If a patient profile was given, run CSP constraint validation.
                 6) Run generate-and-test, best-first search, hill climbing,
                    problem reduction, and means-end analysis as
                    complementary/cross-validating views of the same evidence.
        Time Complexity : dominated by forward chaining's O(D*k); every
                           other step is documented as O(N) or O(N log N)
                           in its own module, N/D/k all bounded by the
                           (small, fixed) knowledge-base size.
        """
        fc_results = self.forward_engine.run(patient_id, reported_symptoms)
        ranked = self.certainty_engine.rank(fc_results, temperature_f, pain_score)
        explanation = self.xai_engine.build_report(ranked)
        graph = self.tree_builder.build(reported_symptoms, ranked)

        violations: List[ConstraintViolation] = []
        if patient is not None and ranked:
            violations = self.constraint_solver.full_report(patient, ranked[0].disease_name)

        gnt_results = self.generate_and_test.run(reported_symptoms)
        bfs_results = self.best_first.search(reported_symptoms)
        hill_weights = self.hill_climber.optimize(fc_results)
        reduction = self.problem_reducer.reduce(reported_symptoms)
        suggestion = self.means_end.next_best_question(ranked, reported_symptoms, set())

        return DiagnosisSessionResult(
            reported_symptoms=reported_symptoms,
            ranked_candidates=ranked,
            explanation=explanation,
            reasoning_graph=graph,
            constraint_violations=violations,
            generate_and_test_results=gnt_results,
            best_first_results=bfs_results,
            hill_climb_weights=hill_weights,
            problem_reduction=reduction,
            means_end_suggestion=suggestion,
        )

    def run_backward_chaining_session(
        self, candidate_diseases: List[str], known_present: Set[str], answer_fn
    ) -> Dict[str, BackwardChainResult]:
        """Expose an interactive backward-chaining Q&A session directly (used by the Diagnosis page)."""
        return self.backward_engine.run_session(candidate_diseases, known_present, answer_fn)

    def get_disease(self, name: str) -> Optional[DiseaseProfile]:
        return self.kb.get_disease(name)

    # -------------------------------------------------------- persistence
    def register_patient(
        self,
        full_name: str,
        age: int,
        gender: str,
        is_pregnant: bool = False,
        bp_systolic: Optional[int] = None,
        bp_diastolic: Optional[int] = None,
        known_allergies: str = "",
        chronic_conditions: str = "",
    ) -> int:
        """
        Purpose: Persist a new patient and return its database id.
        Time Complexity : O(1) (single insert).
        """
        with get_session() as session:
            patient = Patient(
                full_name=full_name,
                age=age,
                gender=gender,
                is_pregnant=is_pregnant,
                blood_pressure_systolic=bp_systolic,
                blood_pressure_diastolic=bp_diastolic,
                known_allergies=known_allergies,
                chronic_conditions=chronic_conditions,
            )
            session.add(patient)
            session.flush()
            patient_id = patient.id
        logger.info("Registered patient #%d: %s", patient_id, full_name)
        return patient_id

    def save_visit_and_diagnosis(
        self,
        patient_id: int,
        reported_symptoms: Set[str],
        result: DiagnosisSessionResult,
        notes: str = "",
    ) -> int:
        """
        Purpose: Persist a completed diagnosis session (Visit +
                 DiagnosisLog rows) so it can be reloaded from the
                 Medical History page later.
        Output : the new DiagnosisLog id.
        Time Complexity : O(1) (two inserts).
        """
        with get_session() as session:
            visit = Visit(
                patient_id=patient_id,
                reported_symptoms=",".join(sorted(reported_symptoms)),
                notes=notes,
            )
            session.add(visit)
            session.flush()

            alt_str = ",".join(f"{n}:{cf}" for n, cf in result.explanation.alternative_diagnoses)
            log = DiagnosisLog(
                visit_id=visit.id,
                disease_name=result.explanation.primary_diagnosis,
                confidence=result.explanation.confidence_percent / 100.0,
                explanation=result.explanation.reasoning_summary,
                alternative_diagnoses=alt_str,
            )
            session.add(log)
            session.flush()
            log_id = log.id
        logger.info(
            "Saved diagnosis log #%d for patient #%d: %s (%.0f%%)",
            log_id, patient_id, result.explanation.primary_diagnosis, result.explanation.confidence_percent,
        )
        return log_id

    def add_medical_history(self, patient_id: int, condition_name: str, notes: str = "") -> int:
        """Add a historical medical event for a patient. O(1)."""
        with get_session() as session:
            entry = MedicalHistory(patient_id=patient_id, condition_name=condition_name, notes=notes)
            session.add(entry)
            session.flush()
            entry_id = entry.id
        return entry_id

    def get_patient_history(self, patient_id: int) -> List[Dict]:
        """
        Purpose: Fetch every visit + diagnosis log for a patient, newest first.
        Time Complexity : O(V) where V = number of visits for the patient.
        """
        with get_session() as session:
            visits = (
                session.query(Visit)
                .filter(Visit.patient_id == patient_id)
                .order_by(Visit.visit_date.desc())
                .all()
            )
            history = []
            for visit in visits:
                logs = (
                    session.query(DiagnosisLog)
                    .filter(DiagnosisLog.visit_id == visit.id)
                    .all()
                )
                history.append(
                    {
                        "visit_date": visit.visit_date,
                        "reported_symptoms": visit.reported_symptoms,
                        "notes": visit.notes,
                        "diagnoses": [
                            {
                                "disease_name": log.disease_name,
                                "confidence": log.confidence,
                                "explanation": log.explanation,
                                "alternative_diagnoses": log.alternative_diagnoses,
                                "created_at": log.created_at,
                            }
                            for log in logs
                        ],
                    }
                )
            return history

    def get_all_patients(self) -> List[Dict]:
        """Return every registered patient as plain dicts (for the Patient Registration / History pages)."""
        with get_session() as session:
            patients = session.query(Patient).order_by(Patient.created_at.desc()).all()
            return [
                {
                    "id": p.id,
                    "full_name": p.full_name,
                    "age": p.age,
                    "gender": p.gender,
                    "is_pregnant": p.is_pregnant,
                    "known_allergies": p.known_allergies,
                    "chronic_conditions": p.chronic_conditions,
                    "created_at": p.created_at,
                }
                for p in patients
            ]

    def get_diagnosis_statistics(self) -> Dict:
        """
        Purpose: Aggregate statistics for the Statistics dashboard page:
                 disease diagnosis frequency and the full list of
                 historical confidence scores.
        Time Complexity : O(L) where L = total DiagnosisLog rows.
        """
        with get_session() as session:
            logs = session.query(DiagnosisLog).all()
            frequency: Dict[str, int] = {}
            confidences: List[float] = []
            for log in logs:
                frequency[log.disease_name] = frequency.get(log.disease_name, 0) + 1
                confidences.append(log.confidence)
            return {
                "total_diagnoses": len(logs),
                "disease_frequency": frequency,
                "confidence_scores": confidences,
            }
