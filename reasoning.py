"""
reasoning.py
=============
The "brains" that turn raw forward/backward-chaining matches into a
ranked, explainable, uncertainty-aware diagnosis - and the collection of
classical AI search/optimisation techniques the project spec requires:

    - Certainty Factors (reasoning under uncertainty)
    - Explainable AI report generation
    - Reasoning Tree construction (networkx + optional Graphviz rendering)
    - Means-End Analysis (choose the next best clarifying question)
    - Generate-and-Test (enumerate & score every plausible disease)
    - Best-First Search (greedy search over the disease-symptom graph)
    - Hill Climbing (local search to refine a confidence estimate)
    - Problem Reduction (decompose a diagnosis into a category sub-problem)

Each technique is implemented as a small, focused class/function so it
can be demonstrated, tested, and explained independently, then composed
together by expert_system.py into one end-to-end diagnosis pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from backward_chaining import BackwardChainingEngine
from config import CERTAINTY, HILL_CLIMB_ITERATIONS, TOP_N_DIAGNOSES, get_logger
from forward_chaining import ForwardChainingEngine, ForwardChainResult
from fuzzy_logic import get_fuzzy_engine
from knowledge_base import MedicalKnowledgeBase

logger = get_logger(__name__)


# ==========================================================================
# 1. Certainty Factors - Reasoning Under Uncertainty
# ==========================================================================
@dataclass
class CertaintyResult:
    """A disease candidate with its computed certainty factor (confidence)."""
    disease_name: str
    certainty_factor: float          # in [0, 1]
    matched_required: List[str]
    missing_required: List[str]
    matched_supporting: List[str]
    fuzzy_severity_bonus: float = 0.0


class CertaintyFactorEngine:
    """
    Purpose: Combine symptom-match evidence into a single Certainty Factor
             (CF) per disease, in the classic MYCIN-style sense: a number
             in [0,1] representing "degree of belief", not a probability.

    Formula (see config.CERTAINTY for the tunable weights):
        CF = base_symptom_weight * matched_required_count
           + base_symptom_weight * 0.5 * matched_supporting_count
           + fuzzy_severity_bonus
           - missing_symptom_penalty * missing_required_count
        clamped to [min_cf, max_cf]
    """

    def __init__(self, kb: MedicalKnowledgeBase) -> None:
        self.kb = kb
        self.fuzzy_engine = get_fuzzy_engine()

    def compute(
        self,
        fc_result: ForwardChainResult,
        temperature_f: Optional[float] = None,
        pain_score: Optional[float] = None,
    ) -> CertaintyResult:
        """
        Purpose: Turn one ForwardChainResult into a CertaintyResult with a
                 numeric confidence score, optionally boosted by fuzzy
                 severity evidence (temperature/pain readings) if the
                 patient provided them.
        Input  : fc_result - output of ForwardChainingEngine.run() for one disease
                 temperature_f / pain_score - optional numeric vitals
        Output : CertaintyResult
        Logic  : Weighted linear combination of matched/missing symptom
                 counts (see class docstring), plus an additive bonus
                 derived from the fuzzy severity score when vitals are
                 supplied (a highly-febrile, high-pain reading nudges
                 confidence for fever-associated diseases upward).
        Time Complexity : O(1) - fixed number of arithmetic operations
                           (list lengths are already computed).
        Space Complexity: O(1)
        """
        n_required_matched = len(fc_result.matched_required)
        n_supporting_matched = len(fc_result.matched_supporting)
        n_missing = len(fc_result.missing_required)

        cf = (
            CERTAINTY.base_symptom_weight * n_required_matched
            + CERTAINTY.base_symptom_weight * 0.5 * n_supporting_matched
            - CERTAINTY.missing_symptom_penalty * n_missing
        )

        fuzzy_bonus = 0.0
        if temperature_f is not None and pain_score is not None:
            severity = self.fuzzy_engine.combined_severity(temperature_f, pain_score)
            fuzzy_bonus = severity * 0.15  # fuzzy evidence can add up to +0.15 CF
            cf += fuzzy_bonus

        cf = max(CERTAINTY.min_cf, min(CERTAINTY.max_cf, cf))

        return CertaintyResult(
            disease_name=fc_result.disease_name,
            certainty_factor=round(cf, 3),
            matched_required=fc_result.matched_required,
            missing_required=fc_result.missing_required,
            matched_supporting=fc_result.matched_supporting,
            fuzzy_severity_bonus=round(fuzzy_bonus, 3),
        )

    def rank(
        self,
        fc_results: List[ForwardChainResult],
        temperature_f: Optional[float] = None,
        pain_score: Optional[float] = None,
        top_n: int = TOP_N_DIAGNOSES,
    ) -> List[CertaintyResult]:
        """
        Purpose: Compute certainty factors for every forward-chaining
                 result and return the top_n ranked by confidence.
        Time Complexity : O(N log N) for the sort, N = candidate diseases.
        """
        scored = [self.compute(r, temperature_f, pain_score) for r in fc_results]
        scored.sort(key=lambda c: c.certainty_factor, reverse=True)
        return scored[:top_n]


# ==========================================================================
# 2. Explainable AI report
# ==========================================================================
@dataclass
class ExplanationReport:
    """A full, human-readable explainable-AI diagnosis report."""
    primary_diagnosis: str
    confidence_percent: float
    matched_symptoms: List[str]
    missing_symptoms: List[str]
    alternative_diagnoses: List[Tuple[str, float]]
    reasoning_summary: str
    safety_notes: List[str] = field(default_factory=list)


class ExplainableAIEngine:
    """
    Purpose: Turn a ranked list of CertaintyResults into the structured,
             plain-English "Diagnosis / Reason / Confidence / Alternatives"
             report format required by the spec.
    """

    def __init__(self, kb: MedicalKnowledgeBase) -> None:
        self.kb = kb

    def build_report(self, ranked: List[CertaintyResult]) -> ExplanationReport:
        """
        Purpose: Construct the final explanation report from a ranked
                 certainty-factor list (best candidate first).
        Time Complexity : O(N) where N = number of ranked candidates (<= TOP_N_DIAGNOSES).
        """
        if not ranked:
            return ExplanationReport(
                primary_diagnosis="Undetermined",
                confidence_percent=0.0,
                matched_symptoms=[],
                missing_symptoms=[],
                alternative_diagnoses=[],
                reasoning_summary="No disease in the knowledge base matched the reported symptoms closely enough.",
            )

        top = ranked[0]
        alternatives = [(r.disease_name, r.certainty_factor) for r in ranked[1:]]

        checklist = "\n".join(f"  ✔ {s}" for s in top.matched_required + top.matched_supporting)
        missing = "\n".join(f"  ✘ {s} (not reported)" for s in top.missing_required)
        summary_lines = [
            f"Diagnosis: {top.disease_name}",
            "Reason:",
            checklist or "  (no individual symptom matches recorded)",
        ]
        if missing:
            summary_lines.append("Symptoms typically expected but not reported:")
            summary_lines.append(missing)
        summary_lines.append(f"Confidence: {top.certainty_factor * 100:.0f}%")
        if alternatives:
            alt_str = ", ".join(f"{name} ({cf * 100:.0f}%)" for name, cf in alternatives)
            summary_lines.append(f"Alternative Diagnoses: {alt_str}")

        safety_notes = []
        disease = self.kb.get_disease(top.disease_name)
        if disease and disease.is_emergency:
            safety_notes.append(
                f"⚠ {top.disease_name} can be a medical emergency. Seek professional care promptly."
            )
        if disease and not disease.doctor_recommended:
            safety_notes.append("This condition is generally mild, but consult a doctor if symptoms worsen.")
        elif disease:
            safety_notes.append("A licensed medical professional should confirm this diagnosis before treatment.")

        return ExplanationReport(
            primary_diagnosis=top.disease_name,
            confidence_percent=round(top.certainty_factor * 100, 1),
            matched_symptoms=top.matched_required + top.matched_supporting,
            missing_symptoms=top.missing_required,
            alternative_diagnoses=alternatives,
            reasoning_summary="\n".join(summary_lines),
            safety_notes=safety_notes,
        )


# ==========================================================================
# 3. Reasoning Tree (Symptoms -> Rules -> Intermediate Facts -> Diagnosis)
# ==========================================================================
class ReasoningTreeBuilder:
    """
    Purpose: Build a networkx directed graph visualising how reported
             symptoms flow through matched rules into the final diagnosis,
             i.e. Symptoms -> Rules -> Intermediate Facts -> Diagnosis, as
             required by the spec. app.py renders this with matplotlib.
    """

    def __init__(self, kb: MedicalKnowledgeBase) -> None:
        self.kb = kb

    def build(self, reported_symptoms: Set[str], ranked: List[CertaintyResult]) -> nx.DiGraph:
        """
        Purpose: Construct the inference graph for the top-ranked diseases.
        Input  : reported_symptoms - patient's reported symptoms
                 ranked - CertaintyResult list (already sorted, best first)
        Output : networkx.DiGraph with node attribute 'layer' in
                 {"symptom", "rule", "fact", "diagnosis"} for layered rendering.
        Time Complexity : O(N * k) where N = number of diseases shown,
                           k = symptoms per disease (small constants).
        Space Complexity: O(N * k) nodes/edges.
        """
        graph = nx.DiGraph()

        for symptom in reported_symptoms:
            graph.add_node(f"symptom::{symptom}", label=symptom, layer="symptom")

        for rank_index, result in enumerate(ranked):
            disease = self.kb.get_disease(result.disease_name)
            if disease is None:
                continue
            rule_node = f"rule::{result.disease_name}"
            graph.add_node(
                rule_node,
                label=f"Rule: {result.disease_name}",
                layer="rule",
            )
            for symptom in result.matched_required + result.matched_supporting:
                sym_node = f"symptom::{symptom}"
                if sym_node not in graph:
                    graph.add_node(sym_node, label=symptom, layer="symptom")
                graph.add_edge(sym_node, rule_node)

            fact_node = f"fact::{result.disease_name}"
            graph.add_node(
                fact_node,
                label=f"CF={result.certainty_factor:.2f}",
                layer="fact",
            )
            graph.add_edge(rule_node, fact_node)

            diagnosis_node = f"diagnosis::{result.disease_name}"
            graph.add_node(
                diagnosis_node,
                label=f"{result.disease_name}\n({result.certainty_factor * 100:.0f}%)",
                layer="diagnosis",
                is_primary=(rank_index == 0),
            )
            graph.add_edge(fact_node, diagnosis_node)

        return graph


# ==========================================================================
# 4. Means-End Analysis - choose the next best clarifying question
# ==========================================================================
@dataclass
class MeansEndSuggestion:
    """The next best question to ask, and why it was chosen."""
    symptom_to_ask: Optional[str]
    reduces_ambiguity_between: List[str]
    rationale: str


class MeansEndAnalyzer:
    """
    Purpose: Implement Means-End Analysis - given the *current state*
             (known symptoms, current candidate confidences) and the
             *goal state* (a single, sufficiently-confident diagnosis),
             repeatedly reduce the "difference" between them by asking
             the single question that most reduces uncertainty across
             the current top candidates.
    """

    def __init__(self, kb: MedicalKnowledgeBase) -> None:
        self.kb = kb
        self.backward_engine = BackwardChainingEngine(kb)

    def next_best_question(
        self,
        ranked: List[CertaintyResult],
        already_known: Set[str],
        already_asked: Set[str],
        confidence_gap_threshold: float = 0.15,
    ) -> MeansEndSuggestion:
        """
        Purpose: Decide what to ask next to close the gap between the
                 current state (ambiguous top candidates) and the goal
                 state (one clearly-confident diagnosis).
        Input  : ranked - current CertaintyResult ranking (best first)
                 already_known - symptoms already confirmed present
                 already_asked - symptoms already asked about this session
                 confidence_gap_threshold - if the top-2 CFs differ by
                     more than this, we consider the goal state reached
                     and no further question is needed.
        Output : MeansEndSuggestion
        Logic  : If there's a clear leader (CF gap > threshold) OR fewer
                 than 2 candidates, the goal is close enough - stop.
                 Otherwise, ask the backward-chaining engine for the
                 unknown required-symptom shared by the most of the
                 top-competing candidates - the question with the
                 greatest expected "difference reduction".
        Time Complexity : O(C * k) via BackwardChainingEngine.next_question.
        """
        if len(ranked) < 2:
            return MeansEndSuggestion(None, [], "Only one (or zero) viable candidate remains - no further questions needed.")

        top, runner_up = ranked[0], ranked[1]
        gap = top.certainty_factor - runner_up.certainty_factor
        if gap >= confidence_gap_threshold:
            return MeansEndSuggestion(
                None, [],
                f"Goal state reached: '{top.disease_name}' leads '{runner_up.disease_name}' "
                f"by {gap * 100:.0f} confidence points - no further questions needed.",
            )

        competing = [r.disease_name for r in ranked[:3]]
        question = self.backward_engine.next_question(
            competing, already_known, set(), already_asked
        )
        if question is None:
            return MeansEndSuggestion(
                None, competing,
                "No further distinguishing symptoms are available in the knowledge base for these candidates.",
            )
        return MeansEndSuggestion(
            symptom_to_ask=question,
            reduces_ambiguity_between=competing,
            rationale=(
                f"'{question}' is a required symptom shared by the top competing diagnoses "
                f"({', '.join(competing)}); confirming or denying it will most reduce ambiguity."
            ),
        )


# ==========================================================================
# 5. Generate-and-Test
# ==========================================================================
class GenerateAndTestEngine:
    """
    Purpose: Classic Generate-and-Test search: (1) GENERATE every disease
             that plausibly could match (shares >=1 symptom), (2) TEST
             each by scoring it against the full reported symptom set,
             (3) rank and return. This is deliberately implemented as a
             standalone, simple exhaustive method - distinct from the
             forward-chaining rule engine - to satisfy the "generate and
             test" search-strategy requirement explicitly.
    """

    def __init__(self, kb: MedicalKnowledgeBase) -> None:
        self.kb = kb

    def generate(self, reported_symptoms: Set[str]) -> List[str]:
        """GENERATE step: candidate diseases sharing at least one symptom. O(|symptoms|)."""
        return sorted(self.kb.diseases_matching_any(reported_symptoms))

    def test(self, disease_name: str, reported_symptoms: Set[str]) -> float:
        """
        TEST step: score = (matched / total_symptoms_of_disease), a
        simple Jaccard-like overlap ratio independent of the CF engine,
        used to cross-validate the certainty-factor ranking.
        Time Complexity : O(k), k = symptoms of the disease.
        """
        disease = self.kb.get_disease(disease_name)
        if disease is None or not disease.all_symptoms:
            return 0.0
        overlap = len(set(disease.all_symptoms) & reported_symptoms)
        union = len(set(disease.all_symptoms) | reported_symptoms)
        return round(overlap / union, 3) if union else 0.0

    def run(self, reported_symptoms: Set[str], top_n: int = TOP_N_DIAGNOSES) -> List[Tuple[str, float]]:
        """
        Purpose: Full generate-and-test pipeline.
        Time Complexity : O(D * k) - D generated diseases, k symptoms each.
        """
        candidates = self.generate(reported_symptoms)
        scored = [(name, self.test(name, reported_symptoms)) for name in candidates]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_n]


# ==========================================================================
# 6. Best-First Search over the disease-symptom graph
# ==========================================================================
class BestFirstSearchEngine:
    """
    Purpose: Model diseases and symptoms as a bipartite graph and run a
             greedy best-first search from the patient's reported symptoms
             to find the single most likely disease node, using symptom
             overlap as the heuristic h(n) (the fewer *unexplained*
             symptoms a disease leaves, and the more of its symptoms are
             explained, the more attractive the node).
    """

    def __init__(self, kb: MedicalKnowledgeBase) -> None:
        self.kb = kb
        self.graph = self._build_bipartite_graph()

    def _build_bipartite_graph(self) -> nx.Graph:
        """Build once: symptom nodes <-> disease nodes, edge if the disease presents that symptom."""
        graph = nx.Graph()
        for disease_name, disease in self.kb.diseases.items():
            graph.add_node(f"disease::{disease_name}", bipartite="disease")
            for symptom in disease.all_symptoms:
                sym_node = f"symptom::{symptom}"
                if sym_node not in graph:
                    graph.add_node(sym_node, bipartite="symptom")
                graph.add_edge(sym_node, f"disease::{disease_name}")
        return graph

    def _heuristic(self, disease_name: str, reported_symptoms: Set[str]) -> float:
        """
        h(n): higher is better. Rewards explaining more reported symptoms
        and penalises diseases with many *unexplained* symptoms (keeps
        the search "focused" rather than favouring huge symptom lists).
        Time Complexity : O(k)
        """
        disease = self.kb.get_disease(disease_name)
        if disease is None or not disease.all_symptoms:
            return 0.0
        disease_symptoms = set(disease.all_symptoms)
        explained = len(disease_symptoms & reported_symptoms)
        unexplained = len(disease_symptoms - reported_symptoms)
        return explained - 0.1 * unexplained

    def search(self, reported_symptoms: Set[str], top_n: int = TOP_N_DIAGNOSES) -> List[Tuple[str, float]]:
        """
        Purpose: Greedy best-first search - expand from every reported
                 symptom node to its neighbouring disease nodes, score
                 each with the heuristic, and return the best top_n
                 (a priority-queue-driven frontier, classic best-first
                 search restricted to this shallow bipartite graph).
        Time Complexity : O(S * avg_degree + D log D) where S = reported
                           symptoms, D = frontier diseases discovered.
        """
        import heapq

        frontier: List[Tuple[float, str]] = []
        visited: Set[str] = set()

        for symptom in reported_symptoms:
            sym_node = f"symptom::{symptom}"
            if sym_node not in self.graph:
                continue
            for neighbor in self.graph.neighbors(sym_node):
                disease_name = neighbor.replace("disease::", "")
                if disease_name in visited:
                    continue
                visited.add(disease_name)
                score = self._heuristic(disease_name, reported_symptoms)
                heapq.heappush(frontier, (-score, disease_name))  # max-heap via negation

        results = []
        seen = set()
        while frontier and len(results) < top_n:
            neg_score, disease_name = heapq.heappop(frontier)
            if disease_name in seen:
                continue
            seen.add(disease_name)
            results.append((disease_name, round(-neg_score, 3)))
        return results


# ==========================================================================
# 7. Hill Climbing - locally optimise a confidence estimate
# ==========================================================================
class HillClimbingOptimizer:
    """
    Purpose: Demonstrate local-search optimisation by hill-climbing over
             the *weights* used to compute a disease's confidence score,
             searching for the weight combination that maximises
             separation between the top candidate and the runner-up
             (i.e. the most *decisive* explanation of the evidence).
             This is a lightweight, self-contained illustration of hill
             climbing applied to the diagnosis-confidence landscape,
             distinct from (and complementary to) the fixed-formula
             CertaintyFactorEngine.
    """

    def __init__(self, iterations: int = HILL_CLIMB_ITERATIONS) -> None:
        self.iterations = iterations

    @staticmethod
    def _score_with_weights(
        result: ForwardChainResult, w_required: float, w_supporting: float, w_missing: float
    ) -> float:
        raw = (
            w_required * len(result.matched_required)
            + w_supporting * len(result.matched_supporting)
            - w_missing * len(result.missing_required)
        )
        return max(0.0, min(1.0, raw))

    def optimize(self, fc_results: List[ForwardChainResult]) -> Dict[str, float]:
        """
        Purpose: Hill-climb over (w_required, w_supporting, w_missing) to
                 maximise the confidence gap between the best and second
                 best candidate - i.e. find weights that make the
                 diagnosis as *decisive* as possible given the evidence.
        Output : dict with the best weights found and resulting top score/gap.
        Logic  : Start from a reasonable seed (matches config.CERTAINTY),
                 then repeatedly try small random perturbations; keep a
                 perturbation only if it improves the objective (gap
                 between best and second-best) - textbook hill climbing
                 with random restarts avoided for simplicity/determinism.
        Time Complexity : O(I * N) where I = iterations, N = candidates scored per step.
        Space Complexity: O(N)
        """
        if len(fc_results) < 2:
            return {"w_required": CERTAINTY.base_symptom_weight, "w_supporting": 0.5, "w_missing": 0.07, "gap": 0.0}

        rng = random.Random(42)  # deterministic for reproducibility
        weights = {"w_required": 0.18, "w_supporting": 0.09, "w_missing": 0.07}

        def objective(w: Dict[str, float]) -> float:
            scores = sorted(
                (self._score_with_weights(r, w["w_required"], w["w_supporting"], w["w_missing"]) for r in fc_results),
                reverse=True,
            )
            return scores[0] - scores[1] if len(scores) >= 2 else scores[0]

        best_score = objective(weights)
        for _ in range(self.iterations):
            candidate = {
                k: max(0.01, v + rng.uniform(-0.02, 0.02)) for k, v in weights.items()
            }
            candidate_score = objective(candidate)
            if candidate_score > best_score:
                weights, best_score = candidate, candidate_score

        return {**weights, "gap": round(best_score, 3)}


# ==========================================================================
# 8. Problem Reduction - decompose diagnosis into category sub-problems
# ==========================================================================
class ProblemReducer:
    """
    Purpose: Implement Problem Reduction / AND-OR graph decomposition:
             break "diagnose the patient" into smaller sub-problems by
             disease category/parent_category hierarchy (e.g. Respiratory
             > Viral > Flu), narrowing the search space at each level
             before doing detailed symptom matching - mirroring how a
             clinician first narrows to a body system, then a
             sub-category, before pinpointing a specific disease.
    """

    def __init__(self, kb: MedicalKnowledgeBase) -> None:
        self.kb = kb

    def reduce(self, reported_symptoms: Set[str]) -> Dict[str, List[str]]:
        """
        Purpose: Group candidate diseases (sharing >=1 reported symptom)
                 by their top-level category, and within each category by
                 their parent_category sub-hierarchy - producing a
                 drill-down structure the UI can render as an AND-OR tree:
                 Category -> Sub-category -> [Diseases].
        Time Complexity : O(D) where D = candidate diseases.
        Space Complexity: O(D)
        """
        candidates = self.kb.diseases_matching_any(reported_symptoms)
        hierarchy: Dict[str, List[str]] = {}
        for name in sorted(candidates):
            disease = self.kb.get_disease(name)
            if disease is None:
                continue
            key = disease.parent_category or disease.category or "Other"
            hierarchy.setdefault(key, []).append(name)
        return hierarchy

    def narrow_by_category(self, reported_symptoms: Set[str], category: str) -> List[str]:
        """Return only the candidate diseases under a chosen top-level category (sub-problem selection)."""
        candidates = self.kb.diseases_matching_any(reported_symptoms)
        return sorted(
            name for name in candidates
            if self.kb.get_disease(name) and self.kb.get_disease(name).category == category
        )
