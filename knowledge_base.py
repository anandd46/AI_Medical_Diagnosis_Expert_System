"""
knowledge_base.py
==================
Bridges the persisted medical data (diseases, symptoms, rules, medicines
in medical.db) with the domain-agnostic predicate_logic engine, and
exposes a convenient in-memory view of the knowledge base used by every
reasoning engine (forward_chaining.py, backward_chaining.py, fuzzy_logic.py,
constraints.py, reasoning.py).

Predicate vocabulary used throughout this project
---------------------------------------------------
    Symptom(name)                       - name is a known symptom
    Disease(name)                       - name is a known disease
    HasSymptom(patient, symptom)        - patient reports symptom
    Treats(disease, medicine)           - medicine treats disease
    RequiresSymptom(disease, symptom)   - symptom is a defining/required symptom of disease
    Diagnosis(patient, disease)         - derived: patient is diagnosed with disease
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from config import get_logger
from database import SessionLocal, init_db
from models import Disease as DiseaseRow
from models import Medicine as MedicineRow
from models import Rule as RuleRow
from models import Symptom as SymptomRow
from predicate_logic import KnowledgeBase, Predicate, Rule

logger = get_logger(__name__)


@dataclass
class DiseaseProfile:
    """
    An in-memory, reasoning-friendly view of a Disease row - avoids
    re-querying SQLAlchemy objects (and risking DetachedInstanceError)
    throughout the reasoning engines.
    """
    name: str
    category: str
    description: str
    severity_level: str
    is_emergency: bool
    home_remedies: str
    lifestyle_advice: str
    doctor_recommended: bool
    parent_category: str
    required_symptoms: List[str] = field(default_factory=list)
    all_symptoms: List[str] = field(default_factory=list)
    medicines: List[str] = field(default_factory=list)


@dataclass
class SymptomProfile:
    """In-memory view of a Symptom row."""
    name: str
    category: str
    description: str = ""


@dataclass
class MedicineProfile:
    """In-memory view of a Medicine row."""
    name: str
    dosage_info: str
    contraindications: List[str]
    is_otc: bool


class MedicalKnowledgeBase:
    """
    Loads the entire medical knowledge base from medical.db into memory
    once, and exposes both:
      (a) plain Python structures (fast lookups for the UI / CSP / fuzzy engines)
      (b) a predicate_logic.KnowledgeBase populated with Symptom/Disease/
          Treats/RequiresSymptom facts and Disease-inference rules, for
          the forward/backward chaining engines.

    This class is intentionally stateless with respect to any *specific*
    patient - patient facts (HasSymptom(patient, X)) are added to a fresh
    copy of the logic KB per diagnosis session (see expert_system.py).
    """

    def __init__(self, auto_init_db: bool = True) -> None:
        if auto_init_db:
            init_db()

        self.diseases: Dict[str, DiseaseProfile] = {}
        self.symptoms: Dict[str, SymptomProfile] = {}
        self.medicines: Dict[str, MedicineProfile] = {}
        self.symptom_to_diseases: Dict[str, Set[str]] = {}
        self.disease_rules_raw: List[RuleRow_ish] = []  # populated in _load()

        self._logic_template = KnowledgeBase()  # static facts/rules, cloned per session
        self._load()
        self._build_logic_template()
        logger.info(
            "MedicalKnowledgeBase loaded: %d diseases, %d symptoms, %d medicines.",
            len(self.diseases), len(self.symptoms), len(self.medicines),
        )

    # ---------------------------------------------------------------- load
    def _load(self) -> None:
        """
        Purpose: Pull every row out of medical.db into plain dataclasses
                 so the rest of the system never has to hold a live
                 SQLAlchemy session open.
        Time Complexity : O(D + S + M) - one pass over each table.
        Space Complexity: O(D + S + M)
        """
        session = SessionLocal()
        try:
            for row in session.query(SymptomRow).all():
                self.symptoms[row.name] = SymptomProfile(
                    name=row.name, category=row.category, description=row.description
                )

            for row in session.query(MedicineRow).all():
                self.medicines[row.name] = MedicineProfile(
                    name=row.name,
                    dosage_info=row.dosage_info,
                    contraindications=[c.strip() for c in row.contraindications.split(",") if c.strip()],
                    is_otc=row.is_otc,
                )

            for row in session.query(DiseaseRow).all():
                all_syms = [s.name for s in row.symptoms]
                # "required" symptoms are those flagged in the association table;
                # we approximate here by re-deriving from the Rule row instead,
                # which stores the authoritative required-symptom list.
                profile = DiseaseProfile(
                    name=row.name,
                    category=row.category,
                    description=row.description,
                    severity_level=row.severity_level,
                    is_emergency=row.is_emergency,
                    home_remedies=row.home_remedies,
                    lifestyle_advice=row.lifestyle_advice,
                    doctor_recommended=row.doctor_recommended,
                    parent_category=row.parent_category,
                    all_symptoms=all_syms,
                    medicines=[m.name for m in row.medicines],
                )
                self.diseases[row.name] = profile

                for sym_name in all_syms:
                    self.symptom_to_diseases.setdefault(sym_name, set()).add(row.name)

            for rule_row in session.query(RuleRow).all():
                disease_name = rule_row.disease.name
                required = rule_row.symptom_list()
                if disease_name in self.diseases:
                    self.diseases[disease_name].required_symptoms = self._match_case(
                        required, self.diseases[disease_name].all_symptoms
                    )
                self.disease_rules_raw.append(
                    _RawRule(
                        disease_name=disease_name,
                        required_symptoms=self.diseases[disease_name].required_symptoms
                        if disease_name in self.diseases else [],
                        min_matches=rule_row.min_matches,
                        confidence_weight=rule_row.confidence_weight,
                    )
                )
        finally:
            session.close()

    @staticmethod
    def _match_case(required_lower_or_mixed: List[str], canonical_names: List[str]) -> List[str]:
        """
        Purpose: Rule.condition_symptoms may not match the exact casing of
                 Symptom.name; resolve each required symptom string back to
                 its canonical (DB-cased) name using a case-insensitive match.
        Time Complexity : O(r * c) with tiny r (required syms) and c (all syms of a disease) - negligible.
        """
        canon_lower = {c.lower(): c for c in canonical_names}
        resolved = []
        for r in required_lower_or_mixed:
            resolved.append(canon_lower.get(r.lower(), r))
        return resolved

    # ---------------------------------------------------- logic template
    def _build_logic_template(self) -> None:
        """
        Purpose: Populate a predicate_logic.KnowledgeBase with static,
                 patient-independent facts and rules:
                     Symptom(name), Disease(name), Treats(disease, medicine),
                     RequiresSymptom(disease, symptom)
                 plus one inference Rule per disease of the form
                     HasSymptom(?P, s1) ^ HasSymptom(?P, s2) ^ ... -> Diagnosis(?P, disease)
                 This template is deep-copied (via clone_logic_kb) for
                 each new diagnosis session so patient facts never leak
                 between sessions.
        Time Complexity : O(D * k) where k = symptoms/medicines per disease.
        """
        for name in self.symptoms:
            self._logic_template.add_fact(Predicate("Symptom", [name]))

        for name, disease in self.diseases.items():
            self._logic_template.add_fact(Predicate("Disease", [name]))
            for med in disease.medicines:
                self._logic_template.add_fact(Predicate("Treats", [name, med]))
            for sym in disease.required_symptoms:
                self._logic_template.add_fact(Predicate("RequiresSymptom", [name, sym]))

            if disease.required_symptoms:
                body = [Predicate("HasSymptom", ["?P", s]) for s in disease.required_symptoms]
                self._logic_template.add_rule(
                    Rule(body=body, head=Predicate("Diagnosis", ["?P", name]), name=f"Diagnose_{name}")
                )

    def clone_logic_kb(self) -> KnowledgeBase:
        """
        Purpose: Return a fresh predicate_logic.KnowledgeBase pre-loaded
                 with the static medical facts/rules, ready to receive
                 one patient's HasSymptom(...) facts for a single
                 diagnosis session (used by forward_chaining.py).
        Time Complexity : O(F + R) to copy the static facts/rules.
        Space Complexity: O(F + R)
        """
        kb = KnowledgeBase()
        kb.facts = set(self._logic_template.facts)
        kb.rules = list(self._logic_template.rules)
        return kb

    # -------------------------------------------------------------- utils
    def all_symptom_names(self) -> List[str]:
        """Return every known symptom name, sorted alphabetically."""
        return sorted(self.symptoms.keys())

    def all_disease_names(self) -> List[str]:
        """Return every known disease name, sorted alphabetically."""
        return sorted(self.diseases.keys())

    def get_disease(self, name: str) -> DiseaseProfile | None:
        return self.diseases.get(name)

    def diseases_matching_any(self, reported_symptoms: Set[str]) -> Set[str]:
        """
        Purpose: Quickly narrow the search space to diseases that share
                 at least one symptom with what the patient reported -
                 used by generate_and_test (reasoning.py) and best-first
                 search so we don't score all 50 diseases uselessly.
        Time Complexity : O(|reported_symptoms|) average, via the
                           symptom_to_diseases index.
        """
        candidates: Set[str] = set()
        for sym in reported_symptoms:
            candidates |= self.symptom_to_diseases.get(sym, set())
        return candidates


@dataclass
class _RawRule:
    """Lightweight mirror of models.Rule used internally after the DB session closes."""
    disease_name: str
    required_symptoms: List[str]
    min_matches: int
    confidence_weight: float


# Type alias only used for the annotation above (kept name stable for readability).
RuleRow_ish = _RawRule
