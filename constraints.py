"""
constraints.py
================
Constraint Satisfaction (CSP) validation for the medical expert system.

Purpose
-------
A diagnosis or a suggested treatment can be logically well-supported by
symptoms yet still be medically *invalid* or *unsafe* given the patient's
attributes (age, sex, pregnancy status, allergies, existing conditions,
vital signs). This module encodes those attributes as CSP variables with
finite domains and checks a battery of hard constraints against them,
independent of - and applied *after* - the symptom-based reasoning.

Design
------
Each constraint is a small predicate function `(patient, disease_or_med) -> ConstraintViolation | None`.
`ConstraintSolver.validate_diagnosis` / `validate_treatment` run every
registered constraint and collect all violations, rather than stopping at
the first one, so the UI can show the patient a complete safety report
(this mirrors classic CSP "constraint checking" rather than full
backtracking search, since the "assignment" here - the diagnosis - is
already produced by the reasoning engines; the CSP's job is to validate,
not to search for an assignment).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from config import GERIATRIC_AGE_THRESHOLD, MAX_PATIENT_AGE, MIN_PATIENT_AGE, PEDIATRIC_AGE_LIMIT, get_logger
from knowledge_base import DiseaseProfile, MedicalKnowledgeBase, MedicineProfile

logger = get_logger(__name__)


@dataclass
class PatientProfile:
    """CSP-relevant subset of a patient's attributes (mirrors models.Patient)."""
    full_name: str
    age: int
    gender: str                 # "male" / "female" / "other"
    is_pregnant: bool = False
    blood_pressure_systolic: int | None = None
    blood_pressure_diastolic: int | None = None
    known_allergies: List[str] | None = None
    chronic_conditions: List[str] | None = None

    def __post_init__(self):
        self.known_allergies = [a.strip().lower() for a in (self.known_allergies or [])]
        self.chronic_conditions = [c.strip().lower() for c in (self.chronic_conditions or [])]


@dataclass
class ConstraintViolation:
    """A single failed constraint, with a human-readable explanation."""
    constraint_name: str
    severity: str          # "error" (blocks) or "warning" (caution)
    message: str


ConstraintFn = Callable[[PatientProfile], ConstraintViolation | None]
TreatmentConstraintFn = Callable[[PatientProfile, MedicineProfile], ConstraintViolation | None]


# ==========================================================================
# Patient-attribute constraints (age/sex/vitals validity)
# ==========================================================================
def constraint_valid_age(patient: PatientProfile) -> ConstraintViolation | None:
    """A patient's age must fall within [MIN_PATIENT_AGE, MAX_PATIENT_AGE]."""
    if not (MIN_PATIENT_AGE <= patient.age <= MAX_PATIENT_AGE):
        return ConstraintViolation(
            "valid_age", "error",
            f"Age {patient.age} is outside the plausible human range "
            f"({MIN_PATIENT_AGE}-{MAX_PATIENT_AGE}).",
        )
    return None


def constraint_male_not_pregnant(patient: PatientProfile) -> ConstraintViolation | None:
    """Biological constraint: a male patient cannot be recorded as pregnant."""
    if patient.gender.lower() == "male" and patient.is_pregnant:
        return ConstraintViolation(
            "male_not_pregnant", "error",
            "Patient is recorded as male but also marked pregnant - this is a data-entry contradiction.",
        )
    return None


def constraint_pregnancy_requires_female(patient: PatientProfile) -> ConstraintViolation | None:
    """Only patients recorded as female (or other, allowing for edge cases) may be marked pregnant."""
    if patient.is_pregnant and patient.gender.lower() not in ("female", "other"):
        return ConstraintViolation(
            "pregnancy_requires_appropriate_sex", "error",
            "Pregnancy flag is set for a patient not recorded as female/other.",
        )
    return None


def constraint_blood_pressure_plausible(patient: PatientProfile) -> ConstraintViolation | None:
    """Systolic/diastolic BP readings, if provided, must be physiologically plausible."""
    sys_bp, dia_bp = patient.blood_pressure_systolic, patient.blood_pressure_diastolic
    if sys_bp is None or dia_bp is None:
        return None
    if not (40 <= dia_bp <= 200) or not (60 <= sys_bp <= 260):
        return ConstraintViolation(
            "blood_pressure_plausible", "error",
            f"Blood pressure reading {sys_bp}/{dia_bp} mmHg is outside a physiologically plausible range.",
        )
    if sys_bp <= dia_bp:
        return ConstraintViolation(
            "blood_pressure_plausible", "error",
            f"Systolic ({sys_bp}) must be greater than diastolic ({dia_bp}).",
        )
    return None


PATIENT_CONSTRAINTS: List[ConstraintFn] = [
    constraint_valid_age,
    constraint_male_not_pregnant,
    constraint_pregnancy_requires_female,
    constraint_blood_pressure_plausible,
]


# ==========================================================================
# Diagnosis-level constraints (disease vs. patient attributes)
# ==========================================================================
def constraint_pregnancy_disease_flag(patient: PatientProfile, disease: DiseaseProfile) -> ConstraintViolation | None:
    """Flag diseases whose standard treatment requires special caution during pregnancy."""
    high_risk_if_pregnant = {"Malaria", "Typhoid", "Dengue", "Hepatitis A", "Pneumonia", "Tuberculosis"}
    if patient.is_pregnant and disease.name in high_risk_if_pregnant:
        return ConstraintViolation(
            "pregnancy_disease_caution", "warning",
            f"{disease.name} in a pregnant patient requires urgent obstetric-aware management; "
            "standard medicine dosing may not apply.",
        )
    return None


def constraint_pediatric_disease_flag(patient: PatientProfile, disease: DiseaseProfile) -> ConstraintViolation | None:
    """Certain adult-onset chronic diseases are extremely unlikely (not impossible) in young children."""
    adult_onset_diseases = {"Hypertension", "Heart Disease", "Osteoarthritis", "COPD", "Angina", "Gallstones"}
    if patient.age <= 5 and disease.name in adult_onset_diseases:
        return ConstraintViolation(
            "pediatric_plausibility", "warning",
            f"{disease.name} is uncommon in a child aged {patient.age}; consider re-evaluating symptoms "
            "or an alternative diagnosis.",
        )
    return None


def constraint_emergency_flag(patient: PatientProfile, disease: DiseaseProfile) -> ConstraintViolation | None:
    """Surface a hard warning whenever the candidate diagnosis is flagged as a medical emergency."""
    if disease.is_emergency:
        return ConstraintViolation(
            "emergency_condition", "warning",
            f"{disease.name} is flagged as a potential MEDICAL EMERGENCY - seek immediate professional care.",
        )
    return None


DIAGNOSIS_CONSTRAINTS: List[Callable[[PatientProfile, DiseaseProfile], ConstraintViolation | None]] = [
    constraint_pregnancy_disease_flag,
    constraint_pediatric_disease_flag,
    constraint_emergency_flag,
]


# ==========================================================================
# Treatment-level constraints (medicine vs. patient attributes)
# ==========================================================================
def constraint_medicine_allergy(patient: PatientProfile, medicine: MedicineProfile) -> ConstraintViolation | None:
    """
    A medicine must not be suggested if the patient has a recorded
    allergy to it or its drug class. Matches both against the medicine's
    own name (e.g. allergy "aspirin" vs medicine "Aspirin") and against
    its contraindication tags (e.g. allergy "penicillin" vs medicine
    "Amoxicillin" tagged contraindications="penicillin_allergy").
    """
    med_lower = medicine.name.lower()
    for allergy in patient.known_allergies:
        if not allergy:
            continue
        if allergy in med_lower or med_lower in allergy:
            return ConstraintViolation(
                "medicine_allergy", "error",
                f"Patient is allergic to '{allergy}' - {medicine.name} should NOT be administered.",
            )
        for tag in medicine.contraindications:
            if allergy in tag or tag.replace("_allergy", "") == allergy:
                return ConstraintViolation(
                    "medicine_allergy", "error",
                    f"Patient is allergic to '{allergy}', which is a known contraindication class "
                    f"for {medicine.name} - it should NOT be administered.",
                )
    return None


def constraint_medicine_pregnancy(patient: PatientProfile, medicine: MedicineProfile) -> ConstraintViolation | None:
    """Block/caution medicines contraindicated during pregnancy."""
    if patient.is_pregnant and any("pregnan" in c for c in medicine.contraindications):
        return ConstraintViolation(
            "medicine_pregnancy_contraindicated", "error",
            f"{medicine.name} is contraindicated during pregnancy.",
        )
    return None


def constraint_medicine_child_age(patient: PatientProfile, medicine: MedicineProfile) -> ConstraintViolation | None:
    """Block/caution medicines contraindicated in children below the pediatric age limit."""
    if patient.age < PEDIATRIC_AGE_LIMIT and any("child" in c for c in medicine.contraindications):
        return ConstraintViolation(
            "medicine_child_contraindicated", "error",
            f"{medicine.name} is contraindicated for children under {PEDIATRIC_AGE_LIMIT} "
            f"(patient is {patient.age}).",
        )
    return None


def constraint_medicine_chronic_condition(patient: PatientProfile, medicine: MedicineProfile) -> ConstraintViolation | None:
    """Flag medicines contraindicated by one of the patient's recorded chronic conditions."""
    for condition in patient.chronic_conditions:
        normalized = condition.replace(" ", "_")
        if any(normalized in c or c in normalized for c in medicine.contraindications):
            return ConstraintViolation(
                "medicine_chronic_condition", "warning",
                f"{medicine.name} may interact with patient's existing condition '{condition}' - "
                "verify with a physician before use.",
            )
    return None


def constraint_geriatric_caution(patient: PatientProfile, medicine: MedicineProfile) -> ConstraintViolation | None:
    """General caution flag for elderly patients on medicines with narrow safety margins."""
    narrow_margin_meds = {"Diazepam", "Ibuprofen", "Aspirin", "Methotrexate"}
    if patient.age >= GERIATRIC_AGE_THRESHOLD and medicine.name in narrow_margin_meds:
        return ConstraintViolation(
            "geriatric_caution", "warning",
            f"{medicine.name} requires dose caution in patients aged {GERIATRIC_AGE_THRESHOLD}+ "
            f"(patient is {patient.age}).",
        )
    return None


TREATMENT_CONSTRAINTS: List[TreatmentConstraintFn] = [
    constraint_medicine_allergy,
    constraint_medicine_pregnancy,
    constraint_medicine_child_age,
    constraint_medicine_chronic_condition,
    constraint_geriatric_caution,
]


# ==========================================================================
# Constraint solver / orchestrator
# ==========================================================================
class ConstraintSolver:
    """
    Purpose: Run the full battery of CSP-style constraints for a patient,
             a candidate diagnosis, and its associated medicines, and
             return every violation found (errors + warnings) rather than
             stopping at the first failure.
    """

    def __init__(self, kb: MedicalKnowledgeBase) -> None:
        self.kb = kb

    def validate_patient(self, patient: PatientProfile) -> List[ConstraintViolation]:
        """
        Time Complexity : O(C) where C = number of registered patient constraints (fixed, small).
        """
        violations = []
        for constraint in PATIENT_CONSTRAINTS:
            result = constraint(patient)
            if result:
                violations.append(result)
        return violations

    def validate_diagnosis(self, patient: PatientProfile, disease_name: str) -> List[ConstraintViolation]:
        """
        Purpose: Validate a candidate diagnosis against patient attributes.
        Time Complexity : O(C) - fixed number of diagnosis-level constraints.
        """
        disease = self.kb.get_disease(disease_name)
        if disease is None:
            return []
        violations = []
        for constraint in DIAGNOSIS_CONSTRAINTS:
            result = constraint(patient, disease)
            if result:
                violations.append(result)
        return violations

    def validate_treatment(self, patient: PatientProfile, disease_name: str) -> List[ConstraintViolation]:
        """
        Purpose: Validate every medicine associated with a disease against
                 patient attributes (allergies, pregnancy, age, chronic
                 conditions) and return all violations found.
        Time Complexity : O(M * C) where M = medicines linked to the
                           disease (typically 1-3), C = treatment constraints (fixed).
        """
        disease = self.kb.get_disease(disease_name)
        if disease is None:
            return []
        violations = []
        for med_name in disease.medicines:
            medicine = self.kb.medicines.get(med_name)
            if medicine is None:
                continue
            for constraint in TREATMENT_CONSTRAINTS:
                result = constraint(patient, medicine)
                if result:
                    violations.append(result)
        return violations

    def full_report(self, patient: PatientProfile, disease_name: str) -> List[ConstraintViolation]:
        """Convenience method combining patient + diagnosis + treatment validation."""
        return (
            self.validate_patient(patient)
            + self.validate_diagnosis(patient, disease_name)
            + self.validate_treatment(patient, disease_name)
        )
