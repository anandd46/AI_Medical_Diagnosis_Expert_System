"""
models.py
=========
SQLAlchemy ORM models for the AI Medical Diagnosis Expert System.

Purpose
-------
Defines the relational schema backing medical.db: patients, symptoms,
diseases, rules, medicines, treatments, medical history, visits and
diagnosis logs, plus the many-to-many association tables that connect
diseases <-> symptoms (the core of the knowledge base) and
diseases <-> medicines (treatments).

Each class purposefully carries a docstring explaining the medical
concept it represents so the schema itself doubles as documentation.

Time Complexity / Space Complexity notes are given per-model where a
model defines non-trivial helper methods; plain ORM declarations are
O(1) to construct.
"""

from __future__ import annotations

import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base class shared by every ORM model in this project."""
    pass


# --------------------------------------------------------------------------
# Association (many-to-many) tables
# --------------------------------------------------------------------------

disease_symptom_association = Table(
    "disease_symptom",
    Base.metadata,
    Column("disease_id", Integer, ForeignKey("diseases.id"), primary_key=True),
    Column("symptom_id", Integer, ForeignKey("symptoms.id"), primary_key=True),
    Column("weight", Float, default=1.0),          # how strongly this symptom indicates the disease
    Column("is_required", Boolean, default=False),  # required (defining) symptom vs. supporting symptom
)

disease_medicine_association = Table(
    "disease_medicine",
    Base.metadata,
    Column("disease_id", Integer, ForeignKey("diseases.id"), primary_key=True),
    Column("medicine_id", Integer, ForeignKey("medicines.id"), primary_key=True),
)


class Symptom(Base):
    """
    Represents a single observable medical symptom (e.g. Fever, Cough).

    A symptom is a predicate argument in the predicate-logic engine:
    HasSymptom(patient, symptom).
    """
    __tablename__ = "symptoms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    category: Mapped[str] = mapped_column(String(50), default="general")
    description: Mapped[str] = mapped_column(Text, default="")

    diseases: Mapped[List["Disease"]] = relationship(
        secondary=disease_symptom_association, back_populates="symptoms"
    )

    def __repr__(self) -> str:
        return f"<Symptom {self.name}>"


class Medicine(Base):
    """Represents a medicine / drug that can be used to treat one or more diseases."""
    __tablename__ = "medicines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    dosage_info: Mapped[str] = mapped_column(Text, default="")
    contraindications: Mapped[str] = mapped_column(
        Text, default=""
    )  # comma-separated tags e.g. "pregnancy,child,liver_disease"
    is_otc: Mapped[bool] = mapped_column(Boolean, default=True)  # over-the-counter?

    diseases: Mapped[List["Disease"]] = relationship(
        secondary=disease_medicine_association, back_populates="medicines"
    )

    def __repr__(self) -> str:
        return f"<Medicine {self.name}>"


class Disease(Base):
    """
    Represents a diagnosable disease/condition, the central concept of the
    knowledge base. Disease(x) is a unary predicate in the logic engine.
    """
    __tablename__ = "diseases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(150), unique=True, nullable=False)
    category: Mapped[str] = mapped_column(String(80), default="general")
    description: Mapped[str] = mapped_column(Text, default="")
    severity_level: Mapped[str] = mapped_column(String(20), default="moderate")  # mild/moderate/high/critical
    is_emergency: Mapped[bool] = mapped_column(Boolean, default=False)
    home_remedies: Mapped[str] = mapped_column(Text, default="")
    lifestyle_advice: Mapped[str] = mapped_column(Text, default="")
    doctor_recommended: Mapped[bool] = mapped_column(Boolean, default=True)
    parent_category: Mapped[str] = mapped_column(
        String(80), default=""
    )  # used for problem-reduction hierarchy, e.g. "Respiratory > Viral"

    symptoms: Mapped[List["Symptom"]] = relationship(
        secondary=disease_symptom_association, back_populates="diseases"
    )
    medicines: Mapped[List["Medicine"]] = relationship(
        secondary=disease_medicine_association, back_populates="diseases"
    )
    rules: Mapped[List["Rule"]] = relationship(back_populates="disease")

    def __repr__(self) -> str:
        return f"<Disease {self.name}>"


class Rule(Base):
    """
    Represents an explicit production rule used by the forward/backward
    chaining engines, of the form:

        IF <comma separated required symptom names> THEN Disease(name)

    Storing rules in the database (rather than only in Python) lets the
    knowledge base be inspected, edited and displayed from the
    'Knowledge Base' page of the Streamlit UI.
    """
    __tablename__ = "rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    disease_id: Mapped[int] = mapped_column(ForeignKey("diseases.id"))
    condition_symptoms: Mapped[str] = mapped_column(Text, nullable=False)  # "fever,cough,fatigue"
    min_matches: Mapped[int] = mapped_column(Integer, default=2)  # how many must match to fire
    confidence_weight: Mapped[float] = mapped_column(Float, default=0.8)
    description: Mapped[str] = mapped_column(Text, default="")

    disease: Mapped["Disease"] = relationship(back_populates="rules")

    def symptom_list(self) -> List[str]:
        """Return condition_symptoms as a clean python list of strings."""
        return [s.strip().lower() for s in self.condition_symptoms.split(",") if s.strip()]

    def __repr__(self) -> str:
        return f"<Rule -> {self.disease_id}: {self.condition_symptoms}>"


class Patient(Base):
    """Represents a registered patient."""
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    full_name: Mapped[str] = mapped_column(String(150), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(20), nullable=False)  # male/female/other
    is_pregnant: Mapped[bool] = mapped_column(Boolean, default=False)
    blood_pressure_systolic: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    blood_pressure_diastolic: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    known_allergies: Mapped[str] = mapped_column(Text, default="")  # comma-separated
    chronic_conditions: Mapped[str] = mapped_column(Text, default="")  # comma-separated
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    visits: Mapped[List["Visit"]] = relationship(back_populates="patient")
    history_entries: Mapped[List["MedicalHistory"]] = relationship(back_populates="patient")

    def __repr__(self) -> str:
        return f"<Patient {self.full_name} ({self.age}{self.gender[0].upper()})>"


class MedicalHistory(Base):
    """A single historical medical event/condition for a patient (longitudinal record)."""
    __tablename__ = "medical_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    condition_name: Mapped[str] = mapped_column(String(150), nullable=False)
    diagnosed_on: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    notes: Mapped[str] = mapped_column(Text, default="")

    patient: Mapped["Patient"] = relationship(back_populates="history_entries")


class Visit(Base):
    """Represents a single visit/session where a patient underwent symptom checking."""
    __tablename__ = "visits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    visit_date: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    reported_symptoms: Mapped[str] = mapped_column(Text, default="")  # comma-separated symptom names
    notes: Mapped[str] = mapped_column(Text, default="")

    patient: Mapped["Patient"] = relationship(back_populates="visits")
    diagnosis_logs: Mapped[List["DiagnosisLog"]] = relationship(back_populates="visit")


class DiagnosisLog(Base):
    """
    Persisted output of a completed diagnosis run: which disease was
    concluded, with what confidence, and a serialized explanation so it
    can be reloaded later from the 'Medical History' page.
    """
    __tablename__ = "diagnosis_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("visits.id"))
    disease_name: Mapped[str] = mapped_column(String(150), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, default="")  # human readable reasoning trace
    alternative_diagnoses: Mapped[str] = mapped_column(Text, default="")  # "COVID:0.65,Dengue:0.32"
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    visit: Mapped["Visit"] = relationship(back_populates="diagnosis_logs")

    def __repr__(self) -> str:
        return f"<DiagnosisLog {self.disease_name} ({self.confidence:.0%})>"
