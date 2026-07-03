# 🩺 AI Medical Diagnosis Expert System

An explainable, classical-AI medical symptom-checking expert system built entirely in Python — **no black-box machine learning**. Every diagnosis is traceable back to explicit predicate-logic facts, production rules, fuzzy membership functions, and certainty-factor arithmetic.

> ⚠ **Disclaimer**: This is an educational/portfolio AI project. It is **not** a certified medical device and must never replace professional medical advice, diagnosis, or treatment.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [AI Concepts Used](#ai-concepts-used)
5. [Architecture](#architecture)
6. [Workflow](#workflow)
7. [Project Structure](#project-structure)
8. [Explanation of Every Python File](#explanation-of-every-python-file)
9. [Installation](#installation)
10. [Requirements](#requirements)
11. [Running Instructions](#running-instructions)
12. [Database](#database)
13. [Knowledge Representation](#knowledge-representation)
14. [Predicate Logic](#predicate-logic)
15. [Forward Chaining](#forward-chaining)
16. [Backward Chaining](#backward-chaining)
17. [Constraint Satisfaction](#constraint-satisfaction)
18. [Fuzzy Logic](#fuzzy-logic)
19. [Reasoning Under Uncertainty](#reasoning-under-uncertainty)
20. [Explainable AI](#explainable-ai)
21. [Search Algorithms](#search-algorithms)
22. [Means-End Analysis](#means-end-analysis)
23. [Generate-and-Test](#generate-and-test)
24. [Problem Reduction](#problem-reduction)
25. [Screenshots](#screenshots)
26. [Future Improvements](#future-improvements)
27. [Resume Highlights](#resume-highlights)
28. [Interview Questions](#interview-questions)
29. [Advantages](#advantages)
30. [Limitations](#limitations)
31. [References](#references)
32. [License](#license)
33. [Contributors](#contributors)
34. [Acknowledgements](#acknowledgements)

---

## Project Overview

The **AI Medical Diagnosis Expert System** is a Streamlit-based application that diagnoses probable diseases from patient-reported symptoms using **classical, symbolic AI reasoning techniques** rather than statistical machine learning. It represents medical knowledge (diseases, symptoms, treatments, rules) as explicit logical facts and rules, and reasons over them using a hand-built predicate-logic engine, forward and backward chaining, fuzzy logic, certainty factors, constraint satisfaction, and several classical search/optimisation strategies (best-first search, hill climbing, generate-and-test, means-end analysis, problem reduction).

The result is a system where **every** diagnosis comes with a complete, human-readable explanation: which symptoms matched, which were missing, how confident the system is and why, what alternative diagnoses were considered, and what safety constraints (age, pregnancy, allergies, drug interactions) were checked.

## Problem Statement

Machine-learning diagnostic tools are frequently criticised as "black boxes" — they can be accurate but cannot explain *why* they reached a conclusion, which is a serious problem in a medical context where trust, auditability, and safety are paramount. This project asks: **can a purely symbolic, rule-based AI system provide competitive, fully-explainable diagnostic support**, demonstrating the foundational techniques (predicate logic, chaining, fuzzy reasoning, CSPs, heuristic search) that underpinned expert systems long before deep learning, and that remain valuable today specifically *because* they are transparent?

## Objectives

- Represent a non-trivial medical knowledge base (50+ diseases, 115+ symptoms, 40+ medicines) as explicit predicate-logic facts and rules.
- Implement forward chaining, backward chaining, unification and resolution **from scratch** (no external logic-programming library).
- Reason under uncertainty using certainty factors and fuzzy logic rather than hard thresholds.
- Enforce medical safety constraints via a constraint-satisfaction layer independent of the symptom-matching logic.
- Provide complete explainability: every diagnosis includes matched/missing evidence, confidence, alternatives, and a full reasoning trace.
- Demonstrate classical AI search strategies (best-first search, hill climbing, generate-and-test, means-end analysis, problem reduction) applied to a real, useful problem.
- Package everything behind a clean, professional, multi-page Streamlit UI suitable for a public portfolio/demo.

## AI Concepts Used

| Concept | Module | Description |
|---|---|---|
| Knowledge Representation | `models.py`, `sample_data.py` | Diseases, symptoms, medicines, rules as relational + predicate facts |
| Predicate Logic / Unification / Resolution | `predicate_logic.py` | Hand-built first-order Horn-clause engine |
| Forward Chaining | `forward_chaining.py` | Data-driven inference from symptoms to diagnoses |
| Backward Chaining | `backward_chaining.py` | Goal-driven proof + dynamic questioning |
| Constraint Satisfaction (CSP) | `constraints.py` | Age/sex/pregnancy/allergy/interaction validation |
| Fuzzy Logic (Mamdani inference) | `fuzzy_logic.py` | scikit-fuzzy membership functions + fuzzy rule base |
| Certainty Factors | `reasoning.py` | MYCIN-style confidence scoring |
| Explainable AI | `reasoning.py` | Structured diagnosis/reason/confidence/alternatives report |
| Reasoning Tree | `reasoning.py`, `utils.py` | NetworkX inference graph, rendered with Matplotlib |
| Means-End Analysis | `reasoning.py` | Choosing the next best clarifying question |
| Generate-and-Test | `reasoning.py` | Exhaustive candidate generation + scoring |
| Best-First Search | `reasoning.py` | Greedy heuristic search over a disease–symptom graph |
| Hill Climbing | `reasoning.py` | Local search optimising diagnostic decisiveness |
| Problem Reduction | `reasoning.py` | AND-OR category decomposition (e.g. Respiratory → Viral → Flu) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          app.py (Streamlit UI)                   │
│   Home · Registration · Symptom Checker · Diagnosis · Reasoning  │
│   Tree · Medical History · Knowledge Base · Statistics · About   │
└───────────────────────────────┬───────────────────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   expert_system.py         │
                    │   (Facade / Orchestrator)  │
                    └──────┬──────────────┬──────┘
         ┌──────────────────┼──────────────────┼───────────────────┐
         ▼                  ▼                  ▼                   ▼
 forward_chaining.py  backward_chaining.py  constraints.py    reasoning.py
         │                  │                  │            (CF, XAI, tree,
         └──────────┬───────┴──────────┬───────┘             means-end, GNT,
                     ▼                  ▼                     best-first, hill
            predicate_logic.py   fuzzy_logic.py                climb, problem
                     │                                          reduction)
                     ▼
             knowledge_base.py
                     │
                     ▼
              database.py + models.py
                     │
                     ▼
                medical.db (SQLite)
```

## Workflow

1. **Register / select a patient** (age, sex, pregnancy status, allergies, chronic conditions).
2. **Report symptoms** via the Symptom Checker, optionally with temperature/pain readings.
3. **Forward chaining** matches symptoms against every disease's rule, producing ranked candidates.
4. **Certainty factors** (optionally boosted by **fuzzy severity** from temperature/pain) score each candidate.
5. **Explainable AI** builds the final "Diagnosis / Reason / Confidence / Alternatives" report.
6. **Constraint satisfaction** flags unsafe treatments or implausible diagnoses for this specific patient.
7. **Means-end analysis** / **backward chaining** can suggest the next best clarifying question if candidates are close.
8. The **reasoning tree**, **generate-and-test**, **best-first search**, **hill climbing**, and **problem reduction** views are all available for deeper inspection.
9. Results can be **saved to patient history** or **exported as a PDF**.

## Project Structure

Everything lives in a single flat folder (no subfolders), as required for this deliverable:

```
Project/
├── app.py                 # Streamlit UI (all pages)
├── expert_system.py       # Orchestrator facade tying every engine together
├── knowledge_base.py       # Bridges DB data <-> predicate logic engine
├── reasoning.py            # Certainty factors, XAI, tree, search algorithms
├── forward_chaining.py     # Data-driven inference engine
├── backward_chaining.py    # Goal-driven inference + dynamic questioning
├── fuzzy_logic.py           # Fuzzy membership + Mamdani inference system
├── predicate_logic.py       # First-order logic: facts, rules, unification
├── constraints.py           # CSP-based medical safety validation
├── database.py              # SQLAlchemy engine/session + init/reset
├── models.py                 # SQLAlchemy ORM schema
├── utils.py                   # PDF export, chart builders, formatting helpers
├── sample_data.py              # 115 symptoms, 51 diseases, 42 medicines, seeding
├── config.py                    # Central configuration & logging
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── medical.db                    # Created automatically on first run
```

## Explanation of Every Python File

- **`config.py`** — All tunable constants (paths, certainty-factor weights, fuzzy ranges, UI theme) and the shared logger factory. Every other module imports from here instead of hard-coding values.
- **`models.py`** — SQLAlchemy ORM models: `Patient`, `Symptom`, `Disease`, `Medicine`, `Rule`, `MedicalHistory`, `Visit`, `DiagnosisLog`, plus the `disease_symptom` / `disease_medicine` association tables.
- **`database.py`** — Engine/session management, `init_db()` (create + seed), `reset_db()`. Runnable directly: `python database.py`.
- **`sample_data.py`** — The knowledge base *content*: 115 symptoms, 42 medicines, 51 diseases (each with required/supporting symptoms, medicines, remedies, severity, emergency flag), and the `seed()` function that writes it all into `medical.db`.
- **`predicate_logic.py`** — A genuine first-order predicate logic engine: `Predicate`, `Rule`, Robinson `unify()`, and a `KnowledgeBase` class with naive forward-chaining resolution and a proof trace.
- **`knowledge_base.py`** — Loads `medical.db` into fast in-memory dataclasses (`DiseaseProfile`, `SymptomProfile`, `MedicineProfile`) and builds/clones the static `predicate_logic.KnowledgeBase` template used by every reasoning engine.
- **`forward_chaining.py`** — Wraps predicate-logic forward chaining with medical scoring (match ratio, matched/missing symptoms) and produces a ranked candidate list even for partial matches.
- **`backward_chaining.py`** — Proves/disproves a specific disease hypothesis by checking its required symptoms one at a time, and implements `next_question()` / `run_session()` for interactive dynamic questioning.
- **`fuzzy_logic.py`** — Builds trapezoidal fuzzy membership functions for temperature/pain/duration and a 9-rule Mamdani fuzzy inference system (via `scikit-fuzzy`) that outputs one combined severity score.
- **`constraints.py`** — CSP-style validation: `PatientProfile` + a battery of constraint functions (age validity, male-not-pregnant, medicine allergy/pregnancy/child/chronic-condition/geriatric checks) collected by `ConstraintSolver`.
- **`reasoning.py`** — The largest module: `CertaintyFactorEngine`, `ExplainableAIEngine`, `ReasoningTreeBuilder`, `MeansEndAnalyzer`, `GenerateAndTestEngine`, `BestFirstSearchEngine`, `HillClimbingOptimizer`, `ProblemReducer`.
- **`utils.py`** — `generate_diagnosis_pdf()` (ReportLab), and Matplotlib chart builders (confidence bar chart, disease frequency, confidence distribution, reasoning-tree rendering).
- **`expert_system.py`** — The `ExpertSystem` facade: owns the shared knowledge base, exposes `run_diagnosis()` (the full pipeline) and database persistence helpers (`register_patient`, `save_visit_and_diagnosis`, `get_patient_history`, `get_diagnosis_statistics`).
- **`app.py`** — The Streamlit UI: one render function per page, routed from a sidebar radio.

## Installation

```bash
# 1. Clone or copy this folder
cd Project

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.12+
- See `requirements.txt` for the full pinned dependency list (Streamlit, SQLAlchemy, Pandas, NumPy, scikit-fuzzy, NetworkX, Matplotlib, Graphviz, ReportLab).

## Running Instructions

```bash
# (Optional) explicitly initialise + seed the database first
python database.py

# Launch the app
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`). The database is created and seeded automatically on first run if it doesn't already exist.

To reset the knowledge base to a clean state at any time:

```bash
python database.py --reset
```

## Database

SQLite (`medical.db`), accessed through SQLAlchemy. Tables: `patients`, `symptoms`, `diseases`, `disease_symptom` (association), `disease_medicine` (association), `medicines`, `rules`, `medical_history`, `visits`, `diagnosis_logs`.

## Knowledge Representation

Facts and relationships are represented both relationally (SQLite/SQLAlchemy, for persistence and the UI) and as first-order predicates (for reasoning):

```
Symptom(Fever)
Disease(Flu)
RequiresSymptom(Flu, Fever)
HasSymptom(John, Fever)
Treats(Flu, Paracetamol)
Diagnosis(John, Flu)          <- derived by forward chaining
```

## Predicate Logic

`predicate_logic.py` implements `Predicate`, `Rule` (Horn clauses), Robinson `unify()`, and a `KnowledgeBase` with `forward_chain()` performing naive resolution to a fixed point, plus a proof trace via `explain()`.

## Forward Chaining

Given `HasSymptom(patient, s)` facts, `ForwardChainingEngine.run()` asserts them into a cloned knowledge base, calls `forward_chain()`, and additionally scores *every* disease sharing at least one symptom (not just fully-fired rules) so partial evidence is still visible and rankable.

## Backward Chaining

`BackwardChainingEngine.prove()` starts from a disease *goal* and walks its required-symptom subgoals against known facts, classifying each as confirmed/denied/unknown. `next_question()` picks the unknown symptom shared by the most competing candidates (maximising information gain), and `run_session()` drives a full interactive Q&A loop.

## Constraint Satisfaction

`constraints.py` defines CSP variables (`PatientProfile`) and three tiers of constraint functions — patient-level (age validity, male/pregnant contradiction), diagnosis-level (pregnancy/pediatric plausibility, emergency flag), and treatment-level (allergy, pregnancy, child age, chronic-condition interaction, geriatric caution) — all collected without short-circuiting so a complete safety report is always produced.

## Fuzzy Logic

`fuzzy_logic.py` builds trapezoidal membership functions for temperature, pain and duration across `mild/moderate/high/very_high`, and a 9-rule Mamdani `ControlSystem` combining temperature AND pain into one defuzzified severity score in `[0,1]`, which feeds into the certainty-factor calculation as a bonus.

## Reasoning Under Uncertainty

`CertaintyFactorEngine` (in `reasoning.py`) computes a MYCIN-style certainty factor per disease as a weighted combination of matched required/supporting symptoms, a missing-symptom penalty, and an optional fuzzy-severity bonus, clamped to `[0.02, 0.98]` (the system never claims absolute certainty).

## Explainable AI

`ExplainableAIEngine.build_report()` produces the structured report format used throughout the UI and PDF export: primary diagnosis, matched/missing symptom checklist, numeric confidence, ranked alternative diagnoses, and safety notes — every field traceable to the underlying rule and evidence.

## Search Algorithms

- **Best-First Search** (`BestFirstSearchEngine`) — greedy search over a bipartite symptom↔disease graph using an "explained vs. unexplained symptoms" heuristic.
- **Hill Climbing** (`HillClimbingOptimizer`) — local search over confidence-scoring weights to maximise the gap between the top and runner-up diagnosis (decisiveness).
- **Generate-and-Test** (`GenerateAndTestEngine`) — exhaustively generates every disease sharing ≥1 symptom, then tests/scores each with a Jaccard-style overlap ratio, independent of the certainty-factor engine (used as a cross-check).

## Means-End Analysis

`MeansEndAnalyzer.next_best_question()` compares the current state (ambiguous top candidates) to the goal state (one clearly-confident diagnosis) and, if the gap is too small, asks `BackwardChainingEngine` for the single most disambiguating next question.

## Problem Reduction

`ProblemReducer.reduce()` decomposes "diagnose the patient" into a category/sub-category hierarchy (e.g. `Respiratory > Viral` → Flu, COVID-19, Common Cold), letting the UI show an AND-OR style drill-down before detailed symptom matching.

## Screenshots

> Add screenshots of the Home, Symptom Checker, Diagnosis, and Reasoning Tree pages here before publishing, e.g.:
>
> `![Home Page](screenshots/home.png)`
> `![Diagnosis Report](screenshots/diagnosis.png)`
> `![Reasoning Tree](screenshots/reasoning_tree.png)`

## Future Improvements

- Optional PySWIP/Prolog backend as an alternative resolution engine.
- Multi-symptom temporal reasoning (symptom onset order / progression).
- Natural-language symptom input (free text → symptom extraction).
- Multi-language UI.
- User authentication and per-clinician patient scoping.
- Richer drug-interaction database (pairwise medicine-medicine constraints).

## Resume Highlights

- Designed and implemented a **from-scratch first-order predicate logic engine** (unification, Horn-clause resolution, forward chaining) in pure Python.
- Built a **51-disease, 115-symptom, 42-medicine** medical knowledge base with a normalized SQLAlchemy schema.
- Implemented **8 distinct classical AI reasoning/search techniques** (forward/backward chaining, CSP, fuzzy logic, certainty factors, best-first search, hill climbing, generate-and-test, means-end analysis, problem reduction) as independently testable modules.
- Built a **fully explainable diagnosis pipeline** with confidence scoring, alternative-diagnosis ranking, and safety-constraint validation.
- Shipped a **9-page Streamlit dashboard** with PDF export, historical statistics, and an interactive reasoning-tree visualisation.

## Interview Questions

1. How does your unification algorithm differ from a full Prolog resolution engine, and what simplifications did you make?
2. Why use certainty factors instead of Bayesian probabilities here — what are the tradeoffs?
3. How does the fuzzy Mamdani inference system combine two fuzzy inputs (temperature, pain) into one output — walk through fuzzification, rule evaluation, aggregation, and defuzzification.
4. How would you extend the CSP layer to catch pairwise drug-drug interactions rather than just single-medicine contraindications?
5. Why did you separate forward chaining (data-driven) from backward chaining (goal-driven), and when would each be more appropriate clinically?
6. How does best-first search's heuristic differ from the certainty-factor ranking, and why keep both?
7. What would break first if the knowledge base grew to 5,000 diseases, and how would you address it (indexing, algorithmic complexity)?

## Advantages

- **Fully explainable** — every conclusion traces to explicit rules/facts, unlike black-box ML.
- **No training data required** — knowledge is authored directly by domain experts.
- **Deterministic and auditable** — the same inputs always produce the same reasoning trace.
- **Safety-aware** — CSP layer catches unsafe treatment suggestions independent of symptom matching.
- **Modular** — each AI technique is an independent, unit-testable class.

## Limitations

- Knowledge base coverage is inherently limited to what's manually authored (51 diseases here) — it cannot generalise to unseen diseases the way a trained model might.
- Certainty factors are a heuristic scoring scheme, not calibrated clinical probabilities.
- No temporal/longitudinal symptom-progression reasoning yet.
- Not a substitute for professional diagnosis — see the disclaimer at the top of this document.

## References

- Shortliffe, E. H. (1976). *Computer-Based Medical Consultations: MYCIN*.
- Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach* (predicate logic, unification, resolution, search algorithms).
- Zadeh, L. A. (1965). *Fuzzy Sets*. Information and Control.
- scikit-fuzzy documentation: https://pythonhosted.org/scikit-fuzzy/
- SQLAlchemy documentation: https://docs.sqlalchemy.org/
- Streamlit documentation: https://docs.streamlit.io/

## License

MIT License — see [`LICENSE`](./LICENSE). Includes an additional medical-software disclaimer.

## Contributors

- AI Engineering Portfolio Project — built as a demonstration of classical/explainable AI techniques applied to a real-world domain.

## Acknowledgements

Built with Streamlit, SQLAlchemy, scikit-fuzzy, NetworkX, Matplotlib, and ReportLab. Inspired by classical expert systems (MYCIN, DENDRAL) and the foundational AI textbook material on logic, search, and reasoning under uncertainty.
