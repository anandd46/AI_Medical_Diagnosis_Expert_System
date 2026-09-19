"""
app.py
=======
Streamlit front-end for the AI Medical Diagnosis Expert System.

This is the single UI entry point: `streamlit run app.py`. It wires the
ExpertSystem facade (expert_system.py) to a multi-page, sidebar-navigated
dashboard covering every feature in the project spec: Home, Patient
Registration, Symptom Checker, Diagnosis, Reasoning Tree, Medical
History, Knowledge Base, Statistics, and About.

The module is organised as one function per page (render_home,
render_patient_registration, ...) called from a simple router at the
bottom of the file, keeping each page's logic easy to locate and modify.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from config import (
    APP_AUTHOR,
    APP_NAME,
    APP_VERSION,
    STREAMLIT_LAYOUT,
    STREAMLIT_PAGE_ICON,
    STREAMLIT_PAGE_TITLE,
    THEME_COLORS,
)
from constraints import PatientProfile
from expert_system import ExpertSystem
from utils import (
    generate_diagnosis_pdf,
    plot_confidence_bar_chart,
    plot_confidence_distribution,
    plot_disease_frequency,
    plot_reasoning_tree,
)

# --------------------------------------------------------------------------
# Page configuration & theming
# --------------------------------------------------------------------------
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
    .main .block-container {{ padding-top: 2rem; }}
    h1, h2, h3 {{ color: {THEME_COLORS['primary']}; }}
    div[data-testid="stMetricValue"] {{ color: {THEME_COLORS['primary']}; }}
    .diagnosis-card {{
        background: {THEME_COLORS['background_light']};
        border-left: 6px solid {THEME_COLORS['primary']};
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .emergency-banner {{
        background: #FDECEA;
        border-left: 6px solid {THEME_COLORS['danger']};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        color: {THEME_COLORS['danger']};
        font-weight: 600;
    }}
    .warning-banner {{
        background: #FEF5E7;
        border-left: 6px solid {THEME_COLORS['warning']};
        border-radius: 8px;
        padding: 0.8rem 1.1rem;
        margin-bottom: 0.6rem;
    }}
    .stButton>button {{
        border-radius: 6px;
        font-weight: 600;
    }}
    footer {{visibility: hidden;}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached resources / session state
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading medical knowledge base...")
def get_expert_system() -> ExpertSystem:
    """
    Purpose: Build (once per server process) the ExpertSystem facade,
             which loads the knowledge base and wires up every reasoning
             engine. Cached so repeated Streamlit reruns don't reload
             medical.db on every widget interaction.
    """
    return ExpertSystem()


def _init_session_state() -> None:
    """Initialise every st.session_state key this app relies on, if absent."""
    defaults = {
        "selected_patient_id": None,
        "selected_patient_name": None,
        "reported_symptoms": set(),
        "last_result": None,
        "temperature_f": 98.6,
        "pain_score": 0,
        "backward_present": set(),
        "backward_absent": set(),
        "backward_asked": set(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()
es: ExpertSystem = get_expert_system()


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
PAGES = [
    "Home",
    "Patient Registration",
    "Symptom Checker",
    "Diagnosis",
    "Reasoning Tree",
    "Medical History",
    "Knowledge Base",
    "Statistics",
    "About Project",
]

with st.sidebar:
    st.markdown(f"## {STREAMLIT_PAGE_ICON} {APP_NAME}")
    st.caption(f"v{APP_VERSION} — {APP_AUTHOR}")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    if st.session_state.selected_patient_name:
        st.success(f"Active patient: **{st.session_state.selected_patient_name}**")
    else:
        st.info("No patient selected. Register or pick one on the Patient Registration page.")
    st.divider()
    st.caption(
        "⚠ This system is an educational AI portfolio project. "
        "It does NOT replace professional medical advice."
    )


# ==========================================================================
# PAGE: Home
# ==========================================================================
def render_home() -> None:
    st.title(f"{STREAMLIT_PAGE_ICON} {APP_NAME}")
    st.markdown(
        "An explainable, classical-AI expert system that diagnoses diseases from "
        "reported symptoms using **predicate logic, forward & backward chaining, "
        "fuzzy logic, certainty factors, constraint satisfaction, and heuristic search** "
        "— not a black-box machine learning model."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Diseases modeled", len(es.kb.all_disease_names()))
    col2.metric("Symptoms tracked", len(es.kb.all_symptom_names()))
    col3.metric("Medicines catalogued", len(es.kb.medicines))
    stats = es.get_diagnosis_statistics()
    col4.metric("Diagnoses run", stats["total_diagnoses"])

    st.divider()
    st.subheader("How it works")
    steps = st.columns(4)
    with steps[0]:
        st.markdown("**1. Report Symptoms**")
        st.write("Select symptoms (and optionally temperature/pain) on the Symptom Checker page.")
    with steps[1]:
        st.markdown("**2. Reason**")
        st.write("Forward chaining, fuzzy logic and certainty factors rank every plausible disease.")
    with steps[2]:
        st.markdown("**3. Explain**")
        st.write("Every diagnosis comes with a full explanation, reasoning tree, and confidence score.")
    with steps[3]:
        st.markdown("**4. Validate & Act**")
        st.write("Constraint checks flag unsafe treatments; export a PDF report or save to history.")

    st.divider()
    st.subheader("AI techniques implemented")
    technique_cols = st.columns(3)
    techniques = [
        "Predicate Logic & Unification", "Forward Chaining", "Backward Chaining",
        "Constraint Satisfaction (CSP)", "Fuzzy Logic (Mamdani inference)", "Certainty Factors",
        "Explainable AI reporting", "Reasoning Tree (NetworkX)", "Means-End Analysis",
        "Generate-and-Test", "Best-First Search", "Hill Climbing", "Problem Reduction",
    ]
    for i, technique in enumerate(techniques):
        technique_cols[i % 3].markdown(f"✅ {technique}")

    st.divider()
    st.info(
        "New here? Go to **Patient Registration** to add a patient, then use the "
        "**Symptom Checker** to run your first AI-assisted diagnosis."
    )


# ==========================================================================
# PAGE: Patient Registration
# ==========================================================================
def render_patient_registration() -> None:
    st.title("🧾 Patient Registration")

    tab_new, tab_existing = st.tabs(["Register New Patient", "Select Existing Patient"])

    with tab_new:
        with st.form("patient_registration_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                full_name = st.text_input("Full Name *")
                age = st.number_input("Age *", min_value=0, max_value=120, value=30)
                gender = st.selectbox("Gender *", ["Female", "Male", "Other"])
            with col2:
                is_pregnant = st.checkbox("Currently Pregnant", value=False)
                bp_sys = st.number_input("Blood Pressure - Systolic (optional)", min_value=0, max_value=260, value=0)
                bp_dia = st.number_input("Blood Pressure - Diastolic (optional)", min_value=0, max_value=200, value=0)

            allergies = st.text_input("Known Allergies (comma-separated)", placeholder="e.g. penicillin, aspirin")
            chronic = st.text_input("Chronic Conditions (comma-separated)", placeholder="e.g. diabetes, kidney_disease")

            submitted = st.form_submit_button("Register Patient", width='stretch')
            if submitted:
                if not full_name.strip():
                    st.error("Full name is required.")
                elif gender == "Male" and is_pregnant:
                    st.error("Data constraint violation: a male patient cannot be marked pregnant.")
                else:
                    pid = es.register_patient(
                        full_name=full_name.strip(),
                        age=int(age),
                        gender=gender.lower(),
                        is_pregnant=is_pregnant,
                        bp_systolic=int(bp_sys) or None,
                        bp_diastolic=int(bp_dia) or None,
                        known_allergies=allergies,
                        chronic_conditions=chronic,
                    )
                    st.session_state.selected_patient_id = pid
                    st.session_state.selected_patient_name = full_name.strip()
                    st.success(f"Patient '{full_name}' registered (ID #{pid}) and set as active patient.")

    with tab_existing:
        patients = es.get_all_patients()
        if not patients:
            st.info("No patients registered yet.")
        else:
            df = pd.DataFrame(patients)[["id", "full_name", "age", "gender", "is_pregnant", "created_at"]]
            df.columns = ["ID", "Name", "Age", "Gender", "Pregnant", "Registered On"]
            st.dataframe(df, width='stretch', hide_index=True)

            options = {f"#{p['id']} — {p['full_name']} ({p['age']}{p['gender'][0].upper()})": p for p in patients}
            choice = st.selectbox("Select active patient", list(options.keys()))
            if st.button("Set as Active Patient", width='stretch'):
                chosen = options[choice]
                st.session_state.selected_patient_id = chosen["id"]
                st.session_state.selected_patient_name = chosen["full_name"]
                st.success(f"Active patient set to {chosen['full_name']}.")


# ==========================================================================
# PAGE: Symptom Checker
# ==========================================================================
def render_symptom_checker() -> None:
    st.title("🔍 Symptom Checker")

    if st.session_state.selected_patient_id is None:
        st.warning("No active patient selected. You can still run a diagnosis anonymously, "
                    "but it won't be saved to a patient's history unless you register/select one first.")

    all_symptoms = es.kb.all_symptom_names()
    st.subheader("1. Select reported symptoms")
    selected = st.multiselect(
        "Search and select symptoms",
        options=all_symptoms,
        default=sorted(st.session_state.reported_symptoms),
        help="Start typing to filter; select every symptom the patient currently has.",
    )
    st.session_state.reported_symptoms = set(selected)

    st.subheader("2. Optional vitals (improves confidence via fuzzy logic)")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Body Temperature (°F)", 95.0, 106.0, float(st.session_state.temperature_f), 0.1)
        st.session_state.temperature_f = temperature
    with col2:
        pain = st.slider("Overall Pain / Discomfort (0-10)", 0, 10, int(st.session_state.pain_score))
        st.session_state.pain_score = pain

    st.divider()
    run_col, clear_col = st.columns([3, 1])
    with clear_col:
        if st.button("Clear Selections", width='stretch'):
            st.session_state.reported_symptoms = set()
            st.rerun()

    with run_col:
        run_clicked = st.button(
            "🩺 Run AI Diagnosis", type="primary", width='stretch',
            disabled=len(st.session_state.reported_symptoms) == 0,
        )

    if len(st.session_state.reported_symptoms) == 0:
        st.caption("Select at least one symptom to enable diagnosis.")

    if run_clicked:
        patient_profile = None
        if st.session_state.selected_patient_id is not None:
            patients = {p["id"]: p for p in es.get_all_patients()}
            p = patients.get(st.session_state.selected_patient_id)
            if p:
                patient_profile = PatientProfile(
                    full_name=p["full_name"],
                    age=p["age"],
                    gender=p["gender"],
                    is_pregnant=p["is_pregnant"],
                    known_allergies=p["known_allergies"].split(",") if p["known_allergies"] else [],
                    chronic_conditions=p["chronic_conditions"].split(",") if p["chronic_conditions"] else [],
                )

        with st.spinner("Running forward chaining, fuzzy inference, certainty factors, and CSP validation..."):
            result = es.run_diagnosis(
                st.session_state.reported_symptoms,
                patient=patient_profile,
                temperature_f=temperature,
                pain_score=pain,
                patient_id=str(st.session_state.selected_patient_id or "anonymous"),
            )
        st.session_state.last_result = result
        st.success("Diagnosis complete — see the **Diagnosis** page for the full report.")
        st.balloons()


# ==========================================================================
# PAGE: Diagnosis
# ==========================================================================
def render_diagnosis() -> None:
    st.title("🩺 Diagnosis Report")
    result = st.session_state.last_result

    if result is None:
        st.info("No diagnosis has been run yet. Go to **Symptom Checker** to run one.")
        return

    explanation = result.explanation
    disease = es.get_disease(explanation.primary_diagnosis)

    if disease and disease.is_emergency:
        st.markdown(
            f'<div class="emergency-banner">⚠ EMERGENCY WARNING: {disease.name} can be a medical '
            f'emergency. Seek immediate professional care.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""<div class="diagnosis-card">
        <h2>{explanation.primary_diagnosis}</h2>
        <p style="font-size:1.3rem;"><b>Confidence: {explanation.confidence_percent:.0f}%</b></p>
        </div>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✔ Matched Symptoms")
        for s in explanation.matched_symptoms:
            st.write(f"✔ {s}")
    with col2:
        st.subheader("✘ Expected but Not Reported")
        if explanation.missing_symptoms:
            for s in explanation.missing_symptoms:
                st.write(f"✘ {s}")
        else:
            st.write("None — every typical symptom was reported.")

    if explanation.alternative_diagnoses:
        st.subheader("Alternative Diagnoses")
        fig = plot_confidence_bar_chart(
            [(explanation.primary_diagnosis, explanation.confidence_percent / 100)]
            + explanation.alternative_diagnoses
        )
        st.pyplot(fig)

    if result.constraint_violations:
        st.subheader("⚠ Safety & Constraint Checks")
        for v in result.constraint_violations:
            css = "emergency-banner" if v.severity == "error" else "warning-banner"
            st.markdown(f'<div class="{css}">[{v.severity.upper()}] {v.message}</div>', unsafe_allow_html=True)

    if explanation.safety_notes:
        for note in explanation.safety_notes:
            st.warning(note)

    if disease:
        st.subheader("Recommended Treatment")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Medicines**")
            if disease.medicines:
                for m in disease.medicines:
                    med = es.kb.medicines.get(m)
                    st.write(f"- **{m}** — {med.dosage_info if med else ''}")
            else:
                st.write("No specific medication catalogued — supportive care recommended.")
            st.markdown("**Home Remedies**")
            st.write(disease.home_remedies or "None specified.")
        with col2:
            st.markdown("**Lifestyle Advice**")
            st.write(disease.lifestyle_advice or "None specified.")
            st.markdown("**Doctor Visit Recommended?**")
            st.write("Yes" if disease.doctor_recommended else "Not usually required for mild cases.")

    with st.expander("🔬 Full Reasoning Trace"):
        st.text(explanation.reasoning_summary)

    with st.expander("🧭 Means-End Analysis (next best question)"):
        st.write(result.means_end_suggestion.rationale)
        if result.means_end_suggestion.symptom_to_ask:
            st.write(f"Suggested next question: **Does the patient have {result.means_end_suggestion.symptom_to_ask}?**")

    with st.expander("🧪 Generate-and-Test Results"):
        st.table(pd.DataFrame(result.generate_and_test_results, columns=["Disease", "Overlap Score"]))

    with st.expander("🎯 Best-First Search Results"):
        st.table(pd.DataFrame(result.best_first_results, columns=["Disease", "Heuristic Score"]))

    with st.expander("⛰ Hill Climbing (confidence-weight optimisation)"):
        st.json(result.hill_climb_weights)

    with st.expander("🌳 Problem Reduction (category breakdown)"):
        for category, diseases in result.problem_reduction.items():
            st.write(f"**{category}**: {', '.join(diseases)}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.selected_patient_id and st.button("💾 Save Diagnosis to Patient History", width='stretch'):
            es.save_visit_and_diagnosis(
                st.session_state.selected_patient_id, st.session_state.reported_symptoms, result
            )
            st.success("Saved to patient history.")
    with col2:
        patients = {p["id"]: p for p in es.get_all_patients()}
        p = patients.get(st.session_state.selected_patient_id, {})
        pdf_bytes = generate_diagnosis_pdf(
            patient_name=p.get("full_name", "Anonymous Patient"),
            patient_age=p.get("age", 0),
            patient_gender=p.get("gender", "unspecified"),
            diagnosis_name=explanation.primary_diagnosis,
            confidence_percent=explanation.confidence_percent,
            matched_symptoms=explanation.matched_symptoms,
            missing_symptoms=explanation.missing_symptoms,
            alternatives=explanation.alternative_diagnoses,
            home_remedies=disease.home_remedies if disease else "",
            lifestyle_advice=disease.lifestyle_advice if disease else "",
            safety_notes=explanation.safety_notes,
            reasoning_summary=explanation.reasoning_summary,
        )
        st.download_button(
            "📄 Export Diagnosis as PDF", data=pdf_bytes,
            file_name=f"diagnosis_{explanation.primary_diagnosis.replace(' ', '_')}.pdf",
            mime="application/pdf", width='stretch',
        )


# ==========================================================================
# PAGE: Reasoning Tree
# ==========================================================================
def render_reasoning_tree() -> None:
    st.title("🌳 Reasoning Tree")
    result = st.session_state.last_result
    if result is None:
        st.info("No diagnosis has been run yet. Go to **Symptom Checker** to run one.")
        return

    st.caption("Symptoms → Rules → Intermediate Facts (Certainty Factors) → Diagnosis")
    fig = plot_reasoning_tree(result.reasoning_graph)
    st.pyplot(fig)

    st.divider()
    st.subheader("Graph statistics")
    col1, col2 = st.columns(2)
    col1.metric("Nodes", result.reasoning_graph.number_of_nodes())
    col2.metric("Edges", result.reasoning_graph.number_of_edges())


# ==========================================================================
# PAGE: Medical History
# ==========================================================================
def render_medical_history() -> None:
    st.title("📚 Medical History")

    patients = es.get_all_patients()
    if not patients:
        st.info("No patients registered yet. Go to **Patient Registration** first.")
        return

    options = {f"#{p['id']} — {p['full_name']}": p["id"] for p in patients}
    choice = st.selectbox("Select patient", list(options.keys()))
    patient_id = options[choice]

    history = es.get_patient_history(patient_id)
    if not history:
        st.info("This patient has no recorded visits yet.")
        return

    for visit in history:
        with st.expander(f"Visit on {visit['visit_date'].strftime('%Y-%m-%d %H:%M')}"):
            st.write(f"**Reported symptoms:** {visit['reported_symptoms']}")
            if visit["notes"]:
                st.write(f"**Notes:** {visit['notes']}")
            for diag in visit["diagnoses"]:
                st.markdown(f"**Diagnosis:** {diag['disease_name']} ({diag['confidence'] * 100:.0f}%)")
                if diag["alternative_diagnoses"]:
                    st.caption(f"Alternatives: {diag['alternative_diagnoses']}")
                st.text(diag["explanation"])

    st.divider()
    st.subheader("Add a past medical condition")
    with st.form("add_history_form", clear_on_submit=True):
        condition = st.text_input("Condition name")
        notes = st.text_area("Notes")
        if st.form_submit_button("Add to History"):
            if condition.strip():
                es.add_medical_history(patient_id, condition.strip(), notes)
                st.success("Added to medical history.")
                st.rerun()
            else:
                st.error("Condition name is required.")


# ==========================================================================
# PAGE: Knowledge Base
# ==========================================================================
def render_knowledge_base() -> None:
    st.title("📖 Knowledge Base")
    tab_diseases, tab_symptoms, tab_medicines, tab_rules = st.tabs(
        ["Diseases", "Symptoms", "Medicines", "Rules"]
    )

    with tab_diseases:
        search = st.text_input("Search diseases", placeholder="e.g. Flu, Respiratory, ...")
        rows = []
        for name in es.kb.all_disease_names():
            d = es.kb.get_disease(name)
            if search and search.lower() not in name.lower() and search.lower() not in d.category.lower():
                continue
            rows.append(
                {
                    "Disease": d.name,
                    "Category": d.category,
                    "Severity": d.severity_level,
                    "Emergency": "Yes" if d.is_emergency else "No",
                    "Required Symptoms": ", ".join(d.required_symptoms),
                    "Medicines": ", ".join(d.medicines) or "—",
                }
            )
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        st.divider()
        st.subheader("Disease Detail")
        detail_choice = st.selectbox("Choose a disease to inspect", es.kb.all_disease_names())
        d = es.kb.get_disease(detail_choice)
        st.markdown(f"### {d.name}")
        st.write(d.description)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Category:** {d.category} ({d.parent_category})")
            st.write(f"**Severity:** {d.severity_level}")
            st.write(f"**Emergency:** {'Yes' if d.is_emergency else 'No'}")
            st.write(f"**All Symptoms:** {', '.join(d.all_symptoms)}")
        with col2:
            st.write(f"**Home Remedies:** {d.home_remedies}")
            st.write(f"**Lifestyle Advice:** {d.lifestyle_advice}")
            st.write(f"**Medicines:** {', '.join(d.medicines) or 'None catalogued'}")

    with tab_symptoms:
        rows = [{"Symptom": s.name, "Category": s.category} for s in es.kb.symptoms.values()]
        df = pd.DataFrame(rows).sort_values("Category")
        st.dataframe(df, width='stretch', hide_index=True)

    with tab_medicines:
        rows = [
            {
                "Medicine": m.name,
                "Dosage": m.dosage_info,
                "OTC": "Yes" if m.is_otc else "Prescription",
                "Contraindications": ", ".join(m.contraindications) or "None listed",
            }
            for m in es.kb.medicines.values()
        ]
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    with tab_rules:
        rows = [
            {
                "Disease": rule.disease_name,
                "IF (required symptoms)": ", ".join(rule.required_symptoms),
                "Min Matches to Fire": rule.min_matches,
                "Confidence Weight": rule.confidence_weight,
            }
            for rule in es.kb.disease_rules_raw
        ]
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# ==========================================================================
# PAGE: Statistics
# ==========================================================================
def render_statistics() -> None:
    st.title("📊 Statistics Dashboard")
    stats = es.get_diagnosis_statistics()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Diagnoses Run", stats["total_diagnoses"])
    col2.metric("Unique Diseases Diagnosed", len(stats["disease_frequency"]))
    avg_conf = (
        sum(stats["confidence_scores"]) / len(stats["confidence_scores"]) * 100
        if stats["confidence_scores"] else 0.0
    )
    col3.metric("Average Confidence", f"{avg_conf:.0f}%")

    if not stats["disease_frequency"]:
        st.info("No diagnoses recorded yet. Run and save a diagnosis to populate this dashboard.")
        return

    st.divider()
    st.subheader("Disease Diagnosis Frequency")
    st.pyplot(plot_disease_frequency(stats["disease_frequency"]))

    st.subheader("Confidence Score Distribution")
    st.pyplot(plot_confidence_distribution(stats["confidence_scores"]))


# ==========================================================================
# PAGE: About Project
# ==========================================================================
def render_about() -> None:
    st.title("ℹ️ About This Project")
    st.markdown(
        f"""
**{APP_NAME}** (v{APP_VERSION}) is a portfolio-grade demonstration of classical,
*explainable* Artificial Intelligence applied to medical symptom checking —
deliberately built **without** black-box machine learning, so every
conclusion the system reaches can be traced back to an explicit rule,
fact, or fuzzy membership calculation.

### Core AI techniques
- **Knowledge Representation & Predicate Logic** — `predicate_logic.py` implements
  facts, Horn-clause rules, unification, and forward-chaining resolution from scratch.
- **Forward Chaining** — `forward_chaining.py` derives candidate diagnoses data-first from reported symptoms.
- **Backward Chaining** — `backward_chaining.py` proves/disproves a hypothesised disease, asking targeted questions.
- **Constraint Satisfaction** — `constraints.py` validates diagnoses/treatments against age, sex,
  pregnancy, allergy and chronic-condition constraints.
- **Fuzzy Logic** — `fuzzy_logic.py` uses scikit-fuzzy membership functions and a Mamdani
  inference system to reason about symptom severity as a matter of degree.
- **Certainty Factors** — `reasoning.py` combines matched/missing evidence into a MYCIN-style confidence score.
- **Explainable AI** — every diagnosis includes a full "why" trace, not just a label.
- **Search & Optimisation** — Generate-and-Test, Best-First Search, Hill Climbing, Means-End Analysis,
  and Problem Reduction are all implemented as distinct, inspectable engines in `reasoning.py`.

### Technology stack
Python 3.12 · Streamlit · SQLite · SQLAlchemy · Pandas · NumPy · scikit-fuzzy · NetworkX · Matplotlib · ReportLab

### Disclaimer
This system is an educational/portfolio project. It is **not** a certified medical device and must
never be used as a substitute for professional medical advice, diagnosis, or treatment.

### License
MIT License — see `LICENSE`.
"""
    )


# ==========================================================================
# Router
# ==========================================================================
PAGE_RENDERERS = {
    "Home": render_home,
    "Patient Registration": render_patient_registration,
    "Symptom Checker": render_symptom_checker,
    "Diagnosis": render_diagnosis,
    "Reasoning Tree": render_reasoning_tree,
    "Medical History": render_medical_history,
    "Knowledge Base": render_knowledge_base,
    "Statistics": render_statistics,
    "About Project": render_about,
}

PAGE_RENDERERS[page]()
