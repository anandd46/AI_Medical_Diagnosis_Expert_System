"""
sample_data.py
===============
The medical knowledge base "content": symptoms, diseases, medicines and
the rules that connect them. This is intentionally a *data* module (as
opposed to *logic*, which lives in knowledge_base.py / forward_chaining.py
/ backward_chaining.py) so a domain expert could edit it without touching
any reasoning code.

Contents
--------
SYMPTOMS   : 110 symptoms, each tagged with a clinical category.
MEDICINES  : 40+ medicines/remedies with dosage & contraindication notes.
DISEASES   : 50 diseases, each declaring:
                - category / severity / emergency flag
                - required_symptoms  -> defining symptoms (must mostly match)
                - supporting_symptoms -> symptoms that raise confidence
                - medicines -> names referencing MEDICINES
                - home_remedies / lifestyle_advice / doctor_recommended
                - parent_category -> used by problem_reduction search

seed(session_factory) populates medical.db from this data, building the
disease<->symptom and disease<->medicine association tables as well as
a Rule row per disease for the forward/backward chaining engines.

Time Complexity : seed() is O(D * (Sd + Md)) where D = #diseases,
                   Sd/Md = symptoms/medicines per disease (small constants),
                   i.e. effectively O(D).
Space Complexity: O(D + S + M) for the ORM objects created in memory
                   before commit.
"""

from __future__ import annotations

from typing import Dict, List

from config import get_logger
from models import Disease, Medicine, Rule, Symptom

logger = get_logger(__name__)

# ==========================================================================
# 1. SYMPTOMS  (name, category)
# ==========================================================================
SYMPTOMS: List[Dict[str, str]] = [
    # --- General / constitutional ---
    {"name": "Fever", "category": "general"},
    {"name": "High Fever", "category": "general"},
    {"name": "Fatigue", "category": "general"},
    {"name": "Chills", "category": "general"},
    {"name": "Sweating", "category": "general"},
    {"name": "Night Sweats", "category": "general"},
    {"name": "Weight Loss", "category": "general"},
    {"name": "Weight Gain", "category": "general"},
    {"name": "Loss of Appetite", "category": "general"},
    {"name": "Malaise", "category": "general"},
    {"name": "Body Ache", "category": "general"},
    {"name": "Weakness", "category": "general"},
    {"name": "Dehydration", "category": "general"},
    {"name": "Excessive Thirst", "category": "general"},
    {"name": "Excessive Hunger", "category": "general"},
    {"name": "Swollen Lymph Nodes", "category": "general"},
    {"name": "Pale Skin", "category": "general"},
    {"name": "Cold Sweats", "category": "general"},
    {"name": "Delayed Wound Healing", "category": "general"},
    {"name": "Frequent Infections", "category": "general"},

    # --- Respiratory ---
    {"name": "Cough", "category": "respiratory"},
    {"name": "Dry Cough", "category": "respiratory"},
    {"name": "Productive Cough", "category": "respiratory"},
    {"name": "Coughing Blood", "category": "respiratory"},
    {"name": "Shortness of Breath", "category": "respiratory"},
    {"name": "Rapid Breathing", "category": "respiratory"},
    {"name": "Wheezing", "category": "respiratory"},
    {"name": "Chest Tightness", "category": "respiratory"},
    {"name": "Sore Throat", "category": "respiratory"},
    {"name": "Runny Nose", "category": "respiratory"},
    {"name": "Nasal Congestion", "category": "respiratory"},
    {"name": "Sneezing", "category": "respiratory"},
    {"name": "Loss of Smell", "category": "respiratory"},
    {"name": "Loss of Taste", "category": "respiratory"},
    {"name": "Hoarseness", "category": "respiratory"},

    # --- Gastrointestinal ---
    {"name": "Nausea", "category": "gastrointestinal"},
    {"name": "Vomiting", "category": "gastrointestinal"},
    {"name": "Diarrhea", "category": "gastrointestinal"},
    {"name": "Constipation", "category": "gastrointestinal"},
    {"name": "Abdominal Pain", "category": "gastrointestinal"},
    {"name": "Abdominal Cramps", "category": "gastrointestinal"},
    {"name": "Bloating", "category": "gastrointestinal"},
    {"name": "Heartburn", "category": "gastrointestinal"},
    {"name": "Indigestion", "category": "gastrointestinal"},
    {"name": "Blood in Stool", "category": "gastrointestinal"},
    {"name": "Rectal Bleeding", "category": "gastrointestinal"},
    {"name": "Jaundice", "category": "gastrointestinal"},
    {"name": "Yellow Eyes", "category": "gastrointestinal"},
    {"name": "Loss of Bowel Control", "category": "gastrointestinal"},

    # --- Neurological ---
    {"name": "Headache", "category": "neurological"},
    {"name": "Severe Headache", "category": "neurological"},
    {"name": "Dizziness", "category": "neurological"},
    {"name": "Confusion", "category": "neurological"},
    {"name": "Memory Loss", "category": "neurological"},
    {"name": "Seizures", "category": "neurological"},
    {"name": "Numbness", "category": "neurological"},
    {"name": "Tingling Sensation", "category": "neurological"},
    {"name": "Blurred Vision", "category": "neurological"},
    {"name": "Light Sensitivity", "category": "neurological"},
    {"name": "Slurred Speech", "category": "neurological"},
    {"name": "Loss of Balance", "category": "neurological"},
    {"name": "Fainting", "category": "neurological"},
    {"name": "Tremors", "category": "neurological"},
    {"name": "Migraine Aura", "category": "neurological"},
    {"name": "Neck Stiffness", "category": "neurological"},

    # --- Cardiovascular ---
    {"name": "Chest Pain", "category": "cardiovascular"},
    {"name": "Palpitations", "category": "cardiovascular"},
    {"name": "Irregular Heartbeat", "category": "cardiovascular"},
    {"name": "Rapid Heartbeat", "category": "cardiovascular"},
    {"name": "High Blood Pressure", "category": "cardiovascular"},
    {"name": "Low Blood Pressure", "category": "cardiovascular"},
    {"name": "Swelling in Legs", "category": "cardiovascular"},
    {"name": "Cold Extremities", "category": "cardiovascular"},

    # --- Musculoskeletal ---
    {"name": "Joint Pain", "category": "musculoskeletal"},
    {"name": "Muscle Pain", "category": "musculoskeletal"},
    {"name": "Back Pain", "category": "musculoskeletal"},
    {"name": "Joint Swelling", "category": "musculoskeletal"},
    {"name": "Joint Stiffness", "category": "musculoskeletal"},
    {"name": "Muscle Cramps", "category": "musculoskeletal"},
    {"name": "Muscle Weakness", "category": "musculoskeletal"},

    # --- Dermatological ---
    {"name": "Skin Rash", "category": "dermatological"},
    {"name": "Itching", "category": "dermatological"},
    {"name": "Red Spots", "category": "dermatological"},
    {"name": "Blisters", "category": "dermatological"},
    {"name": "Dry Skin", "category": "dermatological"},
    {"name": "Skin Discoloration", "category": "dermatological"},
    {"name": "Hives", "category": "dermatological"},
    {"name": "Bruising Easily", "category": "dermatological"},
    {"name": "Bleeding Gums", "category": "dermatological"},
    {"name": "Nosebleeds", "category": "dermatological"},

    # --- ENT / Eyes ---
    {"name": "Ear Pain", "category": "ent"},
    {"name": "Ear Discharge", "category": "ent"},
    {"name": "Hearing Loss", "category": "ent"},
    {"name": "Sinus Pressure", "category": "ent"},
    {"name": "Difficulty Swallowing", "category": "ent"},
    {"name": "Eye Redness", "category": "ent"},
    {"name": "Eye Pain", "category": "ent"},
    {"name": "Watery Eyes", "category": "ent"},

    # --- Urinary / Renal ---
    {"name": "Frequent Urination", "category": "urinary"},
    {"name": "Painful Urination", "category": "urinary"},
    {"name": "Blood in Urine", "category": "urinary"},
    {"name": "Low Urine Output", "category": "urinary"},
    {"name": "Flank Pain", "category": "urinary"},
    {"name": "Cloudy Urine", "category": "urinary"},

    # --- Psychological ---
    {"name": "Anxiety", "category": "psychological"},
    {"name": "Irritability", "category": "psychological"},
    {"name": "Insomnia", "category": "psychological"},
    {"name": "Low Mood", "category": "psychological"},
    {"name": "Difficulty Concentrating", "category": "psychological"},
    {"name": "Restlessness", "category": "psychological"},

    # --- Other / Endocrine ---
    {"name": "Enlarged Spleen", "category": "other"},
    {"name": "Goiter", "category": "endocrine"},
    {"name": "Heat Intolerance", "category": "endocrine"},
    {"name": "Cold Intolerance", "category": "endocrine"},
    {"name": "Hair Loss", "category": "endocrine"},
]

# ==========================================================================
# 2. MEDICINES  (name, dosage_info, contraindications, is_otc)
# ==========================================================================
MEDICINES: List[Dict] = [
    {"name": "Paracetamol", "dosage_info": "500-1000mg every 6-8h, max 3g/day",
     "contraindications": "liver_disease", "is_otc": True},
    {"name": "Ibuprofen", "dosage_info": "200-400mg every 6-8h with food",
     "contraindications": "pregnancy,peptic_ulcer,kidney_disease", "is_otc": True},
    {"name": "Aspirin", "dosage_info": "300-600mg every 4-6h",
     "contraindications": "child,pregnancy,peptic_ulcer,bleeding_disorder", "is_otc": True},
    {"name": "Oral Rehydration Salts (ORS)", "dosage_info": "1 sachet per litre of water, sip frequently",
     "contraindications": "", "is_otc": True},
    {"name": "Loperamide", "dosage_info": "4mg initially, then 2mg after each loose stool, max 16mg/day",
     "contraindications": "child,blood_in_stool", "is_otc": True},
    {"name": "Omeprazole", "dosage_info": "20mg once daily before food",
     "contraindications": "", "is_otc": True},
    {"name": "Antacid (Aluminium/Magnesium Hydroxide)", "dosage_info": "10-20ml after meals",
     "contraindications": "kidney_disease", "is_otc": True},
    {"name": "Cetirizine", "dosage_info": "10mg once daily",
     "contraindications": "", "is_otc": True},
    {"name": "Loratadine", "dosage_info": "10mg once daily",
     "contraindications": "", "is_otc": True},
    {"name": "Salbutamol Inhaler", "dosage_info": "2 puffs as needed, max 8 puffs/day",
     "contraindications": "", "is_otc": False},
    {"name": "Montelukast", "dosage_info": "10mg once daily at night",
     "contraindications": "", "is_otc": False},
    {"name": "Amoxicillin", "dosage_info": "500mg every 8h for 5-7 days",
     "contraindications": "penicillin_allergy", "is_otc": False},
    {"name": "Azithromycin", "dosage_info": "500mg day 1, then 250mg for 4 days",
     "contraindications": "liver_disease", "is_otc": False},
    {"name": "Doxycycline", "dosage_info": "100mg twice daily for 7 days",
     "contraindications": "pregnancy,child", "is_otc": False},
    {"name": "Ciprofloxacin", "dosage_info": "500mg twice daily for 5-7 days",
     "contraindications": "pregnancy,child", "is_otc": False},
    {"name": "Metronidazole", "dosage_info": "400mg three times daily for 5-7 days",
     "contraindications": "pregnancy_first_trimester", "is_otc": False},
    {"name": "Rifampicin+Isoniazid+Pyrazinamide+Ethambutol (RIPE)", "dosage_info": "As per DOTS protocol, 6 months",
     "contraindications": "liver_disease", "is_otc": False},
    {"name": "Artemisinin Combination Therapy (ACT)", "dosage_info": "As per weight-based protocol, 3 days",
     "contraindications": "pregnancy_first_trimester", "is_otc": False},
    {"name": "Chloroquine", "dosage_info": "As per weight-based protocol",
     "contraindications": "epilepsy", "is_otc": False},
    {"name": "Metformin", "dosage_info": "500mg twice daily with meals",
     "contraindications": "kidney_disease", "is_otc": False},
    {"name": "Insulin", "dosage_info": "As prescribed based on blood glucose monitoring",
     "contraindications": "", "is_otc": False},
    {"name": "Amlodipine", "dosage_info": "5mg once daily",
     "contraindications": "pregnancy", "is_otc": False},
    {"name": "Losartan", "dosage_info": "50mg once daily",
     "contraindications": "pregnancy,kidney_disease", "is_otc": False},
    {"name": "Atorvastatin", "dosage_info": "10-20mg once daily at night",
     "contraindications": "liver_disease,pregnancy", "is_otc": False},
    {"name": "Sumatriptan", "dosage_info": "50-100mg at onset of migraine",
     "contraindications": "heart_disease,pregnancy", "is_otc": False},
    {"name": "Ondansetron", "dosage_info": "4-8mg every 8h as needed",
     "contraindications": "", "is_otc": False},
    {"name": "Iron + Folic Acid Supplement", "dosage_info": "1 tablet daily with vitamin C",
     "contraindications": "", "is_otc": True},
    {"name": "Vitamin B12 Injection", "dosage_info": "1000mcg intramuscular, per schedule",
     "contraindications": "", "is_otc": False},
    {"name": "Calcium Channel Blocker (Diltiazem)", "dosage_info": "As prescribed",
     "contraindications": "heart_block", "is_otc": False},
    {"name": "Nitroglycerin", "dosage_info": "0.4mg sublingual as needed for chest pain",
     "contraindications": "low_blood_pressure", "is_otc": False},
    {"name": "Levothyroxine", "dosage_info": "1.6mcg/kg once daily on empty stomach",
     "contraindications": "", "is_otc": False},
    {"name": "Methimazole", "dosage_info": "As prescribed based on thyroid levels",
     "contraindications": "pregnancy_first_trimester", "is_otc": False},
    {"name": "Hydrocortisone Cream", "dosage_info": "Apply thin layer twice daily",
     "contraindications": "skin_infection", "is_otc": True},
    {"name": "Permethrin Cream", "dosage_info": "Apply once, wash off after 8-14h",
     "contraindications": "infant_under_2_months", "is_otc": False},
    {"name": "Clotrimazole Cream", "dosage_info": "Apply twice daily for 2-4 weeks",
     "contraindications": "", "is_otc": True},
    {"name": "Antihistamine Eye Drops", "dosage_info": "1 drop each eye twice daily",
     "contraindications": "", "is_otc": True},
    {"name": "Pain Relief Gel (Diclofenac Topical)", "dosage_info": "Apply 2-4g up to 4 times daily",
     "contraindications": "pregnancy", "is_otc": True},
    {"name": "Methotrexate", "dosage_info": "As prescribed weekly, with folic acid",
     "contraindications": "pregnancy,liver_disease", "is_otc": False},
    {"name": "Diazepam", "dosage_info": "As prescribed for acute seizures/anxiety",
     "contraindications": "pregnancy,respiratory_disease", "is_otc": False},
    {"name": "Sertraline", "dosage_info": "50mg once daily, titrate per response",
     "contraindications": "child_under_6", "is_otc": False},
    {"name": "Betahistine", "dosage_info": "16mg three times daily",
     "contraindications": "", "is_otc": False},
    {"name": "Cephalexin", "dosage_info": "500mg every 6h for 7-10 days",
     "contraindications": "penicillin_allergy", "is_otc": False},
]

logger.info("Loaded %d symptoms and %d medicines into sample_data module.", len(SYMPTOMS), len(MEDICINES))

# ==========================================================================
# 3. DISEASES
# ==========================================================================
# Each entry:
#   name, category, parent_category (problem-reduction hierarchy),
#   severity_level (mild/moderate/high/critical), is_emergency,
#   required_symptoms (defining symptoms - core rule conditions),
#   supporting_symptoms (raise confidence but not mandatory),
#   medicines (names from MEDICINES), home_remedies, lifestyle_advice,
#   doctor_recommended, description
DISEASES: List[Dict] = [
    {
        "name": "Common Cold", "category": "Respiratory", "parent_category": "Respiratory > Viral",
        "severity_level": "mild", "is_emergency": False,
        "required_symptoms": ["Runny Nose", "Sneezing", "Sore Throat"],
        "supporting_symptoms": ["Nasal Congestion", "Cough", "Headache", "Fatigue"],
        "medicines": ["Paracetamol", "Cetirizine"],
        "home_remedies": "Warm fluids, steam inhalation, honey with warm water, adequate rest.",
        "lifestyle_advice": "Stay hydrated, avoid cold drinks, rest for 2-3 days.",
        "doctor_recommended": False,
        "description": "A mild viral infection of the upper respiratory tract.",
    },
    {
        "name": "Flu", "category": "Respiratory", "parent_category": "Respiratory > Viral",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Fever", "Body Ache", "Fatigue", "Dry Cough"],
        "supporting_symptoms": ["Chills", "Headache", "Sore Throat", "Nasal Congestion"],
        "medicines": ["Paracetamol", "Ibuprofen"],
        "home_remedies": "Rest, warm fluids, steam inhalation, ginger tea.",
        "lifestyle_advice": "Isolate to avoid spreading, sleep 8+ hours, stay hydrated.",
        "doctor_recommended": True,
        "description": "Influenza - an acute viral respiratory illness.",
    },
    {
        "name": "COVID-19", "category": "Respiratory", "parent_category": "Respiratory > Viral",
        "severity_level": "high", "is_emergency": False,
        "required_symptoms": ["Fever", "Dry Cough", "Loss of Smell", "Loss of Taste"],
        "supporting_symptoms": ["Fatigue", "Shortness of Breath", "Body Ache", "Sore Throat"],
        "medicines": ["Paracetamol"],
        "home_remedies": "Isolation, steam inhalation, warm fluids, pulse-oximeter monitoring.",
        "lifestyle_advice": "Isolate immediately, monitor oxygen saturation, seek care if breathless.",
        "doctor_recommended": True,
        "description": "An infectious respiratory disease caused by the SARS-CoV-2 virus.",
    },
    {
        "name": "Pneumonia", "category": "Respiratory", "parent_category": "Respiratory > Bacterial",
        "severity_level": "high", "is_emergency": True,
        "required_symptoms": ["Fever", "Productive Cough", "Chest Pain", "Shortness of Breath"],
        "supporting_symptoms": ["Rapid Breathing", "Chills", "Fatigue", "Coughing Blood"],
        "medicines": ["Amoxicillin", "Azithromycin"],
        "home_remedies": "Warm fluids, steam inhalation (supportive only - not a substitute for antibiotics).",
        "lifestyle_advice": "Seek urgent medical care; hospitalization may be required for severe cases.",
        "doctor_recommended": True,
        "description": "Infection that inflames air sacs in one or both lungs, which may fill with fluid.",
    },
    {
        "name": "Bronchitis", "category": "Respiratory", "parent_category": "Respiratory > Inflammatory",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Productive Cough", "Chest Tightness", "Wheezing"],
        "supporting_symptoms": ["Fatigue", "Fever", "Shortness of Breath"],
        "medicines": ["Salbutamol Inhaler", "Paracetamol"],
        "home_remedies": "Steam inhalation, honey and warm water, avoid smoke exposure.",
        "lifestyle_advice": "Avoid smoking/pollutants, use a humidifier, rest.",
        "doctor_recommended": True,
        "description": "Inflammation of the lining of the bronchial tubes.",
    },
    {
        "name": "Asthma", "category": "Respiratory", "parent_category": "Respiratory > Chronic",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Wheezing", "Shortness of Breath", "Chest Tightness"],
        "supporting_symptoms": ["Cough", "Rapid Breathing", "Insomnia"],
        "medicines": ["Salbutamol Inhaler", "Montelukast"],
        "home_remedies": "Avoid known triggers (dust, smoke, pollen), breathing exercises.",
        "lifestyle_advice": "Carry a rescue inhaler, identify and avoid personal triggers.",
        "doctor_recommended": True,
        "description": "A chronic condition causing airway inflammation and narrowing.",
    },
    {
        "name": "Tuberculosis", "category": "Respiratory", "parent_category": "Respiratory > Bacterial",
        "severity_level": "critical", "is_emergency": False,
        "required_symptoms": ["Coughing Blood", "Night Sweats", "Weight Loss", "Productive Cough"],
        "supporting_symptoms": ["Fever", "Fatigue", "Loss of Appetite", "Chest Pain"],
        "medicines": ["Rifampicin+Isoniazid+Pyrazinamide+Ethambutol (RIPE)"],
        "home_remedies": "None - requires supervised medical treatment (DOTS therapy).",
        "lifestyle_advice": "Complete the full 6-month course, maintain nutrition, avoid alcohol.",
        "doctor_recommended": True,
        "description": "A serious bacterial infection (Mycobacterium tuberculosis) mainly affecting the lungs.",
    },
    {
        "name": "Sinusitis", "category": "Respiratory", "parent_category": "Respiratory > ENT",
        "severity_level": "mild", "is_emergency": False,
        "required_symptoms": ["Sinus Pressure", "Nasal Congestion", "Headache"],
        "supporting_symptoms": ["Runny Nose", "Loss of Smell", "Sore Throat"],
        "medicines": ["Paracetamol", "Cetirizine"],
        "home_remedies": "Steam inhalation, saline nasal rinse, warm compress on face.",
        "lifestyle_advice": "Stay hydrated, avoid allergens, sleep with head elevated.",
        "doctor_recommended": False,
        "description": "Inflammation of the sinus cavities, often following a cold or allergy.",
    },
    {
        "name": "Strep Throat", "category": "Respiratory", "parent_category": "Respiratory > Bacterial",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Sore Throat", "Fever", "Difficulty Swallowing"],
        "supporting_symptoms": ["Swollen Lymph Nodes", "Headache", "Body Ache"],
        "medicines": ["Amoxicillin", "Paracetamol"],
        "home_remedies": "Warm salt-water gargles, honey and warm water.",
        "lifestyle_advice": "Complete full antibiotic course even if symptoms improve early.",
        "doctor_recommended": True,
        "description": "A bacterial infection (Streptococcus) causing inflammation of the throat.",
    },
    {
        "name": "Allergic Rhinitis", "category": "Respiratory", "parent_category": "Respiratory > Allergic",
        "severity_level": "mild", "is_emergency": False,
        "required_symptoms": ["Sneezing", "Runny Nose", "Itching", "Watery Eyes"],
        "supporting_symptoms": ["Nasal Congestion", "Eye Redness"],
        "medicines": ["Cetirizine", "Loratadine", "Antihistamine Eye Drops"],
        "home_remedies": "Avoid allergens (dust, pollen, pet dander), saline rinse.",
        "lifestyle_advice": "Use air purifiers, wash bedding frequently, identify triggers.",
        "doctor_recommended": False,
        "description": "An allergic response of the nasal passages to airborne allergens.",
    },
    {
        "name": "Typhoid", "category": "Infectious", "parent_category": "Infectious > Bacterial",
        "severity_level": "high", "is_emergency": False,
        "required_symptoms": ["High Fever", "Abdominal Pain", "Loss of Appetite", "Weakness"],
        "supporting_symptoms": ["Headache", "Constipation", "Diarrhea", "Weakness"],
        "medicines": ["Ciprofloxacin", "Azithromycin"],
        "home_remedies": "Plenty of fluids, light easily-digestible food (khichdi, soup).",
        "lifestyle_advice": "Complete antibiotic course, maintain strict food and water hygiene.",
        "doctor_recommended": True,
        "description": "A bacterial infection (Salmonella typhi) spread through contaminated food/water.",
    },
    {
        "name": "Malaria", "category": "Infectious", "parent_category": "Infectious > Parasitic",
        "severity_level": "high", "is_emergency": False,
        "required_symptoms": ["High Fever", "Chills", "Sweating", "Headache"],
        "supporting_symptoms": ["Body Ache", "Nausea", "Vomiting", "Fatigue"],
        "medicines": ["Artemisinin Combination Therapy (ACT)", "Chloroquine"],
        "home_remedies": "Rest, fluids - not a substitute for antimalarial treatment.",
        "lifestyle_advice": "Use mosquito nets/repellent, complete full antimalarial course.",
        "doctor_recommended": True,
        "description": "A mosquito-borne parasitic disease (Plasmodium species) causing cyclical fevers.",
    },
    {
        "name": "Dengue", "category": "Infectious", "parent_category": "Infectious > Viral",
        "severity_level": "high", "is_emergency": True,
        "required_symptoms": ["High Fever", "Severe Headache", "Joint Pain", "Skin Rash"],
        "supporting_symptoms": ["Nausea", "Vomiting", "Bruising Easily", "Nosebleeds", "Eye Pain"],
        "medicines": ["Paracetamol"],
        "home_remedies": "Fluids, papaya leaf extract (traditional, unproven), rest.",
        "lifestyle_advice": "Avoid Ibuprofen/Aspirin (bleeding risk), monitor platelet count, seek care if bleeding.",
        "doctor_recommended": True,
        "description": "A mosquito-borne viral infection that can progress to severe bleeding complications.",
    },
    {
        "name": "Chickenpox", "category": "Infectious", "parent_category": "Infectious > Viral",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Skin Rash", "Itching", "Blisters", "Fever"],
        "supporting_symptoms": ["Fatigue", "Loss of Appetite", "Headache"],
        "medicines": ["Paracetamol", "Hydrocortisone Cream"],
        "home_remedies": "Oatmeal baths, calamine lotion, keep nails trimmed to avoid scratching.",
        "lifestyle_advice": "Isolate until all blisters crust over, avoid Aspirin in children.",
        "doctor_recommended": True,
        "description": "A highly contagious viral infection (Varicella-zoster) causing an itchy rash.",
    },
    {
        "name": "Measles", "category": "Infectious", "parent_category": "Infectious > Viral",
        "severity_level": "high", "is_emergency": False,
        "required_symptoms": ["Fever", "Skin Rash", "Runny Nose", "Eye Redness"],
        "supporting_symptoms": ["Cough", "Light Sensitivity", "Swollen Lymph Nodes"],
        "medicines": ["Paracetamol"],
        "home_remedies": "Rest, fluids, vitamin A rich foods.",
        "lifestyle_advice": "Isolate for at least 4 days after rash onset; ensure vaccination of contacts.",
        "doctor_recommended": True,
        "description": "A highly contagious viral infection with a characteristic spreading rash.",
    },
    {
        "name": "Mumps", "category": "Infectious", "parent_category": "Infectious > Viral",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Swollen Lymph Nodes", "Fever", "Difficulty Swallowing"],
        "supporting_symptoms": ["Headache", "Muscle Pain", "Loss of Appetite"],
        "medicines": ["Paracetamol", "Ibuprofen"],
        "home_remedies": "Warm or cold compress on swollen glands, soft foods.",
        "lifestyle_advice": "Isolate for 5 days after gland swelling begins.",
        "doctor_recommended": True,
        "description": "A contagious viral infection primarily affecting the salivary glands.",
    },
    {
        "name": "Whooping Cough", "category": "Infectious", "parent_category": "Respiratory > Bacterial",
        "severity_level": "high", "is_emergency": False,
        "required_symptoms": ["Cough", "Rapid Breathing", "Vomiting"],
        "supporting_symptoms": ["Fever", "Runny Nose", "Fatigue"],
        "medicines": ["Azithromycin"],
        "home_remedies": "Small frequent meals, humidified air, plenty of rest.",
        "lifestyle_advice": "Isolate young children, ensure vaccination (DTaP) is up to date.",
        "doctor_recommended": True,
        "description": "A highly contagious bacterial respiratory infection (Bordetella pertussis).",
    },
    {
        "name": "Hepatitis A", "category": "Infectious", "parent_category": "Infectious > Viral",
        "severity_level": "high", "is_emergency": False,
        "required_symptoms": ["Jaundice", "Yellow Eyes", "Nausea", "Loss of Appetite"],
        "supporting_symptoms": ["Fatigue", "Abdominal Pain", "Fever"],
        "medicines": [],
        "home_remedies": "Rest, hydration, avoid alcohol and fatty foods.",
        "lifestyle_advice": "Strict hand hygiene, avoid contaminated water, rest until liver enzymes normalize.",
        "doctor_recommended": True,
        "description": "A liver infection caused by the hepatitis A virus, usually spread via contaminated food/water.",
    },
    {
        "name": "Food Poisoning", "category": "Gastrointestinal", "parent_category": "Gastrointestinal > Infectious",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Nausea", "Vomiting", "Diarrhea", "Abdominal Cramps"],
        "supporting_symptoms": ["Fever", "Weakness", "Dehydration"],
        "medicines": ["Oral Rehydration Salts (ORS)", "Loperamide", "Ondansetron"],
        "home_remedies": "ORS, bland diet (BRAT: banana, rice, applesauce, toast), rest.",
        "lifestyle_advice": "Practice food safety, wash hands, avoid undercooked food.",
        "doctor_recommended": False,
        "description": "Illness caused by consuming contaminated food, typically self-limiting.",
    },
    {
        "name": "Gastroenteritis", "category": "Gastrointestinal", "parent_category": "Gastrointestinal > Infectious",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Diarrhea", "Vomiting", "Abdominal Cramps", "Fever"],
        "supporting_symptoms": ["Nausea", "Dehydration", "Headache"],
        "medicines": ["Oral Rehydration Salts (ORS)", "Loperamide"],
        "home_remedies": "Fluids, light diet, probiotics (yogurt).",
        "lifestyle_advice": "Maintain hydration, wash hands frequently, isolate if viral.",
        "doctor_recommended": False,
        "description": "Inflammation of the stomach and intestines, usually viral or bacterial in origin.",
    },
    {
        "name": "Gastritis", "category": "Gastrointestinal", "parent_category": "Gastrointestinal > Inflammatory",
        "severity_level": "mild", "is_emergency": False,
        "required_symptoms": ["Abdominal Pain", "Nausea", "Bloating", "Indigestion"],
        "supporting_symptoms": ["Loss of Appetite", "Vomiting", "Heartburn"],
        "medicines": ["Omeprazole", "Antacid (Aluminium/Magnesium Hydroxide)"],
        "home_remedies": "Small frequent meals, avoid spicy/oily food, ginger tea.",
        "lifestyle_advice": "Avoid NSAIDs, alcohol, and smoking; manage stress.",
        "doctor_recommended": False,
        "description": "Inflammation of the stomach lining, often due to diet, stress, or infection.",
    },
    {
        "name": "Peptic Ulcer", "category": "Gastrointestinal", "parent_category": "Gastrointestinal > Inflammatory",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Abdominal Pain", "Heartburn", "Indigestion", "Blood in Stool"],
        "supporting_symptoms": ["Nausea", "Loss of Appetite", "Bloating"],
        "medicines": ["Omeprazole", "Antacid (Aluminium/Magnesium Hydroxide)"],
        "home_remedies": "Avoid spicy/acidic food, smaller frequent meals.",
        "lifestyle_advice": "Avoid NSAIDs/alcohol/smoking, manage H. pylori infection medically.",
        "doctor_recommended": True,
        "description": "An open sore in the stomach or duodenum lining, often linked to H. pylori or NSAID use.",
    },
    {
        "name": "Appendicitis", "category": "Gastrointestinal", "parent_category": "Gastrointestinal > Surgical",
        "severity_level": "critical", "is_emergency": True,
        "required_symptoms": ["Abdominal Pain", "Nausea", "Vomiting", "Fever"],
        "supporting_symptoms": ["Loss of Appetite", "Bloating", "Constipation"],
        "medicines": [],
        "home_remedies": "None - this is a surgical emergency, do not delay care.",
        "lifestyle_advice": "Seek emergency care immediately; do not take painkillers that mask symptoms.",
        "doctor_recommended": True,
        "description": "Inflammation of the appendix, a medical emergency usually requiring surgery.",
    },
    {
        "name": "Gallstones", "category": "Gastrointestinal", "parent_category": "Gastrointestinal > Biliary",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Abdominal Pain", "Nausea", "Bloating", "Jaundice"],
        "supporting_symptoms": ["Vomiting", "Indigestion", "Fever"],
        "medicines": [],
        "home_remedies": "Low-fat diet, adequate hydration.",
        "lifestyle_advice": "Maintain healthy weight, low-fat diet; surgery may be needed if recurrent.",
        "doctor_recommended": True,
        "description": "Hardened deposits in the gallbladder that can block bile flow and cause pain.",
    },
    {
        "name": "Pancreatitis", "category": "Gastrointestinal", "parent_category": "Gastrointestinal > Inflammatory",
        "severity_level": "critical", "is_emergency": True,
        "required_symptoms": ["Abdominal Pain", "Nausea", "Vomiting", "Fever"],
        "supporting_symptoms": ["Rapid Heartbeat", "Bloating", "Weakness"],
        "medicines": [],
        "home_remedies": "None - requires hospitalization and IV fluids.",
        "lifestyle_advice": "Avoid alcohol entirely, seek emergency care for severe abdominal pain.",
        "doctor_recommended": True,
        "description": "Inflammation of the pancreas, often linked to gallstones or alcohol use.",
    },
    {
        "name": "Diabetes", "category": "Endocrine", "parent_category": "Endocrine > Metabolic",
        "severity_level": "high", "is_emergency": False,
        "required_symptoms": ["Excessive Thirst", "Frequent Urination", "Excessive Hunger", "Fatigue"],
        "supporting_symptoms": ["Weight Loss", "Blurred Vision", "Delayed Wound Healing", "Frequent Infections"],
        "medicines": ["Metformin", "Insulin"],
        "home_remedies": "Balanced diet, regular exercise, monitor blood sugar.",
        "lifestyle_advice": "Reduce refined sugar intake, exercise regularly, routine glucose monitoring.",
        "doctor_recommended": True,
        "description": "A chronic metabolic disorder characterized by high blood glucose levels.",
    },
    {
        "name": "Hypertension", "category": "Cardiovascular", "parent_category": "Cardiovascular > Chronic",
        "severity_level": "high", "is_emergency": False,
        "required_symptoms": ["High Blood Pressure", "Headache", "Dizziness"],
        "supporting_symptoms": ["Blurred Vision", "Chest Pain", "Fatigue"],
        "medicines": ["Amlodipine", "Losartan"],
        "home_remedies": "Reduce salt intake, relaxation techniques.",
        "lifestyle_advice": "Regular exercise, reduce sodium/alcohol, manage stress, monitor BP regularly.",
        "doctor_recommended": True,
        "description": "Chronically elevated blood pressure that increases risk of heart disease and stroke.",
    },
    {
        "name": "Heart Disease", "category": "Cardiovascular", "parent_category": "Cardiovascular > Chronic",
        "severity_level": "critical", "is_emergency": True,
        "required_symptoms": ["Chest Pain", "Shortness of Breath", "Palpitations"],
        "supporting_symptoms": ["Fatigue", "Swelling in Legs", "Cold Extremities", "Irregular Heartbeat"],
        "medicines": ["Atorvastatin", "Nitroglycerin"],
        "home_remedies": "None substitute - seek emergency care for chest pain lasting >5 minutes.",
        "lifestyle_advice": "Heart-healthy diet, regular exercise, quit smoking, manage cholesterol.",
        "doctor_recommended": True,
        "description": "A broad term for conditions affecting heart structure and function, including coronary artery disease.",
    },
    {
        "name": "Angina", "category": "Cardiovascular", "parent_category": "Cardiovascular > Chronic",
        "severity_level": "high", "is_emergency": True,
        "required_symptoms": ["Chest Pain", "Shortness of Breath"],
        "supporting_symptoms": ["Fatigue", "Sweating", "Nausea", "Cold Sweats"],
        "medicines": ["Nitroglycerin", "Amlodipine"],
        "home_remedies": "Rest immediately at symptom onset; not a substitute for medical evaluation.",
        "lifestyle_advice": "Avoid strenuous exertion, follow prescribed nitrate therapy, manage risk factors.",
        "doctor_recommended": True,
        "description": "Chest pain caused by reduced blood flow to the heart muscle.",
    },
    {
        "name": "Stroke", "category": "Neurological", "parent_category": "Neurological > Vascular",
        "severity_level": "critical", "is_emergency": True,
        "required_symptoms": ["Slurred Speech", "Numbness", "Confusion", "Loss of Balance"],
        "supporting_symptoms": ["Severe Headache", "Blurred Vision", "Dizziness"],
        "medicines": [],
        "home_remedies": "None - this is a medical emergency, call for emergency help immediately.",
        "lifestyle_advice": "Time-critical emergency: note symptom onset time and get to hospital immediately (F.A.S.T.).",
        "doctor_recommended": True,
        "description": "A sudden interruption of blood supply to the brain; a life-threatening emergency.",
    },
    {
        "name": "Migraine", "category": "Neurological", "parent_category": "Neurological > Functional",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Severe Headache", "Light Sensitivity", "Nausea"],
        "supporting_symptoms": ["Migraine Aura", "Vomiting", "Dizziness"],
        "medicines": ["Sumatriptan", "Ibuprofen"],
        "home_remedies": "Rest in a dark quiet room, cold compress on forehead, hydration.",
        "lifestyle_advice": "Identify and avoid triggers (certain foods, stress, poor sleep), maintain a headache diary.",
        "doctor_recommended": True,
        "description": "A neurological condition causing recurrent, often severe, throbbing headaches.",
    },
    {
        "name": "Epilepsy", "category": "Neurological", "parent_category": "Neurological > Chronic",
        "severity_level": "high", "is_emergency": False,
        "required_symptoms": ["Seizures", "Confusion", "Loss of Balance"],
        "supporting_symptoms": ["Memory Loss", "Fatigue", "Tingling Sensation"],
        "medicines": ["Diazepam"],
        "home_remedies": "Ensure a safe environment during seizures; not a substitute for anti-epileptic medication.",
        "lifestyle_advice": "Maintain medication adherence, avoid known triggers (sleep deprivation, flashing lights).",
        "doctor_recommended": True,
        "description": "A chronic neurological disorder characterized by recurrent unprovoked seizures.",
    },
    {
        "name": "Vertigo (BPPV)", "category": "Neurological", "parent_category": "Neurological > Vestibular",
        "severity_level": "mild", "is_emergency": False,
        "required_symptoms": ["Dizziness", "Loss of Balance", "Nausea"],
        "supporting_symptoms": ["Vomiting", "Blurred Vision", "Fainting"],
        "medicines": ["Betahistine"],
        "home_remedies": "Epley maneuver (guided by a professional), avoid sudden head movements.",
        "lifestyle_advice": "Move slowly when changing position, avoid triggers like lying flat quickly.",
        "doctor_recommended": True,
        "description": "Benign Paroxysmal Positional Vertigo - brief episodes of dizziness triggered by head position changes.",
    },
    {
        "name": "Anemia", "category": "Hematological", "parent_category": "Hematological > Nutritional",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Fatigue", "Pale Skin", "Weakness", "Shortness of Breath"],
        "supporting_symptoms": ["Dizziness", "Cold Extremities", "Headache"],
        "medicines": ["Iron + Folic Acid Supplement", "Vitamin B12 Injection"],
        "home_remedies": "Iron-rich diet (leafy greens, red meat, legumes), vitamin C to aid absorption.",
        "lifestyle_advice": "Address underlying cause (diet, blood loss), routine hemoglobin checks.",
        "doctor_recommended": True,
        "description": "A condition marked by a deficiency of red blood cells or hemoglobin.",
    },
    {
        "name": "Kidney Stones", "category": "Urinary", "parent_category": "Urinary > Structural",
        "severity_level": "high", "is_emergency": False,
        "required_symptoms": ["Flank Pain", "Blood in Urine", "Painful Urination"],
        "supporting_symptoms": ["Nausea", "Vomiting", "Frequent Urination", "Cloudy Urine"],
        "medicines": ["Ibuprofen"],
        "home_remedies": "Increase water intake, warm compress on flank.",
        "lifestyle_advice": "Stay well hydrated, reduce salt and oxalate-rich foods, avoid excess animal protein.",
        "doctor_recommended": True,
        "description": "Hard mineral deposits that form in the kidneys, causing severe pain when passing.",
    },
    {
        "name": "Urinary Tract Infection", "category": "Urinary", "parent_category": "Urinary > Infectious",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Painful Urination", "Frequent Urination", "Cloudy Urine"],
        "supporting_symptoms": ["Flank Pain", "Fever", "Blood in Urine"],
        "medicines": ["Ciprofloxacin", "Cephalexin"],
        "home_remedies": "Increase fluid intake, cranberry juice (supportive, unproven cure).",
        "lifestyle_advice": "Maintain hygiene, urinate after intercourse, avoid holding urine for long periods.",
        "doctor_recommended": True,
        "description": "A bacterial infection anywhere in the urinary system, most often the bladder.",
    },
    {
        "name": "Chronic Kidney Disease", "category": "Urinary", "parent_category": "Urinary > Chronic",
        "severity_level": "critical", "is_emergency": False,
        "required_symptoms": ["Fatigue", "Swelling in Legs", "Low Urine Output"],
        "supporting_symptoms": ["Loss of Appetite", "Nausea", "High Blood Pressure", "Confusion"],
        "medicines": ["Losartan"],
        "home_remedies": "Low-sodium, low-protein diet as advised by a nephrologist.",
        "lifestyle_advice": "Regular kidney function monitoring, control blood pressure and diabetes strictly.",
        "doctor_recommended": True,
        "description": "Progressive loss of kidney function over time, often linked to diabetes/hypertension.",
    },
    {
        "name": "Hyperthyroidism", "category": "Endocrine", "parent_category": "Endocrine > Thyroid",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Weight Loss", "Rapid Heartbeat", "Heat Intolerance", "Tremors"],
        "supporting_symptoms": ["Anxiety", "Sweating", "Insomnia", "Goiter"],
        "medicines": ["Methimazole"],
        "home_remedies": "Avoid iodine-rich foods until evaluated, stress reduction.",
        "lifestyle_advice": "Regular thyroid function tests, avoid stimulants like caffeine.",
        "doctor_recommended": True,
        "description": "Overactive thyroid gland producing excess thyroid hormone.",
    },
    {
        "name": "Hypothyroidism", "category": "Endocrine", "parent_category": "Endocrine > Thyroid",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Weight Gain", "Cold Intolerance", "Fatigue", "Hair Loss"],
        "supporting_symptoms": ["Constipation", "Dry Skin", "Low Mood", "Muscle Weakness"],
        "medicines": ["Levothyroxine"],
        "home_remedies": "Iodine-sufficient diet (as advised), regular exercise.",
        "lifestyle_advice": "Lifelong medication adherence, routine TSH monitoring.",
        "doctor_recommended": True,
        "description": "Underactive thyroid gland producing insufficient thyroid hormone.",
    },
    {
        "name": "Osteoarthritis", "category": "Musculoskeletal", "parent_category": "Musculoskeletal > Degenerative",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Joint Pain", "Joint Stiffness", "Joint Swelling"],
        "supporting_symptoms": ["Muscle Weakness", "Back Pain"],
        "medicines": ["Ibuprofen", "Pain Relief Gel (Diclofenac Topical)"],
        "home_remedies": "Warm compress, gentle stretching, weight management.",
        "lifestyle_advice": "Low-impact exercise (swimming, walking), maintain healthy weight.",
        "doctor_recommended": True,
        "description": "Degeneration of joint cartilage causing pain and stiffness, common with aging.",
    },
    {
        "name": "Rheumatoid Arthritis", "category": "Musculoskeletal", "parent_category": "Musculoskeletal > Autoimmune",
        "severity_level": "high", "is_emergency": False,
        "required_symptoms": ["Joint Pain", "Joint Swelling", "Joint Stiffness", "Fatigue"],
        "supporting_symptoms": ["Low Mood", "Muscle Weakness", "Fever"],
        "medicines": ["Methotrexate", "Ibuprofen"],
        "home_remedies": "Warm/cold therapy, gentle range-of-motion exercises.",
        "lifestyle_advice": "Early diagnosis and treatment critical to prevent joint damage; avoid smoking.",
        "doctor_recommended": True,
        "description": "An autoimmune disorder causing chronic joint inflammation and damage.",
    },
    {
        "name": "Eczema", "category": "Dermatological", "parent_category": "Dermatological > Inflammatory",
        "severity_level": "mild", "is_emergency": False,
        "required_symptoms": ["Skin Rash", "Itching", "Dry Skin"],
        "supporting_symptoms": ["Skin Discoloration", "Blisters"],
        "medicines": ["Hydrocortisone Cream", "Cetirizine"],
        "home_remedies": "Moisturize frequently, lukewarm baths with oatmeal, avoid harsh soaps.",
        "lifestyle_advice": "Identify and avoid irritants/allergens, keep skin moisturized daily.",
        "doctor_recommended": False,
        "description": "A chronic inflammatory skin condition causing itchy, dry, inflamed skin.",
    },
    {
        "name": "Psoriasis", "category": "Dermatological", "parent_category": "Dermatological > Autoimmune",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Skin Rash", "Skin Discoloration", "Itching", "Joint Pain"],
        "supporting_symptoms": ["Dry Skin", "Dry Skin"],
        "medicines": ["Hydrocortisone Cream"],
        "home_remedies": "Moisturizers, careful sun exposure, oatmeal baths.",
        "lifestyle_advice": "Manage stress, avoid skin trauma, avoid alcohol and smoking.",
        "doctor_recommended": True,
        "description": "A chronic autoimmune condition causing rapid skin cell buildup and scaly patches.",
    },
    {
        "name": "Scabies", "category": "Dermatological", "parent_category": "Dermatological > Infestation",
        "severity_level": "mild", "is_emergency": False,
        "required_symptoms": ["Itching", "Skin Rash", "Red Spots"],
        "supporting_symptoms": ["Blisters", "Skin Discoloration"],
        "medicines": ["Permethrin Cream"],
        "home_remedies": "Wash all bedding/clothing in hot water, avoid scratching.",
        "lifestyle_advice": "Treat all household/close contacts simultaneously to prevent reinfestation.",
        "doctor_recommended": True,
        "description": "A contagious skin infestation caused by the microscopic itch mite.",
    },
    {
        "name": "Ringworm", "category": "Dermatological", "parent_category": "Dermatological > Fungal",
        "severity_level": "mild", "is_emergency": False,
        "required_symptoms": ["Skin Rash", "Itching", "Red Spots", "Skin Discoloration"],
        "supporting_symptoms": ["Dry Skin"],
        "medicines": ["Clotrimazole Cream"],
        "home_remedies": "Keep area clean and dry, avoid sharing towels/clothing.",
        "lifestyle_advice": "Complete full antifungal course even after visible improvement.",
        "doctor_recommended": False,
        "description": "A contagious fungal skin infection causing a ring-shaped rash.",
    },
    {
        "name": "Cellulitis", "category": "Dermatological", "parent_category": "Dermatological > Bacterial",
        "severity_level": "high", "is_emergency": False,
        "required_symptoms": ["Skin Rash", "Skin Discoloration", "Fever", "Joint Swelling"],
        "supporting_symptoms": ["Fatigue", "Chills"],
        "medicines": ["Cephalexin", "Amoxicillin"],
        "home_remedies": "Elevate the affected limb, keep the area clean.",
        "lifestyle_advice": "Complete full antibiotic course; seek urgent care if fever or rapid spreading occurs.",
        "doctor_recommended": True,
        "description": "A bacterial skin infection causing redness, swelling, and warmth in the affected area.",
    },
    {
        "name": "Conjunctivitis", "category": "ENT", "parent_category": "ENT > Infectious",
        "severity_level": "mild", "is_emergency": False,
        "required_symptoms": ["Eye Redness", "Eye Pain", "Watery Eyes"],
        "supporting_symptoms": ["Itching", "Watery Eyes"],
        "medicines": ["Antihistamine Eye Drops"],
        "home_remedies": "Cool compress, avoid touching/rubbing eyes, discard old eye makeup.",
        "lifestyle_advice": "Wash hands frequently, avoid sharing towels, discontinue contact lens use.",
        "doctor_recommended": False,
        "description": "Inflammation of the conjunctiva (pink eye), often viral, bacterial, or allergic.",
    },
    {
        "name": "Ear Infection", "category": "ENT", "parent_category": "ENT > Infectious",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Ear Pain", "Ear Discharge", "Hearing Loss"],
        "supporting_symptoms": ["Fever", "Irritability"],
        "medicines": ["Amoxicillin", "Paracetamol"],
        "home_remedies": "Warm compress on the ear, keep ear dry.",
        "lifestyle_advice": "Avoid inserting objects into the ear canal, complete antibiotic course if prescribed.",
        "doctor_recommended": True,
        "description": "Infection of the middle or outer ear, common in children.",
    },
    {
        "name": "Depression (MDD)", "category": "Psychological", "parent_category": "Psychological > Mood",
        "severity_level": "high", "is_emergency": False,
        "required_symptoms": ["Low Mood", "Fatigue", "Loss of Appetite", "Insomnia"],
        "supporting_symptoms": ["Difficulty Concentrating", "Weight Loss", "Irritability"],
        "medicines": ["Sertraline"],
        "home_remedies": "Regular physical activity, social connection, structured daily routine.",
        "lifestyle_advice": "Seek therapy/counselling, maintain sleep hygiene, avoid alcohol.",
        "doctor_recommended": True,
        "description": "Major Depressive Disorder - a persistent mood disorder affecting daily functioning.",
    },
    {
        "name": "Anxiety Disorder", "category": "Psychological", "parent_category": "Psychological > Mood",
        "severity_level": "moderate", "is_emergency": False,
        "required_symptoms": ["Anxiety", "Restlessness", "Insomnia", "Rapid Heartbeat"],
        "supporting_symptoms": ["Difficulty Concentrating", "Irritability", "Muscle Cramps"],
        "medicines": ["Sertraline"],
        "home_remedies": "Deep breathing exercises, meditation, limit caffeine.",
        "lifestyle_advice": "Cognitive behavioural therapy, regular exercise, structured relaxation practice.",
        "doctor_recommended": True,
        "description": "A mental health condition marked by excessive, persistent worry and physical tension.",
    },
    {
        "name": "COPD", "category": "Respiratory", "parent_category": "Respiratory > Chronic",
        "severity_level": "critical", "is_emergency": False,
        "required_symptoms": ["Shortness of Breath", "Productive Cough", "Wheezing"],
        "supporting_symptoms": ["Fatigue", "Chest Tightness", "Weight Loss"],
        "medicines": ["Salbutamol Inhaler", "Montelukast"],
        "home_remedies": "Breathing exercises (pursed-lip breathing), avoid smoke exposure.",
        "lifestyle_advice": "Quit smoking immediately, pulmonary rehabilitation, get flu/pneumonia vaccines.",
        "doctor_recommended": True,
        "description": "Chronic Obstructive Pulmonary Disease - progressive lung disease causing airflow blockage.",
    },
]

logger.info("Loaded %d diseases into sample_data module.", len(DISEASES))


# ==========================================================================
# 4. SEEDING
# ==========================================================================
def seed(session_factory) -> None:
    """
    Populate an empty database with SYMPTOMS, MEDICINES, DISEASES and the
    derived Rule rows used by the forward/backward chaining engines.

    Purpose: Turn the plain-Python data above into persisted ORM rows and
             association-table links, so the rest of the system (which
             reads everything through database.py / knowledge_base.py)
             has a fully populated medical.db to work with.
    Input  : session_factory - a SQLAlchemy sessionmaker (e.g. database.SessionLocal)
    Output : None (writes rows to the database and commits).
    Logic  : 1) Insert all Symptom rows, keyed by name for fast lookup.
             2) Insert all Medicine rows, keyed by name for fast lookup.
             3) For each disease dict: insert the Disease row, link its
                required+supporting symptoms via the association table
                (marking required ones with is_required=True and a higher
                weight), link its medicines, and create one Rule whose
                condition is the required_symptoms list.
    Time Complexity : O(S + M + D * k) where k is the average number of
                       symptoms/medicines per disease (small, bounded).
    Space Complexity: O(S + M + D) for the in-memory ORM objects.
    """
    session = session_factory()
    try:
        # ---- Symptoms -----------------------------------------------------
        symptom_lookup: Dict[str, Symptom] = {}
        for s in SYMPTOMS:
            obj = Symptom(name=s["name"], category=s["category"])
            session.add(obj)
            symptom_lookup[s["name"]] = obj
        session.flush()  # assign primary keys without committing yet

        # ---- Medicines ------------------------------------------------------
        medicine_lookup: Dict[str, Medicine] = {}
        for m in MEDICINES:
            obj = Medicine(
                name=m["name"],
                dosage_info=m["dosage_info"],
                contraindications=m["contraindications"],
                is_otc=m["is_otc"],
            )
            session.add(obj)
            medicine_lookup[m["name"]] = obj
        session.flush()

        # ---- Diseases + associations + rules --------------------------------
        for d in DISEASES:
            disease = Disease(
                name=d["name"],
                category=d["category"],
                description=d["description"],
                severity_level=d["severity_level"],
                is_emergency=d["is_emergency"],
                home_remedies=d["home_remedies"],
                lifestyle_advice=d["lifestyle_advice"],
                doctor_recommended=d["doctor_recommended"],
                parent_category=d["parent_category"],
            )

            for sym_name in d["required_symptoms"]:
                symptom = symptom_lookup.get(sym_name)
                if symptom is None:
                    logger.warning("Disease '%s' references unknown symptom '%s'", d["name"], sym_name)
                    continue
                disease.symptoms.append(symptom)

            for sym_name in d["supporting_symptoms"]:
                symptom = symptom_lookup.get(sym_name)
                if symptom is None or symptom in disease.symptoms:
                    continue
                disease.symptoms.append(symptom)

            for med_name in d["medicines"]:
                medicine = medicine_lookup.get(med_name)
                if medicine is None:
                    logger.warning("Disease '%s' references unknown medicine '%s'", d["name"], med_name)
                    continue
                disease.medicines.append(medicine)

            session.add(disease)
            session.flush()  # need disease.id for the Rule foreign key

            rule = Rule(
                disease_id=disease.id,
                condition_symptoms=",".join(d["required_symptoms"]),
                min_matches=max(2, len(d["required_symptoms"]) - 1),
                confidence_weight=0.85 if d["severity_level"] in ("high", "critical") else 0.75,
                description=f"IF patient has {', '.join(d['required_symptoms'])} THEN consider {d['name']}",
            )
            session.add(rule)

        session.commit()
        logger.info(
            "Seeded database: %d symptoms, %d medicines, %d diseases, %d rules.",
            len(SYMPTOMS), len(MEDICINES), len(DISEASES), len(DISEASES),
        )
    except Exception:
        session.rollback()
        logger.exception("Seeding failed - rolled back.")
        raise
    finally:
        session.close()
