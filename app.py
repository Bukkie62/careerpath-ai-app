"""
CareerPath AI - Prediction API
--------------------------------
Wraps the trained hierarchical KNN model (field_model.pkl, field_scaler.pkl,
field_label_encoder.pkl, role_models_by_field.pkl) in a FastAPI service so
Lovable/Base44 frontends (or anything else) can call it over HTTP.

This reuses the SAME encoding logic from data_pipeline.py so a new
student's raw answers get turned into the exact feature vector shape the
model was trained on - this consistency is the part that's easy to get
wrong if you rebuild the encoding by hand in a different language.

Run locally:
    pip install fastapi uvicorn scikit-learn imbalanced-learn pandas numpy joblib --break-system-packages
    uvicorn app:app --reload

Deploy: push this file + data_pipeline.py + the 4 .pkl files +
requirements.txt to a GitHub repo, then deploy on Render/Railway as a
Python web service with start command: uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from fuzzy_logic.fuzzy_engine import build_fuzzy_system, adjust_scores, get_wellbeing_flag

app = FastAPI(title="CareerPath AI Prediction API")

# Allow your Lovable/Base44 frontend (any origin) to call this API.
# Once you know your frontend's deployed URL, replace "*" with it for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load trained artifacts once, at startup ---
field_model = joblib.load("field_model.pkl")
field_scaler = joblib.load("field_scaler.pkl")
field_encoder = joblib.load("field_label_encoder.pkl")
role_models = joblib.load("role_models_by_field.pkl")

# --- Build the fuzzy inference system once, at startup ---
# Adjusts KNN role compatibility scores based on the student's stress and
# burnout levels - a student with high stress/burnout gets scores nudged
# toward lower-pressure roles, and receives a wellbeing note. See
# fuzzy_logic/fuzzy_engine.py for the full Mamdani FIS implementation
# (membership functions, rule base, centroid defuzzification).
fuzzy_simulation = build_fuzzy_system()

# The list + order of columns the model was trained on. If you don't have
# this saved yet, add `joblib.dump(feature_columns, "feature_columns.pkl")`
# right after `FEATURE_COLUMNS = feature_columns` in main() in
# knn_career_recommender.py, retrain once, and re-download this file too.
FEATURE_COLUMNS = joblib.load("feature_columns.pkl")

# Maps a wide range of REAL course/major names to the nearest of the 14
# categories the model was actually trained on (see data_pipeline.py
# COURSE_TO_FIELD). Checked in order - more specific keywords first, so
# e.g. "veterinary medicine" hits agricultural_sciences before "medicine"
# would otherwise route it to medical_sciences.
#
# KNOWN LIMITATION: "Law" has no good home here. The real 200-student
# training survey never included a Law respondent, and no personality
# type was ever mapped to "Law & Public Service" as a fallback either -
# meaning that field is structurally unreachable by this model no matter
# what a student enters. Law students below are routed to social_sciences
# as the closest available proxy, and the API adds an explicit note
# flagging this. Properly fixing this requires collecting real Law
# (and Pharmacy, Music, etc.) student survey responses and retraining -
# see thesis limitations section.
COURSE_KEYWORD_MAP = [
    # (keywords to check, matched category)
    (["veterinary"], "agricultural_sciences"),
    (["agricultur", "animal science", "crop science", "soil science",
      "forestry", "fisher"], "agricultural_sciences"),
    (["computer science", "software", "information technology",
      "cyber security", "cybersecurity", "computer engineering",
      "information systems", "info systems", "mgt info sys",
      "management information systems", "data science",
      "computing"], "it_computer_science"),
    (["civil engineering", "mechanical engineering", "electrical",
      "chemical engineering", "petroleum engineering", "mechatronic",
      "industrial engineering", "telecommunications engineering",
      "engineering"], "engineering"),
    (["nursing", "physiology", "physiotherapy", "public health",
      "medical laboratory", "radiography", "nutrition", "dietetics",
      "biomedical", "optometry", "medical rehabilitation",
      "anatomy"], "health_sciences"),
    (["medicine", "surgery", "dentistry", "pharmacy", "pharmacology",
      "pharmaceutical", "dental", "mbbs"], "medical_sciences"),
    (["business admin", "accounting", "banking", "finance", "economics",
      "marketing", "management", "insurance", "actuarial"], "business"),
    (["environmental science", "urban planning", "estate management",
      "architecture", "surveying", "geomatics",
      "environmental management"], "environmental_technology_science"),
    (["law", "legal studies", "jurisprudence", "llb"], "social_sciences"),
    (["political science", "sociology", "psychology",
      "international relations", "social work", "mass communication",
      "journalism", "criminology", "public administration"], "social_sciences"),
    (["geology", "mining", "geophysics"], "earth_&_mineral_sciences"),
    (["physics", "chemistry", "mathematics", "statistics",
      "industrial chemistry"], "physical_sciences"),
    (["biology", "microbiology", "biochemistry", "botany", "zoology",
      "genetics", "biotechnology"], "life_sciences"),
    (["education", "guidance and counselling",
      "curriculum"], "education"),
    (["fine art", "music", "theatre", "creative art", "graphic design",
      "fashion design", "english", "literature", "language",
      "linguistics", "history", "philosophy",
      "religious studies"], "arts_&_design"),
]


def match_course_to_category(raw_course: str) -> tuple[str, bool]:
    """
    Returns (matched_category_slug, is_confident_match).
    Tries an exact match first (fastest, most reliable), then falls back
    to keyword matching against COURSE_KEYWORD_MAP, then finally "other"
    if nothing matches at all.
    """
    normalized = raw_course.strip().lower()

    # Exact match against known training-time course strings
    known_exact = {
        "it/computer science": "it_computer_science",
        "engineering": "engineering",
        "health sciences": "health_sciences",
        "medical sciences": "medical_sciences",
        "business": "business",
        "environmental technology/science": "environmental_technology_science",
        "social sciences": "social_sciences",
        "earth & mineral sciences": "earth_&_mineral_sciences",
        "physical sciences": "physical_sciences",
        "agricultural sciences": "agricultural_sciences",
        "life sciences": "life_sciences",
        "education": "education",
        "arts & design": "arts_&_design",
        "other": "other",
    }
    if normalized in known_exact:
        return known_exact[normalized], True

    # Keyword fallback for real course names not in the exact list
    for keywords, category in COURSE_KEYWORD_MAP:
        if any(kw in normalized for kw in keywords):
            return category, True

    return "other", False

RIASEC_KEYS = ["realistic", "investigative", "artistic",
               "social", "enterprising", "conventional"]

WORK_PREF_MAP = {
    "Remote": "pref_remote", "Hybrid": "pref_hybrid",
    "Office-based": "pref_office_based",
    "International placement": "pref_international",
    "Corporate environment": "pref_corporate",
    "Startup environment": "pref_startup",
}


class StudentInput(BaseModel):
    course_of_study: str = Field(..., description="Must match a course seen during training, e.g. 'IT/Computer Science'")
    riasec_cluster: str = Field(..., description="One of: realistic, investigative, artistic, social, enterprising, conventional")
    technical_skill: float = Field(..., ge=1, le=10)
    communication_skill: float = Field(..., ge=1, le=10)
    analytical_skill: float = Field(..., ge=1, le=10)
    creative_skill: float = Field(..., ge=1, le=10)
    work_preference: str = Field(..., description="e.g. Remote, Hybrid, Office-based, International placement, Corporate environment, Startup environment")
    prior_experience: bool = False
    stress_level: float = Field(5, ge=1, le=10)
    burnout_level: int = Field(0, ge=0, le=2, description="0=low, 1=medium, 2=high")
    # Optional real values, if your platform later collects them directly
    # instead of deriving them synthetically:
    cgpa: Optional[float] = Field(None, ge=0, le=5, description="On a 5.0 scale, if known")
    openness: Optional[float] = Field(None, ge=0, le=1)
    conscientiousness: Optional[float] = Field(None, ge=0, le=1)
    extraversion: Optional[float] = Field(None, ge=0, le=1)
    agreeableness: Optional[float] = Field(None, ge=0, le=1)
    neuroticism: Optional[float] = Field(None, ge=0, le=1)


def build_feature_vector(student: StudentInput) -> dict:
    """Reconstructs the exact feature encoding used during training."""
    tech = (student.technical_skill - 1) / 9.0
    comm = (student.communication_skill - 1) / 9.0
    analytic = (student.analytical_skill - 1) / 9.0
    creative = (student.creative_skill - 1) / 9.0
    stress = (student.stress_level - 1) / 9.0
    prior_exp = int(student.prior_experience)

    features = {
        "technical_skill": tech,
        "communication_skill": comm,
        "analytical_skill": analytic,
        "creative_skill": creative,
        "prior_experience": prior_exp,
    }

    riasec_key = student.riasec_cluster.strip().lower()
    for key in RIASEC_KEYS:
        features[f"riasec_{key}"] = 1 if riasec_key == key else 0

    for label, col in WORK_PREF_MAP.items():
        features[col] = 1 if student.work_preference == label else 0

    # OCEAN + CGPA: use real values if the frontend collected them,
    # otherwise fall back to the same derivation formula used to
    # generate synthetic training features (see data_pipeline.py
    # add_synthetic_features for the rationale).
    riasec_artistic = features["riasec_artistic"]
    riasec_social = features["riasec_social"]
    riasec_enterprising = features["riasec_enterprising"]

    features["openness"] = (
        student.openness if student.openness is not None
        else np.clip(0.6 * creative + 0.4 * riasec_artistic, 0, 1)
    )
    features["conscientiousness"] = (
        student.conscientiousness if student.conscientiousness is not None
        else np.clip(0.5 * analytic + 0.5 * prior_exp, 0, 1)
    )
    features["extraversion"] = (
        student.extraversion if student.extraversion is not None
        else np.clip(0.5 * comm + 0.25 * riasec_enterprising + 0.25 * riasec_social, 0, 1)
    )
    features["agreeableness"] = (
        student.agreeableness if student.agreeableness is not None
        else np.clip(0.6 * riasec_social + 0.4 * comm, 0, 1)
    )
    features["neuroticism"] = (
        student.neuroticism if student.neuroticism is not None
        else np.clip(0.6 * stress + 0.4 * (student.burnout_level / 2.0), 0, 1)
    )
    features["cgpa_normalized"] = (
        (student.cgpa / 5.0) if student.cgpa is not None
        else np.clip(0.55 * analytic + 0.45 * features["conscientiousness"], 0, 1)
    )

    # Course one-hot: use smart matching instead of requiring an exact
    # string match, so real course names (Pharmacy, Law, Music, etc.)
    # get routed to their closest trained category instead of defaulting
    # to blank/unknown.
    matched_category, was_confident = match_course_to_category(student.course_of_study)
    for col in FEATURE_COLUMNS:
        if col.startswith("course_"):
            features[col] = 1 if col == f"course_{matched_category}" else 0

    return features, matched_category, was_confident


@app.get("/")
def root():
    return {"status": "CareerPath AI Prediction API is running"}


@app.post("/predict")
def predict(student: StudentInput, top_n: int = 5):
    try:
        feature_dict, matched_category, was_confident = build_feature_vector(student)
        x = np.array([[feature_dict[col] for col in FEATURE_COLUMNS]])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected feature: {e}")

    # Stage 1: predict field
    x_field_scaled = field_scaler.transform(x)
    field_probs = field_model.predict_proba(x_field_scaled)[0]
    top_field_idx = int(np.argmax(field_probs))
    predicted_field = field_encoder.inverse_transform([top_field_idx])[0]
    field_confidence = float(field_probs[top_field_idx])

    # Stage 2: rank roles within that field (if a role model exists for it)
    role_predictions = []
    if predicted_field in role_models:
        sub_model, role_scaler, role_encoder = role_models[predicted_field]
        x_role_scaled = role_scaler.transform(x)
        role_probs = sub_model.predict_proba(x_role_scaled)[0]
        top_indices = np.argsort(role_probs)[::-1][:top_n]
        role_predictions = [
            {"career": role_encoder.inverse_transform([i])[0],
             "compatibility_score": float(role_probs[i])}
            for i in top_indices
        ]

    # Fuzzy wellbeing adjustment: nudge role scores based on the student's
    # stress and burnout levels (Mamdani FIS - see fuzzy_logic/fuzzy_engine.py).
    # KNN scores are 0-1; the fuzzy engine works on a 0-100 scale, so we
    # convert, adjust, then convert back for API consistency.
    wellbeing_note = None
    wellbeing_flag = None
    fuzzy_adjustment = None
    if role_predictions:
        scores_100 = {r["career"]: r["compatibility_score"] * 100 for r in role_predictions}
        adjusted_100, fuzzy_adjustment, wellbeing_note = adjust_scores(
            scores_100, student.stress_level, student.burnout_level
        )
        wellbeing_flag = get_wellbeing_flag(fuzzy_adjustment)

        for r in role_predictions:
            r["compatibility_score"] = round(adjusted_100[r["career"]] / 100, 4)
        role_predictions.sort(key=lambda r: r["compatibility_score"], reverse=True)

    notes = [
        "Role-level rankings are currently based on illustrative proxy "
        "labels pending real role-preference survey data - see thesis "
        "limitations section."
    ]
    if matched_category == "social_sciences" and "law" in student.course_of_study.lower():
        notes.append(
            "Law was matched to Social Sciences as the closest available "
            "category - the model was not trained on any Law student "
            "survey data, so this prediction is a rough proxy, not a "
            "confident Law-specific recommendation."
        )
    elif not was_confident:
        notes.append(
            f"'{student.course_of_study}' didn't closely match a known "
            f"course category, so this prediction relies more heavily on "
            f"personality and skills than on your field of study."
        )

    response = {
        "predicted_field": predicted_field,
        "field_confidence": field_confidence,
        "recommended_roles": role_predictions,
        "note": " ".join(notes),
        "wellbeing": {
            "fuzzy_adjustment": fuzzy_adjustment,
            "flag": wellbeing_flag,
            "message": wellbeing_note,
        },
    }
    return response
