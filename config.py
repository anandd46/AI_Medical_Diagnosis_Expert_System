
"""
config.py
=========
Central configuration module for the AI Medical Diagnosis Expert System.

Purpose
-------
Every other module in this project imports its settings from here instead of
hard-coding constants. This keeps the system easy to reconfigure (database
path, logging behaviour, fuzzy-logic thresholds, certainty-factor weights,
UI theme, etc.) without touching business logic anywhere else.

Time Complexity : O(1) - module-level constants only.
Space Complexity: O(1) - a handful of scalars / small dicts.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATABASE_FILE: str = "medical.db"
DATABASE_PATH: Path = BASE_DIR / DATABASE_FILE
DATABASE_URL: str = f"sqlite:///{DATABASE_PATH}"

LOG_FILE: Path = BASE_DIR / "expert_system.log"
EXPORT_DIR: Path = BASE_DIR  # PDF exports are written next to the project (flat structure rule)

# --------------------------------------------------------------------------
# Application metadata
# --------------------------------------------------------------------------
APP_NAME: str = "AI Medical Diagnosis Expert System"
APP_VERSION: str = "1.0.0"
APP_AUTHOR: str = "AI Engineering Portfolio Project"

# --------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Create (or fetch) a configured logger.

    Purpose: Provide a single, consistent logger factory used by every
             module so log output is uniform and always written both to
             the console and to expert_system.log.
    Input  : name (str) - typically __name__ of the calling module.
    Output : logging.Logger instance, already configured.
    Logic  : If the root expert-system logger has no handlers yet, attach
             a StreamHandler and a FileHandler. Subsequent calls reuse the
             same handlers (logging.getLogger caches by name).
    Time Complexity : O(1)
    Space Complexity: O(1)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)
        formatter = logging.Formatter(LOG_FORMAT)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        try:
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            # If the filesystem is read-only (e.g. some deployment targets)
            # fall back silently to console-only logging.
            pass

        logger.propagate = False
    return logger


# --------------------------------------------------------------------------
# Certainty Factor (uncertainty reasoning) configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CertaintyConfig:
    """Weights used when combining evidence into a certainty factor (CF)."""
    base_symptom_weight: float = 0.18       # CF contribution per matched symptom
    severity_bonus: dict = field(default_factory=lambda: {
        "mild": 0.05, "moderate": 0.10, "high": 0.18, "very_high": 0.25
    })
    missing_symptom_penalty: float = 0.07   # CF penalty per expected-but-absent symptom
    max_cf: float = 0.98                    # never claim 100% certainty
    min_cf: float = 0.02


CERTAINTY = CertaintyConfig()

# --------------------------------------------------------------------------
# Fuzzy Logic configuration (used by fuzzy_logic.py)
# --------------------------------------------------------------------------
FEVER_RANGE = (95.0, 106.0)          # degrees Fahrenheit, universe of discourse
PAIN_RANGE = (0, 10)                  # 0-10 pain scale
DURATION_RANGE = (0, 30)              # symptom duration in days

FUZZY_SEVERITY_LABELS = ["mild", "moderate", "high", "very_high"]

# --------------------------------------------------------------------------
# Constraint Satisfaction configuration
# --------------------------------------------------------------------------
MIN_PATIENT_AGE: int = 0
MAX_PATIENT_AGE: int = 120
PEDIATRIC_AGE_LIMIT: int = 12
GERIATRIC_AGE_THRESHOLD: int = 65

# --------------------------------------------------------------------------
# Search / reasoning configuration
# --------------------------------------------------------------------------
MAX_QUESTIONS_PER_SESSION: int = 8      # cap for means-end / backward chaining Q&A
TOP_N_DIAGNOSES: int = 5                # how many alternative diagnoses to display
HILL_CLIMB_ITERATIONS: int = 50

# --------------------------------------------------------------------------
# UI configuration
# --------------------------------------------------------------------------
STREAMLIT_PAGE_TITLE: str = APP_NAME
STREAMLIT_PAGE_ICON: str = "🩺"
STREAMLIT_LAYOUT: str = "wide"

THEME_COLORS = {
    "primary": "#0B6E4F",
    "secondary": "#08A045",
    "background_light": "#F7F9FA",
    "background_dark": "#0E1117",
    "danger": "#C0392B",
    "warning": "#E67E22",
    "info": "#2E86C1",
}

# --------------------------------------------------------------------------
# Environment overrides (optional, for deployment flexibility)
# --------------------------------------------------------------------------
DEBUG: bool = os.environ.get("MEDICAL_EXPERT_DEBUG", "false").lower() == "true"
