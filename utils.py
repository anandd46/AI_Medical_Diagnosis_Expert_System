"""
utils.py
=========
Reusable helper functions shared across the project: PDF diagnosis
export, matplotlib chart builders (for the Statistics dashboard and the
Reasoning Tree page), and small formatting/validation utilities.

Keeping these here avoids duplicating plotting/export code across the
Streamlit pages in app.py.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend - required for Streamlit/server environments
import matplotlib.pyplot as plt
import networkx as nx
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import THEME_COLORS, get_logger

logger = get_logger(__name__)


# ==========================================================================
# PDF export
# ==========================================================================
def generate_diagnosis_pdf(
    patient_name: str,
    patient_age: int,
    patient_gender: str,
    diagnosis_name: str,
    confidence_percent: float,
    matched_symptoms: List[str],
    missing_symptoms: List[str],
    alternatives: List[Tuple[str, float]],
    home_remedies: str,
    lifestyle_advice: str,
    safety_notes: List[str],
    reasoning_summary: str,
) -> bytes:
    """
    Purpose: Render a complete diagnosis report as a downloadable PDF,
             satisfying the "Export diagnosis PDF" feature requirement.
    Input  : patient/diagnosis fields as produced by expert_system.py
    Output : raw PDF bytes (caller writes these to disk or a download button)
    Logic  : Build a ReportLab Platypus flowable document: title, patient
             info table, diagnosis summary, matched/missing symptom
             lists, alternative diagnoses table, remedies/advice, and
             safety notes, styled with the app's theme colors.
    Time Complexity : O(S + A) where S = symptoms listed, A = alternatives listed.
    Space Complexity: O(S + A) plus the fixed PDF document overhead.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], textColor=colors.HexColor(THEME_COLORS["primary"])
    )
    heading_style = ParagraphStyle(
        "HeadingCustom", parent=styles["Heading2"], textColor=colors.HexColor(THEME_COLORS["primary"]),
        spaceBefore=12, spaceAfter=6,
    )
    body_style = styles["BodyText"]

    story = [
        Paragraph("AI Medical Diagnosis Expert System", title_style),
        Paragraph(f"Diagnosis Report — generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style),
        Spacer(1, 0.5 * cm),
    ]

    patient_table = Table(
        [
            ["Patient Name", patient_name],
            ["Age", str(patient_age)],
            ["Gender", patient_gender.title()],
        ],
        colWidths=[5 * cm, 10 * cm],
    )
    patient_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF3EE")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(patient_table)
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("Primary Diagnosis", heading_style))
    story.append(Paragraph(f"<b>{diagnosis_name}</b> — Confidence: {confidence_percent:.0f}%", body_style))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Matched Symptoms", heading_style))
    story.append(Paragraph(", ".join(matched_symptoms) or "None recorded", body_style))

    if missing_symptoms:
        story.append(Paragraph("Expected but Not Reported", heading_style))
        story.append(Paragraph(", ".join(missing_symptoms), body_style))

    if alternatives:
        story.append(Paragraph("Alternative Diagnoses", heading_style))
        alt_rows = [["Disease", "Confidence"]] + [
            [name, f"{cf * 100:.0f}%"] for name, cf in alternatives
        ]
        alt_table = Table(alt_rows, colWidths=[10 * cm, 5 * cm])
        alt_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(THEME_COLORS["primary"])),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(alt_table)

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Home Remedies", heading_style))
    story.append(Paragraph(home_remedies or "None specified", body_style))

    story.append(Paragraph("Lifestyle Advice", heading_style))
    story.append(Paragraph(lifestyle_advice or "None specified", body_style))

    if safety_notes:
        story.append(Paragraph("Safety Notes", heading_style))
        for note in safety_notes:
            story.append(Paragraph(f"⚠ {note}", body_style))

    story.append(Paragraph("Reasoning Trace", heading_style))
    for line in reasoning_summary.split("\n"):
        story.append(Paragraph(line.replace("✔", "[OK]").replace("✘", "[MISSING]"), body_style))

    story.append(Spacer(1, 0.6 * cm))
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["BodyText"], textColor=colors.HexColor(THEME_COLORS["danger"]), fontSize=8
    )
    story.append(
        Paragraph(
            "DISCLAIMER: This report is generated by an educational AI expert system for portfolio/demo "
            "purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. "
            "Always consult a qualified healthcare provider.",
            disclaimer_style,
        )
    )

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    logger.info("Generated diagnosis PDF for patient '%s' (%d bytes).", patient_name, len(pdf_bytes))
    return pdf_bytes


# ==========================================================================
# Matplotlib chart builders
# ==========================================================================
def plot_confidence_bar_chart(ranked_results: List[Tuple[str, float]]):
    """
    Purpose: Bar chart of disease -> confidence percentage, used on the
             Diagnosis and Statistics pages.
    Input  : list of (disease_name, certainty_factor in [0,1])
    Output : matplotlib Figure
    Time Complexity : O(N) - N candidates plotted.
    """
    names = [n for n, _ in ranked_results]
    values = [round(v * 100, 1) for _, v in ranked_results]

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.5 * len(names))))
    colors_list = [THEME_COLORS["primary"]] + [THEME_COLORS["info"]] * (len(names) - 1)
    ax.barh(names[::-1], values[::-1], color=colors_list[::-1])
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Diagnosis Confidence Comparison")
    for i, v in enumerate(values[::-1]):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_disease_frequency(frequency: Dict[str, int]):
    """
    Purpose: Bar chart of how often each disease has been diagnosed
             historically, for the Statistics dashboard.
    Time Complexity : O(N log N) due to sorting by frequency.
    """
    items = sorted(frequency.items(), key=lambda kv: kv[1], reverse=True)[:15]
    names = [k for k, _ in items]
    counts = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(names))))
    ax.barh(names[::-1], counts[::-1], color=THEME_COLORS["secondary"])
    ax.set_xlabel("Number of Diagnoses")
    ax.set_title("Disease Diagnosis Frequency (Top 15)")
    fig.tight_layout()
    return fig


def plot_confidence_distribution(confidences: List[float]):
    """
    Purpose: Histogram of historical diagnosis confidence scores, for the
             Statistics dashboard.
    Time Complexity : O(N) for the histogram binning (matplotlib internal).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([c * 100 for c in confidences], bins=10, range=(0, 100), color=THEME_COLORS["info"], edgecolor="white")
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Number of Diagnoses")
    ax.set_title("Confidence Score Distribution")
    fig.tight_layout()
    return fig


def plot_reasoning_tree(graph: nx.DiGraph):
    """
    Purpose: Render the reasoning tree (Symptoms -> Rules -> Facts ->
             Diagnosis) built by reasoning.ReasoningTreeBuilder as a
             layered matplotlib figure, colour-coded by node layer.
    Time Complexity : O(V + E) for layout and drawing.
    """
    layer_order = {"symptom": 0, "rule": 1, "fact": 2, "diagnosis": 3}
    layer_colors = {
        "symptom": THEME_COLORS["info"],
        "rule": THEME_COLORS["warning"],
        "fact": THEME_COLORS["secondary"],
        "diagnosis": THEME_COLORS["primary"],
    }

    for node, data in graph.nodes(data=True):
        data["subset"] = layer_order.get(data.get("layer", "symptom"), 0)

    try:
        pos = nx.multipartite_layout(graph, subset_key="subset")
    except Exception:
        pos = nx.spring_layout(graph, seed=42)

    fig, ax = plt.subplots(figsize=(11, max(6, graph.number_of_nodes() * 0.35)))
    node_colors = [layer_colors.get(graph.nodes[n].get("layer"), "#999999") for n in graph.nodes]
    labels = {n: graph.nodes[n].get("label", n) for n in graph.nodes}

    nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True, edge_color="#B0B0B0", arrowsize=12)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=1400, alpha=0.9)
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7)

    ax.set_title("Reasoning Tree: Symptoms → Rules → Facts → Diagnosis")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ==========================================================================
# Formatting / validation utilities
# ==========================================================================
def format_percent(value: float) -> str:
    """Format a [0,1] float as a whole-number percentage string, e.g. 0.873 -> '87%'."""
    return f"{value * 100:.0f}%"


def truncate(text: str, max_len: int = 80) -> str:
    """Truncate long text for compact table display, appending an ellipsis if cut."""
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def parse_csv_field(value: str) -> List[str]:
    """Parse a comma-separated database field (allergies, chronic conditions, etc.) into a clean list."""
    return [v.strip() for v in (value or "").split(",") if v.strip()]
