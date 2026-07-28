# -----------------------------------------------------------------------------
# Standard Python Libraries
# -----------------------------------------------------------------------------

import os
import random
import hashlib
import warnings
import logging

# -----------------------------------------------------------------------------
# Numerical Computing
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd


# =============================================================================
DATA PREPROCESSING AND FEATURE ENGINEERING
# =============================================================================

print_title(DATA PREPROCESSING AND FEATURE ENGINEERING")

# =============================================================================
# Create Working Dataset
# =============================================================================

df = df_raw.copy()

# =============================================================================
# Remove Incomplete Records
# =============================================================================

REQUIRED_COLUMNS = [
    "DOCUENTO_DE_IDENTIDAD",
    "NOMBRE_COMPLETO",
    "PERIODO_INGRESO"
]

initial_records = len(df)

df = df.dropna(subset=REQUIRED_COLUMNS).copy()

removed_records = initial_records - len(df)

# =============================================================================
# Filter Invalid Ages
# =============================================================================

df["EDAD"] = pd.to_numeric(
    df["EDAD"],
    errors="coerce"
)

df = df[df["EDAD"] >= 15].copy()

# =============================================================================
# Convert Academic Variables
# =============================================================================

ACADEMIC_COLUMNS = [
    "NIVEL_ACTUAL",
    "SEMESTRES_TOTAL_POR_CARRERA"
]

for column in ACADEMIC_COLUMNS:

    df[column] = (
        pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)
    )

# =============================================================================
# Keep the Latest Academic Record
# =============================================================================

df = (
    df
    .sort_values(
        by=[
            "DOCUENTO_DE_IDENTIDAD",
            "NIVEL_ACTUAL"
        ],
        ascending=[
            True,
            False
        ]
    )
    .drop_duplicates(
        subset="DOCUENTO_DE_IDENTIDAD",
        keep="first"
    )
    .copy()
)

# =============================================================================
# Calculate Academic Progress
# =============================================================================

def calculate_academic_progress(row):
    """
    Calculate the academic progress percentage.
    """

    if row["SEMESTRES_TOTAL_POR_CARRERA"] > 0:

        progress = (
            row["NIVEL_ACTUAL"]
            /
            row["SEMESTRES_TOTAL_POR_CARRERA"]
        ) * 100

        return min(round(progress, 2), 100)

    return 0


df["ACADEMIC_PROGRESS"] = df.apply(
    calculate_academic_progress,
    axis=1
)

# Numeric variable used by Kaplan-Meier and Machine Learning

df["AVANCE_NU"] = df["ACADEMIC_PROGRESS"]

# =============================================================================
# Apply Institutional Business Rules
# =============================================================================

CURRENT_ACADEMIC_PERIOD = "2026-1S"


def academic_period_to_semester(period):
    """
    Convert an academic period into a sequential semester number.
    """

    try:

        year = int(period[:4])

        semester = 1 if "-1S" in period else 2

        return (year * 2) + semester

    except:

        return 0


def classify_student(row):
    """
    Classify each student according to institutional business rules.
    """

    current_level = row["NIVEL_ACTUAL"]

    program_length = row["SEMESTRES_TOTAL_POR_CARRERA"]

    enrollment_period = str(row["PERIODO_INGRESO"])

    elapsed_semesters = (

        academic_period_to_semester(CURRENT_ACADEMIC_PERIOD)

        -

        academic_period_to_semester(enrollment_period)

    )

    # Rule 1: Graduated

    if (
        current_level >= program_length
        and
        program_length > 0
    ):

        return "Graduated"

    # Rule 2: Dropout

    delay = elapsed_semesters - current_level

    if delay >= 2:

        return "Dropout"

    # Rule 3: Active

    return "Active"


df["ACADEMIC_STATUS"] = df.apply(
    classify_student,
    axis=1
)

# =============================================================================
# Create Target Variable
# =============================================================================

df["target"] = (
    df["ACADEMIC_STATUS"]
    .eq("Dropout")
    .astype(int)
)

# =============================================================================
# Dataset Summary
# =============================================================================

status_summary = df["ACADEMIC_STATUS"].value_counts()

graduates = status_summary.get("Graduated", 0)
active = status_summary.get("Active", 0)
dropout = status_summary.get("Dropout", 0)

print(f"\n{BOLD}{'='*90}")
print("DATA PREPROCESSING SUMMARY")
print(f"{'='*90}{END}")

print(f"Dataset Shape               : {df.shape}")
print(f"Processed Students          : {len(df):,}")
print(f"Total Variables             : {df.shape[1]}")
print(f"Removed Records             : {removed_records:,}")

print("\nAcademic Status Distribution")

print(f"Graduated                   : {graduates:,}")
print(f"Active                      : {active:,}")
print(f"Dropout                     : {dropout:,}")

print(f"\nDropout Rate                : {dropout / len(df):.2%}")

# =============================================================================
# Export Predictive Dataset
# =============================================================================

PREDICTIVE_DATASET = "Predictive_Dataset.csv"

df.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        PREDICTIVE_DATASET
    ),

    sep=";",
    encoding="utf-8-sig",
    index=False

)

print(f"\n{GREEN}✓ Predictive dataset exported successfully.{END}")

print(f"{GREEN}✓  completed successfully.{END}")
