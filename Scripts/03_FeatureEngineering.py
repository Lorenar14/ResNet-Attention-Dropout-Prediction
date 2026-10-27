import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# Configuration
OUTPUT_DIR = Path("/content/Results")
INPUT_FILE = OUTPUT_DIR / "Anonymized_Dataset.csv"
TARGET_PERIOD = "2026-1S"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"Anonymized dataset not found: {INPUT_FILE}. "
        "Run Phase 1 first."
    )


# Load dataset
df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig",
    dtype="string",
    low_memory=False,
)

if df.empty:
    raise ValueError("The input dataset contains no records.")

original_rows = len(df)


# Standardize column names
def normalize_column(column):
    column = unicodedata.normalize("NFKD", str(column))
    column = column.encode("ascii", "ignore").decode("ascii")
    column = column.upper().strip()
    column = re.sub(r"\s+", "_", column)
    column = re.sub(r"[^A-Z0-9_]", "", column)
    return re.sub(r"_+", "_", column).strip("_")


df.columns = [normalize_column(c) for c in df.columns]

if df.columns.duplicated().any():
    raise ValueError("Duplicate column names were found.")


# Standardize source column names
aliases = {
    "DOCUENTO_DE_IDENTIDAD": "STUDENT_ID",
    "DOCUMENTO_DE_IDENTIDAD": "STUDENT_ID",
    "ID_ESTUDIANTE": "STUDENT_ID",
    "PERIODO_INGRESO": "ENTRY_PERIOD",
    "PERIODO_ACTUAL": "CURRENT_PERIOD",
    "EDAD": "AGE",
    "ESTRATO": "STRATUM",
    "NIVEL_ACTUAL": "CURRENT_LEVEL",
    "SEMESTRES_TOTAL_POR_CARRERA": "PROGRAM_DURATION",
    "PROGRAMA_ACADEMICO": "ACADEMIC_PROGRAM",
    "SNIES": "SNIES_CODE",
    "ESTADO_CIVIL": "CIVIL_STATUS",
    "GENERO_SEX": "GENDER",
}

rename_map = {
    col: aliases[col]
    for col in df.columns
    if col in aliases and aliases[col] not in df.columns
}

df.rename(columns=rename_map, inplace=True)

required = [
    "STUDENT_ID",
    "ENTRY_PERIOD",
    "CURRENT_LEVEL",
    "PROGRAM_DURATION",
 ]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(
        f"Required columns are missing: {missing}. "
        f"Available columns: {df.columns.tolist()}"
    )


# Standardize text and remove exact duplicate rows
for col in df.columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA}
    )

df = df.drop_duplicates().copy()
exact_duplicates_removed = original_rows - len(df)


# Convert numeric fields
for col in ["AGE", "STRATUM", "CURRENT_LEVEL", "PROGRAM_DURATION"]:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )


# Convert academic periods to sequential semester numbers
def semester_sequence(value):
    if pd.isna(value):
        return np.nan

    match = re.fullmatch(
        r"\s*(\d{4})\s*[-/]\s*([12])\s*S?\s*",
        str(value).strip().upper(),
    )

    if not match:
        return np.nan

    return int(match.group(1)) * 2 + int(match.group(2)) - 1


target_sequence = semester_sequence(TARGET_PERIOD)

if pd.isna(target_sequence):
    raise ValueError(f"Invalid target period: {TARGET_PERIOD}")

df["ENTRY_SEMESTER_SEQ"] = df["ENTRY_PERIOD"].map(semester_sequence)


# Validate records for the target period
valid = (
    df["STUDENT_ID"].notna()
    & df["ENTRY_SEMESTER_SEQ"].notna()
    & df["CURRENT_LEVEL"].notna()
    & df["PROGRAM_DURATION"].notna()
    & df["CURRENT_LEVEL"].ge(0)
    & df["PROGRAM_DURATION"].gt(0)
    & df["ENTRY_SEMESTER_SEQ"].le(target_sequence)
)

if "CURRENT_PERIOD" in df.columns:
    valid &= (
        df["CURRENT_PERIOD"].str.upper().eq(TARGET_PERIOD)
    )

excluded = df.loc[~valid].copy()
df = df.loc[valid].copy()

if df.empty:
    raise ValueError(
        "No valid records remain. Check the period formats "
        "and target-period filter."
    )


# Review repeated student identifiers and apply a systematic resolution
repeated = df[df.duplicated("STUDENT_ID", keep=False)].copy()

if not repeated.empty:
    repeated.to_csv(
        OUTPUT_DIR / "Phase2_Duplicate_Review.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    # Check whether any substantive field differs within a student.
    review_fields = [
        col for col in df.columns
        if col not in ["STUDENT_ID", "ENTRY_SEMESTER_SEQ"]
    ]

    differences = (
        repeated.groupby("STUDENT_ID")[review_fields]
        .nunique(dropna=False)
        .gt(1)
    )

    conflicting_ids = differences.index[differences.any(axis=1)]

    if len(conflicting_ids) > 0:
        conflicts = repeated[
            repeated["STUDENT_ID"].isin(conflicting_ids)
        ]

        conflicts.to_csv(
            OUTPUT_DIR / "Phase2_Conflicting_Student_Records.csv",
            sep=";",
            index=False,
            encoding="utf-8-sig",
        )

        # Resolving conflicts systematically by choosing the record with the highest academic progress/level and entry period
        # First sort the dataframe to put the most advanced and complete records first
        df = df.sort_values(
            by=["STUDENT_ID", "CURRENT_LEVEL", "ENTRY_SEMESTER_SEQ"],
            ascending=[True, False, False]
        )

    # De-duplicate keeping the most representative record determined by sorting
    df = df.drop_duplicates(subset="STUDENT_ID", keep="first").copy()


# Construct variables for the operational target
df["SEMESTERS_SINCE_ENTRY"] = (
    target_sequence - df["ENTRY_SEMESTER_SEQ"]
)

df["SEMESTER_GAP"] = (
    df["SEMESTERS_SINCE_ENTRY"] - df["CURRENT_LEVEL"]
)

df["ACADEMIC_PROGRESS"] = (
    df["CURRENT_LEVEL"]
    .div(df["PROGRAM_DURATION"])
    .mul(100)
    .clip(0, 100)
    .round(2)
)


# Apply the project-defined operational rule
df["ACADEMIC_STATUS"] = np.select(
    [
        df["CURRENT_LEVEL"] >= df["PROGRAM_DURATION"],
        df["SEMESTER_GAP"] >= 2,
    ],
    ["Graduated", "Dropout"],
    default="Active",
)

df["target"] = (
    df["ACADEMIC_STATUS"].eq("Dropout").astype("int8")
)

df["TARGET_DEFINITION"] = (
    "Operational proxy; enrollment history by period unavailable"
)


# Export modeling dataset
df.drop(
    columns=["ENTRY_SEMESTER_SEQ"],
    errors="ignore",
).to_csv(
    OUTPUT_DIR / "Modeling_Dataset.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig",
)


# Export status distribution
status_summary = (
    df["ACADEMIC_STATUS"]
    .value_counts()
    .rename_axis("ACADEMIC_STATUS")
    .reset_index(name="COUNT")
)

status_summary["PERCENTAGE"] = (
    status_summary["COUNT"] / len(df) * 100
).round(2)

status_summary.to_csv(
    OUTPUT_DIR / "Phase2_Status_Distribution.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig",
)


# Export excluded records when present
if not excluded.empty:
    excluded.to_csv(
        OUTPUT_DIR / "Phase2_Excluded_Records.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )


# Summary
print("\nPHASE 2 SUMMARY")
print("-" * 50)
print(f"Original records: {original_rows:,}")
print(f"Exact duplicates removed: {exact_duplicates_removed:,}")
print(f"Records excluded during validation: {len(excluded):,}")
print(f"Final records: {len(df):,}")
print(f"Unique students: {df['STUDENT_ID'].nunique():,}")
print(f"Target period: {TARGET_PERIOD}")

print("\nAcademic status distribution:")
print(status_summary.to_string(index=False))

print("\nBinary target distribution:")
print(df["target"].value_counts().sort_index().to_string())

print(f"\nModeling dataset: {OUTPUT_DIR / 'Modeling_Dataset.csv'}")
print(
    "Status distribution: "
    f"{OUTPUT_DIR / 'Phase2_Status_Distribution.csv'}"
)

log_progress("Phase 2 completed successfully.")
