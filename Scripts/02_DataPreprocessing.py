import hashlib
import re
import unicodedata

import pandas as pd

# Input and output paths
INPUT_FILE = PROJECT_DIR / "DataSet".csv"
ANONYMIZED_FILE = OUTPUT_DIR / "Anonymized_Dataset.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Validate the input file
if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"The input dataset was not found: {INPUT_FILE}\n"
        "Check the filename and the current Google Colab directory."
    )

log_progress("Phase 1: Loading the institutional dataset.")

df_raw = pd.read_csv(
    INPUT_FILE,
    sep=";",
    encoding="latin-1",
    dtype="string",
    keep_default_na=True,
    low_memory=False,
)

if df_raw.empty:
    raise ValueError("The input dataset contains no records.")

if df_raw.columns.empty:
    raise ValueError("The input dataset contains no columns.")

log_progress(
    f"Dataset loaded: {df_raw.shape[0]:,} rows and "
    f"{df_raw.shape[1]:,} columns."
)

# Standardize column names
def normalize_column_name(column_name):
    """Normalize column names for consistent processing."""
    normalized = unicodedata.normalize("NFKD", str(column_name))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.upper().strip()
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^A-Z0-9_]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")

df_raw.columns = [
    normalize_column_name(column)
    for column in df_raw.columns
]

if df_raw.columns.duplicated().any():
    duplicated_columns = df_raw.columns[
        df_raw.columns.duplicated()
    ].tolist()

    raise ValueError(
        f"Duplicate columns after normalization: {duplicated_columns}"
    )

# Map original column names to standardized English names
column_mapping = {
    "DOCUENTO_DE_IDENTIDAD": "STUDENT_ID",
    "DOCUMENTO_DE_IDENTIDAD": "STUDENT_ID",
    "TIPO_DOC": "DOCUMENT_TYPE",
    "NOMBRE_COMPLETO": "FULL_NAME",
    "PERIODO_INGRESO": "ENTRY_PERIOD",
    "PERIODO_ACTUAL": "CURRENT_PERIOD",
    "FECHA_NACIMIENTO": "BIRTH_DATE",
    "EDAD": "AGE",
    "GENERO_SEX": "GENDER",
    "ESTADO_CIVIL": "CIVIL_STATUS",
    "TIPOESTADO_ACADEMICO": "ACADEMIC_STATUS_TYPE",
    "ESTRATO": "STRATUM",
    "PROGRAMA_ACADEMICO": "ACADEMIC_PROGRAM",
    "NIVEL_ACTUAL": "CURRENT_LEVEL",
    "SNIES": "SNIES_CODE",
    "SEMESTRES_TOTAL_POR_CARRERA": "PROGRAM_DURATION",
    "NOMBRE_SEDE": "CAMPUS_NAME",
    "IMPREB": "IMPREB",
}

unmapped_columns = [
    column
    for column in df_raw.columns
    if column not in column_mapping
]

if unmapped_columns:
    raise ValueError(
        "Review the following original column names before proceeding: "
        f"{unmapped_columns}"
    )

df_raw.rename(columns=column_mapping, inplace=True)

if df_raw.columns.duplicated().any():
    raise ValueError(
        "Column name mapping created duplicate columns. "
        "Review the original headers and column_mapping."
    )

required_columns = ["STUDENT_ID", "FULL_NAME"]

missing_columns = [
    column
    for column in required_columns
    if column not in df_raw.columns
]

if missing_columns:
    raise KeyError(
        f"Required columns are missing: {missing_columns}"
    )

log_progress("Column names standardized successfully.")

# Validate identifiers and missing values
for column in required_columns:
    df_raw[column] = df_raw[column].str.strip()
    df_raw[column] = df_raw[column].replace("", pd.NA)

missing_id_count = int(df_raw["STUDENT_ID"].isna().sum())
missing_name_count = int(df_raw["FULL_NAME"].isna().sum())
unique_id_count = int(df_raw["STUDENT_ID"].nunique())
duplicate_row_count = int(df_raw.duplicated().sum())

log_progress(f"Distinct student identifiers: {unique_id_count:,}.")
log_progress(f"Missing student identifiers: {missing_id_count:,}.")
log_progress(f"Missing student names: {missing_name_count:,}.")
log_progress(f"Exact duplicate rows: {duplicate_row_count:,}.")

# Generate SHA-256 pseudonymous identifiers
def pseudonymize(value):
    """Generate a SHA-256 hash while preserving missing values."""
    if pd.isna(value):
        return pd.NA

    normalized_value = str(value).strip()

    if not normalized_value:
        return pd.NA

    return hashlib.sha256(
        normalized_value.encode("utf-8")
    ).hexdigest()

# Create the pseudonymized dataset
df_processed = df_raw.copy()

df_processed["STUDENT_ID"] = df_processed["STUDENT_ID"].apply(
    pseudonymize
)

# Remove direct personal identifiers
df_processed.drop(
    columns=["FULL_NAME"],
    inplace=True,
)

# Check that direct identifiers are no longer present
sensitive_columns = {
    "FULL_NAME",
    "DOCUMENTO_DE_IDENTIDAD",
    "DOCUENTO_DE_IDENTIDAD",
}

remaining_sensitive_columns = (
    sensitive_columns.intersection(df_processed.columns)
)

if remaining_sensitive_columns:
    raise ValueError(
        "Direct identifiers remain in the analytical dataset: "
        f"{sorted(remaining_sensitive_columns)}"
    )

# Export the pseudonymized dataset
df_processed.to_csv(
    ANONYMIZED_FILE,
    index=False,
    encoding="utf-8-sig",
)

log_progress(
    f"Pseudonymized dataset saved to: {ANONYMIZED_FILE}"
)

# Summarize the processing results
print("\nPHASE 1 SUMMARY")
print(f"Input records: {len(df_raw):,}")
print(f"Output records: {len(df_processed):,}")
print(f"Output columns: {len(df_processed.columns):,}")
print(f"Distinct original student identifiers: {unique_id_count:,}")
print(f"Missing student identifiers: {missing_id_count:,}")
print(f"Missing student names: {missing_name_count:,}")
print(f"Exact duplicate rows: {duplicate_row_count:,}")
print(f"Output file: {ANONYMIZED_FILE}")
print("Direct student names removed from the analytical dataset.")

log_progress("Phase 1 completed successfully.")
