 DATA INGESTION ANONYMIZATION AND STRUCTURAL CLEANING

RAW_DATA_PATH = "DATA.csv"
MASTER_KEY_PATH = os.path.join(OUTPUT_DIR, "Master_Key.xlsx")
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "Anonymized_Dataset.csv")

df_raw = pd.read_csv(RAW_DATA_PATH, sep=";", encoding="latin-1", low_memory=False)

df_raw.columns = (
    df_raw.columns
    .str.strip()
    .str.upper()
    .str.replace(" ", "_")
    .str.replace(r"[^A-Z0-9_]", "", regex=True)
)

column_mapping = {
    "DOCUENTO_DE_IDENTIDAD": "STUDENT_ID",
    "TIPO_DOC": "DOCUMENT_TYPE",
    "NOMBRE_COMPLETO": "FULL_NAME",
    "PERIODO_INGRESO": "ENTRY_PERIOD",
    "PERIODO_ACTUAL": "CURRENT_PERIOD",
    "FECHA_NACIMIENTO": "BIRTH_DATE",
    "EDAD": "AGE",
    "GNERO__SEX": "GENDER",
    "ESTADO_CIVIL": "CIVIL_STATUS",
    "TIPOESTADO_ACADMICO": "ACADEMIC_STATUS_TYPE",
    "ESTRATO": "STRATUM",
    "PROGRAMA_ACADMICO": "ACADEMIC_PROGRAM",
    "NIVEL_ACTUAL": "CURRENT_LEVEL",
    "SNIES": "SNIES_CODE",
    "SEMESTRES_TOTAL_POR_CARRERA": "PROGRAM_DURATION",
    "NOMBRE_SEDE": "CAMPUS_NAME",
    "IMPREB": "IMPREB"
}

df_raw = df_raw.rename(columns=column_mapping)

IDENT_COLS = ["STUDENT_ID", "FULL_NAME"]
if not all(col in df_raw.columns for col in IDENT_COLS):
    raise KeyError("Essential institutional identification columns are missing.")

master_key = df_raw[IDENT_COLS].copy()
master_key.columns = ["ORIGINAL_ID", "ORIGINAL_NAME"]

def apply_sha256(val):
    return hashlib.sha256(str(val).strip().encode("utf-8")).hexdigest()

for col in IDENT_COLS:
    df_raw[col] = df_raw[col].astype(str).str.strip().apply(apply_sha256)

master_key["ANONYMIZED_ID"] = df_raw["STUDENT_ID"]
master_key["ANONYMIZED_NAME"] = df_raw["FULL_NAME"]

master_key.to_excel(MASTER_KEY_PATH, index=False)
df_raw.to_csv(CLEANED_DATA_PATH, sep=";", encoding="utf-8", index=False)

log_progress("Phase 1 completed: Data loaded, translated, and SHA-256 anonymized.")
