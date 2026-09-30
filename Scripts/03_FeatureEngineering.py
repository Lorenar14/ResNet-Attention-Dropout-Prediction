 BUSINESS RULES AND TARGET DEFINITION

df = df_raw.copy()
df = df.dropna(subset=["STUDENT_ID", "ENTRY_PERIOD"]).copy()

df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
df["CURRENT_LEVEL"] = pd.to_numeric(df["CURRENT_LEVEL"], errors="coerce")
df["PROGRAM_DURATION"] = pd.to_numeric(df["PROGRAM_DURATION"], errors="coerce")
df["STRATUM"] = pd.to_numeric(df["STRATUM"], errors="coerce")

df = df[df["AGE"].isna() | (df["AGE"] >= 15)].copy()

df["ACADEMIC_PROGRESS"] = np.where(
    df["PROGRAM_DURATION"] > 0,
    (df["CURRENT_LEVEL"] / df["PROGRAM_DURATION"]) * 100,
    np.nan
).clip(0, 100).round(2)

df = (
    df.sort_values(["STUDENT_ID", "CURRENT_LEVEL"], ascending=[True, False])
      .drop_duplicates(subset="STUDENT_ID", keep="first")
      .copy()
)

TARGET_PERIOD = "2026-1S"

def period_to_sequence(period_str):
    try:
        p = str(period_str).strip().upper()
        return (int(p[:4]) * 2) + (1 if "-1S" in p else 2)
    except (ValueError, TypeError):
        return np.nan

df["ENTRY_SEMESTER_SEQ"] = df["ENTRY_PERIOD"].apply(period_to_sequence)
active_period_seq = period_to_sequence(TARGET_PERIOD)

def evaluate_status(row):
    lvl, dur, entry = row["CURRENT_LEVEL"], row["PROGRAM_DURATION"], row["ENTRY_SEMESTER_SEQ"]
    if pd.notna(lvl) and pd.notna(dur) and dur > 0 and lvl >= dur:
        return "Graduated"
    if pd.notna(lvl) and pd.notna(entry) and pd.notna(active_period_seq):
        if (active_period_seq - entry) - lvl >= 2:
            return "Dropout"
    return "Active"

df["ACADEMIC_STATUS"] = df.apply(evaluate_status, axis=1)
df["target"] = df["ACADEMIC_STATUS"].eq("Dropout").astype(int)

log_progress("Phase 2 completed: Institutional rules and target variable computed.")
