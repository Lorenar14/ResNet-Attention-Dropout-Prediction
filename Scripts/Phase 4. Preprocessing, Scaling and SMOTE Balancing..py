from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTENC
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

BASE_DIR = Path("/content")
OUTPUT_DIR = BASE_DIR / "Results"
FIGURE_DIR = OUTPUT_DIR / "Figures"
INPUT_FILE = OUTPUT_DIR / "Modeling_Dataset.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = globals().get("RANDOM_SEED", 42)

SELECTED_FEATURES = [
    "AGE",
    "STRATUM",
    "CURRENT_LEVEL",
    "PROGRAM_DURATION",
    "ACADEMIC_PROGRESS",
]

TARGET_COLUMN = "TARGET"
CATEGORICAL_FEATURES = ["STRATUM"]

PALETTE_BLUE_DARK = "#17365D"
PALETTE_RED_DARK = "#8B1E2D"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Modeling dataset not found: {INPUT_FILE}. "
        "Run Phase 2 before Phase 4."
    )

# Load and standardize the modeling dataset
df = pd.read_csv(
    INPUT_FILE,
    sep=";",
    encoding="utf-8-sig",
    low_memory=False,
)

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.upper()
)

# Support known alternative names for academic progress
column_aliases = {
    "ACADEMIC_PROGRESS": [
        "ACADEMIC_PROGRESS",
        "ACADEMIC_ADVANCE",
        "AVANCE_ACADEMICO",
        "AVANCE_NU",
    ],
    "TARGET": [
        "TARGET",
        "TARGET_COLUMN",
    ],
}

for standard_name, aliases in column_aliases.items():
    if standard_name not in df.columns:
        matching_column = next(
            (
                alias for alias in aliases
                if alias in df.columns
            ),
            None,
        )

        if matching_column is not None:
            df = df.rename(
                columns={matching_column: standard_name}
            )

missing_columns = [
    column
    for column in SELECTED_FEATURES + [TARGET_COLUMN]
    if column not in df.columns
]

if missing_columns:
    raise KeyError(
        f"Required columns are missing: {missing_columns}. "
        f"Available columns: {df.columns.tolist()}"
    )

if df.empty:
    raise ValueError("The modeling dataset is empty.")

# Prepare predictors and target
X_data = df[SELECTED_FEATURES].copy()
y_data = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce",
)

if y_data.isna().any():
    raise ValueError(
        "TARGET contains missing or invalid values."
    )

if not set(y_data.unique()).issubset({0, 1}):
    raise ValueError(
        "TARGET must contain only binary values 0 and 1."
    )

if y_data.nunique() != 2:
    raise ValueError(
        "Both target classes must be present."
    )

y_data = y_data.astype("int8")

# Convert predictor values to numeric
for column in SELECTED_FEATURES:
    X_data[column] = pd.to_numeric(
        X_data[column],
        errors="coerce",
    )

X_data = X_data.replace([np.inf, -np.inf], np.nan)

# Validate categorical socioeconomic strata
valid_stratum = X_data["STRATUM"].dropna()

if not valid_stratum.empty:
    if not np.allclose(
        valid_stratum,
        np.round(valid_stratum),
    ):
        raise ValueError(
            "STRATUM contains non-integer values. "
            "Review the source data before applying SMOTENC."
        )

# Reset indices to avoid alignment errors
X_data = X_data.reset_index(drop=True)
y_data = y_data.reset_index(drop=True)

print("Modeling dataset loaded.")
print(f"Records: {len(X_data):,}")
print(f"Predictors: {len(SELECTED_FEATURES)}")
print(f"Target distribution:\n{y_data.value_counts().sort_index()}")

# Split data into development and independent test sets
X_dev, X_test_raw, y_dev, y_test = train_test_split(
    X_data,
    y_data,
    test_size=0.20,
    stratify=y_data,
    random_state=RANDOM_SEED,
)

# Split development data into training and validation sets
X_train_raw, X_val_raw, y_train_raw, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    stratify=y_dev,
    random_state=RANDOM_SEED,
)

print("\nData partitioning completed.")
print(f"Training before SMOTE: {len(X_train_raw):,}")
print(f"Validation: {len(X_val_raw):,}")
print(f"Independent test: {len(X_test_raw):,}")

# Impute numeric predictors using training data only
numeric_features = [
    feature
    for feature in SELECTED_FEATURES
    if feature not in CATEGORICAL_FEATURES
]

numeric_imputer = SimpleImputer(strategy="median")

X_train_numeric = pd.DataFrame(
    numeric_imputer.fit_transform(
        X_train_raw[numeric_features]
    ),
    columns=numeric_features,
    index=X_train_raw.index,
)

X_val_numeric = pd.DataFrame(
    numeric_imputer.transform(
        X_val_raw[numeric_features]
    ),
    columns=numeric_features,
    index=X_val_raw.index,
)

X_test_numeric = pd.DataFrame(
    numeric_imputer.transform(
        X_test_raw[numeric_features]
    ),
    columns=numeric_features,
    index=X_test_raw.index,
)

# Impute the categorical socioeconomic stratum
stratum_imputer = SimpleImputer(strategy="most_frequent")

X_train_stratum = stratum_imputer.fit_transform(
    X_train_raw[CATEGORICAL_FEATURES]
)

X_val_stratum = stratum_imputer.transform(
    X_val_raw[CATEGORICAL_FEATURES]
)

X_test_stratum = stratum_imputer.transform(
    X_test_raw[CATEGORICAL_FEATURES]
)

# Scale numeric predictors using training data only
feature_scaler = RobustScaler()

X_train_numeric_scaled = pd.DataFrame(
    feature_scaler.fit_transform(X_train_numeric),
    columns=numeric_features,
    index=X_train_raw.index,
)

X_val_numeric_scaled = pd.DataFrame(
    feature_scaler.transform(X_val_numeric),
    columns=numeric_features,
    index=X_val_raw.index,
)

X_test_numeric_scaled = pd.DataFrame(
    feature_scaler.transform(X_test_numeric),
    columns=numeric_features,
    index=X_test_raw.index,
)

# Reconstruct feature matrices in the original feature order
def assemble_features(numeric_data, stratum_data, index):
    feature_data = numeric_data.copy()

    feature_data["STRATUM"] = np.asarray(
        stratum_data
    ).reshape(-1)

    feature_data = feature_data[SELECTED_FEATURES]
    feature_data.index = index

    return feature_data


X_train_prepared = assemble_features(
    X_train_numeric_scaled,
    X_train_stratum,
    X_train_raw.index,
)

X_val_prepared = assemble_features(
    X_val_numeric_scaled,
    X_val_stratum,
    X_val_raw.index,
)

X_test_prepared = assemble_features(
    X_test_numeric_scaled,
    X_test_stratum,
    X_test_raw.index,
)

# Validate prepared data
for partition_name, partition in [
    ("training", X_train_prepared),
    ("validation", X_val_prepared),
    ("test", X_test_prepared),
]:
    if not np.isfinite(
        partition.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"Non-finite values detected in {partition_name}."
        )

# Apply SMOTENC to training data only
class_counts_before = (
    y_train_raw.value_counts().sort_index()
)

if class_counts_before.min() <= 5:
    raise ValueError(
        "SMOTENC with k_neighbors=5 requires at least "
        "six observations in each training class."
    )

categorical_indices = [
    SELECTED_FEATURES.index(feature)
    for feature in CATEGORICAL_FEATURES
]

smote_sampler = SMOTENC(
    categorical_features=categorical_indices,
    sampling_strategy="auto",
    random_state=RANDOM_SEED,
    k_neighbors=5,
)

X_train_resampled, y_train_resampled = (
    smote_sampler.fit_resample(
        X_train_prepared,
        y_train_raw,
    )
)

X_train = np.asarray(
    X_train_resampled,
    dtype=np.float32,
)

y_train = np.asarray(
    y_train_resampled,
    dtype=np.int32,
)

X_val = X_val_prepared.to_numpy(dtype=np.float32)
y_val = y_val.to_numpy(dtype=np.int32)

X_test = X_test_prepared.to_numpy(dtype=np.float32)
y_test = y_test.to_numpy(dtype=np.int32)

# Verify that SMOTENC preserved valid socioeconomic categories
valid_strata = set(
    X_train_prepared["STRATUM"].astype(int).unique()
)

resampled_strata = set(
    X_train_resampled["STRATUM"].astype(int).unique()
)

if not resampled_strata.issubset(valid_strata):
    raise RuntimeError(
        "SMOTENC generated unexpected socioeconomic categories."
    )

class_counts_after = pd.Series(y_train).value_counts()

if class_counts_after.nunique() != 1:
    raise RuntimeError(
        "Training classes are not balanced after SMOTENC."
    )

print("\nSMOTENC applied to the training partition only.")

# Create a supplementary figure showing class balance
fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 4.5),
    sharey=True,
)

plot_data = [
    (
        axes[0],
        y_train_raw.to_numpy(),
        "Before SMOTENC",
    ),
    (
        axes[1],
        y_train,
        "After SMOTENC",
    ),
]

for ax, labels, panel_title in plot_data:
    counts = (
        pd.Series(labels)
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    bars = ax.bar(
        ["Non-dropout", "Operational dropout"],
        counts.values,
        color=[
            PALETTE_BLUE_DARK,
            PALETTE_RED_DARK,
        ],
        width=0.62,
    )

    ax.set_title(panel_title)
    ax.set_ylabel("Number of training observations")
    ax.tick_params(axis="x", rotation=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    for bar in bars:
        ax.annotate(
            f"{int(bar.get_height()):,}",
            (
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

fig.suptitle(
    "Supplementary Figure S2. Training Class Balance Before and After SMOTENC",
    fontsize=12,
    fontweight="bold",
)

fig.tight_layout(rect=[0, 0, 1, 0.92])

fig.savefig(
    FIGURE_DIR / "Supplementary_S2_SMOTENC_Class_Balance.pdf",
    format="pdf",
    bbox_inches="tight",
)

fig.savefig(
    FIGURE_DIR / "Supplementary_S2_SMOTENC_Class_Balance.png",
    dpi=600,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

# Export partition summary
partition_summary = pd.DataFrame([
    {
        "PARTITION": "Training before SMOTENC",
        "N_SAMPLES": len(y_train_raw),
        "NON_DROPOUT": int((y_train_raw == 0).sum()),
        "OPERATIONAL_DROPOUT": int((y_train_raw == 1).sum()),
    },
    {
        "PARTITION": "Training after SMOTENC",
        "N_SAMPLES": len(y_train),
        "NON_DROPOUT": int((y_train == 0).sum()),
        "OPERATIONAL_DROPOUT": int((y_train == 1).sum()),
    },
    {
        "PARTITION": "Validation",
        "N_SAMPLES": len(y_val),
        "NON_DROPOUT": int((y_val == 0).sum()),
        "OPERATIONAL_DROPOUT": int((y_val == 1).sum()),
    },
    {
        "PARTITION": "Independent test",
        "N_SAMPLES": len(y_test),
        "NON_DROPOUT": int((y_test == 0).sum()),
        "OPERATIONAL_DROPOUT": int((y_test == 1).sum()),
    },
])

partition_summary.to_csv(
    OUTPUT_DIR / "Phase4_Partition_Summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# Save preprocessing objects and feature order
joblib.dump(
    numeric_imputer,
    OUTPUT_DIR / "Numeric_Imputer.joblib",
)

joblib.dump(
    stratum_imputer,
    OUTPUT_DIR / "Stratum_Imputer.joblib",
)

joblib.dump(
    feature_scaler,
    OUTPUT_DIR / "RobustScaler.joblib",
)

joblib.dump(
    smote_sampler,
    OUTPUT_DIR / "SMOTENC_Sampler.joblib",
)

joblib.dump(
    SELECTED_FEATURES,
    OUTPUT_DIR / "Selected_Features.joblib",
)

pd.Series(
    SELECTED_FEATURES,
    name="FEATURE",
).to_csv(
    OUTPUT_DIR / "Selected_Features.csv",
    index=False,
    encoding="utf-8-sig",
)

# Report phase results
print("\nPHASE 4: PREPROCESSING, SCALING AND SMOTE BALANCING")

print("\nPartition summary:")
print(partition_summary.to_string(index=False))

print("\nTraining class distribution before SMOTENC:")
print(class_counts_before.to_string())

print("\nTraining class distribution after SMOTENC:")
print(pd.Series(y_train).value_counts().sort_index().to_string())

print("\nValidation class distribution:")
print(pd.Series(y_val).value_counts().sort_index().to_string())

print("\nIndependent test class distribution:")
print(pd.Series(y_test).value_counts().sort_index().to_string())

print("\nSelected predictors:")
for feature in SELECTED_FEATURES:
    print(f"- {feature}")

print("\nFinal array shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")

print("\nSaved outputs:")
print(OUTPUT_DIR / "Phase4_Partition_Summary.csv")
print(FIGURE_DIR / "Supplementary_S2_SMOTENC_Class_Balance.pdf")
print(OUTPUT_DIR / "Numeric_Imputer.joblib")
print(OUTPUT_DIR / "Stratum_Imputer.joblib")
print(OUTPUT_DIR / "RobustScaler.joblib")
print(OUTPUT_DIR / "SMOTENC_Sampler.joblib")
print(OUTPUT_DIR / "Selected_Features.joblib")

print("\nPhase 4 completed successfully.")
