import json
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

# Configure artifact paths
OUTPUT_DIR = Path(OUTPUT_DIR)
MODEL_DIR = OUTPUT_DIR / "Models"

MODEL_PATH = MODEL_DIR / "ResNet_Attention.keras"
HISTORY_PATH = OUTPUT_DIR / "Phase5_Training_History.csv"
CONFIG_PATH = OUTPUT_DIR / "Phase5_Training_Config.json"
FEATURES_PATH = OUTPUT_DIR / "Selected_Features.csv"

VALIDATION_REPORT_PATH = (
    OUTPUT_DIR / "Phase6_Artifact_Validation.csv"
)

EXPECTED_FEATURES = [
    "AGE",
    "STRATUM",
    "CURRENT_LEVEL",
    "PROGRAM_DURATION",
    "ACADEMIC_PROGRESS",
]

log_progress(
    "Starting Phase 6: training artifact validation."
)

# Validate required files
required_files = {
    "Trained model": MODEL_PATH,
    "Training history": HISTORY_PATH,
    "Training configuration": CONFIG_PATH,
    "Selected features": FEATURES_PATH,
}

missing_files = [
    f"{description}: {path}"
    for description, path in required_files.items()
    if not path.is_file()
]

if missing_files:
    raise FileNotFoundError(
        "Required artifacts are missing:\n"
        + "\n".join(missing_files)
        + "\nRun Phase 5 successfully before continuing."
    )

print("All required artifacts were found.")

# Validate selected predictors
saved_features = (
    pd.read_csv(FEATURES_PATH)["FEATURE"]
    .astype(str)
    .tolist()
)

if saved_features != EXPECTED_FEATURES:
    raise ValueError(
        "The selected features do not match the project configuration. "
        f"Expected: {EXPECTED_FEATURES}; "
        f"found: {saved_features}."
    )

# Validate training configuration
with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
    training_config = json.load(config_file)

required_config_keys = [
    "model_name",
    "input_features",
    "input_dimension",
    "epochs_completed",
    "training_samples_after_smote",
    "validation_samples",
]

missing_config_keys = [
    key
    for key in required_config_keys
    if key not in training_config
]

if missing_config_keys:
    raise KeyError(
        "Training configuration is missing required fields: "
        f"{missing_config_keys}"
    )

if training_config["model_name"] != "ResNet_Attention":
    raise ValueError(
        "The configuration does not identify ResNet_Attention."
    )

if training_config["input_features"] != EXPECTED_FEATURES:
    raise ValueError(
        "The training configuration contains unexpected predictors."
    )

if training_config["input_dimension"] != len(EXPECTED_FEATURES):
    raise ValueError(
        "The configured input dimension does not match "
        "the five selected predictors."
    )

if training_config["epochs_completed"] < 1:
    raise ValueError(
        "The recorded number of training epochs is invalid."
    )

print("\nTraining configuration validated.")
print(f"Model: {training_config['model_name']}")
print(f"Predictors: {training_config['input_features']}")
print(f"Completed epochs: {training_config['epochs_completed']}")

# Validate training history
training_history = pd.read_csv(
    HISTORY_PATH,
    index_col=0,
)

required_history_columns = [
    "loss",
    "val_loss",
    "accuracy",
    "val_accuracy",
    "auc",
    "val_auc",
]

missing_history_columns = [
    column
    for column in required_history_columns
    if column not in training_history.columns
]

if missing_history_columns:
    raise KeyError(
        "The training history is missing required metrics: "
        f"{missing_history_columns}"
    )

if training_history.empty:
    raise ValueError(
        "The training history contains no recorded epochs."
    )

if len(training_history) != training_config["epochs_completed"]:
    raise ValueError(
        "The number of recorded epochs differs from "
        "the training configuration."
    )

history_values = training_history[
    required_history_columns
].to_numpy(dtype=float)

if not np.isfinite(history_values).all():
    raise ValueError(
        "The training history contains missing or non-finite values."
    )

best_epoch = int(
    np.argmin(training_history["val_loss"].to_numpy())
) + 1

best_val_loss = float(
    training_history["val_loss"].iloc[best_epoch - 1]
)

best_val_accuracy = float(
    training_history["val_accuracy"].iloc[best_epoch - 1]
)

best_val_auc = float(
    training_history["val_auc"].iloc[best_epoch - 1]
)

print("\nTraining history validated.")
print(f"Recorded epochs: {len(training_history)}")
print(f"Best validation-loss epoch: {best_epoch}")
print(f"Minimum validation loss: {best_val_loss:.4f}")
print(f"Validation accuracy at best epoch: {best_val_accuracy:.4f}")
print(f"Validation AUC at best epoch: {best_val_auc:.4f}")

# Validate the saved model architecture
saved_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

model_input_dim = saved_model.input_shape[-1]
model_output_dim = saved_model.output_shape[-1]

if model_input_dim != len(EXPECTED_FEATURES):
    raise ValueError(
        "The saved model input dimension does not match "
        f"the expected {len(EXPECTED_FEATURES)} predictors. "
        f"Found: {model_input_dim}."
    )

if model_output_dim != 1:
    raise ValueError(
        "The saved model must have one output for binary classification."
    )

print("\nSaved model validated.")
print(f"Model name: {saved_model.name}")
print(f"Input shape: {saved_model.input_shape}")
print(f"Output shape: {saved_model.output_shape}")

# Create the validation report
validation_summary = pd.DataFrame([
    {
        "CHECK": "Required artifacts",
        "STATUS": "PASS",
        "DETAILS": "All required Phase 5 files exist.",
    },
    {
        "CHECK": "Selected predictors",
        "STATUS": "PASS",
        "DETAILS": (
            f"{len(EXPECTED_FEATURES)} predictors match "
            "the project configuration."
        ),
    },
    {
        "CHECK": "Training configuration",
        "STATUS": "PASS",
        "DETAILS": "Required configuration fields are valid.",
    },
    {
        "CHECK": "Training history",
        "STATUS": "PASS",
        "DETAILS": (
            f"{len(training_history)} epochs validated; "
            f"best validation-loss epoch: {best_epoch}."
        ),
    },
    {
        "CHECK": "Saved model architecture",
        "STATUS": "PASS",
        "DETAILS": (
            f"Input dimension: {model_input_dim}; "
            f"output dimension: {model_output_dim}."
        ),
    },
])

validation_summary.to_csv(
    VALIDATION_REPORT_PATH,
    index=False,
    encoding="utf-8-sig",
)

# Save validation metadata
validation_metadata = {
    "model_name": training_config["model_name"],
    "selected_features": EXPECTED_FEATURES,
    "number_of_features": len(EXPECTED_FEATURES),
    "epochs_completed": int(len(training_history)),
    "best_epoch_by_validation_loss": best_epoch,
    "minimum_validation_loss": best_val_loss,
    "validation_accuracy_at_best_epoch": best_val_accuracy,
    "validation_auc_at_best_epoch": best_val_auc,
    "model_input_dimension": int(model_input_dim),
    "model_output_dimension": int(model_output_dim),
    "validation_status": "PASS",
}

METADATA_PATH = OUTPUT_DIR / "Phase6_Validation_Metadata.json"

with open(METADATA_PATH, "w", encoding="utf-8") as metadata_file:
    json.dump(
        validation_metadata,
        metadata_file,
        indent=4,
        ensure_ascii=False,
    )

print("\nArtifact validation summary:")
print(validation_summary.to_string(index=False))

print("\nSaved outputs:")
print(f"Validation report: {VALIDATION_REPORT_PATH}")
print(f"Validation metadata: {METADATA_PATH}")

log_progress(
    "Phase 6 completed: training artifacts, configuration, "
    "history, and model architecture validated successfully."
)
