import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)

from xgboost import XGBClassifier


# 9.1. Configuration

DECISION_THRESHOLD = 0.50
THRESHOLD_SOURCE = "Fixed threshold applied to all models"
RANDOM_STATE = int(RANDOM_SEED)

OUTPUT_DIR = OUTPUT_DIR
MODEL_DIR = OUTPUT_DIR / "Models"
PHASE9_DIR = OUTPUT_DIR / "Phase9_Model_Comparison"
FIGURE_DIR = OUTPUT_DIR / "Figures"

PHASE9_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_PATH = PHASE9_DIR / "Phase9_Model_Comparison_Fixed_Threshold.csv"
PREDICTIONS_PATH = PHASE9_DIR / "Phase9_Test_Predictions_Fixed_Threshold.csv"
CONFIG_PATH = PHASE9_DIR / "Phase9_Comparison_Config_Fixed_Threshold.json"

FIGURE_PDF_PATH = FIGURE_DIR / "Figure_12_Model_Performance_Comparison_Fixed_Threshold.pdf"
FIGURE_PNG_PATH = FIGURE_DIR / "Figure_12_Model_Performance_Comparison_Fixed_Threshold.png"

MODEL_PATH = MODEL_DIR / "ResNet_Attention.keras"

EXPECTED_FEATURES = [
    "AGE",
    "STRATUM",
    "CURRENT_LEVEL",
    "PROGRAM_DURATION",
    "ACADEMIC_PROGRESS",
]

log_progress(
    "Starting Phase 9: comparative evaluation using a fixed threshold."
)


# 9.2. Validate data and model artifacts

required_objects = [
    "X_train",
    "y_train",
    "X_val",
    "y_val",
    "X_test",
    "y_test",
    "neural_model",
    "RANDOM_SEED",
    "PALETTE_BLUE_DARK",
    "PALETTE_RED_DARK",
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        f"Required objects are missing: {missing_objects}. "
        "Run the preceding phases first."
    )

X_train_array = np.asarray(X_train, dtype=np.float32)
y_train_array = np.asarray(y_train).reshape(-1).astype(int)

X_val_array = np.asarray(X_val, dtype=np.float32)
y_val_array = np.asarray(y_val).reshape(-1).astype(int)

X_test_array = np.asarray(X_test, dtype=np.float32)
y_test_array = np.asarray(y_test).reshape(-1).astype(int)

if not (
    X_train_array.ndim == 2
    and X_val_array.ndim == 2
    and X_test_array.ndim == 2
):
    raise ValueError("All feature matrices must be two-dimensional.")

if not (
    X_train_array.shape[1]
    == X_val_array.shape[1]
    == X_test_array.shape[1]
    == len(EXPECTED_FEATURES)
):
    raise ValueError(
        "The training, validation, and test matrices must contain "
        f"the same {len(EXPECTED_FEATURES)} predictors."
    )

if not (
    len(X_train_array) == len(y_train_array)
    and len(X_val_array) == len(y_val_array)
    and len(X_test_array) == len(y_test_array)
):
    raise ValueError(
        "Feature matrices and target arrays have inconsistent lengths."
    )

for split_name, features, labels in [
    ("training", X_train_array, y_train_array),
    ("validation", X_val_array, y_val_array),
    ("test", X_test_array, y_test_array),
]:
    if not np.isfinite(features).all():
        raise ValueError(
            f"The {split_name} matrix contains missing or infinite values."
        )

    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(
            f"The {split_name} target must contain both classes, 0 and 1."
        )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"The trained ResNet-Attention model was not found: {MODEL_PATH}"
    )

if not np.isclose(DECISION_THRESHOLD, 0.50):
    raise ValueError("The fixed decision threshold must be 0.50.")

print(f"Training records: {len(y_train_array):,}")
print(f"Validation records: {len(y_val_array):,}")
print(f"Independent test records: {len(y_test_array):,}")
print(f"Number of predictors: {len(EXPECTED_FEATURES)}")
print(f"Fixed decision threshold: {DECISION_THRESHOLD:.2f}")


# 9.3. Define the baseline models

baseline_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# 9.4. Train baseline models and generate test probabilities

model_probabilities = {}
model_predictions = {}

for model_name, model in baseline_models.items():
    log_progress(f"Training baseline model: {model_name}")

    model.fit(X_train_array, y_train_array)

    probabilities = model.predict_proba(X_test_array)[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"{model_name} generated invalid prediction probabilities."
        )

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(
            f"{model_name} generated probabilities outside [0, 1]."
        )

    model_probabilities[model_name] = probabilities

    model_predictions[model_name] = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    print(f"{model_name}: evaluation completed.")


# 9.5. Evaluate the trained ResNet-Attention model

resnet_probabilities = neural_model.predict(
    X_test_array,
    verbose=0,
).reshape(-1)

if len(resnet_probabilities) != len(y_test_array):
    raise ValueError(
        "ResNet-Attention predictions do not match the test sample count."
    )

if (
    not np.isfinite(resnet_probabilities).all()
    or (resnet_probabilities < 0).any()
    or (resnet_probabilities > 1).any()
):
    raise ValueError(
        "ResNet-Attention generated invalid prediction probabilities."
    )

model_probabilities["ResNet-Attention"] = resnet_probabilities

model_predictions["ResNet-Attention"] = (
    resnet_probabilities >= DECISION_THRESHOLD
).astype(int)


# 9.6. Calculate evaluation metrics at the same threshold

comparison_records = []
prediction_records = []

for model_name, probabilities in model_probabilities.items():

    predicted_labels = model_predictions[model_name]

    tn, fp, fn, tp = confusion_matrix(
        y_test_array,
        predicted_labels,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {
        "Model": model_name,
        "Decision_Threshold": DECISION_THRESHOLD,
        "Threshold_Source": THRESHOLD_SOURCE,
        "Test_Samples": len(y_test_array),
        "Accuracy": accuracy_score(
            y_test_array, predicted_labels
        ),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_test_array, predicted_labels
        ),
        "Precision": precision_score(
            y_test_array,
            predicted_labels,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test_array,
            predicted_labels,
            zero_division=0,
        ),
        "Specificity": specificity,
        "F1_Score": f1_score(
            y_test_array,
            predicted_labels,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y_test_array,
            probabilities,
        ),
        "Average_Precision": average_precision_score(
            y_test_array,
            probabilities,
        ),
        "MCC": matthews_corrcoef(
            y_test_array,
            predicted_labels,
        ),
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
    }

    comparison_records.append(metrics)

    prediction_records.append(
        pd.DataFrame({
            "MODEL": model_name,
            "TRUE_LABEL": y_test_array,
            "PREDICTED_LABEL": predicted_labels,
            "DROPOUT_PROBABILITY": probabilities,
            "DECISION_THRESHOLD": DECISION_THRESHOLD,
        })
    )

comparison_df = pd.DataFrame(comparison_records)

comparison_df = comparison_df.sort_values(
    "ROC_AUC",
    ascending=False,
).reset_index(drop=True)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False,
    encoding="utf-8",
    float_format="%.6f",
)

predictions_df = pd.concat(
    prediction_records,
    ignore_index=True,
)

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
    encoding="utf-8",
    float_format="%.6f",
)

print("\nPHASE 9: COMPARATIVE MODEL EVALUATION")
print("Fixed decision threshold: 0.50\n")

display_columns = [
    "Model",
    "Decision_Threshold",
    "Accuracy",
    "Balanced_Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1_Score",
    "ROC_AUC",
    "Average_Precision",
    "MCC",
]

print(
    comparison_df[display_columns]
    .round(4)
    .to_string(index=False)
)


# 9.7. Generate the comparative performance figure

plot_metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC",
    "Average_Precision",
]

plot_labels = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC",
    "Average Precision",
]

model_order = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "ResNet-Attention",
]

plot_df = (
    comparison_df
    .set_index("Model")
    .reindex(model_order)
)

x = np.arange(len(plot_metrics))
bar_width = 0.19

fig, ax = plt.subplots(figsize=(11, 6))

for index, model_name in enumerate(model_order):

    values = plot_df.loc[model_name, plot_metrics].to_numpy(
        dtype=float
    )

    if model_name == "ResNet-Attention":
        color = PALETTE_RED_DARK
        hatch = "//"
    else:
        color = PALETTE_BLUE_DARK
        hatch = None

    ax.bar(
        x + (index - 1.5) * bar_width,
        values,
        width=bar_width,
        label=model_name,
        color=color,
        hatch=hatch,
        edgecolor="black",
        linewidth=0.5,
    )

ax.set_ylabel("Score")
ax.set_xlabel("Evaluation Metric")
ax.set_title(
    "Comparative Model Performance at a Fixed Threshold of 0.50",
    pad=14,
    fontweight="bold",
)

ax.set_xticks(x)
ax.set_xticklabels(plot_labels)
ax.set_ylim(0, 1.05)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.25,
)

ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="lower right",
    frameon=False,
)

fig.tight_layout()

fig.savefig(
    FIGURE_PDF_PATH,
    format="pdf",
    bbox_inches="tight",
)

fig.savefig(
    FIGURE_PNG_PATH,
    format="png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# 9.8. Save experimental configuration

config = {
    "phase": 9,
    "purpose": "Comparative model evaluation with a common fixed threshold",
    "decision_threshold": DECISION_THRESHOLD,
    "threshold_source": THRESHOLD_SOURCE,
    "models": model_order,
    "predictors": EXPECTED_FEATURES,
    "training_samples": int(len(y_train_array)),
    "validation_samples": int(len(y_val_array)),
    "independent_test_samples": int(len(y_test_array)),
    "test_set_used_for_threshold_selection": False,
    "smote_applied_to_validation": False,
    "smote_applied_to_test": False,
    "evaluation_note": (
        "All models use a common probability threshold of 0.50 "
        "for threshold-dependent classification metrics. "
        "ROC-AUC and Average Precision are calculated from predicted "
        "probabilities and do not depend on this threshold."
    ),
    "target_interpretation": (
        "Operational dropout label derived from administrative rules; "
        "not independently verified dropout."
    ),
    "outputs": [
        str(COMPARISON_PATH),
        str(PREDICTIONS_PATH),
        str(FIGURE_PDF_PATH),
        str(FIGURE_PNG_PATH),
    ],
}

with open(CONFIG_PATH, "w", encoding="utf-8") as config_file:
    json.dump(
        config,
        config_file,
        indent=4,
        ensure_ascii=False,
    )


# 9.9. Validate generated artifacts

generated_files = [
    COMPARISON_PATH,
    PREDICTIONS_PATH,
    FIGURE_PDF_PATH,
    FIGURE_PNG_PATH,
    CONFIG_PATH,
]

for output_path in generated_files:
    if not output_path.exists():
        raise FileNotFoundError(
            f"Expected output was not generated: {output_path}"
        )

print("\nGenerated files:")

for output_path in generated_files:
    print(output_path)

log_progress(
    "Phase 9 completed: all models evaluated using a fixed threshold of 0.50."
)

print("\nPhase 9 completed successfully.")
