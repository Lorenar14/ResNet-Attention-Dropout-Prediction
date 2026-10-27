import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

# 8.1. Configure paths and visualization settings

OUTPUT_DIR = Path("/content/Results")
MODEL_DIR = OUTPUT_DIR / "Models"
FIGURE_DIR = OUTPUT_DIR / "Figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "ResNet_Attention.keras"
FEATURES_PATH = OUTPUT_DIR / "Selected_Features.csv"

METRICS_PATH = OUTPUT_DIR / "Phase8_Model_Performance.csv"
REPORT_PATH = OUTPUT_DIR / "Phase8_Classification_Report.csv"
CONFIG_PATH = OUTPUT_DIR / "Phase8_Evaluation_Config.json"
PREDICTIONS_PATH = OUTPUT_DIR / "Phase8_Test_Predictions.csv"

CONFUSION_MATRIX_PATH = FIGURE_DIR / "Figure_09_Confusion_Matrix"
ROC_CURVE_PATH = FIGURE_DIR / "Figure_10_ROC_Curve"
PR_CURVE_PATH = FIGURE_DIR / "Figure_11_Precision_Recall_Curve"

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
    "ps.fonttype": 42,
})

log_progress("Starting Phase 8: model performance evaluation.")

# 8.2. Validate required files

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. "
        "Run Phase 5 before Phase 8."
    )

if not FEATURES_PATH.is_file():
    raise FileNotFoundError(
        f"Selected features file not found: {FEATURES_PATH}."
    )

if "X_test" not in globals() or "y_test" not in globals():
    raise RuntimeError(
        "X_test and y_test are not available. "
        "Run Phase 4 before Phase 8."
    )

EXPECTED_FEATURES = [
    "AGE",
    "STRATUM",
    "CURRENT_LEVEL",
    "PROGRAM_DURATION",
    "ACADEMIC_PROGRESS",
]

SELECTED_FEATURES = (
    pd.read_csv(FEATURES_PATH)["FEATURE"]
    .astype(str)
    .tolist()
)

if SELECTED_FEATURES != EXPECTED_FEATURES:
    raise ValueError(
        "The selected features do not match the project configuration. "
        f"Expected: {EXPECTED_FEATURES}; "
        f"found: {SELECTED_FEATURES}."
    )

# 8.3. Validate the independent test set

X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test).reshape(-1)

if X_test.ndim != 2:
    raise ValueError("X_test must be a two-dimensional array.")

if X_test.shape[1] != len(SELECTED_FEATURES):
    raise ValueError(
        f"Expected {len(SELECTED_FEATURES)} predictors, "
        f"but found {X_test.shape[1]}."
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "X_test and y_test have different numbers of observations."
    )

if len(y_test) == 0:
    raise ValueError("The independent test set is empty.")

if not np.isfinite(X_test).all():
    raise ValueError("X_test contains non-finite values.")

if not np.isin(y_test, [0, 1]).all():
    raise ValueError("The target must contain only classes 0 and 1.")

if len(np.unique(y_test)) != 2:
    raise ValueError(
        "The independent test set must contain both target classes."
    )

y_test = y_test.astype(np.int32)

print("Independent test set validated.")
print(f"Test observations: {len(y_test):,}")
print(f"Number of predictors: {X_test.shape[1]}")
print(f"Predictors: {SELECTED_FEATURES}")

# 8.4. Load the trained model

neural_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

if neural_model.input_shape[-1] != X_test.shape[1]:
    raise ValueError(
        "The model input dimension does not match the test data."
    )

if neural_model.output_shape[-1] != 1:
    raise ValueError(
        "The model must produce one output for binary classification."
    )

print("\nResNet-Attention model loaded successfully.")

# 8.5. Set the classification threshold

OPTIMAL_THRESHOLD = 0.50
THRESHOLD_SOURCE = "Fixed threshold; no threshold optimization applied"

print(f"Classification threshold: {OPTIMAL_THRESHOLD:.2f}")
print(f"Threshold source: {THRESHOLD_SOURCE}")

# 8.6. Generate predictions on the independent test set

y_probability = neural_model.predict(
    X_test,
    batch_size=256,
    verbose=1,
).reshape(-1)

if len(y_probability) != len(y_test):
    raise ValueError(
        "The number of predicted probabilities differs from "
        "the number of test observations."
    )

if not np.isfinite(y_probability).all():
    raise ValueError(
        "The model generated non-finite prediction probabilities."
    )

if np.any((y_probability < 0) | (y_probability > 1)):
    raise ValueError(
        "Predicted probabilities must be between 0 and 1."
    )

y_pred = (
    y_probability >= OPTIMAL_THRESHOLD
).astype(np.int32)

# 8.7. Calculate performance metrics

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred,
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0,
)

specificity_matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
)

tn, fp, fn, tp = specificity_matrix.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0,
)

auc_roc = roc_auc_score(
    y_test,
    y_probability,
)

average_precision = average_precision_score(
    y_test,
    y_probability,
)

negative_predictive_value = (
    tn / (tn + fn)
    if (tn + fn) > 0
    else 0.0
)

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0.0
)

false_negative_rate = (
    fn / (fn + tp)
    if (fn + tp) > 0
    else 0.0
)

mcc = matthews_corrcoef(
    y_test,
    y_pred,
)

metrics = {
    "N_TEST": int(len(y_test)),
    "THRESHOLD": OPTIMAL_THRESHOLD,
    "ACCURACY": accuracy,
    "BALANCED_ACCURACY": balanced_accuracy,
    "PRECISION": precision,
    "RECALL": recall,
    "SPECIFICITY": specificity,
    "F1_SCORE": f1,
    "AUC_ROC": auc_roc,
    "AVERAGE_PRECISION": average_precision,
    "NEGATIVE_PREDICTIVE_VALUE": negative_predictive_value,
    "FALSE_POSITIVE_RATE": false_positive_rate,
    "FALSE_NEGATIVE_RATE": false_negative_rate,
    "MATTHEWS_CORRELATION_COEFFICIENT": mcc,
    "TRUE_NEGATIVES": int(tn),
    "FALSE_POSITIVES": int(fp),
    "FALSE_NEGATIVES": int(fn),
    "TRUE_POSITIVES": int(tp),
}

metrics_df = pd.DataFrame([
    {"METRIC": key, "VALUE": value}
    for key, value in metrics.items()
])

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
    encoding="utf-8-sig",
)

# 8.8. Generate and export the classification report

classification_results = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=[
        "Non-dropout",
        "Operational dropout",
    ],
    output_dict=True,
    zero_division=0,
)

classification_df = pd.DataFrame(
    classification_results
).transpose()

classification_df.index.name = "CLASS"

classification_df.to_csv(
    REPORT_PATH,
    encoding="utf-8-sig",
)

# 8.9. Export individual test predictions

predictions_df = pd.DataFrame({
    "ACTUAL_TARGET": y_test,
    "PREDICTED_TARGET": y_pred,
    "PREDICTED_DROPOUT_PROBABILITY": y_probability,
    "CLASSIFICATION_THRESHOLD": OPTIMAL_THRESHOLD,
})

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
    encoding="utf-8-sig",
)

# 8.10. Plot the confusion matrix

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
)

fig, ax = plt.subplots(figsize=(7, 5.5))

image = ax.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues",
)

fig.colorbar(
    image,
    ax=ax,
    fraction=0.046,
    pad=0.04,
)

ax.set(
    xticks=np.arange(2),
    yticks=np.arange(2),
    xticklabels=["Non-dropout", "Operational dropout"],
    yticklabels=["Non-dropout", "Operational dropout"],
    xlabel="Predicted class",
    ylabel="Actual class",
    title="Figure 9. Confusion Matrix",
)

threshold_value = cm.max() / 2.0

for row in range(cm.shape[0]):
    for col in range(cm.shape[1]):
        ax.text(
            col,
            row,
            f"{cm[row, col]:,}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=(
                "white"
                if cm[row, col] > threshold_value
                else "black"
            ),
        )

fig.tight_layout()

fig.savefig(
    f"{CONFUSION_MATRIX_PATH}.pdf",
    format="pdf",
    bbox_inches="tight",
)

fig.savefig(
    f"{CONFUSION_MATRIX_PATH}.png",
    format="png",
    dpi=600,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

# 8.11. Plot the ROC curve

fpr, tpr, _ = roc_curve(
    y_test,
    y_probability,
)

fig, ax = plt.subplots(figsize=(7, 5.5))

ax.plot(
    fpr,
    tpr,
    color=PALETTE_BLUE_DARK,
    linewidth=2.2,
    label=f"ResNet-Attention (AUC = {auc_roc:.3f})",
)

ax.plot(
    [0, 1],
    [0, 1],
    color=PALETTE_RED_DARK,
    linewidth=1.5,
    linestyle="--",
    label="No-discrimination reference",
)

ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Figure 10. Receiver Operating Characteristic Curve",
    xlim=(0, 1),
    ylim=(0, 1.02),
)

ax.legend(frameon=False, loc="lower right")
ax.grid(axis="both", linestyle=":", linewidth=0.7, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    f"{ROC_CURVE_PATH}.pdf",
    format="pdf",
    bbox_inches="tight",
)

fig.savefig(
    f"{ROC_CURVE_PATH}.png",
    format="png",
    dpi=600,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

# 8.12. Plot the precision-recall curve

pr_precision, pr_recall, _ = precision_recall_curve(
    y_test,
    y_probability,
)

positive_prevalence = float(np.mean(y_test))

fig, ax = plt.subplots(figsize=(7, 5.5))

ax.plot(
    pr_recall,
    pr_precision,
    color=PALETTE_RED_DARK,
    linewidth=2.2,
    label=f"ResNet-Attention (AP = {average_precision:.3f})",
)

ax.axhline(
    positive_prevalence,
    color=PALETTE_BLUE_DARK,
    linewidth=1.5,
    linestyle="--",
    label=f"Positive-class prevalence = {positive_prevalence:.3f}",
)

ax.set(
    xlabel="Recall",
    ylabel="Precision",
    title="Figure 11. Precision-Recall Curve",
    xlim=(0, 1),
    ylim=(0, 1.02),
)

ax.legend(frameon=False, loc="best")
ax.grid(axis="both", linestyle=":", linewidth=0.7, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    f"{PR_CURVE_PATH}.pdf",
    format="pdf",
    bbox_inches="tight",
)

fig.savefig(
    f"{PR_CURVE_PATH}.png",
    format="png",
    dpi=600,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

# 8.13. Save evaluation configuration

evaluation_config = {
    "model_name": "ResNet_Attention",
    "evaluation_partition": "Independent test set",
    "number_of_test_observations": int(len(y_test)),
    "selected_features": SELECTED_FEATURES,
    "classification_threshold": OPTIMAL_THRESHOLD,
    "threshold_source": THRESHOLD_SOURCE,
    "positive_class": 1,
    "negative_class": 0,
    "smote_applied_to_test": False,
    "metrics_file": str(METRICS_PATH),
    "classification_report_file": str(REPORT_PATH),
    "predictions_file": str(PREDICTIONS_PATH),
}

with open(CONFIG_PATH, "w", encoding="utf-8") as config_file:
    json.dump(
        evaluation_config,
        config_file,
        indent=4,
        ensure_ascii=False,
    )

# 8.14. Display evaluation results

print("\nPHASE 8: MODEL PERFORMANCE EVALUATION")

print("\nPerformance metrics:")
print(metrics_df.to_string(index=False))

print("\nClassification report:")
print(classification_df.round(4).to_string())

print("\nConfusion matrix:")
print(pd.DataFrame(
    cm,
    index=["Actual non-dropout", "Actual operational dropout"],
    columns=["Predicted non-dropout", "Predicted operational dropout"],
).to_string())

print("\nSaved outputs:")
print(f"Metrics: {METRICS_PATH}")
print(f"Classification report: {REPORT_PATH}")
print(f"Test predictions: {PREDICTIONS_PATH}")
print(f"Evaluation configuration: {CONFIG_PATH}")
print(f"Confusion matrix: {CONFUSION_MATRIX_PATH}.pdf")
print(f"ROC curve: {ROC_CURVE_PATH}.pdf")
print(f"Precision-Recall curve: {PR_CURVE_PATH}.pdf")

log_progress(
    "Phase 8 completed: independent test evaluation, "
    "performance metrics, predictions, and figures saved."
)
