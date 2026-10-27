import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# 7.1. Define input and output paths

OUTPUT_DIR = Path(OUTPUT_DIR)
FIGURES_DIR = OUTPUT_DIR / "Figures"

HISTORY_PATH = OUTPUT_DIR / "Phase5_Training_History.csv"

FIGURE_PATH = (
    FIGURES_DIR / "Supplementary_S1_Training_Validation_Loss_Curve.pdf"
)

FIGURE_PNG_PATH = (
    FIGURES_DIR / "Supplementary_S1_Training_Validation_Loss_Curve.png"
)

SUMMARY_PATH = OUTPUT_DIR / "Phase7_Learning_Curve_Summary.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

if not HISTORY_PATH.is_file():
    raise FileNotFoundError(
        f"Training history not found: {HISTORY_PATH}. "
        "Run Phase 5 before generating learning curves."
    )

log_progress("Starting Phase 7: learning-curve generation.")

# 7.2. Load and validate the training history

training_history = pd.read_csv(
    HISTORY_PATH,
    index_col=0,
)

required_columns = ["loss", "val_loss"]

missing_columns = [
    column
    for column in required_columns
    if column not in training_history.columns
]

if missing_columns:
    raise KeyError(
        f"Required loss metrics are missing: {missing_columns}"
    )

if training_history.empty:
    raise ValueError(
        "The training history contains no recorded epochs."
    )

loss_values = training_history[
    required_columns
].to_numpy(dtype=float)

if not np.isfinite(loss_values).all():
    raise ValueError(
        "The training history contains missing or "
        "non-finite loss values."
    )

if (loss_values < 0).any():
    raise ValueError(
        "Loss values cannot be negative."
    )

training_loss = training_history["loss"].to_numpy(dtype=float)
validation_loss = training_history["val_loss"].to_numpy(dtype=float)

if len(training_loss) != len(validation_loss):
    raise ValueError(
        "Training and validation loss series must have "
        "the same number of epochs."
    )

epochs_range = np.arange(1, len(training_history) + 1)

best_epoch = int(np.argmin(validation_loss)) + 1
best_val_loss = float(validation_loss[best_epoch - 1])

final_training_loss = float(training_loss[-1])
final_validation_loss = float(validation_loss[-1])

# 7.3. Plot training and validation loss

fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.plot(
    epochs_range,
    training_loss,
    color=PALETTE_BLUE_DARK,
    linewidth=2.0,
    label="Training Loss",
)

ax.plot(
    epochs_range,
    validation_loss,
    color=PALETTE_RED_DARK,
    linestyle="--",
    linewidth=2.0,
    label="Validation Loss",
)

ax.axvline(
    x=best_epoch,
    color=PALETTE_RED_DARK,
    linestyle=":",
    linewidth=1.2,
    alpha=0.8,
    label=f"Best validation loss (epoch {best_epoch})",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Binary Cross-Entropy Loss")
ax.set_title("Training and Validation Loss", pad=8)

tick_step = max(1, int(np.ceil(len(epochs_range) / 10)))

ax.set_xticks(
    np.unique(
        np.append(
            np.arange(1, len(epochs_range) + 1, tick_step),
            len(epochs_range),
        )
    )
)

ax.grid(
    linestyle="--",
    alpha=0.3,
)

ax.set_axisbelow(True)

ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

# 7.4. Save the figure

fig.savefig(
    FIGURE_PATH,
    format="pdf",
    bbox_inches="tight",
)

fig.savefig(
    FIGURE_PNG_PATH,
    format="png",
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)

# 7.5. Save numerical results

learning_curve_summary = pd.DataFrame([
    {
        "METRIC": "Completed epochs",
        "VALUE": len(training_history),
    },
    {
        "METRIC": "Best validation-loss epoch",
        "VALUE": best_epoch,
    },
    {
        "METRIC": "Minimum validation loss",
        "VALUE": best_val_loss,
    },
    {
        "METRIC": "Final training loss",
        "VALUE": final_training_loss,
    },
    {
        "METRIC": "Final validation loss",
        "VALUE": final_validation_loss,
    },
])

learning_curve_summary.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
)

# 7.6. Report results

print("\nPHASE 7: TRAINING AND VALIDATION LEARNING CURVES")
print(f"Completed epochs: {len(training_history)}")
print(f"Best validation-loss epoch: {best_epoch}")
print(f"Minimum validation loss: {best_val_loss:.4f}")
print(f"Final training loss: {final_training_loss:.4f}")
print(f"Final validation loss: {final_validation_loss:.4f}")
print(f"PDF figure: {FIGURE_PATH}")
print(f"PNG figure: {FIGURE_PNG_PATH}")
print(f"Summary: {SUMMARY_PATH}")

log_progress(
    "Phase 7 completed: learning curves and numerical summary "
    "generated successfully."
)
