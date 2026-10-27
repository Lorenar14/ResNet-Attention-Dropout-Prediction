import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras import Input, Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure experiment paths and parameters
OUTPUT_DIR = Path("/content/Results")
MODEL_DIR = OUTPUT_DIR / "Models"
FIGURE_DIR = OUTPUT_DIR / "Figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "ResNet_Attention.keras"
HISTORY_PATH = OUTPUT_DIR / "Phase5_Training_History.csv"
CONFIG_PATH = OUTPUT_DIR / "Phase5_Training_Config.json"

TRAINING_FIGURE_PDF = FIGURE_DIR / "Figure_08_Training_History.pdf"
TRAINING_FIGURE_PNG = FIGURE_DIR / "Figure_08_Training_History.png"

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5

RANDOM_SEED = int(globals().get("RANDOM_SEED", 42))
tf.keras.utils.set_random_seed(RANDOM_SEED)

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

# Validate the outputs from Phase 4
EXPECTED_FEATURES = [
    "AGE",
    "STRATUM",
    "CURRENT_LEVEL",
    "PROGRAM_DURATION",
    "ACADEMIC_PROGRESS",
]

required_arrays = [
    "X_train",
    "y_train",
    "X_val",
    "y_val",
]

missing_arrays = [
    name for name in required_arrays
    if name not in globals()
]

if missing_arrays:
    raise RuntimeError(
        f"Missing Phase 4 arrays: {missing_arrays}. "
        "Execute Phase 4 before Phase 5."
    )

features_path = OUTPUT_DIR / "Selected_Features.csv"

if not features_path.exists():
    raise FileNotFoundError(
        f"Selected feature file not found: {features_path}"
    )

saved_features = (
    pd.read_csv(features_path)["FEATURE"]
    .astype(str)
    .tolist()
)

if saved_features != EXPECTED_FEATURES:
    raise ValueError(
        "The selected features do not match the Phase 5 configuration. "
        f"Expected: {EXPECTED_FEATURES}; found: {saved_features}"
    )

X_train = np.asarray(X_train, dtype=np.float32)
X_val = np.asarray(X_val, dtype=np.float32)

y_train = np.asarray(y_train).reshape(-1).astype(np.int32)
y_val = np.asarray(y_val).reshape(-1).astype(np.int32)

if X_train.ndim != 2 or X_val.ndim != 2:
    raise ValueError(
        "Training and validation features must be two-dimensional."
    )

if X_train.shape[1] != len(EXPECTED_FEATURES):
    raise ValueError(
        f"Expected {len(EXPECTED_FEATURES)} predictors, "
        f"but received {X_train.shape[1]}."
    )

if X_val.shape[1] != X_train.shape[1]:
    raise ValueError(
        "Training and validation feature dimensions do not match."
    )

if len(X_train) != len(y_train):
    raise ValueError(
        "Training features and labels have different lengths."
    )

if len(X_val) != len(y_val):
    raise ValueError(
        "Validation features and labels have different lengths."
    )

if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
    raise ValueError(
        "Non-finite values were detected in the model inputs."
    )

if not np.isin(y_train, [0, 1]).all():
    raise ValueError("Training labels must contain only 0 and 1.")

if not np.isin(y_val, [0, 1]).all():
    raise ValueError("Validation labels must contain only 0 and 1.")

if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
    raise ValueError(
        "Both classes must be present in training and validation."
    )

print("Phase 4 inputs validated.")
print(f"Training samples: {len(y_train):,}")
print(f"Validation samples: {len(y_val):,}")
print(f"Number of predictors: {X_train.shape[1]}")
print(f"Selected features: {EXPECTED_FEATURES}")

# Define the ResNet-Attention architecture
def construct_resnet_attention_model(input_dim):
    """Build a residual neural network with gated feature attention."""

    inputs = Input(
        shape=(input_dim,),
        name="input_features",
    )

    initial = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(0.001),
        name="initial_dense",
    )(inputs)

    initial = layers.BatchNormalization(
        name="initial_batch_normalization",
    )(initial)

    initial = layers.Activation(
        "swish",
        name="initial_activation",
    )(initial)

    residual = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(0.001),
        name="residual_dense_1",
    )(initial)

    residual = layers.BatchNormalization(
        name="residual_batch_normalization_1",
    )(residual)

    residual = layers.Activation(
        "swish",
        name="residual_activation_1",
    )(residual)

    residual = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(0.001),
        name="residual_dense_2",
    )(residual)

    residual = layers.BatchNormalization(
        name="residual_batch_normalization_2",
    )(residual)

    residual = layers.Add(
        name="residual_connection",
    )([initial, residual])

    residual = layers.Activation(
        "swish",
        name="residual_output",
    )(residual)

    attention_weights = layers.Dense(
        128,
        activation="sigmoid",
        name="attention_gate",
    )(residual)

    attended_features = layers.Multiply(
        name="attention_weighted_features",
    )([residual, attention_weights])

    classifier = layers.Dense(
        64,
        activation="swish",
        name="classification_dense",
    )(attended_features)

    classifier = layers.Dropout(
        0.50,
        name="classification_dropout",
    )(classifier)

    predictions = layers.Dense(
        1,
        activation="sigmoid",
        name="dropout_probability",
    )(classifier)

    model = Model(
        inputs=inputs,
        outputs=predictions,
        name="ResNet_Attention",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


# Initialize the model
neural_model = construct_resnet_attention_model(
    input_dim=X_train.shape[1],
)

neural_model.summary()

# Configure training callbacks
callback_early_stop = EarlyStopping(
    monitor="val_loss",
    patience=EARLY_STOPPING_PATIENCE,
    mode="min",
    restore_best_weights=True,
    verbose=1,
)

callback_reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=REDUCE_LR_PATIENCE,
    mode="min",
    min_lr=1e-6,
    verbose=1,
)

# Train the model
print("\nStarting ResNet-Attention training.")

history = neural_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[
        callback_early_stop,
        callback_reduce_lr,
    ],
    verbose=1,
)

completed_epochs = len(history.history["loss"])

best_epoch = int(
    np.argmin(history.history["val_loss"])
) + 1

best_val_loss = float(
    np.min(history.history["val_loss"])
)

print(f"\nCompleted epochs: {completed_epochs}")
print(f"Best epoch by validation loss: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.6f}")

# Export the training history
history_df = pd.DataFrame(history.history)
history_df.index = np.arange(1, len(history_df) + 1)
history_df.index.name = "epoch"

history_df.to_csv(
    HISTORY_PATH,
    encoding="utf-8-sig",
)

# Plot the training and validation curves
fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4.5),
)

epochs_range = np.arange(1, completed_epochs + 1)

axes[0].plot(
    epochs_range,
    history.history["loss"],
    color=PALETTE_BLUE_DARK,
    linewidth=2,
    label="Training loss",
)

axes[0].plot(
    epochs_range,
    history.history["val_loss"],
    color=PALETTE_RED_DARK,
    linewidth=2,
    label="Validation loss",
)

axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary cross-entropy")
axes[0].legend(frameon=False)

axes[1].plot(
    epochs_range,
    history.history["auc"],
    color=PALETTE_BLUE_DARK,
    linewidth=2,
    label="Training AUC",
)

axes[1].plot(
    epochs_range,
    history.history["val_auc"],
    color=PALETTE_RED_DARK,
    linewidth=2,
    label="Validation AUC",
)

axes[1].set_title("Area Under the ROC Curve")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("AUC")
axes[1].set_ylim(0, 1.02)
axes[1].legend(frameon=False)

for ax in axes:
    ax.grid(axis="both", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Figure 8. ResNet-Attention Training History",
    fontsize=13,
    fontweight="bold",
)

fig.tight_layout(rect=[0, 0, 1, 0.92])

fig.savefig(
    TRAINING_FIGURE_PDF,
    format="pdf",
    bbox_inches="tight",
)

fig.savefig(
    TRAINING_FIGURE_PNG,
    dpi=600,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

# Save the trained model
neural_model.save(MODEL_PATH)

# Save experiment configuration
training_config = {
    "model_name": "ResNet_Attention",
    "random_seed": RANDOM_SEED,
    "input_features": EXPECTED_FEATURES,
    "input_dimension": int(X_train.shape[1]),
    "epochs_requested": EPOCHS,
    "epochs_completed": completed_epochs,
    "best_epoch_by_validation_loss": best_epoch,
    "best_validation_loss": best_val_loss,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "loss": "binary_crossentropy",
    "optimizer": "Adam",
    "metrics": [
        "accuracy",
        "auc",
        "precision",
        "recall",
    ],
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "reduce_lr_patience": REDUCE_LR_PATIENCE,
    "training_samples_after_smote": int(len(y_train)),
    "validation_samples": int(len(y_val)),
    "architecture": "Residual dense network with gated attention",
}

with open(CONFIG_PATH, "w", encoding="utf-8") as config_file:
    json.dump(
        training_config,
        config_file,
        indent=4,
        ensure_ascii=False,
    )

print("\nSaved artifacts:")
print(f"Model: {MODEL_PATH}")
print(f"Training history: {HISTORY_PATH}")
print(f"Training configuration: {CONFIG_PATH}")
print(f"Training figure PDF: {TRAINING_FIGURE_PDF}")
print(f"Training figure PNG: {TRAINING_FIGURE_PNG}")

print("\nPhase 5 completed successfully.")
