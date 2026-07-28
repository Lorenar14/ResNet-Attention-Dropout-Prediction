# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Dropout,
    Add,
    Multiply,
    Activation
)

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.optimizers import Adam

# =============================================================================
# RESNET-ATTENTION MODEL DEVELOPMENT AND TRAINING
# =============================================================================

print_title("PHASE 7 - RESNET-ATTENTION MODEL")

# =============================================================================
# RESNET-ATTENTION ARCHITECTURE
# =============================================================================

print_section("Building the ResNet-Attention Architecture")

def build_resnet_optimized(input_dim):
    """
    ResNet-Attention architecture for university dropout prediction.
    """

    inputs = Input(shape=(input_dim,))

    # -------------------------------------------------------------------------
    # Initial Projection Layer
    # -------------------------------------------------------------------------

    x_init = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(0.001)
    )(inputs)

    x_init = layers.BatchNormalization()(x_init)
    x_init = layers.Activation("swish")(x_init)

    # -------------------------------------------------------------------------
    # Residual Block
    # -------------------------------------------------------------------------

    x1 = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(0.001)
    )(x_init)

    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation("swish")(x1)

    x2 = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(0.001)
    )(x1)

    x2 = layers.BatchNormalization()(x2)

    residual = layers.Add()([x_init, x2])

    residual = layers.Activation("swish")(residual)

    # -------------------------------------------------------------------------
    # Attention Mechanism
    # -------------------------------------------------------------------------

    attention = layers.Dense(
        128,
        activation="sigmoid"
    )(residual)

    x = layers.Multiply()([residual, attention])

    # -------------------------------------------------------------------------
    # Classification Head
    # -------------------------------------------------------------------------

    x = layers.Dense(
        64,
        activation="swish"
    )(x)

    x = layers.Dropout(0.50)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    # -------------------------------------------------------------------------
    # Model Compilation
    # -------------------------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001
    )

    model.compile(

        optimizer=optimizer,

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


# =============================================================================
#  MODEL INITIALIZATION
# =============================================================================

print_section("Initializing the Deep Learning Model")

tf.random.set_seed(SEED)

nn_model = build_resnet_optimized(
    X_train.shape[1]
)

print(f"{GREEN}✓ ResNet-Attention model created successfully.{END}")


# =============================================================================
#  CALLBACK CONFIGURATION
# =============================================================================

print_section("Configuring Training Callbacks")

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=15,

    restore_best_weights=True

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.50,

    patience=5,

    min_lr=1e-5,

    verbose=1

)

print(f"{GREEN}✓ EarlyStopping configured.{END}")
print(f"{GREEN}✓ ReduceLROnPlateau configured.{END}")


# =============================================================================
# MODEL TRAINING
# =============================================================================

print_section("Training the ResNet-Attention Model")

history = nn_model.fit(

    X_train,

    y_train_res,

    validation_data=(X_test, y_test),

    epochs=50,

    batch_size=32,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1

)

print(f"\n{GREEN}✓ Model training completed successfully.{END}")
