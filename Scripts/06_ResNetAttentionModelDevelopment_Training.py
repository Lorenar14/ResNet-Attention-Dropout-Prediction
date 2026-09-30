RESNET ATTENTION ARCHITECTURE TRAINING

def construct_resnet_attention_model(input_dim):
    inputs = Input(shape=(input_dim,), name="input_features")
    x_init = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(inputs)
    x_init = layers.BatchNormalization()(x_init)
    x_init = layers.Activation("swish")(x_init)

    x1 = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x_init)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation("swish")(x1)

    x2 = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x1)
    x2 = layers.BatchNormalization()(x2)

    residual_path = layers.Add()([x_init, x2])
    residual_path = layers.Activation("swish")(residual_path)

    attention_gate = layers.Dense(128, activation="sigmoid")(residual_path)
    gated_features = layers.Multiply()([residual_path, attention_gate])

    dense_out = layers.Dense(64, activation="swish")(gated_features)
    dropout_out = layers.Dropout(0.50)(dense_out)
    predictions = layers.Dense(1, activation="sigmoid")(dropout_out)

    model = Model(inputs=inputs, outputs=predictions, name="ResNet_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")]
    )
    return model

tf.keras.utils.set_random_seed(RANDOM_SEED)
neural_model = construct_resnet_attention_model(input_dim=X_train.shape[1])

callback_early_stop = EarlyStopping(monitor="val_loss", patience=15, mode="min", restore_best_weights=True, verbose=1)
callback_reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.50, patience=5, mode="min", min_lr=1e-5, verbose=1)

print("Starting ResNet-Attention model training...")
history = neural_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    shuffle=True,
    callbacks=[callback_early_stop, callback_reduce_lr],
    verbose=1
)

log_progress("Phase 5 completed: ResNet-Attention neural network successfully trained.")
