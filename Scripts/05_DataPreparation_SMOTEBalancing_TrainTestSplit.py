PREPROCESSING SCALING AND SMOTE BALANCING

SELECTED_FEATURES = ["AGE", "STRATUM", "CURRENT_LEVEL", "PROGRAM_DURATION", "ACADEMIC_PROGRESS"]
X_data = df[SELECTED_FEATURES].copy()
y_data = df["target"].copy()

X_dev_set, X_test_raw, y_dev_set, y_test = train_test_split(X_data, y_data, test_size=0.20, stratify=y_data, random_state=RANDOM_SEED)
X_train_raw, X_val_raw, y_train_raw, y_val = train_test_split(X_dev_set, y_dev_set, test_size=0.20, stratify=y_dev_set, random_state=RANDOM_SEED)

training_medians = X_train_raw.median()
X_train_raw = X_train_raw.fillna(training_medians)
X_val_raw = X_val_raw.fillna(training_medians)
X_test_raw = X_test_raw.fillna(training_medians)

feature_scaler = RobustScaler()
X_train_scaled = feature_scaler.fit_transform(X_train_raw)
X_val = feature_scaler.transform(X_val_raw)
X_test = feature_scaler.transform(X_test_raw)

smote_sampler = SMOTE(sampling_strategy="auto", random_state=RANDOM_SEED, k_neighbors=5)
X_train, y_train = smote_sampler.fit_resample(X_train_scaled, y_train_raw)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(X_train_scaled[y_train_raw == 0, 0], X_train_scaled[y_train_raw == 0, 2], alpha=0.5, color=PALETTE_BLUE_DARK, label="No Dropout", s=15)
axes[0].scatter(X_train_scaled[y_train_raw == 1, 0], X_train_scaled[y_train_raw == 1, 2], alpha=0.7, color=PALETTE_RED_DARK, label="Dropout (Minority)", s=25)
axes[0].set_title("(a) Before SMOTE (Imbalanced)")
axes[0].set_xlabel("Age (Scaled)")
axes[0].set_ylabel("Academic Level (Scaled)")
axes[0].legend(frameon=False)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 2], alpha=0.5, color=PALETTE_BLUE_DARK, label="No Dropout", s=15)
axes[1].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 2], alpha=0.7, color=PALETTE_SLATE, label="Synthetic Dropout (SMOTE)", s=25)
axes[1].set_title("(b) After SMOTE (Balanced)")
axes[1].set_xlabel("Age (Scaled)")
axes[1].set_ylabel("Academic Level (Scaled)")
axes[1].legend(frameon=False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_04_SMOTE_Effect.pdf"), format="pdf", bbox_inches="tight")
plt.close()

log_progress("Phase 4 completed: Robust scaling, SMOTE balancing, and effect visualization executed.")
