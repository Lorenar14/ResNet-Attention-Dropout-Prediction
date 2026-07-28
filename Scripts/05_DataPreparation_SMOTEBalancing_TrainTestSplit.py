# =============================================================================
#  DATA PREPROCESSING, SMOTE BALANCING AND TRAIN-TEST SPLIT
# =============================================================================

print_title("PHASE 5 - DATA PREPROCESSING")

# =============================================================================
# FEATURE SELECTION
# =============================================================================

print_section("Feature Selection")

FEATURES = [
    "EDAD",
    "ESTRATO",
    "NIVEL_ACTUAL",
    "SEMESTRES_TOTAL_POR_CARRERA",
    "AVANCE_NU"
]

X = df[FEATURES].copy()
y = df["target"].copy()

print("\nSelected Features:")
print(FEATURES)

print(f"\nTotal Records      : {len(X):,}")
print(f"Number of Features : {X.shape[1]}")


# =============================================================================
# TRAIN-TEST SPLIT (BEFORE IMPUTATION TO PREVENT DATA LEAKAGE)
# =============================================================================

print_section("Train-Test Split")

X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

# =============================================================================
# MISSING VALUE IMPUTATION (TRAIN-SET MEDIANS)
# =============================================================================

print_section("Missing Value Imputation")

# Compute medians ONLY on the training set to prevent data leakage
train_medians = X_train_raw.median()

# Apply the training medians to both training and test sets
X_train_raw = X_train_raw.fillna(train_medians)
X_test_raw = X_test_raw.fillna(train_medians)

print(f"{GREEN}✓ Missing values imputed using training set medians.{END}")

# =============================================================================
# ORIGINAL CLASS DISTRIBUTION
# =============================================================================

print_section("Original Class Distribution")

# Print class distribution after the train-test split
print("\nTraining Set (Before SMOTE)")
print(y_train_raw.value_counts().sort_index())

print("\nTesting Set")
print(y_test.value_counts().sort_index())

# =============================================================================
# SMOTE OVERSAMPLING
# =============================================================================

print_section("SMOTE Oversampling")

smote = SMOTE(
    sampling_strategy="auto",
    random_state=42,
    k_neighbors=5
)

X_train_res, y_train_res = smote.fit_resample(
    X_train_raw,
    y_train_raw
)

print("\nTraining Set (After SMOTE)")
print(y_train_res.value_counts().sort_index())

# =============================================================================
# ROBUST FEATURE SCALING
# =============================================================================

print_section("Robust Feature Scaling")

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train_res)

X_test = scaler.transform(X_test_raw)

# =============================================================================
# PREPROCESSING SUMMARY
# =============================================================================

print(f"\n{BOLD}{'='*90}")
print("PREPROCESSING SUMMARY")
print(f"{'='*90}{END}")

print(f"Original Dataset                 : {X.shape}")
print(f"Training Dataset (Before SMOTE)  : {X_train_raw.shape}")
print(f"Testing Dataset                  : {X_test_raw.shape}")
print(f"Balanced Training Dataset        : {X_train.shape}")

print(f"\nTotal Students                  : {len(df):,}")
print(f"Training Students               : {len(X_train_raw):,}")
print(f"Testing Students                : {len(X_test_raw):,}")
print(f"Balanced Training Samples       : {len(X_train_res):,}")

print(
    f"\nSynthetic Samples Generated     : "
    f"{len(X_train_res)-len(X_train_raw):,}"
)

print(f"\n{GREEN}✓ Train-test split completed successfully.")
print("✓ SMOTE applied only to the training dataset.")
print("✓ Testing dataset remained completely unseen.")
print("✓ RobustScaler applied successfully.")
print(f"{END}")
