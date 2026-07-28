
# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

# Standard libraries
import pandas as pd

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =============================================================================
# COMPARATIVE EVALUATION OF PREDICTIVE MODELS
# =============================================================================

print_title( COMPARATIVE MODEL EVALUATION")

print_section("Defining Evaluation Metrics")

# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate_model(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_score)
    }

# =============================================================================
# Traditional Machine Learning Models
# =============================================================================

print_section("Training Traditional Machine Learning Models")

traditional_models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=SEED
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=SEED,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        random_state=SEED,
        eval_metric="logloss",
        n_jobs=-1
    )
}

results = {}

# =============================================================================
# Train and Evaluate Baseline Models
# =============================================================================

for model_name, model in traditional_models.items():

    model.fit(
        X_train,
        y_train_res
    )

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    results[model_name] = evaluate_model(
        y_test,
        predictions,
        probabilities
    )

# =============================================================================
# Evaluate Proposed ResNet + Attention Model
# =============================================================================

print_section("Evaluating Proposed Deep Learning Model")

results["ResNet + Attention"] = evaluate_model(
    y_test,
    y_pred_final,
    y_scores
)

# =============================================================================
# Create Comparative Results Table
# =============================================================================

df_results = (
    pd.DataFrame(results)
      .T
      .astype(float)
)

baseline_models = list(traditional_models.keys())

df_results.loc["Baseline Average"] = (
    df_results.loc[baseline_models]
              .mean()
)

display_order = baseline_models + [
    "Baseline Average",
    "ResNet + Attention"
]

df_results = df_results.reindex(display_order)

# =============================================================================
# Display Results
# =============================================================================

print_section("Comparative Performance Summary")

print(
    df_results.round(4)
)

# =============================================================================
# Best Performing Model
# =============================================================================

best_model = df_results.drop(
    index="Baseline Average"
)["AUC-ROC"].idxmax()

best_auc = df_results.loc[
    best_model,
    "AUC-ROC"
]

print("\nBest Predictive Model")
print("-"*60)

print(f"Model   : {best_model}")
print(f"AUC-ROC : {best_auc:.4f}")

print("\n✓ Comparative evaluation completed successfully.")
