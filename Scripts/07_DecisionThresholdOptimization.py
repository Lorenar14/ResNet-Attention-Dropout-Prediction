# =============================================================================
# DECISION THRESHOLD OPTIMIZATION
# =============================================================================

print_title("PHASE 7 - DECISION THRESHOLD OPTIMIZATION")

# =============================================================================
# PREDICT CLASS PROBABILITIES
# =============================================================================

print_section("Predicting Class Probabilities")

# Generate probability scores using the trained ResNet-Attention model

y_scores = nn_model.predict(
    X_test,
    verbose=0
).ravel()

print(f"{GREEN}✓ Probability prediction completed successfully.{END}")

# =============================================================================
# PRECISION-RECALL ANALYSIS
# =============================================================================

print_section("Precision-Recall Analysis")

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_scores
)

# =============================================================================
# OPTIMAL THRESHOLD SEARCH
# =============================================================================

print_section("Searching for the Optimal Classification Threshold")

# Align arrays by discarding the last synthetic point from precision and recall
precision_adj = precision[:-1]
recall_adj = recall[:-1]

# Compute denominator avoiding division by zero
denominator = np.where(
    (precision_adj + recall_adj) == 0,
    1.0,
    precision_adj + recall_adj
)

f1_scores = (
    2 * precision_adj * recall_adj
) / denominator

best_index = np.argmax(f1_scores)

optimal_threshold = thresholds[best_index]

optimal_f1 = f1_scores[best_index]

print(f"\nOptimal Threshold : {optimal_threshold:.4f}")
print(f"Maximum F1-Score  : {optimal_f1:.4f}")

# =============================================================================
# FINAL CLASSIFICATION
# =============================================================================

print_section("Generating Final Predictions")

# Convert probabilities into binary predictions

y_pred_final = (
    y_scores >= optimal_threshold
).astype(int)

print(f"{GREEN}✓ Final predictions generated successfully.{END}")
