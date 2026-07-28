# =============================================================================
# MODEL PERFORMANCE EVALUATION
# =============================================================================

print_title("PHASE 8 - MODEL PERFORMANCE EVALUATION")

# =============================================================================
# CLASSIFICATION REPORT
# =============================================================================

print_section("Classification Report")

report = classification_report(
    y_test,
    y_pred_final,
    target_names=[
        "Persistence",
        "Dropout"
    ]
)

print(report)

# =============================================================================
#  PERFORMANCE METRICS
# =============================================================================

print_section("Performance Metrics")

accuracy = accuracy_score(
    y_test,
    y_pred_final
)

precision = precision_score(
    y_test,
    y_pred_final
)

recall = recall_score(
    y_test,
    y_pred_final
)

f1 = f1_score(
    y_test,
    y_pred_final
)

auc_score = roc_auc_score(
    y_test,
    y_scores
)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc_score:.4f}")

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

print_section("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred_final
)

print(cm)

# =============================================================================
# EVALUATION SUMMARY
# =============================================================================

print_section("Evaluation Summary")

print(f"{GREEN}✓ Classification report generated successfully.")
print("✓ Performance metrics computed successfully.")
print("✓ Confusion matrix generated successfully.")
print("✓ ROC-AUC score computed successfully.")
print(f"{END}")
