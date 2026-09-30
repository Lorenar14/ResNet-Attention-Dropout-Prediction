DECISION THRESHOLD OPTIMIZATION

val_probabilities = neural_model.predict(X_val, verbose=0).ravel()
precisions, recalls, thresholds = precision_recall_curve(y_val, val_probabilities)
p_subset, r_subset = precisions[:-1], recalls[:-1]
f1_harmonic_scores = np.divide(2 * p_subset * r_subset, p_subset + r_subset, out=np.zeros_like(p_subset), where=(p_subset + r_subset) != 0)
optimal_threshold_idx = np.argmax(f1_harmonic_scores)
optimal_decision_threshold = float(thresholds[optimal_threshold_idx])

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(thresholds, f1_harmonic_scores, color=PALETTE_BLUE_DARK, linewidth=2.0, label="F1-Score")
ax.axvline(optimal_decision_threshold, color=PALETTE_RED_DARK, linestyle="--", linewidth=1.5, label=f"Optimal Threshold ({optimal_decision_threshold:.4f})")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("F1-Score")
ax.set_title("Decision Threshold Optimization", pad=8)
ax.grid(linestyle="--", alpha=0.3)
ax.legend(frameon=False, loc="lower left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_06_Threshold_Optimization_Curve.pdf"), format="pdf", bbox_inches="tight")
plt.close()

test_probabilities = neural_model.predict(X_test, verbose=0).ravel()
test_predictions = (test_probabilities >= optimal_decision_threshold).astype(int)

log_progress("Phase 7 completed: Threshold optimization and test predictions evaluated.")
