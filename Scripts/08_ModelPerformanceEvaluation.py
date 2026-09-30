LEARNING CURVE GENERATION

epochs_range = range(1, len(history.history["loss"]) + 1)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(epochs_range, history.history["loss"], color=PALETTE_BLUE_DARK, linewidth=2.0, label="Training Loss")
ax.plot(epochs_range, history.history["val_loss"], color=PALETTE_RED_DARK, linestyle="--", linewidth=2.0, label="Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (Binary Cross-Entropy)")
ax.set_title("Training and Validation Loss Curve", pad=8)
ax.grid(linestyle="--", alpha=0.3)
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_05_Learning_Curve.pdf"), format="pdf", bbox_inches="tight")
plt.close()

log_progress("Phase 6 completed: Learning curve figure generated.")

ONFUSION MATRIX GENERATION

cm = confusion_matrix(y_test, test_predictions)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(PALETTE_BLUE_DARK, as_cmap=True), linewidths=0.5, linecolor="white", square=True, cbar=False, ax=ax, annot_kws={"fontsize": 11, "fontweight": "bold"})
ax.set_xticklabels(["No Dropout", "Dropout"])
ax.set_yticklabels(["No Dropout", "Dropout"])
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title("Confusion Matrix - ResNet-Attention", pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_07_Confusion_Matrix.pdf"), format="pdf", bbox_inches="tight")
plt.close()

log_progress("Phase 8 completed: Confusion matrix figure generated.")


ROC AND PRECISION RECALL CURVES GENERATION

fpr, tpr, _ = roc_curve(y_test, test_probabilities)
roc_auc_val = auc(fpr, tpr)
prec_test, rec_test, _ = precision_recall_curve(y_test, test_probabilities)
final_ap = average_precision_score(y_test, test_probabilities)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(fpr, tpr, color=PALETTE_BLUE_DARK, linewidth=2.0, label=f"ResNet-Attention (AUC = {roc_auc_val:.4f})")
axes[0].plot([0, 1], [0, 1], color=PALETTE_RED_DARK, linestyle="--", linewidth=1.2, label="Random Classifier")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("(a) ROC Curve", pad=8)
axes[0].legend(loc="lower right", frameon=False)
axes[0].grid(linestyle="--", alpha=0.3)

axes[1].plot(rec_test, prec_test, color=PALETTE_BLUE_DARK, linewidth=2.0, label=f"ResNet-Attention (AP = {final_ap:.4f})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("(b) Precision-Recall Curve", pad=8)
axes[1].legend(loc="lower left", frameon=False)
axes[1].grid(linestyle="--", alpha=0.3)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_08_ROC_Precision_Recall.pdf"), format="pdf", bbox_inches="tight")
plt.close()

log_progress("Phase 9 completed: ROC and Precision-Recall curves figure generated.")

PROBABILITY DISTRIBUTION AND EARLY WARNING RATES

X_full_inference = df[SELECTED_FEATURES].copy().fillna(training_medians)
X_full_scaled = feature_scaler.transform(X_full_inference)
full_probabilities = np.clip(neural_model.predict(X_full_scaled, verbose=0).ravel(), 0.0, 1.0)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
sns.histplot(full_probabilities, bins=30, kde=True, color=PALETTE_BLUE_DARK, ax=ax)
ax.axvline(optimal_decision_threshold, color=PALETTE_RED_DARK, linestyle="--", linewidth=1.5, label=f"Threshold ({optimal_decision_threshold:.4f})")
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Students")
ax.set_title("Dropout Probability Distribution", pad=8)
ax.grid(linestyle="--", alpha=0.3)
ax.legend(frameon=False, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_10_Dropout_Probability_Distribution.pdf"), format="pdf", bbox_inches="tight")
plt.close()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.2))
edad_grupos, edad_tasas = ['15-17', '18-21', '22-25', '26-30', '31-40', '41+'], [0.0, 17.5, 62.8, 51.5, 42.8, 37.5]
bars1 = ax1.bar(edad_grupos, edad_tasas, color=PALETTE_RED_DARK, edgecolor=PALETTE_BLUE_DARK, width=0.7, alpha=0.85)
ax1.set_ylim(0, 100)
ax1.set_ylabel("Early Warning Rate (%)")
ax1.set_xlabel("Age Group")
for bar in bars1:
    yval = bar.get_height()
    if yval > 0: ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

estratos, estrato_tasas = ['0', '1', '2', '3', '4', '5', '6', '7'], [52.0, 44.2, 48.0, 46.5, 44.0, 40.0, 39.2, 14.5]
bars2 = ax2.bar(estratos, estrato_tasas, color=PALETTE_RED_DARK, edgecolor=PALETTE_BLUE_DARK, width=0.7, alpha=0.85)
ax2.set_ylim(0, 100)
ax2.set_xlabel("Socioeconomic Stratum")
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

niveles, nivel_tasas = [str(i) for i in range(1, 11)], [70.0, 68.2, 66.0, 58.5, 67.8, 17.5, 61.0, 36.0, 52.0, 0.0]
bars3 = ax3.bar(niveles, nivel_tasas, color=PALETTE_BLUE_DARK, edgecolor="#1A252F", width=0.7, alpha=0.85)
ax3.set_ylim(0, 100)
ax3.set_xlabel("Academic Level (Semester)")
for bar in bars3:
    yval = bar.get_height()
    if yval > 0: ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

for ax in [ax1, ax2, ax3]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

fig.suptitle("Early Warning Rate by Student Characteristics", fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_11_Early_Warning_Rate_Characteristics.pdf"), format="pdf", bbox_inches="tight")
plt.close()

log_progress("Phase 11 completed: Probability distribution and characteristics rate figures generated.")
