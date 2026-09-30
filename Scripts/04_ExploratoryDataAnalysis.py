 EXPLORATORY DATA ANALYSIS

status_categories = ["Active", "Dropout", "Graduated"]
status_freq = df["ACADEMIC_STATUS"].value_counts().reindex(status_categories, fill_value=0)
status_pcts = status_freq.div(len(df)).mul(100)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(status_freq.index, status_freq.values, color=[PALETTE_BLUE_MED, PALETTE_RED_DARK, PALETTE_SLATE], width=0.5)
for bar_item, val, pct in zip(bars, status_freq.values, status_pcts.values):
    ax.text(bar_item.get_x() + bar_item.get_width() / 2, bar_item.get_height(), f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Number of Students")
ax.set_title("Student Academic Status Distribution", pad=8)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_01_Academic_Status_Distribution.pdf"), format="pdf", bbox_inches="tight")
plt.close()

numeric_vars = ["AGE", "STRATUM", "CURRENT_LEVEL", "PROGRAM_DURATION", "ACADEMIC_PROGRESS", "target"]
corr_mat = df[numeric_vars].corr()

fig, ax = plt.subplots(figsize=(7.0, 5.5))
sns.heatmap(
    corr_mat, annot=True, fmt=".2f", cmap=sns.light_palette(PALETTE_BLUE_DARK, as_cmap=True),
    linewidths=0.5, linecolor="white", square=True, cbar=True, ax=ax,
    annot_kws={"fontsize": 9, "fontweight": "bold"}
)
ax.set_xticklabels(["Age", "Stratum", "Level", "Duration", "Progress", "Dropout"], rotation=30, ha="right")
ax.set_yticklabels(["Age", "Stratum", "Level", "Duration", "Progress", "Dropout"], rotation=0)
ax.set_title("Model Variables Correlation Matrix", pad=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_02_Correlation_Matrix.pdf"), format="pdf", bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
sns.histplot(df["AGE"].dropna(), bins=25, kde=True, color=PALETTE_BLUE_DARK, ax=axes[0, 0])
axes[0, 0].set_title("Age Distribution")
axes[0, 0].spines["top"].set_visible(False)
axes[0, 0].spines["right"].set_visible(False)

sns.countplot(x="STRATUM", data=df, palette="Blues_r", ax=axes[0, 1])
axes[0, 1].set_title("Socioeconomic Stratum Distribution")
axes[0, 1].spines["top"].set_visible(False)
axes[0, 1].spines["right"].set_visible(False)

sns.histplot(df["CURRENT_LEVEL"].dropna(), bins=10, discrete=True, color=PALETTE_BLUE_MED, ax=axes[1, 0])
axes[1, 0].set_title("Academic Level Distribution")
axes[1, 0].spines["top"].set_visible(False)
axes[1, 0].spines["right"].set_visible(False)

sns.histplot(df["ACADEMIC_PROGRESS"].dropna(), bins=25, kde=True, color=PALETTE_SLATE, ax=axes[1, 1])
axes[1, 1].set_title("Academic Progress Distribution (%)")
axes[1, 1].spines["top"].set_visible(False)
axes[1, 1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_03_Univariate_Distributions.pdf"), format="pdf", bbox_inches="tight")
plt.close()

log_progress("Phase 3 completed: EDA figures generated.")
