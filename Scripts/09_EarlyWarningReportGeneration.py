EXCEL REPORT CONSOLIDATION

full_student_report = pd.DataFrame({
    "ANONYMIZED_ID": df["STUDENT_ID"].values,
    "ENTRY_PERIOD": df["ENTRY_PERIOD"].values,
    "AGE": df["AGE"].values,
    "STRATUM": df["STRATUM"].values,
    "CURRENT_LEVEL": df["CURRENT_LEVEL"].values,
    "ACADEMIC_PROGRESS_PCT": df["ACADEMIC_PROGRESS"].values,
    "DROPOUT_PROBABILITY": pd.Series(full_probabilities).astype(float).round(1).values,
    "RISK_LEVEL": np.where(full_probabilities >= 0.70, "High Risk",
                   np.where(full_probabilities >= 0.40, "Medium Risk", "Low Risk"))
})

summary_report = pd.DataFrame({
    "Alert Status": ["High Risk (Early Warning)", "Medium Risk", "Low Risk (Active)"],
    "Students": [
        sum(full_probabilities >= 0.70),
        sum((full_probabilities >= 0.40) & (full_probabilities < 0.70)),
        sum(full_probabilities < 0.40)
    ],
    "Percentage": [
        (sum(full_probabilities >= 0.70) / len(full_probabilities)) * 100,
        (sum((full_probabilities >= 0.40) & (full_probabilities < 0.70)) / len(full_probabilities)) * 100,
        (sum(full_probabilities < 0.40) / len(full_probabilities)) * 100
    ]
})

performance_report = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", "Average Precision"],
    "Value": [
        df_results.loc["ResNet-Attention", "Accuracy"],
        df_results.loc["ResNet-Attention", "Precision"],
        df_results.loc["ResNet-Attention", "Recall"],
        df_results.loc["ResNet-Attention", "F1-Score"],
        df_results.loc["ResNet-Attention", "AUC-ROC"],
        df_results.loc["ResNet-Attention", "Average Precision"]
    ]
}).round(4)

FINAL_EXCEL_PATH = os.path.join(OUTPUT_FOLDER, "AI_EWS_Final_Results.xlsx")
with pd.ExcelWriter(FINAL_EXCEL_PATH, engine="openpyxl") as writer:
    performance_report.to_excel(writer, sheet_name="Model_Performance", index=False)
    df_results.to_excel(writer, sheet_name="Model_Comparison")
    summary_report.to_excel(writer, sheet_name="Warning_Summary", index=False)
    full_student_report.to_excel(writer, sheet_name="Student_Details", index=False)

print(f"{GREEN}✓ Phase 14 completed: Professional report successfully saved in {FINAL_EXCEL_PATH}.{END}")
