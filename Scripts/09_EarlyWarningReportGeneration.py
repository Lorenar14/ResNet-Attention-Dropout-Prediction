# =============================================================================
#  EARLY WARNING REPORT GENERATION
# =============================================================================

print_title("PHASE 13 - EARLY WARNING REPORT")

print_section("Preparing Active Students Dataset")

# =============================================================================
# Output File
# =============================================================================

output_file = "Early_Warning_Report_ResNet.xlsx"

# =============================================================================
# Select Active Students
# =============================================================================

active_students = df[
    df["ACADEMIC_STATUS"] == "Active"
].copy()

# =============================================================================
# Feature Scaling
# =============================================================================

X_active = scaler.transform(
    active_students[FEATURES].fillna(
        active_students[FEATURES].median()
    )
)

print(
    f"Active students evaluated : {len(active_students):,}"
)

# =============================================================================
# Predict Dropout Probability
# =============================================================================

print_section("Predicting Dropout Risk")

dropout_probability = (
    nn_model.predict(X_active)
    .flatten()
)

# =============================================================================
# Create Early Warning Report
# =============================================================================

report = pd.DataFrame({

    "Student ID":
        active_students["DOCUENTO_DE_IDENTIDAD"],

    "Academic Progress":
        active_students["ACADEMIC_PROGRESS"],

    "Dropout Probability":
        np.round(dropout_probability,1)

})

# =============================================================================
# Risk Classification
# =============================================================================

try:
    threshold = best_threshold
except NameError:
    threshold = 0.34

report["Risk Level"] = report[
    "Dropout Probability"
].apply(

    lambda x:
        "HIGH" if x >= 0.70
        else "MEDIUM" if x >= threshold
        else "LOW"

)

# =============================================================================
# Excel Formatting
# =============================================================================

print_section("Generating Excel Report")

high_fill = PatternFill(
    start_color="FFC7CE",
    end_color="FFC7CE",
    fill_type="solid"
)

medium_fill = PatternFill(
    start_color="FFEB9C",
    end_color="FFEB9C",
    fill_type="solid"
)

high_font = Font(
    color="9C0006",
    bold=True
)

medium_font = Font(
    color="9C6500",
    bold=True
)

# =============================================================================
# Export Report
# =============================================================================

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    report.to_excel(
        writer,
        sheet_name="Early Warning Report",
        index=False
    )

    ws = writer.sheets[
        "Early Warning Report"
    ]

    probability_column = (
        report.columns
        .get_loc("Dropout Probability") + 1
    )

    risk_column = (
        report.columns
        .get_loc("Risk Level") + 1
    )

    for row in range(2, len(report)+2):

        ws.cell(
            row=row,
            column=probability_column
        ).number_format = "0.0"

        risk_cell = ws.cell(
            row=row,
            column=risk_column
        )

        if risk_cell.value == "HIGH":

            risk_cell.fill = high_fill
            risk_cell.font = high_font

        elif risk_cell.value == "MEDIUM":

            risk_cell.fill = medium_fill
            risk_cell.font = medium_font

        risk_cell.alignment = Alignment(
            horizontal="center"
        )

# =============================================================================
# Summary
# =============================================================================

high = (report["Risk Level"]=="HIGH").sum()
medium = (report["Risk Level"]=="MEDIUM").sum()
low = (report["Risk Level"]=="LOW").sum()

print_section("Report Summary")

print(f"Total Active Students : {len(report):,}")
print(f"High Risk             : {high:,}")
print(f"Medium Risk           : {medium:,}")
print(f"Low Risk              : {low:,}")

print(f"\nDecision Threshold : {threshold:.1f}")

print("\n✓ Early Warning Report successfully generated.")
print(f"✓ File exported as: {output_file}")
