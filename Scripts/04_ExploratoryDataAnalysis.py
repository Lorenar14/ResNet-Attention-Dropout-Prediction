# -----------------------------------------------------------------------------
# Standard Python Libraries
# -----------------------------------------------------------------------------

import os
import random
import hashlib
import warnings
import logging

# -----------------------------------------------------------------------------
# Numerical Computing
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Data Visualization
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import seaborn as sns

# =============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print_title("EXPLORATORY DATA ANALYSIS")

# =============================================================================
# DATASET OVERVIEW
# =============================================================================

print_section("Dataset Overview")

print(f"Dataset Shape              : {df.shape}")
print(f"Number of Observations     : {df.shape[0]:,}")
print(f"Number of Variables        : {df.shape[1]}")

print("\nVariable Types")

display(df.dtypes.to_frame("Data Type"))

# =============================================================================
# Missing Values
# =============================================================================

missing = (
    df.isnull()
      .sum()
      .to_frame("Missing Values")
)

missing["Percentage (%)"] = (
    missing["Missing Values"]
    / len(df)
    * 100
).round(2)

display(
    missing.sort_values(
        "Missing Values",
        ascending=False
    )
)

# =============================================================================
# Duplicate Records
# =============================================================================

duplicates = df.duplicated().sum()

print(f"\nDuplicate Records          : {duplicates:,}")

# =============================================================================
# Memory Usage
# =============================================================================

memory_usage = (
    df.memory_usage(deep=True).sum()
    / 1024**2
)

print(f"Memory Usage               : {memory_usage:.2f} MB")

# =============================================================================
# Target Distribution
# =============================================================================

print("\nTarget Distribution")

display(

    df["ACADEMIC_STATUS"]

    .value_counts()

    .rename_axis("Academic Status")

    .reset_index(name="Students")

)

# =============================================================================
# Numerical Variables
# =============================================================================

numerical_variables = df.select_dtypes(
    include=np.number
).columns.tolist()

print("\nNumerical Variables")

print(numerical_variables)

# =============================================================================
# Categorical Variables
# =============================================================================

categorical_variables = df.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nCategorical Variables")

print(categorical_variables)

print(f"\n{GREEN}✓ Dataset overview completed successfully.{END}")

# =============================================================================
# DISTRIBUTION DIAGRAM
# Violin Plot + Boxplot
# =============================================================================

print_section("Distribution Diagram")

# =============================================================================
# Prepare Data
# =============================================================================

active_students = df[df["ACADEMIC_STATUS"] == "Active"]
dropout_students = df[df["ACADEMIC_STATUS"] == "Dropout"]
graduated_students = df[df["ACADEMIC_STATUS"] == "Graduated"]

# =============================================================================
# Create Figure
# =============================================================================

fig, ax = plt.subplots(figsize=(7.2, 5.2))

# =============================================================================
# Violin Plot
# =============================================================================

sns.violinplot(
    data=df[df["ACADEMIC_STATUS"] != "Graduated"],
    x="ACADEMIC_STATUS",
    y="AVANCE_NU",
    order=["Active", "Dropout"],
    palette=[BLUE_DARK, RED_DARK],
    inner=None,
    cut=0,
    linewidth=1.3,
    saturation=0.95,
    ax=ax
)

# =============================================================================
# Boxplot
# =============================================================================

sns.boxplot(
    data=df,
    x="ACADEMIC_STATUS",
    y="AVANCE_NU",
    order=["Active", "Dropout", "Graduated"],
    width=0.18,
    showfliers=False,
    boxprops={
        "facecolor": "white",
        "edgecolor": "black",
        "linewidth": 1.2
    },
    medianprops={
        "color": "black",
        "linewidth": 2
    },
    whiskerprops={
        "linewidth": 1.2
    },
    capprops={
        "linewidth": 1.2
    },
    ax=ax
)

# =============================================================================
# Graduated Reference
# =============================================================================

ax.plot(
    2,
    100,
    marker="o",
    markersize=7,
    color="gray",
    zorder=5
)

ax.hlines(
    100,
    1.75,
    2.25,
    colors="gray",
    linewidth=2
)

# =============================================================================
# Median Labels
# =============================================================================

for i, status in enumerate(["Active", "Dropout", "Graduated"]):

    median = df.loc[
        df["ACADEMIC_STATUS"] == status,
        "AVANCE_NU"
    ].median()

    ax.text(
        i,
        median + 2,
        f"{median:.1f}",
        ha="center",
        fontsize=9,
        fontweight="bold"
    )

# =============================================================================
# Sample Size
# =============================================================================

labels = [

    f"Active\n(n={len(active_students):,})",

    f"Dropout\n(n={len(dropout_students):,})",

    f"Graduated\n(n={len(graduated_students):,})"

]

ax.set_xticklabels(labels)

# =============================================================================
# Axis Labels
# =============================================================================

ax.set_xlabel("")

ax.set_ylabel("Academic Progress (%)")

ax.set_ylim(5, 102)

# =============================================================================
# Grid
# =============================================================================

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.20
)

sns.despine(
    top=True,
    right=True
)

# =============================================================================
# Layout
# =============================================================================

plt.tight_layout()

# =============================================================================
# Export Figures
# =============================================================================

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Figure_01_Distribution_Diagram.pdf"
    ),
    bbox_inches="tight"
)

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Figure_01_Distribution_Diagram.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# =============================================================================
# PAIRPLOT ANALYSIS
# =============================================================================

print_section("Pairplot Analysis")

# =============================================================================
# Variables
# =============================================================================

variables = [
    "EDAD",
    "AVANCE_NU",
    "SEMESTRES_TOTAL_POR_CARRERA"
]

# =============================================================================
# Stratified Sampling
# =============================================================================

MAX_SAMPLES = 1500

df_plot = (
    df.groupby("target", group_keys=False)
      .apply(
          lambda x: x.sample(
              min(MAX_SAMPLES, len(x)),
              random_state=SEED
          )
      )
      .reset_index(drop=True)
)

# =============================================================================
# Color Palette
# =============================================================================

palette = {
    0: BLUE_DARK,
    1: RED_DARK
}

# =============================================================================
# Pearson Correlation Function
# =============================================================================

def correlation_annotation(x, y, **kwargs):

    ax = plt.gca()

    if len(ax.texts) == 0:

        r, p = pearsonr(x, y)

        color = RED_DARK if r >= 0 else BLUE_DARK

        ax.annotate(
            f"$r$ = {r:.2f}",
            xy=(0.50, 0.60),
            xycoords="axes fraction",
            ha="center",
            fontsize=12,
            color=color,
            fontweight="bold"
        )

        if p < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {p:.3f}"

        ax.annotate(
            p_text,
            xy=(0.50, 0.42),
            xycoords="axes fraction",
            ha="center",
            fontsize=9
        )

    ax.set_xticks([])
    ax.set_yticks([])

# =============================================================================
# PairGrid
# =============================================================================

g = sns.PairGrid(
    df_plot,
    vars=variables,
    hue="target",
    palette=palette,
    height=2.8,
    diag_sharey=False
)

# =============================================================================
# Lower Triangle
# =============================================================================

g.map_lower(
    sns.scatterplot,
    s=18,
    alpha=0.35,
    edgecolor=None
)

# =============================================================================
# Diagonal
# =============================================================================

g.map_diag(
    sns.kdeplot,
    fill=True,
    alpha=0.50,
    linewidth=1.5
)

# =============================================================================
# Upper Triangle
# =============================================================================

g.map_upper(
    correlation_annotation
)

# =============================================================================
# Legend
# =============================================================================

g.add_legend(title="Student Status")

legend = g._legend

legend.texts[0].set_text("No Dropout")
legend.texts[1].set_text("Dropout")

legend.get_frame().set_linewidth(0)

# =============================================================================
# Remove Spines
# =============================================================================

for ax in g.axes.flatten():

    sns.despine(
        ax=ax,
        top=True,
        right=True
    )

# =============================================================================
# Layout
# =============================================================================

g.fig.subplots_adjust(
    left=0.08,
    right=0.97,
    bottom=0.08,
    top=0.96,
    wspace=0.03,
    hspace=0.03
)

# =============================================================================
# Export Figures
# =============================================================================

g.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Figure_02_Pairplot.pdf"
    ),
    bbox_inches="tight"
)

g.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Figure_02_Pairplot.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# =============================================================================
# PAIRPLOT ANALYSIS
# =============================================================================

print_section("Pairplot Analysis")

# =============================================================================
# Variables
# =============================================================================

variables = [
    "EDAD",
    "AVANCE_NU",
    "SEMESTRES_TOTAL_POR_CARRERA"
]

# =============================================================================
# Stratified Sampling
# =============================================================================

MAX_SAMPLES = 1500

df_plot = (
    df.groupby("target", group_keys=False)
      .apply(
          lambda x: x.sample(
              min(MAX_SAMPLES, len(x)),
              random_state=SEED
          )
      )
      .reset_index(drop=True)
)

# =============================================================================
# Color Palette
# =============================================================================

palette = {
    0: BLUE_DARK,
    1: RED_DARK
}

# =============================================================================
# Pearson Correlation Function
# =============================================================================

def correlation_annotation(x, y, **kwargs):

    ax = plt.gca()

    if len(ax.texts) == 0:

        r, p = pearsonr(x, y)

        color = RED_DARK if r >= 0 else BLUE_DARK

        ax.annotate(
            f"$r$ = {r:.2f}",
            xy=(0.50, 0.60),
            xycoords="axes fraction",
            ha="center",
            fontsize=12,
            color=color,
            fontweight="bold"
        )

        if p < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {p:.3f}"

        ax.annotate(
            p_text,
            xy=(0.50, 0.42),
            xycoords="axes fraction",
            ha="center",
            fontsize=9
        )

    ax.set_xticks([])
    ax.set_yticks([])

# =============================================================================
# PairGrid
# =============================================================================

g = sns.PairGrid(
    df_plot,
    vars=variables,
    hue="target",
    palette=palette,
    height=2.8,
    diag_sharey=False
)

# =============================================================================
# Lower Triangle
# =============================================================================

g.map_lower(
    sns.scatterplot,
    s=18,
    alpha=0.35,
    edgecolor=None
)

# =============================================================================
# Diagonal
# =============================================================================

g.map_diag(
    sns.kdeplot,
    fill=True,
    alpha=0.50,
    linewidth=1.5
)

# =============================================================================
# Upper Triangle
# =============================================================================

g.map_upper(
    correlation_annotation
)

# =============================================================================
# Legend
# =============================================================================

g.add_legend(title="Student Status")

legend = g._legend

legend.texts[0].set_text("No Dropout")
legend.texts[1].set_text("Dropout")

legend.get_frame().set_linewidth(0)

# =============================================================================
# Remove Spines
# =============================================================================

for ax in g.axes.flatten():

    sns.despine(
        ax=ax,
        top=True,
        right=True
    )

# =============================================================================
# Layout
# =============================================================================

g.fig.subplots_adjust(
    left=0.08,
    right=0.97,
    bottom=0.08,
    top=0.96,
    wspace=0.03,
    hspace=0.03
)

# =============================================================================
# Export Figures
# =============================================================================

g.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Figure_02_Pairplot.pdf"
    ),
    bbox_inches="tight"
)

g.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Figure_02_Pairplot.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# =============================================================================
# DROPOUT PROBABILITY ACROSS ACADEMIC SEMESTERS
# =============================================================================

print_section("Dropout Probability Across Academic Semesters")

from matplotlib.ticker import MaxNLocator

# =============================================================================
# Calculate Dropout Probability by Semester
# =============================================================================

dropout = (
    df.groupby("NIVEL_ACTUAL")
      .agg(
          Dropout_Probability=("target", "mean"),
          Students=("target", "count")
      )
      .reset_index()
      .sort_values("NIVEL_ACTUAL")
)

# =============================================================================
# Create Figure
# =============================================================================

fig, ax = plt.subplots(figsize=(7.8, 5.6))

# =============================================================================
# Line
# =============================================================================

ax.plot(
    dropout["NIVEL_ACTUAL"],
    dropout["Dropout_Probability"],
    color=BLUE_DARK,
    linewidth=2.5,
    marker="o",
    markersize=6,
    markerfacecolor=RED_DARK,
    markeredgecolor="white",
    markeredgewidth=0.8
)

# =============================================================================
# Fill Area (optional, improves visualization)
# =============================================================================

ax.fill_between(
    dropout["NIVEL_ACTUAL"],
    dropout["Dropout_Probability"],
    color=BLUE_DARK,
    alpha=0.08
)

# =============================================================================
# Axis Labels
# =============================================================================

ax.set_xlabel(
    "Academic Semester",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "Dropout Probability",
    fontsize=12,
    fontweight="bold"
)

# =============================================================================
# Axis Limits
# =============================================================================

ax.set_xlim(
    dropout["NIVEL_ACTUAL"].min(),
    dropout["NIVEL_ACTUAL"].max()
)

ax.set_ylim(
    0,
    dropout["Dropout_Probability"].max() * 1.10
)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# =============================================================================
# Grid
# =============================================================================

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.25
)

# =============================================================================
# Remove Spines
# =============================================================================

sns.despine(
    top=True,
    right=True
)

# =============================================================================
# Tick Labels
# =============================================================================

ax.tick_params(
    axis="both",
    labelsize=11
)

# =============================================================================
# Layout
# =============================================================================

plt.tight_layout()

# =============================================================================
# Export Figures
# =============================================================================

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Figure_04_Dropout_Probability.pdf"
    ),
    bbox_inches="tight"
)

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Figure_04_Dropout_Probability.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.show()
