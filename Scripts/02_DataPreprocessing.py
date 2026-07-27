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


# =============================================================================
# DATA LOADING AND ANONYMIZATION
# =============================================================================

print_title(" DATA LOADING AND ANONYMIZATION")

# =============================================================================
# Dataset Configuration
# =============================================================================

DATASET_PATH = DATA_DIR / "anonymized_student_dataset.csv"

MASTER_KEY_FILE = "Master_Key.xlsx"

ANONYMIZED_DATASET_FILE = "Anonymized_Dataset.csv"

EXPORT_MASTER_KEY = True

EXPORT_ANONYMIZED_DATASET = True

# =============================================================================
# Load Dataset
# =============================================================================

print_section("Loading Dataset")

df_raw = pd.read_csv(
    DATASET_PATH,
    sep=";",
    encoding="latin-1"
)

print(f"{GREEN}✓ Dataset loaded successfully.{END}")

# =============================================================================
# Standardize Column Names
# =============================================================================

print_section("Standardizing Column Names")

df_raw.columns = (

    df_raw.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
        .str.replace(r"[^A-Z0-9_]", "", regex=True)

)

print(f"{GREEN}✓ Column names standardized.{END}")

# =============================================================================
#  Clean Identification Fields
# =============================================================================

print_section("Cleaning Identification Fields")

IDENTIFICATION_COLUMNS = [

    "DOCUENTO_DE_IDENTIDAD",
    "NOMBRE_COMPLETO"

]

for column in IDENTIFICATION_COLUMNS:

    df_raw[column] = (

        df_raw[column]
            .astype(str)
            .str.strip()

    )

print(f"{GREEN}✓ Identification fields cleaned.{END}")

# =============================================================================
#  Create Master Key
# =============================================================================

print_section("Creating Master Key")

master_key = pd.DataFrame({

    "DOCUENTO_DE_IDENTIDAD_ORIGINAL":

        df_raw["DOCUENTO_DE_IDENTIDAD"],

    "NOMBRE_COMPLETO_ORIGINAL":

        df_raw["NOMBRE_COMPLETO"]

})

print(f"{GREEN}✓ Master key created.{END}")

# =============================================================================
#  SHA-256 Anonymization
# =============================================================================

print_section("Applying SHA-256 Anonymization")


def sha256_hash(value):
    """
    Generate a SHA-256 hash from a string.

    Parameters
    ----------
    value : str
        Original sensitive value.

    Returns
    -------
    str
        SHA-256 encrypted value.
    """

    return hashlib.sha256(
        str(value).encode("utf-8")
    ).hexdigest()


#------------------------------------------------------------------------------
# Generate cryptographic hashes
#------------------------------------------------------------------------------

id_hash = df_raw["DOCUENTO_DE_IDENTIDAD"].apply(sha256_hash)

name_hash = df_raw["NOMBRE_COMPLETO"].apply(sha256_hash)

#------------------------------------------------------------------------------
# Replace sensitive information
#------------------------------------------------------------------------------

master_key["DOCUENTO_DE_IDENTIDAD"] = id_hash

df_raw["DOCUENTO_DE_IDENTIDAD"] = id_hash

df_raw["NOMBRE_COMPLETO"] = name_hash

print(f"{GREEN}✓ Sensitive information anonymized successfully.{END}")

# =============================================================================
#  Export Files
# =============================================================================

print_section("Exporting Files")

if EXPORT_MASTER_KEY:

    master_key.to_excel(

        os.path.join(
            OUTPUT_FOLDER,
            MASTER_KEY_FILE
        ),

        index=False

    )

    print(f"{GREEN}✓ Master Key exported successfully.{END}")

if EXPORT_ANONYMIZED_DATASET:

    df_raw.to_csv(

        os.path.join(
            OUTPUT_FOLDER,
            ANONYMIZED_DATASET_FILE
        ),

        sep=";",
        encoding="utf-8",
        index=False

    )

    print(f"{GREEN}✓ Anonymized dataset exported successfully.{END}")

# =============================================================================
# Dataset Summary
# =============================================================================

print_section("Dataset Summary")

print(f"Dataset Shape             : {df_raw.shape}")

print(f"Total Records             : {df_raw.shape[0]:,}")

print(f"Total Variables           : {df_raw.shape[1]}")

print(f"Missing Values            : {df_raw.isnull().sum().sum():,}")

print(f"Duplicate Records         : {df_raw.duplicated().sum():,}")

print(f"\n{GREEN}{'='*90}")
print("PHASE 2 COMPLETED SUCCESSFULLY")
print(f"{'='*90}{END}")
