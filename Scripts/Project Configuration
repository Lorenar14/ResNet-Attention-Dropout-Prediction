# Standard libraries
import logging
import os
import random
from pathlib import Path

# Reproducibility settings
RANDOM_SEED = 42

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# Project directories
PROJECT_DIR = Path.cwd()

OUTPUT_DIR = PROJECT_DIR / "Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def log_progress(message):
    """Log the progress of the analysis pipeline."""
    logger.info(message)


# Data processing and visualization libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# Class balancing
from imblearn.over_sampling import SMOTE

# Machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Model evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from xgboost import XGBClassifier

# ResNet-Attention architecture
from tensorflow.keras import Input, Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Random seed initialization
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

try:
    tf.config.experimental.enable_op_determinism()
    log_progress("TensorFlow deterministic operations enabled.")
except (AttributeError, RuntimeError):
    logger.warning(
        "TensorFlow deterministic operations could not be enabled."
    )

# Visualization color palette
PALETTE_BLUE_DARK = "#17365D"
PALETTE_RED_DARK = "#8B1E2D"
PALETTE_TEAL = "#2A7F7F"
PALETTE_OCHRE = "#C08A24"

# Model color assignments
MODEL_COLORS = {
    "ResNet-Attention": PALETTE_BLUE_DARK,
    "Logistic Regression": PALETTE_RED_DARK,
    "Random Forest": PALETTE_TEAL,
    "XGBoost": PALETTE_OCHRE,
}

# Global visualization settings
sns.set_theme(style="whitegrid")

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# Configuration summary
log_progress("Phase 1: Project configuration completed.")
log_progress(f"Project directory: {PROJECT_DIR}")
log_progress(f"Output directory: {OUTPUT_DIR}")
log_progress(f"Random seed: {RANDOM_SEED}")
log_progress("Four-color visualization palette configured.")
