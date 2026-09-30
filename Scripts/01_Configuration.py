import os
import random
import warnings
import logging
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Input, Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, roc_curve, auc
)

from imblearn.over_sampling import SMOTE


# =============================================================================
# SETUP AND REPRODUCIBILITY
# =============================================================================

RANDOM_SEED = 42

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    "savefig.bbox": "tight"
})

PALETTE_BLUE_DARK = "#2C3E50"
PALETTE_BLUE_MED = "#34495E"
PALETTE_SLATE = "#7F8C8D"
PALETTE_RED_DARK = "#C0392B"

SUCCESS_GREEN = "\033[92m"
COLOR_RESET = "\033[0m"

def log_progress(msg):
    print(f"{SUCCESS_GREEN}✓ {msg}{COLOR_RESET}")

log_progress("Environment and visualization parameters successfully configured.")
