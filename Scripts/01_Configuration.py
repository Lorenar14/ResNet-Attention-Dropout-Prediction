# =============================================================================
# EARLY WARNING SYSTEM FOR UNIVERSITY DROPOUT PREDICTION
# Environment Configuration
# =============================================================================

%%capture

# -----------------------------------------------------------------------------
# Install required Python packages (Google Colab)
# -----------------------------------------------------------------------------

!pip install -q lifelines
!pip install -q graphviz
!pip install -q openpyxl

# -----------------------------------------------------------------------------
# Install Microsoft TrueType fonts (Optional)
# -----------------------------------------------------------------------------

!apt-get -qq install ttf-mscorefonts-installer
!fc-cache -fv

import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
