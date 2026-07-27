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
