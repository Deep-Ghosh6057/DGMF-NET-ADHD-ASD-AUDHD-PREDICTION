"""
===============================================================================
DGMF-Net Configuration
===============================================================================
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

FIGURE_DIR = PROJECT_ROOT / "figures"

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

TABLE_DIR = OUTPUT_DIR / "tables"

REPORT_DIR = OUTPUT_DIR / "reports"

LOG_DIR = OUTPUT_DIR / "logs"

# =============================================================================
# CREATE DIRECTORIES IF THEY DON'T EXIST
# =============================================================================

for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    TABLE_DIR,
    REPORT_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)