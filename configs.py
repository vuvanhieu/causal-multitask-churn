# configs.py
import os

# ==== Risk Threshold Config ====
LOW_CREDIT_THRESHOLD = 580
HIGH_CREDIT_THRESHOLD = 700
LOW_BALANCE_THRESHOLD = 1000
HIGH_BALANCE_THRESHOLD = 100000
NEW_CUSTOMER_TENURE = 2
SENIOR_AGE_THRESHOLD = 60
# ===============================

# configs.py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==== Font size config for plots ====
BASE_FONT_SIZE = 14
TITLE_FONT_SIZE = 18
LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12

SHAP_NSAMPLES = 200
EXPLANATION_TOP_K = 5
JACCARD_BOXPLOT_COLORS = ['#ff9999', '#66b3ff', '#99ff99']


# ==============================
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Thư mục gốc của project
# ROOT_DIR = os.getcwd()
# Thư mục OUTPUT chung cho toàn bộ project
OUTPUT_DIR = Path(PROJECT_PATH) / "OUTPUT"

def create_directories(path: str) -> str:
    """
    Đảm bảo tồn tại thư mục path, sau đó trả lại path.
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_output_subdir(*subdirs) -> str:
    """
    Tạo đường dẫn con trong OUTPUT, ví dụ:
    get_output_subdir("Multitask_Deep_Learning", "Multitask DNN")
    """
    path = os.path.join(OUTPUT_DIR, *subdirs)
    os.makedirs(path, exist_ok=True)
    return path
