from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR/"data"/"train.csv"
TEST_PATH = BASE_DIR/"data"/"test.csv"

TARGET_COL = "SalePrice"
DROP_COLS = ["Id"]