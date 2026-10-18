
import pickle
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

MODEL_VERSION = "1.0.0"


def model_predict(user_input: dict):
    input_df = pd.DataFrame([user_input])
    output = model.predict(input_df)[0]
    return output
