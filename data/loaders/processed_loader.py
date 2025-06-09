import os
import pandas as pd

def load_processed_ett(path, target_col="OT"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)

    features = df.drop(columns=[target_col])
    target = df[target_col]

    return features, target.values