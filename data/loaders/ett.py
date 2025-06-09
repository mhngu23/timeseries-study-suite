import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

RAW_ROOT = os.path.join(os.path.dirname(__file__), "..", "raw")
PROCESSED_ROOT = os.path.join(os.path.dirname(__file__), "..", "processed")
# print(RAW_ROOT)
# print(PROCESSED_ROOT)
def load_ett(dataset_name="ETTh1", raw_dir=RAW_ROOT, save_root=PROCESSED_ROOT, split_ratios=(0.7, 0.2, 0.1)):
    assert dataset_name in {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}, f"Unknown dataset: {dataset_name}"

    # Prepare paths
    save_dir = os.path.join(save_root, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # print(save_dir)
    # print(raw_dir)
    # exit()
    filename = os.path.join(raw_dir, f"{dataset_name}.csv")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Please download {dataset_name}.csv and place it in {raw_dir}")

    df = pd.read_csv(filename, parse_dates=["date"])
    # Set the data column as the index
    df.set_index("date", inplace=True)

    # Drop rows with NaN values
    # This is important to ensure that the data is clean before processing
    df.dropna(inplace=True)

    # Separate target and features
    target_col = "OT"
    features = df.drop(columns=[target_col])
    target = df[target_col]

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Save stats
    stats = {
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "feature_names": features.columns.tolist()
    }
    with open(os.path.join(save_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Reconstruct DataFrame
    all_data = np.concatenate([features_scaled, target.values.reshape(-1, 1)], axis=1)
    all_df = pd.DataFrame(
        all_data,
        index=features.index,
        columns=features.columns.tolist() + [target_col]
    )

    # Split
    n = len(all_df)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    train_df = all_df.iloc[:n_train]
    val_df = all_df.iloc[n_train:n_train + n_val]
    test_df = all_df.iloc[n_train + n_val:]

    # Save splits
    train_df.to_csv(os.path.join(save_dir, "train.csv"))
    val_df.to_csv(os.path.join(save_dir, "val.csv"))
    test_df.to_csv(os.path.join(save_dir, "test.csv"))

    return train_df, val_df, test_df


