import argparse
from loaders.ett import load_ett

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
        help="Which dataset to preprocess"
    )
    args = parser.parse_args()

    print(f"Preprocessing {args.dataset} ...")
    train, val, test = load_ett(args.dataset)
    print(f"Done! Train shape: {train.shape}, Val shape: {val.shape}, Test shape: {test.shape}")

if __name__ == "__main__":
    main()
