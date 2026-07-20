import pandas as pd


def load_catalog(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["model_name"] = df["model_name"].str.strip()
    df["required_gpu"] = df["required_gpu"].str.strip()

    duplicates = df[df.duplicated(subset="model_name")]
    if not duplicates.empty:
        print("Warning: duplicate models found:")
        print(duplicates)

    return df


if __name__ == "__main__":
    catalog = load_catalog("../data/model_catalog/model_catalog.csv")
    print(catalog)
    print(f"\nLoaded {len(catalog)} models")
