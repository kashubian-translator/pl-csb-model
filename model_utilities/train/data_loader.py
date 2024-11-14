import pandas as pd
import datasets as ds


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path, sep="\t", index_col=0)


def load_dataset(train_path: str, validation_path: str, test_path: str) -> ds.DatasetDict:
    paths = {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
    }

    dataset = ds.load_dataset("csv", data_files=paths, sep="\t")

    # Removing the incremental id column as we dont need it
    dataset = dataset.remove_columns("Unnamed: 0")

    return dataset
