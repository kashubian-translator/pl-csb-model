import pandas as pd

def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path, sep="\t", index_col=0)
