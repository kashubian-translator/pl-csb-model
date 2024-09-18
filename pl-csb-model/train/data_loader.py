import pandas as pd

def load_train() -> pd.DataFrame:
    data_path = "../pl-csb-data/data/output/train.tsv"
    return pd.read_csv(data_path, sep="\t", index_col=0)
