import pandas as pd
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
import re

import train_model

MODEL_NAME = "facebook/nllb-200-distilled-600M"

def word_tokenize(text: str) -> list:
    return re.findall("(\w+|[^\w\s])", text)

def prepare_translation_dataset(trans_df: pd.DataFrame, tokenizer: NllbTokenizer) -> pd.DataFrame:
    dataset = pd.DataFrame()

    dataset["csb_toks"] = trans_df.csb.apply(lambda x: tokenizer.tokenize(str(x)))
    dataset["pl_toks"] = trans_df.pl.apply(lambda x: tokenizer.tokenize(str(x)))

    dataset["csb_words"] = trans_df.csb.apply(lambda x: word_tokenize(str(x)))
    dataset["pl_words"] = trans_df.pl.apply(lambda x: word_tokenize(str(x)))

    return dataset

def main() -> None:
    data_path = "../pl-csb-data/data/train.tsv"
    trans_df = pd.read_csv(data_path, sep="\t")

    tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME, additional_special_tokens=["csb_Latn"])
    tokenizer.lang_code_to_id["csb_Latn"] = len(tokenizer.lang_code_to_id)
    tokenizer.id_to_lang_code[tokenizer.lang_code_to_id["csb_Latn"]] = "csb_Latn"

    dataset = prepare_translation_dataset(trans_df, tokenizer)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_model.train(model, trans_df, tokenizer)
