import pandas as pd
from transformers import NllbTokenizer
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES
from tqdm.auto import tqdm, trange
import re

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

def get_updated_tokenizer() -> NllbTokenizer:
    tokenizer = tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME, additional_special_tokens=["csb_Latn"])

    print(tokenizer.added_tokens_encoder)

    #Setting this makes the tokenizer automatically pre-pend tokenised text with the given language code.
    # tokenizer.src_lang = 'csb_Latn'

    #This should display the given text, pre-pended with the language code.
    # print(tokenizer.decode(tokenizer("Domëszlnô farwa").input_ids))
    return tokenizer
    

def main() -> None:
    data_path = "../pl-csb-data/data/train.tsv"
    trans_df = pd.read_csv(data_path, sep="\t")

    tokenizer = get_updated_tokenizer()

    model = NllbTokenizer.from_pretrained(MODEL_NAME)

    dataset = prepare_translation_dataset(trans_df, tokenizer)
