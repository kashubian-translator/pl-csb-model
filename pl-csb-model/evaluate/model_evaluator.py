import pandas as pd
import sacrebleu
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

import translate.translator as translator


def evaluate(model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer, sentences: pd.Series, references: pd.Series) -> sacrebleu.metrics.BLEUScore:
    source_lang = sentences.name
    target_lang = references.name

    translated_sentences = sentences.map(lambda s: translator.translate(s, model, tokenizer, source_lang, target_lang))
    translated_sentences = translated_sentences.map(lambda s: translator.normalize_translation(s))

    bleu = sacrebleu.corpus_bleu(translated_sentences.to_list(), references.map(lambda s: [s]).to_list())
    return bleu
