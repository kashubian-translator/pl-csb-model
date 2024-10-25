from logging import Logger

import pandas as pd
import sacrebleu
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

from translate.translator import Translator


class ModelEvaluator:
    __logger: Logger

    def __init__(self, logger) -> None:
        self.__logger = logger

    def __remove_corrupted_sentences(self, sentences: pd.Series, references: pd.Series) -> tuple[pd.Series, pd.Series]:
        df = pd.concat([sentences, references], axis=1).dropna()
        valid_sentences, valid_references = df.iloc[:, 0], df.iloc[:, 1]

        removed_sentence_count = len(sentences) - len(valid_sentences)
        if (removed_sentence_count > 0):
            self.__logger.info(f"Removing {removed_sentence_count} corrupted translations from evaluation dataset")

        return valid_sentences, valid_references

    def evaluate(self, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer, sentences: pd.Series, references: pd.Series) -> sacrebleu.metrics.BLEUScore:
        source_lang = sentences.name
        target_lang = references.name
        translator = Translator(self.__logger, model, tokenizer, source_lang, target_lang)

        translated_sentences = sentences.map(lambda s: translator.translate(s))
        translated_sentences = translated_sentences.map(lambda s: translator.normalize_translation(s))

        translated_sentences, references = self.__remove_corrupted_sentences(translated_sentences, references)

        bleu = sacrebleu.corpus_bleu(translated_sentences.to_list(), references.map(lambda s: [s]).to_list())
        return bleu
