from logging import Logger

import pandas as pd
import sacrebleu
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

from translate.translator import Translator


class ModelEvaluator:
    __logger: Logger

    def __init__(self, logger, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer) -> None:
        self.__logger = logger
        self.__translator = Translator(self.__logger, model, tokenizer)

    # def __remove_corrupted_sentences(self, sentences: pd.Series, references: pd.Series) -> tuple[pd.Series, pd.Series]:
    #     df = pd.concat([sentences, references], axis=1).dropna()
    #     valid_sentences, valid_references = df.iloc[:, 0], df.iloc[:, 1]

    #     removed_sentence_count = len(sentences) - len(valid_sentences)
    #     if (removed_sentence_count > 0):
    #         self.__logger.info(f"Removing {removed_sentence_count} corrupted translations from evaluation dataset")

    #     return valid_sentences, valid_references

    def evaluate_bleu(self, sentences: pd.Series, references: pd.Series) -> sacrebleu.metrics.BLEUScore:
        source_lang = sentences.name
        target_lang = references.name

        translated_sentences: pd.Series = sentences.map(lambda s: self.__translator.translate(s, source_lang, target_lang))
        # translated_sentences, references = self.__remove_corrupted_sentences(translated_sentences, references)

        bleu = sacrebleu.corpus_bleu(translated_sentences.to_list(), references.map(lambda s: [s]).to_list())
        return bleu

    def evaluate_chrf(self, sentences: pd.Series, references: pd.Series) -> sacrebleu.metrics.CHRFScore:
        source_lang = sentences.name
        target_lang = references.name

        translated_sentences = sentences.map(lambda s: self.__translator.translate(s, source_lang, target_lang))
        # translated_sentences, references = self.__remove_corrupted_sentences(translated_sentences, references)

        # Word order = 2 means chrf++ instead of chrf
        chrfpp = sacrebleu.corpus_chrf(translated_sentences.to_list(), references.map(lambda s: [s]).to_list(), word_order=2)
        return chrfpp
