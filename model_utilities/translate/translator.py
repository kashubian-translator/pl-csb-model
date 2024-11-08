import re
from logging import Logger

from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, Pipeline, pipeline


class Translator:
    __logger: Logger
    __model: AutoModelForSeq2SeqLM
    __tokenizer: NllbTokenizer
    __pipeline: Pipeline

    def __init__(self, logger: Logger, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer, max_length=200) -> None:
        self.__logger = logger

        self.__model = model

        self.__tokenizer = tokenizer
        self.__max_length = max_length

        self.__pipeline = pipeline(task="translation",
                                   model=self.__model,
                                   tokenizer=self.__tokenizer,
                                   max_length=self.__max_length)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        model_output = self.__pipeline(text, src_lang=source_lang, tgt_lang=target_lang)
        translation_text = model_output[0]["translation_text"]
        self.__logger.info(f"Translation from {source_lang} to {target_lang}: {translation_text}")
        return translation_text
