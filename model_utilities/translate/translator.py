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

        self.__pipeline = pipeline("translation",
                                   model=self.__model,
                                   tokenizer=self.__tokenizer,
                                   max_length=self.__max_length)

    # def normalize_translation(self, input: list[str]) -> str | None:
    #     # Technically matching against 'Latn' will break if we do languages other than PL and CSB but we won't
    #     match = re.match(r"^.*Latn (.*)<.*$", input[0])
    #     # Workaround for when model spits out a translation that is too long and doesn't match regex.
    #     if match == None:
    #         self.__logger.warning(f"Warning translation does not match regex pattern: {input[0]}")
    #         return None
    #     return match.group(1)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        return self.__pipeline(text, src_lang=source_lang, tgt_lang=target_lang)[0]["translation_text"]
