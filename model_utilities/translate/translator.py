import re
from logging import Logger

from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

class Translator:
    __logger: Logger
    __model: AutoModelForSeq2SeqLM
    __tokenizer: NllbTokenizer
    __forced_bos_token_id: int
    __max_input_length: int
    __num_beams: int
    __a: int
    __b: int

    def __init__(self, logger: Logger, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer,
                 src_lang = "pol_Latn", tgt_lang = "csb_Latn", max_input_length = 1024, num_beams = 4, a = 32, b = 3) -> None:
        self.__logger = logger

        self.__model = model
        self.__model.eval()

        self.__tokenizer = tokenizer
        self.__tokenizer.src_lang = src_lang
        self.__tokenizer.tgt_lang = tgt_lang

        self.__forced_bos_token_id = self.__tokenizer.convert_tokens_to_ids(tgt_lang)
        self.__max_input_length = max_input_length
        self.__num_beams = num_beams
        self.__a = a
        self.__b = b

    def normalize_translation(self, input: list[str]) -> str | None:
        # Technically matching against 'Latn' will break if we do languages other than PL and CSB but we won't
        match = re.match(r"^.*Latn (.*)<.*$", input[0])
        # Workaround for when model spits out a translation that is too long and doesn't match regex.
        if match == None:
            self.__logger.warning(f"Warning translation does not match regex pattern: {input[0]}")
            return None
        return match.group(1)


    def translate(self, text: str, **kwargs) -> list[str]:
        inputs = self.__tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.__max_input_length)
        result = self.__model.generate(
            **inputs.to(self.__model.device),
            forced_bos_token_id = self.__forced_bos_token_id,
            max_new_tokens = int(self.__a + self.__b * inputs.input_ids.shape[1]),
            num_beams = self.__num_beams, **kwargs
        )
        return self.__tokenizer.batch_decode(result)
