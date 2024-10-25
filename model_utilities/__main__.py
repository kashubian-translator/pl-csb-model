import argparse
from logging import Logger

from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

import config_loader
import train.data_loader as data_loader
from evaluate.model_evaluator import ModelEvaluator
from train.logger import set_up_logger
from train.model_finetuner import ModelFinetuner
from translate.translator import Translator


def train_model(config: dict, logger: Logger) -> None:
    pretrained_model_name = config["MODEL"]["pretrained_model_name"]

    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
    tokenizer = NllbTokenizer.from_pretrained(pretrained_model_name, additional_special_tokens=["csb_Latn"])
    train_data = data_loader.load_data(config["DATA"]["training_data_file"])

    ModelFinetuner(logger).finetune(pretrained_model, tokenizer, train_data, config)


def use_model(config: dict, logger: Logger) -> None:
    output_model_name = config["MODEL"]["output_model_name"]

    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(output_model_name)
    tokenizer = NllbTokenizer.from_pretrained(output_model_name)
    message = "Wsiądźmy do tego autobusu"
    translated_message = Translator(logger, pretrained_model, tokenizer).translate(message)
    print(f"Message {message} has been translated to: {translated_message}")


def evaluate_model(config: dict, logger: Logger) -> None:
    output_model_name = config["MODEL"]["output_model_name"]

    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(output_model_name)
    tokenizer = NllbTokenizer.from_pretrained(output_model_name)
    eval_data = data_loader.load_data(config["DATA"]["evaluation_data_file"])

    source_language = eval_data[eval_data.columns[0]]
    target_language = eval_data[eval_data.columns[1]]

    evaluator = ModelEvaluator(logger)

    result = evaluator.evaluate(pretrained_model, tokenizer, sentences=source_language, references=target_language)
    print(f"BLEU Score ({source_language.name} -> {target_language.name}): {result.score}")

    result = evaluator.evaluate(pretrained_model, tokenizer, sentences=target_language, references=source_language)
    print(f"BLEU Score ({target_language.name} -> {source_language.name}): {result.score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "translate", "evaluate"], help="Mode to run the application with")

    args = parser.parse_args()

    logger = set_up_logger(__name__)

    config = config_loader.load()

    if args.mode == "train":
        train_model(config, logger)
    elif args.mode == "translate":
        use_model(config, logger)
    elif args.mode == "evaluate":
        evaluate_model(config, logger)
