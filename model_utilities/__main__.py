import argparse
from logging import Logger

from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, pipeline

import config_loader
import train.data_loader as data_loader
from evaluate.evaluator import ModelEvaluator
from train.logger import set_up_logger
from train.finetuner import ModelFinetuner
from translate.translator import Translator


def train_model(config: dict, logger: Logger) -> None:
    pretrained_model_name = config["MODEL"]["pretrained_model_name"]

    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
    tokenizer = NllbTokenizer.from_pretrained(pretrained_model_name, additional_special_tokens=["csb_Latn"])
    train_data = data_loader.load_data(config["DATA"]["training_data_file"])

    ModelFinetuner(logger).finetune(pretrained_model, tokenizer, train_data, config)


def use_model(config: dict, logger: Logger) -> None:
    output_model_name = config["MODEL"]["output_model_name"]

    model = AutoModelForSeq2SeqLM.from_pretrained(output_model_name)
    tokenizer = NllbTokenizer.from_pretrained(output_model_name)
    message = "Wsiądźmy do tego autobusu"

    translated_message = Translator(logger, model, tokenizer).translate(message, "pol_Latn", "csb_Latn")

    print(f"Message {message} has been translated to: {translated_message}")


def evaluate_model(config: dict, logger: Logger) -> None:
    output_model_name = config["MODEL"]["output_model_name"]

    model = AutoModelForSeq2SeqLM.from_pretrained(output_model_name)
    tokenizer = NllbTokenizer.from_pretrained(output_model_name)
    eval_data = data_loader.load_data(config["DATA"]["evaluation_data_file"])

    source_data = eval_data[eval_data.columns[0]]
    target_data = eval_data[eval_data.columns[1]]

    evaluator = ModelEvaluator(logger, model, tokenizer)

    bleu, chrfpp = evaluator.evaluate(sentences=source_data, references=target_data)
    logger.info(f"BLEU Score ({source_data.name} -> {target_data.name}): {bleu.score}")
    logger.info(f"chrF++ Score ({source_data.name} -> {target_data.name}): {chrfpp.score}")

    bleu, chrfpp = evaluator.evaluate(sentences=target_data, references=source_data)
    logger.info(f"BLEU Score ({target_data.name} -> {source_data.name}): {bleu.score}")
    logger.info(f"chrF++ Score ({target_data.name} -> {source_data.name}): {chrfpp.score}")


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
