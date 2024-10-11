import argparse
import sacrebleu
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

import config_loader
import evaluate.model_evaluator as model_evaluator
import train.data_loader as data_loader
import train.model_finetuner as model_finetuner
import translate.translator as translator

def train_model(config: dict) -> None:
    pretrained_model_name = config["MODEL"]["pretrained_model_name"]

    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
    tokenizer = NllbTokenizer.from_pretrained(pretrained_model_name, additional_special_tokens=["csb_Latn"])
    train_data = data_loader.load_data(config["DATA"]["training_data_file"])

    model_finetuner.finetune(pretrained_model, tokenizer, train_data, config)
    
def use_model(config: dict) -> None:
    output_model_name = config["MODEL"]["output_model_name"]

    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(output_model_name)
    tokenizer = NllbTokenizer.from_pretrained(output_model_name)
    message="Wsiądźmy do tego autobusu"
    translated_message=translator.translate(message, pretrained_model, tokenizer, 'pol_Latn', 'csb_Latn')
    print(f"Message {message} has been translated to: {translated_message}")

def evaluate_model(config: dict) -> None:
    output_model_name = config["MODEL"]["output_model_name"]
    
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(output_model_name)
    tokenizer = NllbTokenizer.from_pretrained(output_model_name)
    eval_data = data_loader.load_data(config["DATA"]["evaluation_data_file"])

    source_language = eval_data[eval_data.columns[0]]
    target_language = eval_data[eval_data.columns[1]]

    result = model_evaluator.evaluate(pretrained_model, tokenizer, sentences=source_language, references=target_language)
    print(f"BLEU Score ({source_language.name} -> {target_language.name}): {result.score}")

    result = model_evaluator.evaluate(pretrained_model, tokenizer, sentences=target_language, references=source_language)
    print(f"BLEU Score ({target_language.name} -> {source_language.name}): {result.score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "translate", "evaluate"], help="Mode to run the application with")
    
    args = parser.parse_args()
    
    config = config_loader.load()
    
    if args.mode == "train":
        train_model(config)
    elif args.mode == "translate":
        use_model(config)
    elif args.mode == "evaluate":
        evaluate_model(config)

