from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
import argparse

import config_loader
import train.data_loader as data_loader
import train.model_finetuner as model_finetuner
import translate.translator as translator

def train_model(config: dict) -> None:
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(config["MODEL"]["PretrainedModelName"])
    train_data = data_loader.load_train()
    tokenizer = NllbTokenizer.from_pretrained(config["MODEL"]["PretrainedModelName"], additional_special_tokens=["csb_Latn"])

    model_finetuner.finetune(pretrained_model, train_data, tokenizer, config)
    
def use_model(config: dict) -> None:
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(config["MODEL"]["OutputModelName"])
    tokenizer = NllbTokenizer.from_pretrained(config["MODEL"]["OutputModelName"])
    message="Wsiądźmy do tego autobusu"
    translated_message=translator.translate(message, pretrained_model, tokenizer, 'pol_Latn', 'csb_Latn')
    print(f"Message {message} has been translated to: {translated_message}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "translate"], help="Mode to run the application with")
    
    args = parser.parse_args()
    
    config = config_loader.load()
    
    if args.mode == "train":
        train_model(config)
    elif args.mode == "translate":
        use_model(config)

