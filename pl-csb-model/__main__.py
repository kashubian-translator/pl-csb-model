from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

import config_loader
import data_loader
import model_finetuner

if __name__ == "__main__":
    default_config = config_loader.load_default()

    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(default_config["PretrainedModelName"])
    train_data = data_loader.load_train()
    tokenizer = NllbTokenizer.from_pretrained(default_config["PretrainedModelName"], additional_special_tokens=["csb_Latn"])

    model_finetuner.finetune(pretrained_model, train_data, tokenizer)
