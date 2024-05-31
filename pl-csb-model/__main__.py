from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

import config_loader
import data_loader
import model_finetuner

if __name__ == "__main__":
    config = config_loader.load()

    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(config["MODEL"]["PretrainedModelName"])
    train_data = data_loader.load_train()
    tokenizer = NllbTokenizer.from_pretrained(config["MODEL"]["PretrainedModelName"], additional_special_tokens=["csb_Latn"])

    model_finetuner.finetune(pretrained_model, train_data, tokenizer, config)
