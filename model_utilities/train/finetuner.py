import gc
import random
from logging import Logger

import matplotlib.pyplot as plt
import torch
import pandas as pd
import datasets as ds
from tqdm.auto import trange
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, get_constant_schedule_with_warmup
from transformers.optimization import Adafactor

from configparser import ConfigParser


class ModelFinetuner:
    __logger: Logger

    def __init__(self, logger) -> None:
        self.__logger = logger

    def __cleanup(self) -> None:
        """Try to free GPU memory"""
        gc.collect()
        torch.cuda.empty_cache()

    def __log_train_config(self, config: ConfigParser) -> None:
        self.__logger.info("=" * 40)
        self.__logger.info("CONFIGURATION SETTINGS")
        self.__logger.info("=" * 40)
        for section in config.sections():
            self.__logger.info(f"[{section}]")
            for key, value in config.items(section):
                self.__logger.info(f"{key}: {value}")
            self.__logger.info("-" * 40)
        self.__logger.info("=" * 40 + "\n")

    def __plot_losses(self, training_losses: list[float], validation_losses: list[float]) -> None:
        plt.plot(training_losses, label="Training Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("./debug/graphs/losses.png")

    def __train(self, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer, dataset: ds.Dataset, optimizer: Adafactor,
                config: ConfigParser) -> None:
        self.__log_train_config(config)

        train_conf = config["TRAINING"]

        training_losses = []
        validation_losses = []
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(train_conf["warmup_steps"]))

        train_batches = dataset["train"].batch(batch_size=int(train_conf["batch_size"]))
        val_batches = dataset["validation"].batch(batch_size=int(train_conf["batch_size"]))

        self.__logger.debug("Starting the training process")
        model.train()
        x, y, loss = None, None, None
        self.__cleanup()

        num_epochs = int(train_conf["num_epochs"])
        batch_num = len(train_batches)
        patience = int(train_conf.get("early_stop_patience_in_epochs"))
        best_val_loss = float("inf")
        patience_counter = 0

        self.__logger.debug(f"Batch number per epoch: {batch_num}")

        progress_bar = trange(num_epochs * batch_num)
        for epoch in range(num_epochs):
            epoch_training_losses = []
            for i in range(batch_num):
                batch = train_batches[i]
                # Swap the direction of translation for some batches
                batch = list(batch.items())
                random.shuffle(batch)

                (lang1, xx), (lang2, yy) = batch
                try:
                    tokenizer.src_lang = lang1
                    x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True,
                                  max_length=int(train_conf["max_length"])).to(model.device)
                    tokenizer.src_lang = lang2
                    y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True,
                                  max_length=int(train_conf["max_length"])).to(model.device)
                    y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

                    loss = model(**x, labels=y.input_ids).loss
                    loss.backward()
                    epoch_training_losses.append(loss.item())

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                except Exception as e:
                    optimizer.zero_grad(set_to_none=True)
                    x, y, loss = None, None, None
                    self.__cleanup()
                    self.__logger.error("Error: unexpected exception during training, exception: %s", str(e))
                    continue

                progress_bar.update()

            avg_training_loss = sum(epoch_training_losses) / len(epoch_training_losses)
            training_losses.append(avg_training_loss)

            epoch_validation_losses = []
            with torch.no_grad():
                for batch in val_batches:
                    batch = list(batch.items())
                    (lang1, xx), (lang2, yy) = batch
                    try:
                        tokenizer.src_lang = lang1
                        x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True,
                                      max_length=int(train_conf["max_length"])).to(model.device)
                        tokenizer.src_lang = lang2
                        y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True,
                                      max_length=int(train_conf["max_length"])).to(model.device)
                        y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

                        val_loss = model(**x, labels=y.input_ids).loss
                        epoch_validation_losses.append(val_loss.item())
                    except Exception as e:
                        self.__logger.error("Error: unexpected exception during validation, exception: %s", str(e))
                        continue

            avg_validation_loss = sum(epoch_validation_losses) / len(epoch_validation_losses)
            validation_losses.append(avg_validation_loss)

            if avg_validation_loss < best_val_loss:
                best_val_loss = avg_validation_loss
                patience_counter = 0
                output_model_name = config["MODEL"]["output_model_name"]
                model.save_pretrained(output_model_name)
                tokenizer.save_pretrained(output_model_name)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.__logger.info("Early stopping triggered")
                    break

            self.__logger.info(f"Epoch {epoch + 1}/{num_epochs}: Training Loss: {avg_training_loss:.4f}, Validation "
                               f"Loss: {avg_validation_loss:.4f}")

        self.__plot_losses(training_losses, validation_losses)

    def finetune(self, model: AutoModelForSeq2SeqLM, dataset: ds.Dataset, tokenizer: NllbTokenizer,
                 config: ConfigParser) -> None:
        if torch.cuda.is_available():
            self.__logger.info("CUDA is available. Using GPU for training")
            model.cuda()
        else:
            self.__logger.info("CUDA is not available. Using CPU for training")

        try:
            optimizer = Adafactor(
                [p for p in model.parameters() if p.requires_grad],
                scale_parameter=False,
                relative_step=False,
                lr=1e-4,
                clip_threshold=1.0,
                weight_decay=1e-3,
            )
        except Exception as e:
            self.__logger.error(f"Error occurred while initializing Adafactor: {e}")
            raise

        self.__train(model, dataset, tokenizer, optimizer, config)
