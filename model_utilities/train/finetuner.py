import gc
import random
from configparser import ConfigParser
from logging import Logger

import datasets as ds
import matplotlib.pyplot as plt
import torch
from distutils.util import strtobool
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import trange
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, get_constant_schedule_with_warmup

from shared.optimizer_utils import get_optimizer_class


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
        plt.title("Training and validation losses")
        plt.legend()
        plt.savefig("./debug/graphs/losses.png")

    def __train_and_return_losses(self, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer, dataset: ds.Dataset,
                                  optimizer: Optimizer, scheduler: LambdaLR, config: ConfigParser) -> tuple[list[float], list[float]]:
        self.__log_train_config(config)

        train_conf = config["TRAINING"]

        training_losses = []
        validation_losses = []

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
        return training_losses, validation_losses

    def finetune(self, model: AutoModelForSeq2SeqLM, tokenizer: NllbTokenizer, dataset: ds.Dataset,
                 config: ConfigParser, optimizer=None) -> None:
        if torch.cuda.is_available():
            self.__logger.info("CUDA is available. Using GPU for training")
            model.cuda()
        else:
            self.__logger.info("CUDA is not available. Using CPU for training")

        if not optimizer:
            optimizer = self.__initialize_optimizer(model, config)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(config["TRAINING"]["warmup_steps"]))

        self.__train_and_return_losses(model, tokenizer, dataset, optimizer, scheduler, config)

    def __initialize_optimizer(self, model: AutoModelForSeq2SeqLM, config: ConfigParser):
        try:
            optimizer_name = config["TRAINING"]["optimizer_class_name"]
            optimizer_class = get_optimizer_class(optimizer_name)
            learning_rate = float(config["TRAINING"]["optimizer_learning_rate"])
            relative_step = strtobool(config["TRAINING"]["optimizer_relative_step"])
            clip_threshold = float(config["TRAINING"]["optimizer_clip_threshold"])
            decay_rate = float(config["TRAINING"]["optimizer_decay_rate"])
            weight_decay = float(config["TRAINING"]["optimizer_weight_decay"])
            scale_parameter = strtobool(config["TRAINING"]["optimizer_scale_parameter"])

            self.__logger.debug("Initializing optimizer with the following parameters:")
            self.__logger.debug(f"Optimizer class: {optimizer_name}")
            self.__logger.debug(f"Learning rate: {learning_rate}")
            self.__logger.debug(f"Clip threshold: {clip_threshold}")
            self.__logger.debug(f"Decay rate: {decay_rate}")
            self.__logger.debug(f"Weight decay: {weight_decay}")
            self.__logger.debug(f"Scale parameter: {scale_parameter}")
            self.__logger.debug(f"Relative step: {relative_step}")

            optimizer_params = {
                "lr": learning_rate,
                "clip_threshold": clip_threshold,
                "weight_decay": weight_decay,
                "scale_parameter": scale_parameter,
                "relative_step": relative_step
            }

            optimizer = optimizer_class([p for p in model.parameters() if p.requires_grad], **optimizer_params)
        except Exception as e:
            self.__logger.error(f"Error occurred while initializing optimizer: {e}")
            raise
        return optimizer
