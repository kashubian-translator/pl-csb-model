from itertools import product
from logging import Logger

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSeq2SeqLM, get_constant_schedule_with_warmup, Adafactor, AdamW

from shared.optimizer_utils import get_optimizer_class
from train.finetuner import ModelFinetuner


class HyperparameterSearcher:
    __logger: Logger
    __model_finetuner = ModelFinetuner

    def __init__(self, logger, model_finetuner):
        self.__logger = logger
        self.__model_finetuner = model_finetuner

    def hyperparameter_search(self, tokenizer, dataset, config, hyperparameter_space):
        writer = SummaryWriter(log_dir=config["DIRECTORIES"]["hyperparameter_log_dir"])

        keys, values = zip(*hyperparameter_space.items())
        hyperparameter_combinations = [dict(zip(keys, v)) for v in product(*values)]

        self.__logger.info(f"Starting hyperparameter tuning with {len(hyperparameter_combinations)} combinations.")
        best_hyperparams = None
        best_val_loss = float("inf")

        for i, hyperparams in enumerate(hyperparameter_combinations):
            self.__logger.info(f"Testing combination {i + 1}/{len(hyperparameter_combinations)}: {hyperparams}")
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(config["MODEL"]["pretrained_model_name"])
                optimizer = self.__configure_optimizer(model, hyperparams)
                training_losses, validation_losses = self.__model_finetuner.finetune(
                    model, tokenizer, dataset, config, optimizer
                )
                writer.add_hparams(
                    hyperparams,
                    {"hparam/val_loss": validation_losses[-1], "hparam/train_loss": training_losses[-1]},
                    run_name=f"run_{i + 1}"
                )
                if validation_losses[-1] < best_val_loss:
                    best_val_loss = validation_losses[-1]
                    best_hyperparams = hyperparams
            except Exception as e:
                self.__logger.error(f"Error occurred with hyperparameters {hyperparams}: {e}")
                continue

        self.__logger.info(f"Hyperparameter tuning complete. Best hyperparameters: {best_hyperparams}")
        writer.close()

    def __configure_optimizer(self, model, hyperparams):
        optimizer_class = get_optimizer_class(hyperparams.get("optimizer"))
        optimizer_params = {
            "lr": hyperparams.get("lr"),
            "relative_step": hyperparams.get("relative_step"),
            "clip_threshold": hyperparams.get("clip_threshold"),
            "decay_rate": hyperparams.get("decay_rate"),
            "weight_decay": hyperparams.get("weight_decay")
        }
        return optimizer_class([p for p in model.parameters() if p.requires_grad], **optimizer_params)
