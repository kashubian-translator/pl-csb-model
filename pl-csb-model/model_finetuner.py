import gc
import random
from configparser import ConfigParser

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup

def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def get_random_language_pairs(batch_size, langs, data):
    (l1, long1), (l2, long2) = random.sample(langs, 2)
    xx, yy = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        xx.append(item[l1])
        yy.append(item[l2])
    return xx, yy, long1, long2

def train(model: AutoModelForSeq2SeqLM, data: pd.DataFrame, tokenizer: NllbTokenizer, optimizer: Adafactor, config: ConfigParser) -> None:
    losses = []
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(config["TRAINING"]["WarmupSteps"]))

    LANGS = [("pl", "pol_Latn"), ("csb", "csb_Latn")]

    model.train()
    x, y, loss = None, None, None
    cleanup()

    tq = trange(len(losses), int(config["TRAINING"]["TrainingSteps"]))
    for _ in tq:
        xx, yy, lang1, lang2 = get_random_language_pairs(int(config["TRAINING"]["BatchSize"]), LANGS, data)
        try:
            tokenizer.src_lang = lang1
            x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=int(config["TRAINING"]["MaxLength"])).to(model.device)
            tokenizer.src_lang = lang2
            y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=int(config["TRAINING"]["MaxLength"])).to(model.device)
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

            loss = model(**x, labels=y.input_ids).loss
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        except RuntimeError as e:
            optimizer.zero_grad(set_to_none=True)
            x, y, loss = None, None, None
            cleanup()
            print('error', max(len(s) for s in xx + yy), e)
            continue

    model.save_pretrained(config["MODEL"]["OutputModelName"])
    tokenizer.save_pretrained(config["MODEL"]["OutputModelName"])

def finetune(model: AutoModelForSeq2SeqLM, data: pd.DataFrame, tokenizer: NllbTokenizer, config: ConfigParser) -> None:
    if torch.cuda.is_available():
        model.cuda()

    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )

    train(model, data, tokenizer, optimizer, config)
    