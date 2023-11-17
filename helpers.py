import csv
import torch
import pandas as pd
from tqdm import tqdm
from config import Config
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup

FULL_FINETUNING = False

def df_for_ner(file_paths, le):
    sentences = []
    words = []
    tags = []
    numeric_tags = []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                line = line.strip()
                if line:
                    word, tag = line.split('\t')
                    words.append(word)
                    tags.append(tag)
                    numeric_tags.append(le.transform([tag])[0])
                else:
                    sentences.append((words, tags, numeric_tags))
                    words = []
                    tags = []
                    numeric_tags = []
    if words:
        sentences.append((words, tags, numeric_tags))
    return pd.DataFrame(data=sentences, columns=['words','tags', 'numeric_tags'])


def train_fn(train_data_loader, model, optimizer, device, scheduler):
    #Train the Model
    model.train()
    loss_ = 0
    acc_ = 0
    for data in tqdm(train_data_loader, total = len(train_data_loader)):
        for i, j in data.items():
            data[i] = j.to(device)

        #Backward Propagation
        optimizer.zero_grad()
        tag, loss = model(**data)

        preds = torch.argmax(torch.softmax(tag, dim=0), dim=-1).flatten()
        targets = data['target_tags'].flatten()
        acc_ += accuracy_score(targets.cpu().data.numpy(), preds.cpu().data.numpy())

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_ += loss.item()
    return model, loss_ / len(train_data_loader), acc_ / len(train_data_loader)

def val_fn(val_data_loader, model, device, test=False):
    model.eval()
    loss_ = 0
    acc_ = 0
    if test:
        f1_ = 0
    for data in tqdm(val_data_loader, total = len(val_data_loader)):
        for i, j in data.items():
            data[i] = j.to(device)
        tag, loss = model(**data)
        preds = torch.argmax(torch.softmax(tag, dim=0), dim=-1).flatten()
        targets = data['target_tags'].flatten()
        acc_ += accuracy_score(targets.cpu().data.numpy(), preds.cpu().data.numpy())

        if test:
            f1_ += f1_score(targets, preds, average="weighted")

        loss_ += loss.item()
    if test:
        return loss_ / len(val_data_loader), acc_ / len(val_data_loader), f1_ / len(val_data_loader)
    return loss_ / len(val_data_loader), acc_ / len(val_data_loader)


#Function for getparameters
def _get_hyperparameters(model, ff):

    # ff: full_finetuning
    if ff:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    return optimizer_grouped_parameters

def get_optimizer_scheduler(model, num_train_samples, ff):
    optimizer_grouped_parameters = model._get_hyperparameters(ff if ff else FULL_FINETUNING)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-5)

    num_train_steps = int(num_train_samples / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )
    return optimizer, scheduler