import os
import torch
import pickle
import pandas as pd
from config import Config
from helpers import df_for_ner
from pympler.asizeof import asizeof
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Dataset:
  
  def __init__(self, texts, tags, tokenizer=None):
    
    #Texts: [['Diana', 'is', 'a', 'girl], ['she', 'plays'. 'football']]
    #tags: [[0, 1, 2, 5], [1, 3. 5]]
    
    self.texts = texts
    self.tags = tags
    self.tokenizer_name = tokenizer if tokenizer else 'bert-base-uncased'
    self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name) if self.tokenizer_name else Config.TOKENIZER
  
  def __len__(self):
    return len(self.texts)
  
  
  def __getitem__(self, index):
    texts = self.texts[index]
    tags = self.tags[index]
  
    #Tokenise
    ids = []
    target_tag = []

    for i, s in enumerate(texts):
        inputs = self.tokenizer.encode(s, add_special_tokens=False)
     
        input_len = len(inputs)
        ids.extend(inputs)
        target_tag.extend(input_len * [tags[i]])
    
    #To Add Special Tokens, subtract 2 from MAX_LEN
    ids = ids[:Config.MAX_LEN - 2]
    target_tag = target_tag[:Config.MAX_LEN - 2]

    mask = [1] * len(ids)
    token_type_ids = [0] * len(ids)

    #Add Padding if the input_len is small

    padding_len = Config.MAX_LEN - len(ids)
    ids = ids + ([0] * padding_len)
    target_tags = target_tag + ([0] * padding_len)
    mask = mask + ([0] * padding_len)
    token_type_ids = token_type_ids + ([0] * padding_len)

    return {
        "ids" : torch.tensor(ids, dtype=torch.long),
        "mask" : torch.tensor(mask, dtype=torch.long),
        "token_type_ids" : torch.tensor(token_type_ids, dtype=torch.long),
        "target_tags" : torch.tensor(target_tags, dtype=torch.long)
      }
  
def get_dataloaders(file_paths, sample_size=None, tokenizer=None, batch_size=None):

    with open("./le.pkl", "rb") as f:
       le = pickle.load(f)
    sdf = df_for_ner(file_paths=file_paths, le=le)
    if sample_size:
       sdf = sdf.sample(sample_size).reset_index()

    print(f"Memory used by sdf: {asizeof(sdf)}")

    train_sent, val_sent, train_tag, val_tag = train_test_split(sdf['words'], sdf['numeric_tags'], test_size=0.01, random_state=10)
    train_sent = train_sent.reset_index().drop("index", axis=1)['words']
    val_sent = val_sent.reset_index().drop("index", axis=1)['words']
    train_tag = train_tag.reset_index().drop("index", axis=1)['numeric_tags']
    val_tag = val_tag.reset_index().drop("index", axis=1)['numeric_tags']

    train_dataset = Dataset(texts = train_sent, tags = train_tag, tokenizer=tokenizer)
    val_dataset = Dataset(texts = val_sent, tags = val_tag, tokenizer=tokenizer)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size if batch_size else Config.TRAIN_BATCH_SIZE)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size if batch_size else Config.VAL_BATCH_SIZE)

    return train_data_loader, val_data_loader

def get_test_dataloader(file_paths, batch_size=8, sample_size=None, tokenizer=None):
    with open("./le.pkl", "rb") as f:
       le = pickle.load(f)
    sdf = df_for_ner(file_paths=file_paths, le=le)
    if sample_size:
       sdf = sdf.sample(sample_size).reset_index()

    test_sent = sdf['words'].reset_index().drop("index", axis=1)['words']
    test_tags = sdf['numeric_tags'].reset_index().drop("index", axis=1)['numeric_tags']

    test_dataset = Dataset(texts= test_sent, tags= test_tags, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return test_dataloader
    