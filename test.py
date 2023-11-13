import sys
import torch
# import pickle
import argparse
from config import Config
from helpers import val_fn
from model import NERBertModel
from prepare_data import get_test_dataloader

parser = argparse.ArgumentParser(
    description="Argument parser for BMM-NER project"
)
parser.add_argument("--model", default=None)
args = parser.parse_args()
print(args)

test_dataloader = get_test_dataloader(file_paths=[
    "./bmmnerdataset/Bhojpuri_ner.remove[1].test"
])
print("Dataloader created")

model = NERBertModel(num_tag=45)
model.load_state_dict(torch.load(f"./trained_models/{args.model}"))
print("Model loaded")

loss, acc, f1 = val_fn(val_data_loader=test_dataloader, 
                       model=model, 
                       test=True,
                       device="cpu")
print("Test metrics")
print(f"Loss: {loss}")
print(f"Accuracy: {acc}")
print(f"F1-Score: {f1}")