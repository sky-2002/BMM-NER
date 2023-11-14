import sys
import torch
import pickle
import argparse
from config import Config
from model import NERBertModel, BiLSTMBert
from prepare_data import get_dataloaders
from helpers import get_optimizer_scheduler, train_fn, val_fn

parser = argparse.ArgumentParser(
    description="Argument parser for BMM-NER project"
)
parser.add_argument("--epochs", default=2)
parser.add_argument("--saved_model_name", default=None)
parser.add_argument("--ff", default=True, help="Full Finetuning or not")
parser.add_argument("--base_model", 
                    default="bert-base-uncased",
                    help="Model name from huggingface") # l3cube-pune/hindi-bert-scratch

args = parser.parse_args()

epochs = int(args.epochs)
saved_model_name = args.saved_model_name
ff = args.ff
base_model_name = args.base_model

print(args)

with open("./le.pkl", "rb") as f:
    label_encoder = pickle.load(f)

train_dataloader, val_dataloader = get_dataloaders(["./bmmnerdataset/Bhojpuri_ner.remove[1].train",
                 "./bmmnerdataset/Magahi_ner[1].train",
                 "./bmmnerdataset/Maithili_ner[1].train"], sample_size=50, tokenizer=base_model_name)
print("Data loaded")

# model = NERBertModel(num_tag=45)
model = BiLSTMBert(num_tag=45, hidden_dim=768, lstm_layers=2, model_name=base_model_name)
print("Model initialized")
optimizer, scheduler = get_optimizer_scheduler(model, len(train_dataloader), ff)
print("Optimizer and scheduler initialized")


device = "cpu"
for epoch in range(epochs if epochs else Config.EPOCHS):
    # try:
    model, train_loss, train_acc = train_fn(train_dataloader, model, optimizer, device, scheduler)
    val_loss, val_acc = val_fn(val_dataloader, model, device)
    print("------------------------------------------------------------")
    print(f"Epoch: {epoch+1}")
    print(f"Train_loss: {train_loss}, Val_loss: {val_loss}")
    print(f"Train acc: {train_acc}, Val_acc: {val_acc}")
    print("------------------------------------------------------------")
    # except Exception as e:
    #     print(f"Error: {e}")

print("Completed training")
if saved_model_name:
    torch.save(model.state_dict(), f"./trained_models/{saved_model_name}.bin")