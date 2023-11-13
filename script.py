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

args = parser.parse_args()

epochs = int(args.epochs)
saved_model_name = args.saved_model_name

print(args)

with open("./le.pkl", "rb") as f:
    label_encoder = pickle.load(f)

train_dataloader, val_dataloader = get_dataloaders(["./bmmnerdataset/Bhojpuri_ner.remove[1].train",
                 "./bmmnerdataset/Magahi_ner[1].train",
                 "./bmmnerdataset/Maithili_ner[1].train"], sample_size=50)
print("Data loaded")

# model = NERBertModel(num_tag=45)
model = BiLSTMBert(num_tag=45, hidden_dim=768, lstm_layers=2)
print("Model initialized")
optimizer, scheduler = get_optimizer_scheduler(model, len(train_dataloader))
print("Optimizer and scheduler initialized")


device = "cpu"
for epoch in range(epochs if epochs else Config.EPOCHS):
    # try:
    model, train_loss, train_acc = train_fn(train_dataloader, model, optimizer, device, scheduler)
    val_loss, val_acc = val_fn(val_dataloader, model, optimizer, device, scheduler)
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