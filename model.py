import torch
from torch import nn
from transformers import BertModel

class NERBertModel(nn.Module):
    
    def __init__(self, num_tag, model_name=None, class_weights=None):
        super(NERBertModel, self).__init__()
        self.num_tag = num_tag
        self.model_name = model_name if model_name else 'bert-base-uncased'
        self.bert = BertModel.from_pretrained(self.model_name)
        self.bert_drop = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
        self.class_weights = class_weights if class_weights else None

    def forward(self, ids, mask, token_type_ids, target_tags):
        output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        bert_out = self.bert_drop(output) 
        tag = self.out_tag(bert_out)
    
        #Calculate the loss
        Critirion_Loss = nn.CrossEntropyLoss(weight=self.class_weights)
        active_loss = mask.view(-1) == 1
        active_logits = tag.view(-1, self.num_tag)
        active_labels = torch.where(active_loss, target_tags.view(-1), torch.tensor(Critirion_Loss.ignore_index).type_as(target_tags))
        # print(active_logits.shape, active_labels.shape)
        loss = Critirion_Loss(active_logits, active_labels)
        return tag, loss
    
    #Function for getparameters
    def _get_hyperparameters(self, ff):
        # ff: full_finetuning
        if ff:
            param_optimizer = list(self.named_parameters())
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
            param_optimizer = list(self.bert_drop.named_parameters()) + list(self.out_tag.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        return optimizer_grouped_parameters
    
class BiLSTMBert(nn.Module):
    def __init__(self, num_tag, model_name=None, hidden_dim=768, lstm_layers=1, class_weights=None) -> None:
        super(BiLSTMBert, self).__init__()
        self.num_tag = num_tag
        self.model_name = model_name if model_name else 'bert-base-uncased'
        self.bert = BertModel.from_pretrained(self.model_name)
        self.bert_drop = nn.Dropout(0.3)

        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0
        )
        self.out_tag = nn.Linear(in_features=2*hidden_dim, out_features=self.num_tag)
        self.class_weights = class_weights if class_weights else None

    def forward(self, ids, mask, token_type_ids, target_tags):
        output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        bert_out = self.bert_drop(output) 

        lstm_out, _ = self.bilstm(bert_out)
        tag = self.out_tag(lstm_out)

        #Calculate the loss
        Critirion_Loss = nn.CrossEntropyLoss(weight=self.class_weights)
        active_loss = mask.view(-1) == 1
        active_logits = tag.view(-1, self.num_tag)
        active_labels = torch.where(active_loss, target_tags.view(-1), torch.tensor(Critirion_Loss.ignore_index).type_as(target_tags))
        # print(active_logits.shape, active_labels.shape)
        loss = Critirion_Loss(active_logits, active_labels)
        return tag, loss
    
    #Function for getparameters
    def _get_hyperparameters(self, ff):
        # ff: full_finetuning
        if ff:
            param_optimizer = list(self.named_parameters())
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
            param_optimizer = list(self.bert_drop.named_parameters())
            + list(self.bilstm.named_parameters()) + list(self.out_tag.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        return optimizer_grouped_parameters