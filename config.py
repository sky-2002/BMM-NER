from transformers import BertTokenizer

class Config:
    CLS = [101]
    SEP = [102]
    VALUE_TOKEN = [0]
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    EPOCHS = 3
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)