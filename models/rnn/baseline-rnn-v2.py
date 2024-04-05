# from __future__ import unicode_literals, print_function, division
# from torch.utils.data import DataLoader, TensorDataset, RandomSampler
# from torch.nn.utils.rnn import pad_sequence
# import datasets

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets
from datasets import Dataset, DatasetDict, load_dataset
import torchtext
import tqdm
import evaluate
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sentencepiece as spm
from typing import Union, List
import os
import time
import pandas as pd

torch.cuda.empty_cache()

# TODO : I WILL CHANGE THIS PART SOON (adapted for python 3.9 from prev spm tokenizer) dont @ me
en_model = spm.SentencePieceProcessor(model_file=r"C:\Users\user\OneDrive\Documents\GitHub\cs4248-neural-translation\tokenisation\sentencepiece_custom\en.model")
zh_model = spm.SentencePieceProcessor(model_file=r"C:\Users\user\OneDrive\Documents\GitHub\cs4248-neural-translation\tokenisation\sentencepiece_custom\zh.model")

# For consistency 
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# get dataset
dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
train_data, valid_data, test_data = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)

# reduced dataset (10k)
train_data = train_data.select(range(10000))

################################ 1. SPM TOKENIZER FOR PYTHON VER < 3.10 

max_length_g = 285 # From EDA 

class Tokeniser(object):
    def __init__(self, lang: str):
        self.model = spm.SentencePieceProcessor(model_file=f'./{lang}.model')
        self.special_ids = [
            self.model.unk_id(),
            self.model.pad_id(),
            self.model.bos_id(),
            self.model.eos_id()
        ]

    def __len__(self):
        return len(self.model)
    
    def encode_batch(self, sents: List[str], pad_len=None, truncate_len=None):
        return [self.encode(sent, pad_len, truncate_len) for sent in sents]

    def encode(self, sent: Union[str, List[str]], pad_len=None, truncate_len=None):
        if isinstance(sent, list):
            return self.encode_batch(sent, pad_len, truncate_len)
        ids = self.model.encode(sent)
        if pad_len is not None and len(ids) < int(pad_len):
            ids.extend([self.model.pad_id()] * (int(pad_len) - len(ids)))
        elif truncate_len is not None and len(ids) > int(truncate_len):
            ids = ids[:int(truncate_len)]
        return ids

    def decode(self, ids: List[int]):
        return self.model.decode(list(filter(lambda id: id >= 0 and id < len(self), ids)))
    
# TODO : DON'T @ ME i will update this to work with the latest tokenizer
en_tokeniser = Tokeniser(lang='en')
zh_tokeniser = Tokeniser(lang='zh')

def tokenize_example(example, bos_token_id=zh_model.bos_id(), eos_token_id=zh_model.eos_id()):
    en_tokens = en_tokeniser.encode(example['translation']["en"], truncate_len=max_length_g)
    zh_tokens = zh_tokeniser.encode(example['translation']["zh"], truncate_len=max_length_g)
    en_tokens = [bos_token_id] + en_tokens + [eos_token_id]
    zh_tokens = [bos_token_id] + zh_tokens + [eos_token_id]
    return {"en_ids": en_tokens, "zh_ids": zh_tokens}

train_data = train_data.map(tokenize_example)
valid_data = valid_data.map(tokenize_example)
test_data = test_data.map(tokenize_example)

# change to tensors 
data_type = "torch"
format_columns = ["en_ids", "zh_ids"]

train_data = train_data.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)

valid_data = valid_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

test_data = test_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

################################ 2. DATALOADERS

batch_size = 16
pad_index = 3

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_zh_ids = [example["zh_ids"] for example in batch]
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_zh_ids = nn.utils.rnn.pad_sequence(batch_zh_ids, padding_value=pad_index)
        batch = {
            "en_ids": batch_en_ids,
            "zh_ids": batch_zh_ids,
        }
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

################################ 3. MODEL ARCHITECTURE 

class Encoder(nn.Module):
    def __init__(
        self, input_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, bidirectional=False)
        self.fc = nn.Linear(encoder_hidden_dim, decoder_hidden_dim) # encoder_hidden  * 2 if bidirectional!
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(hidden[-1, :, :]))
        return outputs, hidden
    
class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear(
            (encoder_hidden_dim) + decoder_hidden_dim, decoder_hidden_dim # IF BIDIRECTIONAL MUST encoder_hidden_dim * 2 
        )
        self.v_fc = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_length = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_length, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn_fc(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v_fc(energy).squeeze(2)
        return torch.softmax(attention, dim=1)
    
class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        dropout,
        attention,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU((encoder_hidden_dim) + embedding_dim, decoder_hidden_dim) # IF BIDIRECTIONAL MUST encoder_hidden_dim * 2 
        self.fc_out = nn.Linear(
            (encoder_hidden_dim) + decoder_hidden_dim + embedding_dim, output_dim # IF BIDIRECTIONAL MUST encoder_hidden_dim * 2 
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        a = self.attention(hidden, encoder_outputs)
        # a = [batch size, src length]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src length]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src length, encoder hidden dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch size, 1, encoder hidden dim * 2]
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, encoder hidden dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, (encoder hidden dim * 2) + embedding dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [seq length, batch size, decoder hid dim * n directions]
        # hidden = [n layers * n directions, batch size, decoder hid dim]
        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, decoder hidden dim]
        # hidden = [1, batch size, decoder hidden dim]
        # this also means that output == hidden
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [batch size, output dim]
        return prediction, hidden.squeeze(0), a.squeeze(1)
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        batch_size = src.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
        # outputs = [src length, batch size, encoder hidden dim * 2]
        # hidden = [batch size, decoder hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, decoder hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs
    
input_dim = 16384   #len(en_vocab)
output_dim = 16384  #len(zh_vocab)
encoder_embedding_dim = 256
decoder_embedding_dim = 256
encoder_hidden_dim = 512
decoder_hidden_dim = 512
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attention = Attention(encoder_hidden_dim, decoder_hidden_dim)

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    encoder_hidden_dim,
    decoder_hidden_dim,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    encoder_hidden_dim,
    decoder_hidden_dim,
    decoder_dropout,
    attention,
)

model = Seq2Seq(encoder, decoder, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

# For info 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_parameters(model):,} trainable parameters")
    
################################ 4. TRAINING 

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

def train_fn(
    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = batch["en_ids"].to(device)
        trg = batch["zh_ids"].to(device)
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].view(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

################################ 5. HELPERS 

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time % 60)
    return elapsed_mins, elapsed_secs

################################ 6. EVAL

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src = batch["en_ids"].to(device)
            trg = batch["zh_ids"].to(device)
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].view(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

################################ 7. EXECUTE TRAINING

n_epochs = 10
clip = 1.0
teacher_forcing_ratio = 0.5
best_valid_loss = float("inf")
CHECKPOINT_DIR = "./spm_model_checkpoints"
patience = 3  
no_improvement_epochs = 0

# Check if a checkpoint exists
checkpoint_path = f"{CHECKPOINT_DIR}/model_checkpoint.pt"
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_valid_loss = checkpoint['loss']
else:
    print("Training started, no checkpoint found.")

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

for epoch in tqdm.tqdm(range(n_epochs)):
    start_time = time.time()
    train_loss = train_fn(model, train_data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device)
    valid_loss = evaluate_fn(model, valid_data_loader, criterion, device)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    # Save model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        checkpoint_filename = f"{CHECKPOINT_DIR}/model_checkpoint.pt"
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_valid_loss}, checkpoint_filename)
        print(f"Checkpoint saved to {checkpoint_filename}")
        no_improvement_epochs = 0 
    else:
        no_improvement_epochs += 1

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):7.3f}')
    print(f'\t Valid Loss: {valid_loss:.3f} | Valid PPL: {np.exp(valid_loss):7.3f}')

    # Auto stop
    if no_improvement_epochs >= patience:
        print(f"No improvement in val loss for {patience} consecutive epochs. Stopping training.")
        break

print("Training completed.")

################################ 8. EXECUTE EVAL 

checkpoint = torch.load(r"C:\Users\user\OneDrive\Documents\GitHub\cs4248-neural-translation\models\rnn\model_checkpoints\model_checkpoint.pt")
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict']) 
else:
    model.load_state_dict(checkpoint)  

test_loss = evaluate_fn(model, test_data_loader, criterion, device)
print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")

################################ 9. TEST TRANSLATION 

model.load_state_dict(torch.load("spm-model.pt"))

def translate_sentence(
    sentence,
    model,
    device,
    max_output_length=max_length_g,
):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            en_tokens = en_tokeniser.encode(sentence, truncate_len=max_length_g) # TODO UPDATE TOKENISER
        else:
            print('sentence is not an instance!')
        ids = [1] + en_tokens + [2] # BOS = 1 , EOS = 2 
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        encoder_outputs, hidden = model.encoder(tensor)
        inputs = [1]
        attentions = torch.zeros(max_output_length, 1, len(ids))
        for i in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, attention = model.decoder(
                inputs_tensor, hidden, encoder_outputs
            )
            attentions[i] = attention
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == 2:
                break
        zh_tokens = zh_model.Decode(inputs)
        translated = ' '.join(zh_tokens)
    return translated, en_tokens, attentions[: len(inputs) - 1]

sentence = test_data[2]['translation']["en"]
expected_translation = test_data[2]['translation']["zh"]
translation, sentence_tokens, attention = translate_sentence(sentence,model,device)

print(f"Expected {expected_translation}. Predicted : {translation}")

############################ 10. PLOT ATTENTION 

def plot_attention(sentence, translation, attention):
    fig, ax = plt.subplots(figsize=(10, 10))
    attention = attention.squeeze(1).numpy()
    cax = ax.matshow(attention, cmap="bone")
    ax.set_xticks(ticks=np.arange(len(sentence)), labels=sentence, rotation=90, size=15)
    translation = translation[1:]
    ax.set_yticks(ticks=np.arange(len(translation)), labels=translation, size=15)
    plt.show()
    plt.close()

plot_attention(sentence_tokens, translation, attention)

########################## 11. SAVE TO CSV 

translation, sentence_tokens, attention = translate_sentence(sentence,model,device)
en = [] 
zh = [] 
pred = []

for i in range(0,8549):
    en.append(test_data[i]['translation']['en'])
    zh.append(test_data[i]['translation']['zh'])
    t,s,a = translate_sentence(test_data[i]['translation']['en'],model,device)
    pred.append(t)

df = pd.DataFrame()
df['en'] = en
df['zh'] = zh 
df['pred'] = pred

df.to_csv("spm_10000_rows_285_max_g.csv")
