# from __future__ import unicode_literals, print_function, division
# from torch.utils.data import DataLoader, TensorDataset, RandomSampler
# from torch.nn.utils.rnn import pad_sequence
# import datasets

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random
import matplotlib.ticker as ticker
from transformers import MarianTokenizer
import pandas as pd
from datasets import load_dataset
import time
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get dataset
dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
train_data, valid_data, test_data = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)

################################ 1. MARIANTOKENIZER 
print('tokenization step')
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Max length 
max_length_g = 50

# Encode en and zh
def encode_pair(pair, tokenizer, max_length):
    # Encoding en
    en_encoded = tokenizer(pair['translation']['en'],
                           return_tensors="pt",
                           padding="max_length",
                           truncation=True,
                           max_length=max_length)

    # Encoding zh
    zh_encoded = tokenizer(pair['translation']['zh'],
                           return_tensors="pt",
                           padding="max_length",
                           truncation=True,
                           max_length=max_length)

    return {
        "input_ids": en_encoded['input_ids'],
        "attention_mask": en_encoded['attention_mask'],
        "labels": zh_encoded['input_ids']  # Labels for training
    }

fn_kwargs = {
    "tokenizer": tokenizer,
    "max_length": max_length_g,
}

# Encode our dataset en , zh 
train_tokens = train_data.map(encode_pair, fn_kwargs=fn_kwargs)
valid_tokens = valid_data.map(encode_pair, fn_kwargs=fn_kwargs)
test_tokens = test_data.map(encode_pair, fn_kwargs=fn_kwargs)

################################ 2. DATALOADERS

def get_collate_fn():
    def collate_fn(batch):
        # Extract from batch + stack them
        input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels
        }
    return collate_fn

def get_data_loader(dataset, batch_size, shuffle=False):
    collate_fn = get_collate_fn()
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

################################ 3. MODEL ARCHITECTURE 

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # input_size is set to the tokenizer's vocabulary size.
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Using GRU as the RNN unit.
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # Dropout for regularization.
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        # Input is the tokenized and numerical representation of the sentences.
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query: [batch size, hidden size], keys: [batch size, seq len, hidden size]
        query = query.unsqueeze(1)  # [batch size, 1, hidden size]
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # [batch size, seq len, 1]
        scores = scores.squeeze(2)  # [batch size, seq len]
        weights = F.softmax(scores, dim=1).unsqueeze(1)  # [batch size, 1, seq len]
        context = torch.bmm(weights, keys)  # [batch size, 1, hidden size]

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(hidden_size + hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        if target_tensor is None:
            raise ValueError("Target tensor is required for training.")

        batch_size = encoder_outputs.size(0)
        target_length = target_tensor.size(1)

        decoder_outputs = torch.zeros(target_length, batch_size, self.output_size).to(device)
        decoder_input = target_tensor[:, 0].unsqueeze(1)  # Use the first token of the target sequence
        decoder_hidden = encoder_hidden

        for t in range(1, target_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs[t] = decoder_output
            decoder_input = target_tensor[:, t].unsqueeze(1)  # Next input is the next token in the target sequence

        return decoder_outputs, decoder_hidden, attn_weights

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))  # [batch size, 1, hidden size]
        context, attn_weights = self.attention(hidden[:, -1, :], encoder_outputs)  # Context: [batch size, 1, hidden size]
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch size, 1, 2 * hidden size]
        output, hidden = self.gru(rnn_input, hidden)  # Output: [batch size, 1, hidden size]
        output = self.out(output.squeeze(1))  # [batch size, output size]

        return output, hidden, attn_weights
    
################################ 4. TRAINING 

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device):
    total_loss = 0
    for data in dataloader:
        input_tensor = data['input_ids'].to(device)  # [batch size, seq length]
        target_tensor = data['labels'].to(device)  # [batch size, seq length]

        # Assuming input_tensor is [batch size, seq length], add .squeeze() if it's [batch size, seq length, 1]
        input_tensor = input_tensor.squeeze()  # Remove if not necessary
        target_tensor = target_tensor.squeeze()  # Remove if not necessary

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Encoder expects [seq len, batch size, features] if not batch_first
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        # Adjust decoder inputs/outputs based on your decoder's design
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        # Calculate loss; you might need to adjust dimensions based on your specific case
        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

################################ 5. HELPERS 

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, print_every=100, plot_every=100, device='cuda'):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # Using CrossEntropyLoss for multi-class classification

    for epoch in range(1, n_epochs + 1):
        encoder.train()
        decoder.train()
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

################################ 6. EVAL

def evaluate(encoder, decoder, sentence, tokenizer, device='cuda'):
    with torch.no_grad():
        # Tokenize the sentence using the MarianTokenizer
        tokenized_input = tokenizer.encode(sentence, return_tensors="pt").to(device)

        # Pass the tokenized input to the encoder
        encoder_outputs, encoder_hidden = encoder(tokenized_input)

        # Initialize the sequence with the EOS token
        decoder_input = torch.tensor([tokenizer.eos_token_id], device=device).unsqueeze(0)

        decoded_tokens = []
        for _ in range(max_length_g): 
            decoder_output, decoder_hidden = decoder(decoder_input, encoder_hidden)

            # Get the most likely next token ID
            topi = decoder_output.argmax(1)

            # If the EOS token is produced, stop decoding
            if topi.item() == tokenizer.eos_token_id:
                break

            # Add the predicted token to the sequence
            decoded_tokens.append(topi.item())

            # Use the predicted token as the next input to the decoder
            decoder_input = topi.unsqueeze(0)

        # Decode the token IDs to a string
        decoded_sentence = tokenizer.decode(decoded_tokens, skip_special_tokens=True)
        return decoded_sentence

def evaluateRandomly(encoder, decoder, dataset, tokenizer, n=5, device='cuda'):
    rnn_output_df = pd.DataFrame()
    en, zh, rnn = [], [], [] 
    for i in range(n):
        # Randomly select a sample from the test dataset
        sample = random.choice(dataset['test'])
        original_sentence = sample['en']
        target_translation = sample['zh']

        print('Original :', original_sentence)
        en.append(original_sentence)
        print('Translation :', target_translation)
        zh.append(target_translation)

        # Translate the sentence using the evaluate function
        output_sentence = evaluate(encoder, decoder, original_sentence, tokenizer, device=device)

        print('Predicted :', output_sentence)
        rnn.append(output_sentence)
        print('')

    rnn_output_df['en'] = en 
    rnn_output_df['zh'] = zh 
    rnn_output_df['rnn'] = rnn 
    rnn_output_df.to_csv("rnn_output_df.csv",index=False)

################################ 7. RUN 
# Define the hidden size for the RNN
hidden_size = 128
batch_size = 32
# tokenizer's vocabulary size
vocab_size = tokenizer.vocab_size
print('dataloaders step')
# data loaders
train_data_loader = get_data_loader(train_tokens, batch_size, shuffle=True)
valid_data_loader = get_data_loader(valid_tokens, batch_size)
test_data_loader = get_data_loader(test_tokens, batch_size)

# encoder and decoder
encoder = EncoderRNN(vocab_size, hidden_size, dropout_p=0.1).to(device)
decoder = AttnDecoderRNN(hidden_size, vocab_size, dropout_p=0.1).to(device)
print('training step')
# Train the model dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion
train(train_data_loader, encoder, decoder, 1, print_every=1, plot_every=1)
