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
from torch.utils.data import DataLoader, Dataset, RandomSampler

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
model_name = "Helsinki-NLP/opus-mt-en-zh" # en-zh model doesn't work for decoding chi 
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Max length 
max_length_g = 10

# Encode en and zh
def encode_pair(pair, tokenizer, max_length):    
    tr = tokenizer(preprocess(pair['translation']['en']), text_target=preprocess(pair['translation']['zh']),truncation=True, return_tensors="pt",max_length=max_length_g, padding="max_length")
    if (len(tr['input_ids']) != len(tr['labels'])):
        print('mismatched')

    return {
        "input_ids": tr['input_ids'],
        "labels": tr['labels']  
    }

fn_kwargs = {
    "tokenizer": tokenizer,
    "max_length": max_length_g,
}

# Encode our dataset en , zh 
train_tokens = train_data.map(encode_pair, fn_kwargs=fn_kwargs)
valid_tokens = valid_data.map(encode_pair, fn_kwargs=fn_kwargs)
test_tokens = test_data.map(encode_pair, fn_kwargs=fn_kwargs)


################################ 2. MODEL ARCHITECTURE 

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
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        #self.gru = nn.GRU(hidden_size + hidden_size, hidden_size, batch_first=True)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []
        #decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=device) 

        for i in range(max_length_g):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

################################ 3. DATALOADERS
    
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        input_ids = torch.tensor(self.data['input_ids'][idx], dtype=torch.long).squeeze(0)
        #print(input_ids.shape) # torch.Size([15]) instead of torch.Size([1,15])
        labels = torch.tensor(self.data['labels'][idx], dtype=torch.long).squeeze(0)
        return {'input_ids': input_ids, 'labels': labels}
    
def get_dataloader(data, batch_size, shuffle=False):
    dataset = TranslationDataset(data)

    # Use a DataLoader to handle batching
    res_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return res_dataloader


################################ 4. TRAINING 

def create_mask(input_tensor, pad_token_id):
    # input_tensor shape [batch_size, sequence_length]
    mask = (input_tensor != pad_token_id).float()
    return mask

def masked_loss(output, target, mask):
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
    loss = loss.view(mask.shape)  # Reshape loss to [batch_size, sequence_length]
    masked_loss = loss * mask  # Apply mask
    return masked_loss.sum() / mask.sum()  # Avg loss

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device):
    total_loss = 0
    count = 0 
    total = len(dataloader)
    for data in dataloader:
        #print(data['input_ids'].shape)
        count += 1
        input_tensor = data['input_ids'].to(device)  # [batch size, seq length]
        target_tensor = data['labels'].to(device)  # [batch size, seq length]
        #print(input_tensor.shape) #torch.Size([32, 15])

        mask = create_mask(target_tensor, tokenizer.pad_token_id).to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        #print(encoder_outputs.shape) #torch.Size([32, 15, 128])
        #print(encoder_hidden.shape) #torch.Size([1, 32, 128])
        #print(target_tensor.shape) # torch.Size([32, 15])
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        
        loss = masked_loss(decoder_outputs, target_tensor, mask)
        
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss += loss.item()


    return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, print_every=2, plot_every=2, device='cuda'):
    start = time.time()
    #plot_losses = []
    print_loss_total = 0  # Reset every print_every
    #plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss() 

    for epoch in range(1, n_epochs + 1):
        #encoder.train()
        #decoder.train()
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device)
        print_loss_total += loss
        #plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))


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

################################ 6. RUN 

# Define the hidden size for the RNN
hidden_size = 128
batch_size = 32
epochs = 2
vocab_size = tokenizer.vocab_size

# Initialize the data loaders
train_data_loader = get_dataloader(train_tokens, batch_size, shuffle=True)
valid_data_loader = get_dataloader(valid_tokens, batch_size)
test_data_loader = get_dataloader(test_tokens, batch_size)

# Initialize the encoder and decoder
encoder = EncoderRNN(vocab_size, hidden_size, dropout_p=0.1).to(device)
decoder = AttnDecoderRNN(hidden_size, vocab_size,dropout_p=0.1).to(device)

# Train the model dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion
train(train_data_loader, encoder, decoder, 1, print_every=1, plot_every=1)

############ 7. TRANSLATE 1 SENTENCE  

def translate_eng_sentence(encoder, decoder, sentence):
    with torch.no_grad():
        #input_tensor = tensorFromSentence(input_lang, sentence)
        tokenized = tokenizer(sentence,
                           return_tensors="pt",
                           padding="max_length",
                           truncation=True,
                           max_length=max_length_g)
        
        input_tensor = tokenized['input_ids'].to(device) # this doesn't have sos but have eos id 0 at the end

        encoder_outputs, encoder_hidden = encoder(input_tensor.to(device))
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()
        token_ids_list = decoded_ids.tolist()
        decoded_text = tokenizer.decode(token_ids_list, skip_special_tokens=True)
        return(decoded_text)

############ 8. TRANSLATE 1 SENTENCE  

english_sentence = train_tokens['en'][0]
target_sentence = train_tokens['zh'][0]
translated_sentence = translate_eng_sentence(encoder, decoder, english_sentence)
print(f"Translated: {translated_sentence}")
print(f"target_sentence: {target_sentence}")
print(f"english_sentence: {english_sentence}")
