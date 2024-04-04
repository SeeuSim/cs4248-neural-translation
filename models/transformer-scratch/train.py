import math
from timeit import default_timer as timer

from tqdm import tqdm
import torch
import torch.nn as nn
from datasets import load_dataset
from torch import Tensor
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sacrebleu.metrics import BLEU, CHRF, TER
from rouge_chinese import Rouge
import jieba  # you can use any other word cutting library
from bert_score import score

from transformers import MarianTokenizer
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.tensorboard import SummaryWriter

from ...tokenisation.sentencepiece_custom.tokeniser import BaseBPETokeniser

torch.manual_seed(0)


dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
# model_name = "Helsinki-NLP/opus-mt-en-zh"
# tokeniser = MarianTokenizer.from_pretrained(model_name)

tokeniser = BaseBPETokeniser(
    # en_model_file="../../tokenisation/sentencepiece_custom/en.model", 
    # zh_model_file="../../tokenisation/sentencepiece_custom/zh.model"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = (
#     tokenizer.unk_token_id,
#     tokenizer.pad_token_id,
#     tokenizer.pad_token_id,
#     # according to source code, marian does not use bos_token (retrieves from config.decoder_start_token_id)
#     # - https://github.com/huggingface/transformers/blob/main/src/transformers/models/marian/tokenization_marian.py
#     # from "config = AutoConfig.from_pretrained(model_name)", we see decoder_start_token_id = 65000 = pad_token_ide
#     tokenizer.eos_token_id,
# )
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = tokeniser.get_special_ids('en')
ZH_UNK_IDX, ZH_PAD_IDX, ZH_BOS_IDX, ZH_EOS_IDX = tokeniser.get_special_ids('zh')

# print(UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX)  # prints 1 65000 65000 0

SRC_VOCAB_SIZE = len(tokeniser)
TGT_VOCAB_SIZE = len(tokeniser)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == ZH_PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for row in batch:
        tr = row["translation"]
        
        tr = tokeniser(tr["en"], text_target=tr["zh"], max_len=EMB_SIZE)
        src_batch.append(torch.tensor(tr["input_ids"]).to(DEVICE))
        tgt_batch.append(torch.tensor(tr["labels"]).to(DEVICE)) # should we tokenize tgt with the zh-en tokenizer instead?

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=ZH_PAD_IDX)
    return src_batch, tgt_batch


def train_epoch(model, optimizer):
    model.train()
    losses = 0

    # if want to train on only a subset of the train
    # train_iter = load_dataset("iwslt2017", "iwslt2017-en-zh", split="train[:128]")

    # if want to train on all the train data
    train_iter = dataset["train"]

    train_dataloader = DataLoader(
        train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    for src, tgt in tqdm(train_dataloader, "train"):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))

def calc_bleu_scores(pred, tgt): 
    r , c = tgt.rstrip(), pred.rstrip()
    r = [[r]] # list of list of text
    c = [c] # list of text
    
    sacrebleu_scores = []
    for i in range(1, 5):
        bleu = BLEU(smooth_method='exp', tokenize='zh', max_ngram_order=i)
        s = bleu.corpus_score(c, r).score  
        sacrebleu_scores.append(s)
    
    sentence_bleu_score = sentence_bleu(r, c)
    return sentence_bleu_score, sum(sacrebleu_scores) / len(sacrebleu_scores)



def evaluate(model):
    model.eval()
    losses = 0
    nltk_bleu_score = 0
    sacrebleu_score = 0

    # if want to eval on only a subset of the validation data
    # val_iter = load_dataset("iwslt2017", "iwslt2017-en-zh", split="validation[:128]")

    # if want to eval on all the validation data
    val_iter = dataset["validation"]

    debug = True
    for val in tqdm(val_iter["translation"], "eval - sacrebleu"):
        src = val["en"]
        tgt = val["zh"]
        pred = translate(model, src)
        if debug:
            print(f"source: {src}")
            print(f"translated: {pred}")
            print(f"target: {tgt}")
            debug = False
        nltk, sacrebleu = calc_bleu_scores(pred, tgt)
        # print(f"nltk: {nltk} vs sacreblue: {sacrebleu}")
        nltk_bleu_score += nltk
        sacrebleu_score += sacrebleu

    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in tqdm(val_dataloader, "eval - loss"):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(list(val_dataloader)), nltk_bleu_score / len(val_iter['translation']), sacrebleu_score / len(val_iter['translation'])


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            DEVICE
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    # src = torch.tensor(tokenizer(src_sentence)["input_ids"]).view(-1, 1)
    src = torch.tensor(tokeniser(src_sentence)).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX
    ).flatten()
    return " ".join(tokeniser.decode(list(tgt_tokens.cpu().numpy())))


if __name__ == "__main__":
    transformer = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE,
        NHEAD,
        SRC_VOCAB_SIZE,
        TGT_VOCAB_SIZE,
        FFN_HID_DIM,
    )

    transformer = transformer.to(DEVICE)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

    LOADED_EPOCHS = 0
    DIR = "./checkpoints"
    PATH = ""
    # PATH = f"{DIR}/transformer_cp_e1.pt" # fill in with path to trained transformer's checkpoint .pt file if wanna continue training
    if PATH != "":
        print(f"Loaded checkpoint from {PATH.split('/')[-1]}")
        checkpoint = torch.load(PATH, map_location=f"cuda:0")
        transformer.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        LOADED_EPOCHS = checkpoint["epoch"]

    NUM_EPOCHS = 5
    CHECKPOINT_PATH = f"{DIR}/transformer_cp_e{NUM_EPOCHS+LOADED_EPOCHS}.pt"

    # initialise tensorboard
    writer = SummaryWriter()
    # will output to ./runs/ directory by default.

    for epoch in tqdm(range(1, NUM_EPOCHS + 1), "training-epoch"):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss, nltk_bleu_score, sacrebleu_score = evaluate(transformer)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Scores/nltksentencebleu", nltk_bleu_score, epoch)
        writer.add_scalar("Scores/sacrebleu", sacrebleu_score, epoch)
        print(
            (
                f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, NLTK Bleu Score: {nltk_bleu_score:.3f}, Sacrebleu Score: {sacrebleu_score}"
                f"Epoch time = {(end_time - start_time):.3f}s"
            )
        )
    writer.flush()  # make sure that all pending events have been written to disk.
    writer.close()

    # to save general checkpoint
    print(f"Saving checkpoint for {CHECKPOINT_PATH.split('/')[-1]}")
    torch.save({
                'epoch': NUM_EPOCHS + LOADED_EPOCHS,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
                }, CHECKPOINT_PATH)