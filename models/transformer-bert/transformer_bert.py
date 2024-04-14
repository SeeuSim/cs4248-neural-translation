import copy
import datetime
import heapq
import math
import os
import sys

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

"""
Configuration params
"""
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

CONFIG = {
    "SEQUENCE_LENGTH": 288,
    "D_MODEL": 256,  # Originally 512
    "NUM_HEADS": 8,
    "NUM_LAYERS": 6,
    "D_FF": 1024,  # Originally 2048
    "D_K": 32,  # D_MODEL // NUM_HEADS, # Originally 64
    "DROP_OUT_RATE": 0.1,
}

BEAM_SIZE = 8

MAX_DATASET_ROWS = 300_000
BATCH_SIZE = 90
LEARNING_RATE = 1e-4
S_EPOCH = 1
N_EPOCHS = 20

SPM_SRC_MODEL_PATH = "../../tokenisation/sentencepiece_custom/en.model"
SPM_TGT_MODEL_PATH = "../../tokenisation/sentencepiece_custom/zh.model"
CHECKPOINT_DIR_PATH = None
CHECKPOINT_NAME = None

# Contains the dataset plain text files, with .en containing
# the English sentences and .zh containing the matching Chinese
# translations, line by line.
DATASET_DIR = "../../tokenisation/data"
TRAIN_NAME = "iwslt2017-en-zh-train"
VALID_NAME = "iwslt2017-en-zh-validation"
TEST_NAME = "iwslt2017-en-zh-test.en"
"""
Tokeniser
    - Uses the Google SentencePiece BPE Algorithm to learn a number of merges to output a vocabulary of a fixed size.
"""


class SentencePieceBPETokeniser(object):
    PAD_ID = 3  # Defined as sentencepiece custom token

    def __init__(self, lang: str, model_file=None):
        self.model = spm.SentencePieceProcessor(
            model_file=model_file or f"./{lang}.model"
        )
        self.special_ids = (
            self.model.unk_id(),
            SentencePieceBPETokeniser.PAD_ID,  # self.model.pad_id(), # this is -1 and may give errors.
            self.model.bos_id(),
            self.model.eos_id(),
        )

    def __len__(self):
        return len(self.model)

    def encode_batch(self, sents: list[str], max_len=None):
        return [self.encode(sent, max_len) for sent in sents]

    def encode(self, sent: str | list[str], max_len=None):
        if type(sent) == list:
            return self.encode_batch(sent, max_len)
        ids = self.model.encode(sent)
        return ids

    def decode(self, ids: list[int]):
        return self.model.decode(
            list(filter(lambda id: id >= 0 and id < len(self), ids))
        )

    def decode_batch(self, ids: list[list[int]]):
        return [self.decode(id) for id in ids]

    def get_special_ids(self):
        UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = self.special_ids
        return (UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX)


"""
Model Classes
"""


class EncoderLayer(nn.Module):
    def __init__(
        self, n_attn_heads=8, d_model=512, d_k=64, d_ff=2048, dropout_rate=0.1
    ):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model=d_model)
        self.multihead_attention = MultiheadAttention(
            n_heads=n_attn_heads, d_model=d_model, d_k=d_k, dropout_rate=dropout_rate
        )
        self.drop_out_1 = nn.Dropout(dropout_rate)

        self.layer_norm_2 = LayerNormalization(d_model=d_model)
        self.feed_forward = FeedFowardLayer(
            d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate
        )
        self.drop_out_2 = nn.Dropout(dropout_rate)

    def forward(self, x, e_mask):
        x_1 = self.layer_norm_1(x)  # (B, L, d_model)
        x = x + self.drop_out_1(
            self.multihead_attention(x_1, x_1, x_1, mask=e_mask)
        )  # (B, L, d_model)
        x_2 = self.layer_norm_2(x)  # (B, L, d_model)
        x = x + self.drop_out_2(self.feed_forward(x_2))  # (B, L, d_model)

        return x  # (B, L, d_model)


class DecoderLayer(nn.Module):
    def __init__(
        self, n_attn_heads=8, d_model=512, d_k=64, d_ff=2048, dropout_rate=0.1
    ):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model=d_model)
        self.masked_multihead_attention = MultiheadAttention(
            n_heads=n_attn_heads, d_model=d_model, d_k=d_k, dropout_rate=dropout_rate
        )
        self.drop_out_1 = nn.Dropout(dropout_rate)

        self.layer_norm_2 = LayerNormalization(d_model=d_model)
        self.multihead_attention = MultiheadAttention(
            n_heads=n_attn_heads, d_model=d_model, d_k=d_k, dropout_rate=dropout_rate
        )
        self.drop_out_2 = nn.Dropout(dropout_rate)

        self.layer_norm_3 = LayerNormalization(d_model=d_model)
        self.feed_forward = FeedFowardLayer(
            d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate
        )
        self.drop_out_3 = nn.Dropout(dropout_rate)

    def forward(self, x, e_output, e_mask, d_mask):
        x_1 = self.layer_norm_1(x)  # (B, L, d_model)
        x = x + self.drop_out_1(
            self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask)
        )  # (B, L, d_model)
        x_2 = self.layer_norm_2(x)  # (B, L, d_model)
        x = x + self.drop_out_2(
            self.multihead_attention(x_2, e_output, e_output, mask=e_mask)
        )  # (B, L, d_model)
        x_3 = self.layer_norm_3(x)  # (B, L, d_model)
        x = x + self.drop_out_3(self.feed_forward(x_3))  # (B, L, d_model)

        return x  # (B, L, d_model)


class MultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, dropout_rate):
        super().__init__()
        self.inf = 1e9
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k

        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Final output linear transformation
        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(
            input_shape[0], -1, self.n_heads, self.d_k
        )  # (B, L, num_heads, d_k)
        k = self.w_k(k).view(
            input_shape[0], -1, self.n_heads, self.d_k
        )  # (B, L, num_heads, d_k)
        v = self.w_v(v).view(
            input_shape[0], -1, self.n_heads, self.d_k
        )  # (B, L, num_heads, d_k)

        # For convenience, convert all tensors in size (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values = self.self_attention(q, k, v, mask=mask)  # (B, num_heads, L, d_k)
        concat_output = (
            attn_values.transpose(1, 2)
            .contiguous()
            .view(input_shape[0], -1, self.d_model)
        )  # (B, L, d_model)

        return self.w_0(concat_output)

    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(self.d_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(
                1
            )  # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_distribs = self.attn_softmax(attn_scores)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v)  # (B, num_heads, L, d_k)

        return attn_values


class FeedFowardLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout_rate=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.linear_1(x))  # (B, L, d_ff)
        x = self.dropout(x)
        x = self.linear_2(x)  # (B, L, d_model)

        return x


class LayerNormalization(nn.Module):
    def __init__(self, d_model=512, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([d_model], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)

        return x


class PositionalEncoder(nn.Module):
    def __init__(self, sequence_len=288, d_model=512):
        super().__init__()
        self.d_model = d_model

        # Make initial positional encoding matrix with 0
        pe_matrix = torch.zeros(sequence_len, d_model)  # (L, d_model)

        # Calculating position encoding values
        for pos in range(sequence_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

        pe_matrix = pe_matrix.unsqueeze(0)  # (1, L, d_model)
        self.positional_encoding = pe_matrix.to(device=DEVICE).requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)  # (B, L, d_model)
        x = x + self.positional_encoding  # (B, L, d_model)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers=6,
        n_attn_heads=8,
        d_model=512,
        d_k=64,
        d_ff=2048,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    n_attn_heads=n_attn_heads, d_model=d_model, d_k=d_k, d_ff=d_ff
                )
                for _i in range(num_layers)
            ]
        )
        self.layer_norm = LayerNormalization(d_model=d_model)

    def forward(self, x, e_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self, num_layers=6, n_attn_heads=8, d_model=512, d_k=64, d_ff=2048):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    n_attn_heads=n_attn_heads, d_model=d_model, d_k=d_k, d_ff=d_ff
                )
                for i in range(num_layers)
            ]
        )
        self.layer_norm = LayerNormalization(d_model=d_model)

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(self.num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size=16384,
        trg_vocab_size=16384,
        max_seq_len=288,
        n_enc_layers=6,
        n_dec_layers=6,
        n_attn_heads=8,
        d_model=512,
        d_k=64,
        d_ff=2048,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.SEQUENCE_LEN = max_seq_len

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(
            sequence_len=max_seq_len, d_model=d_model
        )
        self.encoder = Encoder(
            num_layers=n_enc_layers,
            n_attn_heads=n_attn_heads,
            d_model=d_model,
            d_k=d_k,
            d_ff=d_ff,
        )
        self.decoder = Decoder(
            num_layers=n_dec_layers,
            n_attn_heads=n_attn_heads,
            d_model=d_model,
            d_k=d_k,
            d_ff=d_ff,
        )
        self.output_linear = nn.Linear(d_model, self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, trg_input, e_mask=None, d_mask=None):
        src_input = self.src_embedding(src_input)  # (B, L) => (B, L, d_model)
        trg_input = self.trg_embedding(trg_input)  # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(
            src_input
        )  # (B, L, d_model) => (B, L, d_model)
        trg_input = self.positional_encoder(
            trg_input
        )  # (B, L, d_model) => (B, L, d_model)

        e_output = self.encoder(src_input, e_mask)  # (B, L, d_model)
        d_output = self.decoder(trg_input, e_output, e_mask, d_mask)  # (B, L, d_model)

        output = self.softmax(
            self.output_linear(d_output)
        )  # (B, L, d_model) => # (B, L, trg_vocab_size)

        return output


"""
Utility Classes for Beam Search
"""


class BeamNode:
    def __init__(self, cur_idx, prob, decoded):
        self.cur_idx = cur_idx
        self.prob = prob
        self.decoded = decoded
        self.is_finished = False

    def __gt__(self, other):
        return self.prob > other.prob

    def __ge__(self, other):
        return self.prob >= other.prob

    def __lt__(self, other):
        return self.prob < other.prob

    def __le__(self, other):
        return self.prob <= other.prob

    def __eq__(self, other):
        return self.prob == other.prob

    def __ne__(self, other):
        return self.prob != other.prob

    def print_spec(self):
        print(
            f"ID: {self} || cur_idx: {self.cur_idx} || prob: {self.prob} || decoded: {self.decoded}"
        )


class PriorityQueue:
    def __init__(self):
        self.queue = []

    def put(self, obj):
        heapq.heappush(self.queue, (obj.prob, obj))

    def get(self):
        return heapq.heappop(self.queue)[1]

    def qsize(self):
        return len(self.queue)

    def print_scores(self):
        scores = [t[0] for t in self.queue]
        print(scores)

    def print_objs(self):
        objs = [t[1] for t in self.queue]
        print(objs)


"""
Utility Functions for Dataset Processing
"""


class CustomDataset(Dataset):
    def __init__(self, src_list, input_trg_list, output_trg_list):
        super().__init__()
        self.src_data = torch.LongTensor(src_list)
        self.input_trg_data = torch.LongTensor(input_trg_list)
        self.output_trg_data = torch.LongTensor(output_trg_list)

        assert np.shape(src_list) == np.shape(
            input_trg_list
        ), f"The shape of src_list and input_trg_list are different: {np.shape(src_list)} {np.shape(input_trg_list)}"
        assert np.shape(input_trg_list) == np.shape(
            output_trg_list
        ), f"The shape of input_trg_list and output_trg_list are different: {np.shape(input_trg_list)} {np.shape(output_trg_list)}"

    def __getitem__(self, idx):
        return self.src_data[idx], self.input_trg_data[idx], self.output_trg_data[idx]

    def __len__(self):
        return np.shape(self.src_data)[0]


def pad_or_truncate(tokenized_text, pad_idx, max_len=None):
    if len(tokenized_text) < max_len:
        left = max_len - len(tokenized_text)
        padding = [pad_idx] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:max_len]

    return tokenized_text


def process_src(text_list, en_tokeniser=None):
    global MAX_SRC_LEN
    _, PAD_IDX, _, EOS_IDX = en_tokeniser.get_special_ids()
    src_input_ids = []
    for text in tqdm(text_list[:MAX_DATASET_ROWS]):
        ids = en_tokeniser.encode(text.strip())
        src_input_ids.append(pad_or_truncate(ids + [EOS_IDX], PAD_IDX))
    return src_input_ids


def process_trg(text_list, zh_tokeniser=None):
    global MAX_TGT_LEN
    _, PAD_IDX, BOS_IDX, EOS_IDX = zh_tokeniser.get_special_ids()
    input_tokenized_list = []
    output_tokenized_list = []
    for text in tqdm(text_list[:MAX_DATASET_ROWS]):
        tokenized = zh_tokeniser.encode(text.strip())
        trg_input = [BOS_IDX] + tokenized
        trg_output = tokenized + [EOS_IDX]
        input_tokenized_list.append(pad_or_truncate(trg_input, PAD_IDX))
        output_tokenized_list.append(pad_or_truncate(trg_output, PAD_IDX))
    return input_tokenized_list, output_tokenized_list


def get_data_loader(file_name, en_tokeniser, zh_tokeniser):
    print(f"Getting source/target {file_name}...")
    with open(f"{DATASET_DIR}/{file_name}.en", "r") as f:
        src_text_list = f.readlines()

    with open(f"{DATASET_DIR}/{file_name}.zh", "r") as f:
        trg_text_list = f.readlines()

    print("Tokenizing & Padding src data...")
    src_list = process_src(src_text_list, en_tokeniser=en_tokeniser)  # (sample_num, L)
    print(f"The shape of src data: {np.shape(src_list)}")

    print("Tokenizing & Padding trg data...")
    input_trg_list, output_trg_list = process_trg(
        trg_text_list, zh_tokeniser=zh_tokeniser
    )  # (sample_num, L)
    print(f"The shape of input trg data: {np.shape(input_trg_list)}")
    print(f"The shape of output trg data: {np.shape(output_trg_list)}")

    dataset = CustomDataset(src_list, input_trg_list, output_trg_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return dataloader


"""
Utility Functions
"""


def init_model(SRC_VOCAB, TGT_VOCAB):
    config = CONFIG

    model = Transformer(
        src_vocab_size=SRC_VOCAB,
        trg_vocab_size=TGT_VOCAB,
        max_seq_len=config["SEQUENCE_LENGTH"],
        n_enc_layers=config["NUM_LAYERS"],
        n_dec_layers=config["NUM_LAYERS"],
        n_attn_heads=config["NUM_HEADS"],
        d_model=config["D_MODEL"],
        d_ff=config["D_FF"],
        d_k=config["D_K"],
    ).to(DEVICE)
    # sdict = torch.load(MODEL_PATH, map_location=DEVICE)
    # model.load_state_dict(sdict["model_state_dict"], strict=False)
    # model.eval()
    return model


"""
Utility classes
"""


class Manager:
    def __init__(self, is_train=True, ckpt_name=None):

        # Load Transformer model & Adam optimizer
        print("Loading Tokeniser SPM files")
        self.en_tokeniser = SentencePieceBPETokeniser(
            "en", model_file=SPM_SRC_MODEL_PATH
        )
        self.zh_tokeniser = SentencePieceBPETokeniser(
            "zh", model_file=SPM_TGT_MODEL_PATH
        )

        print("Loading Transformer model & Adam optimizer...")
        self.model = init_model(len(self.en_tokeniser), len(self.zh_tokeniser))
        self.optim = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.best_loss = sys.float_info.max

        if ckpt_name is not None:
            assert os.path.exists(
                f"{CHECKPOINT_DIR_PATH}/{ckpt_name}"
            ), f"There is no checkpoint named {ckpt_name}."

            print("Loading checkpoint...")
            checkpoint = torch.load(
                f"{CHECKPOINT_DIR_PATH}/{ckpt_name}", map_location=DEVICE
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optim.load_state_dict(checkpoint["optim_state_dict"])
            self.best_loss = checkpoint["loss"]
        else:
            print("Initializing the model...")
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        if is_train:
            # Load loss function
            print("Loading loss function...")
            self.criterion = nn.NLLLoss()

            # Load dataloaders
            print("Loading dataloaders...")
            self.train_loader = get_data_loader(
                TRAIN_NAME,
                en_tokeniser=self.en_tokeniser,
                zh_tokeniser=self.zh_tokeniser,
            )
            self.valid_loader = get_data_loader(
                VALID_NAME,
                en_tokeniser=self.en_tokeniser,
                zh_tokeniser=self.zh_tokeniser,
            )

        print("Setting finished.")

    def train(self):
        print("Training starts.")

        for epoch in range(S_EPOCH, N_EPOCHS + S_EPOCH):
            self.model.train()

            train_losses = []
            start_time = datetime.datetime.now()

            for _i, batch in tqdm(enumerate(self.train_loader)):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = (
                    src_input.to(DEVICE),
                    trg_input.to(DEVICE),
                    trg_output.to(DEVICE),
                )

                e_mask, d_mask = self.make_mask(src_input, trg_input)

                output = self.model(
                    src_input, trg_input, e_mask, d_mask
                )  # (B, L, vocab_size)

                trg_output_shape = trg_output.shape
                self.optim.zero_grad()
                loss = self.criterion(
                    output.view(-1, OUTPUT_VOCAB_SIZE),
                    trg_output.view(trg_output_shape[0] * trg_output_shape[1]),
                )

                loss.backward()
                self.optim.step()

                train_losses.append(loss.item())

                del src_input, trg_input, trg_output, e_mask, d_mask, output
                torch.cuda.empty_cache()

            end_time = datetime.datetime.now()
            training_time = end_time - start_time
            seconds = training_time.seconds
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60

            mean_train_loss = np.mean(train_losses)
            print(f"#################### Epoch: {epoch} ####################")
            print(
                f"Train loss: {mean_train_loss} || One epoch training time: {hours}hrs {minutes}mins {seconds}secs"
            )

            valid_loss, valid_time = self.validation()

            if not os.path.exists(CHECKPOINT_DIR_PATH):
                os.mkdir(CHECKPOINT_DIR_PATH)

            self.best_loss = valid_loss
            state_dict = {
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": self.optim.state_dict(),
                "loss": self.best_loss,
            }
            torch.save(state_dict, f"{CHECKPOINT_DIR_PATH}/ckpt-{epoch}.tar")
            if valid_loss < self.best_loss:
                print(f"***** Current best checkpoint is saved. *****")
                torch.save(state_dict, f"{CHECKPOINT_DIR_PATH}/ckpt-{epoch}-best.tar")

            print(f"Best valid loss: {self.best_loss}")
            print(f"Valid loss: {valid_loss} || One epoch training time: {valid_time}")

        print(f"Training finished!")

    def validation(self):
        print("Validation processing...")
        self.model.eval()

        valid_losses = []
        start_time = datetime.datetime.now()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valid_loader)):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = (
                    src_input.to(DEVICE),
                    trg_input.to(DEVICE),
                    trg_output.to(DEVICE),
                )

                e_mask, d_mask = self.make_mask(src_input, trg_input)

                output = self.model(
                    src_input, trg_input, e_mask, d_mask
                )  # (B, L, vocab_size)

                trg_output_shape = trg_output.shape
                loss = self.criterion(
                    output.view(-1, OUTPUT_VOCAB_SIZE),
                    trg_output.view(trg_output_shape[0] * trg_output_shape[1]),
                )

                valid_losses.append(loss.item())

                del src_input, trg_input, trg_output, e_mask, d_mask, output
                torch.cuda.empty_cache()

        end_time = datetime.datetime.now()
        validation_time = end_time - start_time
        seconds = validation_time.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        mean_valid_loss = np.mean(valid_losses)

        return mean_valid_loss, f"{hours}hrs {minutes}mins {seconds}secs"

    def make_mask(self, src_input, trg_input):
        e_mask = (src_input != self.en_tokeniser.PAD_ID).unsqueeze(1)  # (B, 1, L)
        d_mask = (trg_input != self.zh_tokeniser.PAD_ID).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones(
            [1, CONFIG["SEQUENCE_LENGTH"], CONFIG["SEQUENCE_LENGTH"]], dtype=torch.bool
        )  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask).to(
            DEVICE
        )  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

        return e_mask, d_mask

    def beam_search(self, e_output, e_mask):

        _, TGT_PAD_IDX, TGT_BOS_IDX, TGT_EOS_IDX = self.zh_tokeniser.get_special_ids()
        cur_queue = PriorityQueue()
        for k in range(BEAM_SIZE):
            cur_queue.put(BeamNode(TGT_BOS_IDX, -0.0, [TGT_BOS_IDX]))

        finished_count = 0

        SEQUENCE_LENGTH = self.model.SEQUENCE_LEN

        for pos in range(SEQUENCE_LENGTH):
            new_queue = PriorityQueue()
            for k in range(BEAM_SIZE):
                node = cur_queue.get()
                if node.is_finished:
                    new_queue.put(node)
                else:
                    trg_input = torch.LongTensor(
                        node.decoded
                        + [TGT_PAD_IDX] * (SEQUENCE_LENGTH - len(node.decoded))
                    ).to(
                        DEVICE
                    )  # (L)
                    d_mask = (
                        (trg_input.unsqueeze(0) != TGT_PAD_IDX).unsqueeze(1).to(DEVICE)
                    )  # (1, 1, L)
                    nopeak_mask = torch.ones(
                        [1, SEQUENCE_LENGTH, SEQUENCE_LENGTH], dtype=torch.bool
                    ).to(DEVICE)
                    nopeak_mask = torch.tril(
                        nopeak_mask
                    )  # (1, L, L) to triangular shape
                    d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

                    trg_embedded = self.model.trg_embedding(trg_input.unsqueeze(0))
                    trg_positional_encoded = self.model.positional_encoder(trg_embedded)
                    decoder_output = self.model.decoder(
                        trg_positional_encoded, e_output, e_mask, d_mask
                    )  # (1, L, d_model)

                    output = self.model.softmax(
                        self.model.output_linear(decoder_output)
                    )  # (1, L, trg_vocab_size)

                    output = torch.topk(output[0][pos], dim=-1, k=BEAM_SIZE)
                    last_word_ids = output.indices.tolist()  # (k)
                    last_word_prob = output.values.tolist()  # (k)

                    for i, idx in enumerate(last_word_ids):
                        new_node = BeamNode(
                            idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx]
                        )
                        if idx == TGT_EOS_IDX:
                            new_node.prob = new_node.prob / float(len(new_node.decoded))
                            new_node.is_finished = True
                            finished_count += 1
                        new_queue.put(new_node)

            cur_queue = copy.deepcopy(new_queue)

            if finished_count == BEAM_SIZE:
                break

        decoded_output = cur_queue.get().decoded

        if decoded_output[-1] == TGT_EOS_IDX:
            decoded_output = decoded_output[1:-1]
        else:
            decoded_output = decoded_output[1:]

        return self.zh_tokeniser.decode(decoded_output)

    def translate(self, text: str, verbose=False):
        _, SRC_PAD, _, _ = self.en_tokeniser.get_special_ids()
        tokenized = self.en_tokeniser.encode(text)
        src_data = (
            torch.LongTensor(
                pad_or_truncate(
                    tokenized, pad_idx=SRC_PAD, max_len=self.model.SEQUENCE_LEN
                )
            )
            .unsqueeze(0)
            .to(DEVICE)
        )  # (1, L)
        e_mask = (src_data != SRC_PAD).unsqueeze(1).to(DEVICE)  # (1, 1, L)

        start_time = datetime.datetime.now()

        if verbose:
            print("Encoding input sentence...")
        src_data = self.model.src_embedding(src_data)
        src_data = self.model.positional_encoder(src_data)
        e_output = self.model.encoder(src_data, e_mask)  # (1, L, d_model)

        result = self.beam_search(e_output, e_mask)

        end_time = datetime.datetime.now()

        total_inference_time = end_time - start_time
        seconds = total_inference_time.seconds
        minutes = seconds // 60
        seconds = seconds % 60

        if verbose:
            print(f"Input: {text}")
            print(f"Result: {result}")
            print(
                f"Inference finished! || Total inference time: {minutes}mins {seconds}secs"
            )

        return result


if __name__ == "__main__":
    IS_TRAIN = True
    if CHECKPOINT_NAME is not None:
        manager = Manager(IS_TRAIN, CHECKPOINT_NAME)
    else:
        manager = Manager(IS_TRAIN)

    if IS_TRAIN:
        manager.train()
    else:
        import json
        # Translation, generate output
        with open(f"{DATASET_DIR}/{TEST_NAME}", "r") as test_ds, open("../../evaluations/predictions/230k-transformer-bert.json", "w") as out_ds:
            t_inp = test_ds.readlines()
            predicted = []
            for line in tqdm(t_inp):
                predicted.append(manager.translate(line.strip()))

            json.dump({"predicted": predicted}, out_ds)
