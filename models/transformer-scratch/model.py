import math
import sentencepiece as spm
import torch
import torch.nn as nn

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# SEQUENCE_LENGTH = 288
# NUM_HEADS = 8
# NUM_LAYERS = 6
# D_MODEL = 256
# D_FF = 1024
# D_K = D_MODEL // NUM_HEADS
#
# DROP_OUT_RATE = 0.1
# NUM_EPOCHS = 10
# BEAM_SIZE = 8


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
