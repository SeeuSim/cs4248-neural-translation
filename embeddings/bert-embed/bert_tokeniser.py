from collections.abc import Mapping
from transformers import BatchEncoding
import sentencepiece as spm

SEQUENCE_LENGTH = 288


class BPEBertTokeniser:
    out_keys = [
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "special_tokens_mask",
    ]

    def __init__(self, lang, model_file):
        self.lang = lang
        self.model = spm.SentencePieceProcessor(model_file=model_file)

        self.pad_token_id = 3
        self.bos_id = self.model.bos_id()
        self.eos_id = self.model.eos_id()

    def _process_id(self, input_ids):
        input_ids = [self.bos_id, *input_ids, self.eos_id]
        o_len = len(input_ids)
        token_type_ids = [0] * o_len
        attention_mask = [1] * o_len
        special_tokens_mask = [1] + [0] * (o_len - 2) + [1]

        if o_len > SEQUENCE_LENGTH:
            input_ids = input_ids[: SEQUENCE_LENGTH - 1] + [self.eos_id]
            token_type_ids = token_type_ids[:SEQUENCE_LENGTH]
            attention_mask = attention_mask[:SEQUENCE_LENGTH]
            special_tokens_mask = special_tokens_mask[: SEQUENCE_LENGTH - 1] + [1]

        elif o_len < SEQUENCE_LENGTH:
            # EOS
            input_ids += [self.eos_id]

            # Padding
            input_ids += [self.pad_token_id] * (SEQUENCE_LENGTH - len(input_ids))

            token_type_ids += [0] * (SEQUENCE_LENGTH - len(token_type_ids))
            attention_mask += [0] * (SEQUENCE_LENGTH - len(attention_mask))

            # Padding
            special_tokens_mask += [1] * (SEQUENCE_LENGTH - len(special_tokens_mask))

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
        }

    def encode(self, row):
        if isinstance(row, list):
            return self.encode_batch(row)

        raw_ids = self.model.encode(row)
        return self._process_id(raw_ids)

    def encode_batch(self, rows):
        ids = list(map(lambda row: self._process_id(row), self.model.encode(rows)))
        return {key: [example[key] for example in ids] for key in BPEBertTokeniser.out_keys}

    def pad(self, inputs, **_kwargs):
        if (
            isinstance(inputs, (list, tuple))
            and len(inputs) > 0
            and isinstance(inputs[0], Mapping)
        ):
            inputs = {
                key: [example[key] for example in inputs] for key in inputs[0].keys()
            }
        return BatchEncoding(inputs, tensor_type="pt")

    def __call__(self, inputs, **_kwargs):
        return self.encode(inputs)

    def __len__(self):
        return len(self.model)

