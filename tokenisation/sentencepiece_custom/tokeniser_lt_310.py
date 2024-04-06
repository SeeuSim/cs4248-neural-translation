import sentencepiece as spm
from typing import List, Union

class LangTokeniser(object):
    PAD_ID = 3  # Defined as sentencepiece custom token

    def __init__(self, lang: str, model_file=None):
        self.model = spm.SentencePieceProcessor(model_file=model_file or f"./{lang}.model")
        self.special_ids = (
            self.model.unk_id(),
            LangTokeniser.PAD_ID,  # self.model.pad_id(), # this is -1 and may give errors.
            self.model.bos_id(),
            self.model.eos_id(),
        )

    def __len__(self):
        return len(self.model)
    
    def encode_no_padding(self, sent: Union[str, List[str]], max_len=None):
        ids = self.model.encode(sent)
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def encode_batch(self, sents: List[str], max_len=None):
        return [self.encode(sent, max_len) for sent in sents]

    def encode(self, sent: Union[str, List[str]], max_len=None):
        if isinstance(sent, list):
            return self.encode_batch(sent, max_len)
        ids = self.model.encode(sent)
        if max_len is not None:
            if len(ids) < max_len:
                ids.extend([LangTokeniser.PAD_ID] * (max_len - len(ids)))
            elif len(ids) > max_len:
                ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int]):
        return self.model.decode([id for id in ids if 0 <= id < len(self) and id != LangTokeniser.PAD_ID])

    def decode_batch(self, ids: List[List[int]]):
        return [self.decode(id) for id in ids]

    def get_special_ids(self):
        UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = self.special_ids
        return UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX


class BaseBPETokeniser(object):
    """
    The class to tokenise input English sentences, and decode output Chinese Vocab IDs.

    Examples:
    ```py
    from tokenisation.sentencepiece_custom import BaseBPETokeniser

    tokeniser = BaseBPETokeniser()
    # or initialise with the model files in a separate path:
    tokeniser = BaseBPETokeniser(en_model_file="/path/to/en.model", zh_model_file="/path/to/zh.model")

    row = dataset[0]['translation']

    # Tokenise and truncate to max length of 512 for both.
    inputs = tokeniser(row['en'], text_target=row['zh'], max_len=512)
    # {
    #     'input_ids': [...],       # The English IDs
    #     'attention_mask': [...],
    #     'labels': [...]           # The Chinese IDs
    # }

    # should generate the Chinese tokens output.
    translated = tokeniser.decode(ids)

    ```
    """

    def __init__(self, en_model_file=None, zh_model_file=None):
        self.en_model = LangTokeniser("en", model_file=en_model_file)
        self.zh_model = LangTokeniser("zh", model_file=zh_model_file)

    def __len__(self):
        """
        Both the english and chinese tokenisers have the same length.
        """
        return len(self.en_model)

    def __call__(self, sent: str, text_target=None, max_len=128, max_zh_len=None):
        out = {
            "input_ids": self.en_model.encode(sent, max_len=max_len),
            "attention_mask": [1] * max_len,
        }
        if text_target:
            out["labels"] = self.zh_model.encode(
                text_target, max_len=max_zh_len or max_len
            )
        return out

    def encode_zh(self, sent: str, max_len=128):
        return self.zh_model.encode(sent, max_len=max_len)

    def decode(self, labels: list[int]):
        return self.zh_model.decode(labels)

    def decode_src(self, labels: list[int]):
        return self.en_model.decode(labels)

    def get_special_ids(self, lang: str):
        if lang == "en":
            return self.en_model.get_special_ids()
        elif lang == "zh":
            return self.zh_model.get_special_ids()

    def encode_en_no_padding(self, sent: str, max_len=None):
        return self.en_model.encode_no_padding(sent, max_len=max_len)

    def encode_zh_no_padding(self, sent: str, max_len=None):
        return self.zh_model.encode_no_padding(sent, max_len=max_len)