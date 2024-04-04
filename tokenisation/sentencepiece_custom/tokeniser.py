import sentencepiece as spm


class LangTokeniser(object):
    def __init__(self, lang: str):
        self.model = spm.SentencePieceProcessor(model_file=f"./{lang}.model")
        self.special_ids = [
            self.model.unk_id(),
            self.model.pad_id(),
            self.model.bos_id(),
            self.model.eos_id(),
        ]

    def __len__(self):
        return len(self.model)

    def encode_batch(self, sents: list[str], pad_len=None, truncate_len=None):
        return [self.encode(sent, pad_len, truncate_len) for sent in sents]

    def encode(self, sent: str | list[str], pad_len=None, truncate_len=None):
        if type(sent) == list:
            return self.encode_batch(sent, pad_len, truncate_len)
        ids = self.model.encode(sent)
        if pad_len is not None and len(ids) < int(pad_len):
            ids = [*ids, *([self.model.pad_id()] * (int(pad_len) - len(ids)))]
        elif truncate_len is not None and len(ids) > int(truncate_len):
            ids = ids[: int(truncate_len)]
        return ids

    def decode(self, ids: list[int] | list[list[int]]):
        return self.model.decode(
            list(filter(lambda id: id >= 0 and id < len(self), ids))
        )

    def get_special_ids(self):
        UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = (
            self.model.unk_id(),
            self.model.pad_id(),
            self.model.bos_id(),
            self.model.eos_id(),
        )
        return (UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX)


class BaseBPETokeniser(object):
    def __init__(self):
        self.en_model = LangTokeniser('en')
        self.zh_model = LangTokeniser('zh')
    
    def __len__(self):
        """
        Both the english and chinese tokenisers have the same length.
        """
        return len(self.en_model)
    
    def __call__(self, sent: str, text_target=None,):
        pass