import sentencepiece as spm

VOCAB_SIZE = 16384
# vocab_size = 32000

# <unk>, <s>, </s> all added
SYMBOLS = [
    "<pad>",
]

def train(input_path: str, lang: str):
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=lang,
        model_type="bpe",
        character_coverage=1.0 if lang == "en" else 0.98,
        vocab_size=VOCAB_SIZE,
        user_defined_symbols=SYMBOLS,
    )


if __name__ == "__main__":
    for lang in ['en', 'zh']:
        train(f"./iwslt2017-en-zh.{lang}", lang)
