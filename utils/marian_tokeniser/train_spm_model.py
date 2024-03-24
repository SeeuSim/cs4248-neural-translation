import datasets
import sentencepiece as spm
import re
import json
from transformers import MarianTokenizer, AutoTokenizer

TRAIN_BATCH_SIZE = 1000
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-en-zh"

dataset = datasets.load_dataset("iwslt2017", "iwslt2017-en-zh")

train, test, validation = dataset["train"], dataset["test"], dataset["validation"]

langs = ["en", "zh"]
character_coverage = {"en": 1.0, "zh": 0.995}


def write_training_input_file():
    def get_sent(sent, lang="zh"):
        """
        Normalise the text and lowercase.
        """
        if lang == "zh":
            return sent + "\n"
        return sent.lower() + "\n"

    for lang in langs:
        with open(f"./texts-raw-{lang}.txt", "w") as f:
            f.writelines(
                map(lambda row: get_sent(row[lang], lang=lang), train["translation"])
            )
            f.close()


def train_model():
    for lang in langs:
        spm.SentencePieceTrainer.train(
            input=f"./texts-raw-{lang}.txt",
            model_prefix=f"sp-{lang}",
            vocab_size=32_000,
            character_coverage=character_coverage[lang],
            model_type="bpe",
            user_defined_symbols=['<pad>'],
        )


def vocab_to_json():
    get_path = lambda lang: f'./sp-{lang}.vocab'

    for lang in langs:
        v = {}
        path = get_path(lang)
        with open(path, 'r') as f:
            for line in f.readlines():
                word, _, index = re.split(r'(\t|\s)+-?', line.rstrip())               
                v[word] = int(index)
        

        with open(f'sp-{lang}.json', 'w') as f:
            json.dump(v, f, indent=2)

if __name__ == "__main__":
    write_training_input_file()
    train_model()
    vocab_to_json()
