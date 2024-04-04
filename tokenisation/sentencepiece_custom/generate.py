from datasets import load_dataset
from itertools import chain
from tqdm import tqdm


def format_dataset():
    ds = load_dataset("iwslt2017", "iwslt2017-en-zh")
    it = chain.from_iterable(
        [
            ds["train"],
            ds["test"],
            ds["validation"],
        ]
    )

    with open("./iwslt2017-en-zh.en", "w") as en_file, open(
        "./iwslt2017-en-zh.zh", "w"
    ) as zh_file:
        for row in tqdm(map(lambda r: r["translation"], it)):
            en_file.write(row["en"] + "\n")
            zh_file.write(row["zh"] + "\n")
        en_file.close()
        zh_file.close()


if __name__ == "__main__":
    format_dataset()
