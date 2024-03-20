from itertools import chain
import datasets
import gensim.models
import gensim.downloader
from tqdm import tqdm
from gensim.models.fasttext import load_facebook_model

from utils.tokenise import tokenise

dataset = datasets.load_dataset("iwslt2017", "iwslt2017-en-zh")

train, test, validation = dataset['train'], dataset['test'], dataset['validation']

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __iter__(self):
        get_eng_sent = lambda x: x['en']
        for line in tqdm(chain.from_iterable([
            map(get_eng_sent, train['translation']),
            map(get_eng_sent, test['translation']),
            map(get_eng_sent, validation['translation'])
        ])):
            # TODO: yield list of tokens (list[str])
            yield tokenise(line)

def train_fasttext(custom=True):
    """Trains fasttext embeddings
    - either from scratch or transfer learning with pretrained embeddings
    """
    path_to_dir = "./finetuned_embeddings"
    sentences = MyCorpus()

    if custom:
        print(f"Custom Training FastText model")
        model = gensim.models.FastText(vector_size=4, window=3, min_count=1, sentences=sentences, epochs=10)
        print(f"Custom word vectors: {model.wv}") # FastTextKeyedVectors<vector_size=4, 70914 keys>
        model.save(f'{path_to_dir}/custom-fasttext.model')
    else:
        # Download pretrained english fasttext model here: https://fasttext.cc/docs/en/crawl-vectors.html
        print(f"Retraining pretrained fasttext model")
        model_path = "./pretrained_embeddings/cc.en.300.bin"
        model = load_facebook_model(model_path)
        print(f"Pretrained word vectors: {model.wv}") #FastTextKeyedVectors<vector_size=300, 2000000 keys>
        model.build_vocab(sentences, update=True)
        training_examples_count = model.corpus_count
        print(f"Transfer Learning Training Examples Count = {training_examples_count}") # 240694
        model.train(sentences, total_examples=training_examples_count, epochs=model.epochs)
        print(f"Retrained word vectors: {model.wv}")  # FastTextKeyedVectors<vector_size=300, 2000281 keys> vocab increased by 281 
        model.save(f'{path_to_dir}/retrained_fasttext.model')


if __name__ == "__main__":
    train_fasttext()
    train_fasttext(custom=False)