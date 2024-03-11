from itertools import chain
import datasets
import gensim.models
from tqdm import tqdm

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

def train_w2v():
    sentences = MyCorpus()
    model = gensim.models.Word2Vec(sentences=sentences, workers=5)
    model.save('./custom-w2v.txt')

if __name__ == "__main__":
    train_w2v()
