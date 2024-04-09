from itertools import chain
import datasets
import gensim.models
import gensim.downloader
from tqdm import tqdm
import numpy as np

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

def train_w2v(custom=True):
    """
    Trains w2v embeddings on MyCorpus
    - either from scratch or transfer learning with pretrained google news embeddings
    """
    path_to_dir = "./finetuned_embeddings"
    sentences = MyCorpus()

    if custom:
        print(f"Custom Training W2V model")
        model = gensim.models.Word2Vec(sentences=sentences, workers=5, epochs=10)
        print(f"Transfer Learning Training Examples Count = {model.corpus_count}") # 240694
        print(f"Custom word vectors: {model.wv}") # KeyedVectors<vector_size=100, 23359 keys>
        model.save(f'{path_to_dir}/custom-w2v.model')
    else:
        # Note: transfer learning from google news pretrained embeddings is experimental.
        # Uses experimental intersect_word2vec_format function
        # See function docs here: https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.intersect_word2vec_format.html
        # 
        # Pretrained embeddings "lacks hidden weights, vocabulary frequencies, 
        # and the binary tree"
        # See: 
        # https://stackoverflow.com/questions/70574351/word-vectors-transfer-learning
        # https://phdstatsphys.wordpress.com/2018/12/27/word2vec-how-to-train-and-update-it/
        # Example of transfer learning with word2vec https://gist.github.com/AbhishekAshokDubey/054af6f92d67d5ef8300fac58f59fcc9
        # Download pretrained google news embeddings from: https://code.google.com/archive/p/word2vec/

        print(f"Retraining pretrained google news W2V model")
        google_wv = gensim.downloader.load('word2vec-google-news-300')
        print("Downloaded KeyedVectors for 'word2vec-google-news-300'")

        model = gensim.models.Word2Vec(vector_size=300) # size = 300 to match gnews w2v size
        model.build_vocab(sentences)
        training_examples_count = model.corpus_count
        print(f"Transfer Learning Training Examples Count = {training_examples_count}") # 240694
        # below line will make it 1, so saving it before
        model.build_vocab([list(google_wv.key_to_index.keys())], update=True)

        model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32) # hacky workaround
        model_path = "./pretrained_embeddings/GoogleNews-vectors-negative300.bin"
        model.wv.intersect_word2vec_format(model_path, binary=True, lockf=1.0)
        model.train(sentences, total_examples=training_examples_count, epochs=10) # about 28 sec per epoch
        print(f"Retrained word vectors: {model.wv}") # KeyedVectors<vector_size=100, 23359 keys>
        model.save(f'{path_to_dir}/retrained_gnews_w2v.model')
        # model.wv.save("./gnews_w2v_embeddings")


if __name__ == "__main__":
    train_w2v()
    train_w2v(custom=False)