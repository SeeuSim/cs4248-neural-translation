from itertools import chain
import datasets
from tqdm import tqdm
import fasttext
import fasttext.util
import sentencepiece as spm

# https://www.kaggle.com/code/mschumacher/using-fasttext-models-for-robust-embeddings

dataset = datasets.load_dataset("iwslt2017", "iwslt2017-en-zh")

train, test, validation = dataset['train'], dataset['test'], dataset['validation']

tokeniser = spm.SentencePieceProcessor(model_file="../../tokenisation/sentencepiece_custom/en.model")    

def tokenise(sent):
    return tokeniser.EncodeAsPieces(sent)

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

def train_fasttext():
    """Trains fasttext embeddings
    - transfer learning with pretrained embeddings
    """
    sentences = MyCorpus()
    fasttext.util.download_model('en', if_exists='ignore')  # English    # Download pretrained english fasttext model here: https://fasttext.cc/docs/en/crawl-vectors.html
    ft = fasttext.load_model('cc.en.300.bin')
    fasttext.util.reduce_model(ft, 256)
    ft.get_vocab()

    ft.train(sentences, total_examples=ft.corpus_count, epochs=10)
    ft.save_model('reduced_model.bin')

    

    path_to_dir = "./finetuned_embeddings"
    print(f"Retraining pretrained fasttext model")
    model_path = "./pretrained_embeddings/cc.en.300.bin"
    model = load_facebook_model(model_path)
    print(f"Pretrained word vectors: {model.wv}") #FastTextKeyedVectors<vector_size=300, 2000000 keys>
    model.build_vocab(sentences, update=True)
    training_examples_count = model.corpus_count
    print(f"Transfer Learning Training Examples Count = {training_examples_count}") # 240694
    model.train(sentences, total_examples=training_examples_count, epochs=model.epochs)
    print(f"Retrained word vectors: {model.wv}")  # FastTextKeyedVectors<vector_size=300, 2000281 keys> vocab increased by 281 
    model.save(f'{path_to_dir}/retrained-sentencepiece-fasttext.model')


if __name__ == "__main__":
    fasttext.util.download_model('en', if_exists='ignore')  # English    # Download pretrained english fasttext model here: https://fasttext.cc/docs/en/crawl-vectors.html
    ft = fasttext.load_model('cc.en.300.bin')
    fasttext.util.reduce_model(ft, 256)
    # ft.save_model('reduced_model.bin')
    ft.get_vocab()
    # train_fasttext()
    print(tokenise("Hello, how are you?"))
    # train_fasttext(custom=False)