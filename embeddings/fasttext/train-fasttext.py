from itertools import chain
import datasets
from tqdm import tqdm
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model

import sentencepiece as spm

LANG = 'en' # 'zh' for Chinese, 'en' for English
VOCAB_SIZE = 16384

dataset = datasets.load_dataset("iwslt2017", "iwslt2017-en-zh")
train, test, validation = dataset['train'], dataset['test'], dataset['validation']
tokeniser = spm.SentencePieceProcessor(model_file="../../tokenisation/sentencepiece_custom/en.model")    

def tokenise(sent):
    return tokeniser.EncodeAsPieces(sent)

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __iter__(self):
        get_sent = lambda x: x[LANG]
        for line in tqdm(chain.from_iterable([
            map(get_sent, train['translation']),
            map(get_sent, test['translation']),
            map(get_sent, validation['translation'])
        ])):
            # yield list of tokens (list[str])
            yield tokenise(line)

def train_fasttext(lang=LANG, from_scratch=True):
    """Trains fasttext embeddings
    - transfer learning with pretrained embeddings
    """
    sentences = MyCorpus()
    path_to_dir = "./finetuned_embeddings"
    if from_scratch:
        print("Training fasttext embeddings from scratch")
        sentences = MyCorpus()
        model = FastText(sentences, vector_size=256, workers=4, sorted_vocab=1, min_count=1, max_final_vocab=VOCAB_SIZE)
        model.build_vocab(corpus_iterable=sentences)
        model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=model.epochs)
        print(f"Training Examples Count = {model.corpus_count}") # 240694
        print(f"Trained word vectors: {model.wv}") 
        # zh: <27461 with min_count = 2> <13006 with min_count = 3> <8593 with min_count = 4> <6571 with min_count = 5> 
        # en: <16142 with min_count = 1> <15972 with min_count = 2> 
        # en use min count = 1, zh use min count = 3
        model.save(f'{path_to_dir}/scratch-sp-{lang}-fasttext.model')
    else:
        model_path = f"./cc.{lang}.256.bin" # download the 300 model then reduce dimension to 256 using fasttext library 
        model = load_facebook_model(model_path)
        print(f"Pretrained word vectors: {model.wv}") # FastTextKeyedVectors<vector_size=256, 2000000 keys>

        model.build_vocab(corpus_iterable=sentences, update=True)
        training_examples_count = model.corpus_count
        print(f"Transfer Learning Training Examples Count = {training_examples_count}") # 240694

        model.train(corpus_iterable=sentences, total_examples=training_examples_count, epochs=model.epochs)

        print(f"Retrained word vectors: {model.wv}")  # FastTextKeyedVectors<vector_size=256, 2012529 keys> vocab increased by 12529 
        model.save(f'{path_to_dir}/finetuned-sp-{lang}-fasttext.model')
    return


if __name__ == "__main__":
    train_fasttext(lang=LANG, from_scratch = True)
