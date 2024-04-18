# Investigation into the Impact of Tokenization, Model Architecture and Word Embeddings on Neural Machine Translation

Team 42's Submission for CS4248, taken in AY23/24 S2 at the National University of Singapore.

This repository hosts our experiments on Neural Machine Translation approaches, using the [iwslt-2017 en-zh](https://huggingface.co/datasets/iwslt2017/viewer/iwslt2017-en-zh) dataset.

## Introduction

Given the large task of Neural Machine Translation, we hope to explore various approaches for different parts of the translation process, namely:

- Tokenization
- Embedding Representations
- Model Architecture

## Tokenisation

For Tokenisation, we evaluated three Tokenisation approaches, namely:

1. Word Based

    The notebook can be found [here](models/rnn/baseline-rnn-config-1.ipynb).

    -  English: spacy `en_core_web_lg`
    - Chinese: Stanza 


2. SentencePiece BPE

    The notebook can be found [here](models/rnn/baseline-rnn-config-2.ipynb).

    We use the SentencePiece BPE algorithm to train two SentencePiece BPE models, 
    each with a vocabulary size of 16,384 and one for each language.

3. BERT WordPiece 
  
    The notebook can be found [here](models/rnn/baseline-rnn-config-3.ipynb).

    Using the HuggingFace Tokeniser Finetuning [API](models/rnn/baseline-rnn-config-2.ipynb), we train two tokenisers
    on our training dataset, from the configurations listed below:

    - English: `bert_base_uncased`: This outputs a vocabulary size of 32,000, from the original 30,522.
    - Chinese: `bert_base_chinese`: This outputs a vocabulary size of 32,000, from the original 21,128.

    The notebook for training the tokeniser can be found [here](tokenisation/train_tokenizer_on_dataset.ipynb)

We standardise training at 10,000 rows of the train dataset, while all tokenisers are trained on the full training dataset.


## Members

- Seeu Sim Ong [@SeeuSim](https://github.com/SeeuSim)
- Jean Ong Hui Jia [@jeanong2](https://github.com/jeanong2)
- Cayden Chen Ningjia  [@caydencnj](https://github.com/caydencnj)
- Lian Kok Hai [@unfazing](https://github.com/unfazing)
- Glen Lim Fu Yong [@glyfy](https://github.com/glyfy)

