# SentencePiece Tokeniser 

This directory hosts the preparatory files for training a custom tokeniser
using the SentencePiece algorithm and library, which uses C for fast training.

The MarianTokeniser can then use the custom .model and vocab files, like so:

```py
from transformers import MarianTokenizer

tokenizer = MarianTokenizer(
  './sp-en.model',
  './sp-zh.model',
  './sp-en.json',
  target_vocab_file='./sp-zh.json',
  source_lang='en',
  target_lang='zh',
  separate_vocabs=True,
)

```

**NOTE**: This is still WIP. Use at own discretion.

