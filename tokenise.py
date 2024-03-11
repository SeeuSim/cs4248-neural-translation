import re
from string import punctuation

from nltk.tokenize import word_tokenize

punc_pat = r"[" + re.escape(punctuation) + r']'

def tokenise(sent):
    return list(
        filter(
            lambda tok: len(tok) > 0,
            map(
                lambda tok: re.sub(punc_pat, '', tok),
                word_tokenize(sent)
            )
        )
    )

if __name__ == "__main__":
    print(tokenise(''))
