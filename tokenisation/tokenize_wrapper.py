from datasets import load_dataset
from transformers import AutoTokenizer

def initialise_tokenizer(pretrained): # Pretrained options: "gpt2", "bert-base-uncased", "t5-base", etc FOR ENGLISH
                                                          # "bert-base-chinese" FOR CHINESE
                                      # Full list of pretrained at https://huggingface.co/transformers/v3.5.1/pretrained_models.html
                                      # FOR PRETRAINED TOKENISER ON OUR DATASET:
                                          # "njcay/gpt2_dataset_tokenizer" for gpt2 tokeniser trained on our ENGLISH dataset
                                          # "njcay/bert_dataset_en_tokenizer" for bert tokeniser trained on our ENGLISH dataset
                                          # "njcay/bert_dataset_zh_tokenizer" for bert tokeniser trained on our CHINESE dataset               
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # can add other token types: bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token, additional_special_tokens
    return tokenizer

def tokenize_wrap(sentence, tokenizer, max_length=512, return_token_type_ids=False): 
    try:
        tokenized = tokenizer.encode_plus(sentence, 
                                          max_length = max_length, # None for padding to max length in training set (1024), but max 512 for BERT
                                          truncation=True,
                                          padding='max_length', 
                                          return_token_type_ids=return_token_type_ids, # set True for BERT
                                          return_attention_mask=True,
                                          return_tensors="pt") # "pt" for PyTorch, "tf" for TensorFlow, or None
        return tokenized
        
    except Exception as e:
        print(f"Error encountered during tokenisation of \"{sentence}\"")
        print(e)
        
        return None 


# Uncomment to test on one sentence in command line
"""
raw_datasets = load_dataset("iwslt2017", "iwslt2017-en-zh") 
example = raw_datasets["train"][123456]["translation"]["zh"] # change language here

tokenizer = initialise_tokenizer(pretrained="njcay/bert_dataset_zh_tokenizer") # initialise pretrained tokeniser
tokens = tokenizer.tokenize(example)
tokenized = tokenize_wrap(example, tokenizer, return_token_type_ids=False) # tokenise

print(f"Example: \n{example}")
print(f"Tokens: \n{tokens}")
print(f"Tokenized: \n{tokenized}")
"""