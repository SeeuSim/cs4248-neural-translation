from datasets import load_dataset
from transformers import AutoTokenizer

def initialise_tokenizer(pretrained="gpt2", add_pad=True): # more pretrained options: "gpt2", "bert-base-uncased", "t5-base"
                                                            # full list of pretrained at https://huggingface.co/transformers/v3.5.1/pretrained_models.html
                                                            # "njcay/gpt2_dataset_tokenizer" for gpt2 tokeniser trained on our dataset
    if pretrained == "bert-base-uncased":
        add_pad=False
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    if add_pad:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
    # can add other token types: bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token, additional_special_tokens
    return tokenizer

def tokenize_en(sentence, tokenizer, max_length=512, return_token_type_ids=False): 
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


# Test on one sentence
raw_datasets = load_dataset("iwslt2017", "iwslt2017-en-zh") 
example = raw_datasets["train"][123456]["translation"]["en"]

tokenizer = initialise_tokenizer(pretrained="bert-base-uncased") # initialise pretrained tokeniser
tokenized = tokenize_en(example, tokenizer, return_token_type_ids=True) # tokenise

print(f"Example: \n{example}")
print(f"Tokenized: \n{tokenized}")