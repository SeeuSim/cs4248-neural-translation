import datasets
import numpy as np
import torch
from torch.utils.data import RandomSampler
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Polyfill
np.object = object

IS_KAGGLE = False

IS_CUSTOM_TOK = True

LANG = "en"

if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    import wandb

    user_secrets = UserSecretsClient()
    _WANDB_API_KEY = user_secrets.get_secret("wandb_sec")
    wandb.login(key=_WANDB_API_KEY)

    # upload the dataset to Kaggle first
    dataset = datasets.load_from_disk("/kaggle/input/iwslt-en-zh/ds/")
else:
    dataset = datasets.load_dataset("iwslt2017", "iwslt2017-en-zh")

if IS_CUSTOM_TOK:
    from sentencepiece_custom.tokeniser import BPEforBERTTokenizer

    # Add own class to output vocab indices, decode and attention mask
    tokenizer = BPEforBERTTokenizer(zh_model_file=".\\embeddings\\bert-embed\\sentencepiece_custom\\zh.model", en_model_file=".\\embeddings\\bert-embed\\sentencepiece_custom\\en.model")
else:
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

"""
Training Device
"""
dev = "CPU"
device = torch.device("cpu")
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    dev = "MPS"
    device = torch.device("mps")
elif torch.cuda.is_available() and torch.cuda.device_count():
    dev = "CUDA"
    device = torch.device("cuda")
torch.set_default_device(device)


"""
Dataset
"""
train = datasets.concatenate_datasets([dataset["train"], dataset["test"]]) # combine train and test dataset
test = dataset["validation"]

"""
Preprocess and tokenise
"""

def get_row_data(batch):
    if IS_CUSTOM_TOK:
      output = {'token_type_ids': [], 'attention_mask': [], 'input_ids': [], 'special_tokens_mask' : []}
      for sent in list(map(lambda r: r[LANG], batch["translation"])):
        tokenised = tokenizer(sent, lang=LANG, max_len=288)
        for key, value in tokenised.items():
            output[key].append(value)
      return output
      
    else:
      return tokenizer(
        list(map(lambda r: r["en"], batch["translation"])),
        return_special_tokens_mask=True,
        )

train_dataset = train.map(get_row_data, batched=True)
train_dataset.set_format("torch")
test_dataset = test.map(get_row_data, batched=True)
test_dataset.set_format("torch")

"""
Model Config
"""

vocab_size = 16_384
output_path = "./models"

config = BertConfig(
    vocab_size=vocab_size,
    max_position_embeddings=288,  # or 512 (sentence length for attn mask)
    hidden_size=252, # need to be a multiple of 12 (num of attention heads in BERT)
    # Add or modify other config parameters as needed
)

model = BertForMaskedLM(config)

# Important: Tokenizer impls __len__ for output vocab size
model.resize_token_embeddings(len(tokenizer))

"""
Training Utilities
"""

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Disable Masked Language Modeling
)

training_args = TrainingArguments(
    output_dir=output_path,
    evaluation_strategy="steps",
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=64,
    logging_steps=1000,
    save_steps=1000,
    load_best_model_at_end=True,
    save_total_limit=3,
    use_cpu=dev == "CPU",
    dataloader_pin_memory=False,
)

# Step 6: Create the trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer._get_train_sampler = lambda: RandomSampler(
    trainer.train_dataset, generator=torch.Generator(device)
)

"""
Actual Training
"""
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(output_path)
