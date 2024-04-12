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

"""
Variables to configure.
"""
DATASET_DISK_PATH = None  # Save the dataset using the Huggingface Datasets API save_to_disk, then load the file path.
CUSTOM_TOKENISER_CLASS = None  # Import the class, then reference it here.

VOCAB_SIZE = 16_384
TRAINER_OUTPUT_PATH = "./models"

"""
Dataset
"""
if DATASET_DISK_PATH is not None:
    dataset = datasets.load_from_disk(DATASET_DISK_PATH)
else:
    dataset = datasets.load_dataset("iwslt2017", "iwslt2017-en-zh")

train, test = dataset["train"], dataset["test"]


if CUSTOM_TOKENISER_CLASS is not None:
    """
    This tokeniser will have to implement:
        - __call__(self, text: str | list[str], return_special_tokens_mask=False)
            Given a sentence or a list of sentences, performs these steps for each sentence:
                1. Tokenises it
                2. Maps each token to a numeric ID within its learned vocabulary
                3. Generates a dict of 4 equal length int arrays:
                    - input_ids: The token IDs
                    - attention_mask: An array of 1s for non-padding tokens, 0 for padding tokens
                    - token_type_ids: 0s. We do not use this for the multi-context BERT task.
                    - special_tokens_mask: 1s for BOS/EOS/PAD/UNK, 0 otherwise.
        - __len__(self)
            Returns the size of its learned vocabulary. Do add at least 4 for the 4 tokens: UNK, SOS, EOS, PAD.
    """
    # tokeniser = YourCustomTokeniser()
    pass
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
Preprocess and tokenise
"""


def get_row_data(batch):
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
config = BertConfig(
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=288,  # or 512 (sentence length for attn mask)
    hidden_size=256,
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
    output_dir=TRAINER_OUTPUT_PATH,
    evaluation_strategy="steps",  # Eval by steps instead of epoch
    overwrite_output_dir=True,
    num_train_epochs=10,
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

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Polyfill for weird MPS bug
trainer._get_train_sampler = lambda: RandomSampler(
    trainer.train_dataset, generator=torch.Generator(device)
)


"""
Actual Training
"""
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(TRAINER_OUTPUT_PATH)
