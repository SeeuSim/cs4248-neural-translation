import datasets
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

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

"""
Dataset
"""
dataset = datasets.load_dataset("iwslt2017", "iwslt2017-en-zh")

train, test = dataset["train"], dataset["test"]


def get_row_data(batch):
    return tokenizer(
        list(map(lambda r: r['en'], batch['translation'])), return_special_tokens_mask=True
    )


train_dataset = train.map(get_row_data, batched=True)
train_dataset.set_format("torch")
test_dataset = test.map(get_row_data, batched=True)
test_dataset.set_format("torch")

"""
Model Config
"""

vocab_size = 10_000
output_path = "./models"

config = BertConfig(
    vocab_size=vocab_size,
    max_position_embeddings=300,  # or 512
    # Add or modify other config parameters as needed
)

model = BertForMaskedLM(config)

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
    num_train_epochs=10,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=64,
    logging_steps=1000,
    save_steps=1000,
    load_best_model_at_end=True,
    save_total_limit=3,
    use_cpu=dev == "CPU",
)

# Step 6: Create the trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer._get_train_sampler = \
    lambda: RandomSampler(
        trainer.train_dataset, 
        generator=torch.Generator(device)
    )

"""
Actual Training
"""
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(output_path)
