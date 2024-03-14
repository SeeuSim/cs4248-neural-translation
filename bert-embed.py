from transformers import (
    BertConfig,
    BertModel,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

vocab_size = 10_000
output_path = "./models"

config = BertConfig(
    vocab_size=vocab_size,
    max_position_embeddings=300,  # or 512
    # Add or modify other config parameters as needed
)

model = BertModel(config)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Disable Masked Language Modeling
)

training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
)

# Step 6: Create the trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(output_path)
