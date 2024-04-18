import datasets
import evaluate
import numpy as np
import torch
from torch.utils.data import RandomSampler
from transformers import (
    DataCollatorForSeq2Seq,
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

output_path = "./trainer"
trainer_output = f"{output_path}/output"

model_output_path = "./marianmt"

metric = evaluate.load("sacrebleu")

dev = "CPU"
device = torch.device("cpu")
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    dev = "MPS"
    device = torch.device("mps")
elif torch.cuda.is_available() and torch.cuda.device_count():
    dev = "CUDA"
    device = torch.device("cuda")
torch.set_default_device(device)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def get_model():
    # Initialize the tokenizer and model (replace 'model_name' with the appropriate model for English to Chinese)
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def train():
    dataset = datasets.load_from_disk("/kaggle/input/iwslt-en-zh/ds/")
    model, tokenizer = get_model()

    def preprocess_function(examples):
        source_lang = "en"
        target_lang = "zh"
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=512, padding=False, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets, max_length=512, padding=False, truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=trainer_output,
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        dataloader_pin_memory=False,
    )

    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer._get_train_sampler = lambda: RandomSampler(
        trainer.train_dataset, generator=torch.Generator(device)
    )

    np.object = object

    trainer.train()
    trainer.save_model(model_output_path)


if __name__ == "__main__":
    train()
