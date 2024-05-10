import torch
from adapters import AutoAdapterModel, AdapterTrainer
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate


if __name__ == "__main__":

    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    model = AutoAdapterModel.from_pretrained("bert-base-cased")
    model.add_adapter("rotten_tomatoes", config="seq_bn")

    model.add_classification_head(
        "rotten_tomatoes",
        num_labels=5
    )

    model.train_adapter("rotten_tomatoes")

    training_args = TrainingArguments(output_dir="test_trainer", logging_steps=30, report_to=None)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    print('start training')
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()