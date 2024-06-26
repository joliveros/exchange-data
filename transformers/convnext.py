#!/usr/bin/env python

from datasets import load_metric
import datasets
from pathlib import Path
from transformers import (
    TrainingArguments,
    Trainer,
    ConvNextForImageClassification,
    ConvNextImageProcessor,
)

import alog
import click
import numpy as np
import torch


model_name_or_path = "facebook/convnext-base-224-22k"

metric = load_metric("accuracy")
processor = ConvNextImageProcessor.from_pretrained(model_name_or_path)


def transform(example_batch):
    inputs = processor([x for x in example_batch["pixel_values"]], return_tensors="pt")

    inputs["labels"] = example_batch["labels"]

    return inputs


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


@click.command()
def main(**kwargs):
    ds = datasets.load_from_disk(str(Path.home() / ".exchange-data/orderbook"))

    prepared_ds = ds.with_transform(transform)

    model = ConvNextForImageClassification.from_pretrained(model_name_or_path)

    training_args = TrainingArguments(
        output_dir="./vit_output",
        per_device_train_batch_size=64,
        evaluation_strategy="steps",
        num_train_epochs=10000,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-7,
        torch_compile=True,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["test"],
        tokenizer=processor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(prepared_ds["test"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
