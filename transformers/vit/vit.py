#!/usr/bin/env python

from dataclasses import dataclass
from datasets import load_metric
import datasets
from pathlib import Path
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)

import dataclasses
import alog
import click
import numpy as np
import torch
import json
from exchange_data.data.orderbook_dataset import orderbook_dataset

@dataclass
class Config():
    output_path="./vit_output/pretrained"
    model_name_or_path="google/vit-large-patch16-224"

config = Config()


def train():
    if Path(config.output_path).exists():
        config.model_name_or_path = config.output_path

    alog.info((config.model_name_or_path, config.output_path))

    metric = load_metric("accuracy")
    
    processor = ViTImageProcessor.from_pretrained(config.model_name_or_path)


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



    # ds = datasets.load_from_disk(str(Path.home() / ".exchange-data/orderbook"))
    ds = orderbook_dataset(**dict(
        split=True,
        shuffle=True,
        cache=False,
        database_name='binance_futures',
        depth=72,
        futures=True,
        group_by='1m',
        interval='2d',
        max_volume_quantile=0.99,
        offset_interval='0h',
        plot=False,
        round_decimals=3,
        sequence_length=72,
        symbol='UNFIUSDT',
        window_size='10m',
        additional_group_by='5Min',
        frame_width=299
    ))
    prepared_ds = ds.with_transform(transform)

    model = ViTForImageClassification.from_pretrained(
        config.model_name_or_path, num_labels=2, ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir="./vit_output",
        per_device_train_batch_size=17,
        evaluation_strategy="steps",
        num_train_epochs=12,
        fp16=False,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=8e-8,
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
    trainer.save_model(config.output_path)
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    # model.save_pretrained("./vit_output/pretrained")
    trainer.save_state()

    metrics = trainer.evaluate(prepared_ds["test"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

@click.command()
def main(**kwargs):
    train()

if __name__ == "__main__":
    main()
