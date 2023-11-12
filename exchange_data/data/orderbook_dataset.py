#!/usr/bin/env python
from pathlib import Path

from exchange_data.data import OrderBookFrame
from exchange_data.data.orderbook_change_frame import OrderBookChangeFrame

from PIL import Image as im
from datasets import Dataset
import alog
import click
import numpy as np
import pandas as pd


def orderbook_dataset(save=False, split=True, **kwargs):
    ob_frame = OrderBookFrame(frame_width=224, **kwargs)
    df = ob_frame.frame
    best_bid = df["best_bid"]
    length = best_bid.shape[0]
    df["pct_change"] = 0
    pct_change = df["pct_change"].copy()

    max_ix = length - 1
    for ix in range(0, length):
        next_ix = ix + 1
        if next_ix <= max_ix:
            pct_change.iloc[ix] = (best_bid.iloc[ix] / best_bid.iloc[next_ix]) - 1
        else:
            pct_change.iloc[ix] = 0

    df["pct_change"] = pct_change
    df["labels"] = 0

    df.loc[(df["pct_change"] <= -0.003), "labels"] = 1

    short_df = pd.DataFrame(df[df["labels"] == 1])

    alog.info((df.shape[0], short_df.shape[0]))

    flat_df = df[df["labels"] == 0]

    num_flat = flat_df.shape[0]

    short_df = short_df.sample(num_flat, replace=True)

    balanced_df = pd.concat([short_df, flat_df])

    df = balanced_df

    df["orderbook_img"] = df["orderbook_img"].apply(lambda x: x.flatten())

    dataset = Dataset.from_pandas(df)

    if split:
        dataset = dataset.train_test_split(test_size=0.2)

    alog.info(dataset)

    def transforms(examples):
        examples["pixel_values"] = [
            im.fromarray(np.array(image).reshape((224, 224, 3)), "RGB")
            for image in examples["orderbook_img"]
        ]
        return examples

    dataset = dataset.map(
        transforms, remove_columns=["orderbook_img"], batched=True, batch_size=2
    )

    if save:
        dataset.save_to_disk(Path.home() / ".exchange-data/orderbook")

    return dataset


@click.command()
@click.option("--database_name", "-d", default="binance", type=str)
@click.option("--depth", default=72, type=int)
@click.option("--group-by", "-g", default="30s", type=str)
@click.option("--interval", "-i", default="10m", type=str)
@click.option("--offset-interval", "-o", default="3h", type=str)
@click.option("--plot", "-p", is_flag=True)
@click.option("--sequence-length", "-l", default=48, type=int)
@click.option("--round-decimals", "-D", default=4, type=int)
@click.option("--tick", is_flag=True)
@click.option("--cache", is_flag=True)
@click.option("--save", is_flag=True)
@click.option("--split", is_flag=True)
@click.option("--max-volume-quantile", "-m", default=0.99, type=float)
@click.option("--window-size", "-w", default="3m", type=str)
@click.argument("symbol", type=str)
def main(**kwargs):
    orderbook_dataset(**kwargs)


if __name__ == "__main__":
    main()
