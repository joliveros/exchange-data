#!/usr/bin/env python
from PIL import Image as im
from datasets import Dataset
from exchange_data.data import OrderBookFrame
from pathlib import Path
from scipy.signal import argrelextrema

import alog
import click
import numpy as np
import pandas as pd


def orderbook_dataset(save=False, split=True, **kwargs):
    frame_width = 224
    ob_frame = OrderBookFrame(frame_width=frame_width, **kwargs)
    df = ob_frame.frame
    best_bid = df["best_bid"].to_numpy()
    n = 9

    min_ix = argrelextrema(best_bid, np.less_equal, order=n)[0]
    max_ix = argrelextrema(best_bid, np.greater_equal, order=n)[0]
    position = []
    active_trade = False

    for ix in range(0, df.shape[0]):
        if ix in max_ix:
            active_trade = True
        if ix in min_ix:
            active_trade = False

        if active_trade:
            position.append(1)
        else:
            position.append(0)

    alog.info(df)

    df["labels"] = position

    short_df = pd.DataFrame(df[df["labels"] == 1])

    alog.info((df.shape[0], short_df.shape[0]))

    flat_df = df[df["labels"] == 0]

    num_flat = flat_df.shape[0]

    short_df = short_df.sample(num_flat, replace=True)

    balanced_df = pd.concat([short_df, flat_df])

    df = balanced_df

    df["orderbook_img"] = df["orderbook_img"].apply(lambda x: x.flatten())

    dataset = Dataset.from_pandas(df)

    dataset = dataset.shuffle()

    if split:
        dataset = dataset.train_test_split(test_size=0.2)

    alog.info(dataset)

    def transforms(examples):
        examples["pixel_values"] = [
            im.fromarray(np.array(image).reshape((frame_width, frame_width, 3)), "RGB")
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
@click.option("--additional-group-by", "-G", default="10Min", type=str)
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
