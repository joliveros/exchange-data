#!/usr/bin/env python
from PIL import Image as im
from datasets import Dataset
from exchange_data.data.orderbook_change_frame import OrderBookChangeFrame

# from exchange_data.data import OrderBookFrame
from pathlib import Path
from scipy.signal import argrelextrema

import alog
import click
import numpy as np
import pandas as pd


def orderbook_dataset(
    save=False,
    split=True,
    shuffle=True,
    labeled=True,
    show=False,
    **kwargs
):
    ob_frame = OrderBookChangeFrame(show=show, **kwargs)
    df = ob_frame.frame

    if labeled:
        best_bid = df["best_bid"].to_numpy()
        best_ask = df["best_ask"].to_numpy()
        n = 5

        min_ix = argrelextrema(best_bid, np.less_equal, order=n)[0]
        max_ix = argrelextrema(best_bid, np.greater_equal, order=n)[0]
        position = np.zeros(best_bid.shape)

        capital = 1
        price_in = 0.0
        ix_in = None
        ix_out = None

        for ix in range(0, df.shape[0]):
            if ix in max_ix:
                ix_in = ix
                price_in = best_bid[ix]
                continue

            if ix in min_ix:
                if price_in > 0:
                    ix_out = ix
                    pnl = (price_in - best_ask[ix]) / price_in
                    capital = capital + (capital * pnl * (1 - 0.005))

                    if pnl > 0.009:
                        alog.info((price_in, best_ask[ix], pnl))
                        position[ix_in:ix_out] = 1

                price_in = 0.0

        df["labels"] = position
        df["labels"] = df["labels"].astype("int")

        short_df = pd.DataFrame(df[df["labels"] == 1])
        flat_df = pd.DataFrame(df[df["labels"] == 0])

        flat_len = flat_df.shape[0]
        short_len = short_df.shape[0]

        alog.info((flat_len, short_len))

        # if flat_len > short_len:
        #     short_df = short_df.sample(flat_len, replace=True)
        # else:
        #     flat_df = flat_df.sample(short_len, replace=True)

        short_fraction = int(flat_len * 0.85)

        short_df = short_df.sample(short_fraction, replace=True)

        balanced_df = pd.concat([short_df, flat_df])

        df = balanced_df

        alog.info(df)

    df.dropna(inplace=True)

    df["orderbook_img"] = df["orderbook_img"].apply(lambda x: x.flatten())

    dataset = Dataset.from_pandas(df)

    if shuffle:
        dataset = dataset.shuffle()

    if split:
        dataset = dataset.train_test_split(test_size=0.1)

    alog.info(dataset)

    def transforms(examples):
        examples["pixel_values"] = [
            im.fromarray(
                np.array(image).reshape(
                    (ob_frame.frame_width, ob_frame.frame_width, 3)
                ),
                "RGB",
            )
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
@click.option("--frame-width", "-F", default=224, type=int)
@click.option("--group-by", "-g", default="30s", type=str)
@click.option("--additional-group-by", "-G", default="10Min", type=str)
@click.option("--interval", "-i", default="10m", type=str)
@click.option("--offset-interval", "-o", default="3h", type=str)
@click.option("--plot", "-p", is_flag=True)
@click.option("--sequence-length", "-l", default=48, type=int)
@click.option("--round-decimals", "-D", default=4, type=int)
@click.option("--tick", is_flag=True)
@click.option("--show", is_flag=True)
@click.option("--cache", is_flag=True)
@click.option("--save", is_flag=True)
@click.option("--split", is_flag=True)
@click.option("--max-volume-quantile", "-m", default=0.99, type=float)
@click.option("--window-size", "-w", default="3m", type=str)
@click.argument("symbol", type=str)
def main(**kwargs):
    ds = orderbook_dataset(**kwargs)
    alog.info(ds["train"][-1])


if __name__ == "__main__":
    main()
