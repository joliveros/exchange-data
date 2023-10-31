#!/usr/bin/env python
from exchange_data.data.orderbook_change_frame import OrderBookChangeFrame

from PIL import Image as im
from datasets import Dataset
import alog
import click
import numpy as np
import pandas as pd


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
@click.option("--max-volume-quantile", "-m", default=0.99, type=float)
@click.option("--window-size", "-w", default="3m", type=str)
@click.argument("symbol", type=str)
def main(**kwargs):
    ob_frame = OrderBookChangeFrame(**kwargs)
    df = ob_frame.frame
    df = df.drop("dtype", axis=1)

    # alog.info(df["orderbook_img"].iloc[-1].shape)

    df["orderbook_img"] = df["orderbook_img"].apply(lambda x: x.flatten())

    alog.info(df.tail())

    dataset = Dataset.from_pandas(df)

    def transforms(examples):
        examples["pixel_values"] = [
            im.fromarray(np.array(image).reshape((229, 229, 3)), "RGB")
            for image in examples["orderbook_img"]
        ]
        return examples

    dataset = dataset.map(
        transforms, remove_columns=["orderbook_img"], batched=True, batch_size=2
    )

    alog.info(dataset)


if __name__ == "__main__":
    main()
