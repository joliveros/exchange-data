#!/usr/bin/env python

import alog
import click
from datasets import Dataset

from exchange_data.data.orderbook_change_frame import OrderBookChangeFrame


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
    obFrame = OrderBookChangeFrame(**kwargs)
    df = obFrame.frame

    tds = Dataset.from_pandas(df)

    alog.info(tds)


if __name__ == "__main__":
    main()
