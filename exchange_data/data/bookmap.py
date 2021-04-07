#!/usr/bin/env python

from exchange_data.data.orderbook_frame import OrderBookFrame

import alog
import click
import numpy as np
import pandas as pd


class Bookmap(OrderBookFrame):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            **kwargs)




@click.command()
@click.option('--database_name', '-d', default='binance', type=str)
@click.option('--depth', default=72, type=int)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--interval', '-i', default='3h', type=str)
@click.option('--offset-interval', '-o', default='3h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--round-decimals', '-D', default=4, type=int)
@click.option('--tick', is_flag=True)
@click.option('--max-volume-quantile', '-m', default=0.99, type=float)
@click.option('--volatility-intervals', '-v', is_flag=True)
@click.option('--window-size', '-w', default='3m', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    Bookmap(**kwargs)


if __name__ == '__main__':
    main()
