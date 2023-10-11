#!/usr/bin/env python

import math
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL
from cached_property import cached_property_with_ttl
from decimal import Decimal, getcontext
from exchange_data import settings
from exchange_data.emitters import binance
from exchange_data.emitters.binance import truncate
from exchange_data.emitters.binance.trade_executor_base import TradeExecutorBase
from exchange_data.trading import Positions
from math import floor
from pytimeparse.timeparse import timeparse
from redis import Redis
from redis_cache import RedisCache
import alog
import binance
import click
import signal
import sys

cache = RedisCache(redis_client=Redis(host=settings.REDIS_HOST))


class TradeExecutor(TradeExecutorBase):
    def __int__(self, **kwargs):
        super(**kwargs)


@click.command()
@click.option("--base-asset", "-b", default="BNB", type=str)
@click.option("--futures", "-F", is_flag=True)
@click.option("--leverage", default=2, type=int)
@click.option("--quantity", "-q", default=0.0, type=float)
@click.option("--log-requests", "-l", is_flag=True)
@click.option("--trading-enabled", "-e", is_flag=True)
@click.option("--trade-min", is_flag=True)
@click.option("--once", is_flag=True)
@click.argument("symbol", type=str)
def main(**kwargs):
    emitter = TradeExecutor(**kwargs)
    emitter.start()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda: exit(0))
    signal.signal(signal.SIGTERM, lambda: exit(0))
    main()
