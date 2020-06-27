#!/usr/bin/env python

from collections import deque

from dateutil import tz

from exchange_data import settings, Database, Measurement
from exchange_data.bitmex_orderbook import BitmexOrderBook
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger, TimeChannels, SignalInterceptor
from exchange_data.emitters.bitmex import BitmexEmitterBase
from exchange_data.emitters.bitmex._orderbook_l2_emitter import OrderBookL2Emitter
from exchange_data.orderbook import OrderBook
from exchange_data.orderbook._ordertree import OrderTree
from exchange_data.utils import NoValue, DateTimeUtils, EventEmitterBase
from functools import lru_cache
from numpy.core.multiarray import ndarray

import alog
import click
import gc
import json
import numpy as np
import sys
import traceback


class BitmexOrderBookChannels(NoValue):
    OrderBookFrame = 'OrderBookFrame'


class BinanceOrderBookChannels(
    Messenger,
    OrderBook,
    Database,
    SignalInterceptor,
    DateTimeUtils,
):
    def __init__(
        self, symbol: BitmexChannels,
        depths=None,
        emit_interval=None,
        database_name='bitmex',
        reset_orderbook: bool = True,
        save_data: bool = True,
        subscriptions_enabled: bool = True,
        **kwargs
    ):
        self.symbol = symbol
        self.subscriptions_enabled = subscriptions_enabled
        self.should_reset_orderbook = reset_orderbook

        super().__init__(
            symbol=symbol,
            database_name=database_name,
            database_batch_size=10,
            **kwargs
        )

        if emit_interval is None:
            self.emit_interval = '5s'
        else:
            self.emit_interval = emit_interval

        if depths is None:
            depths = [21]

        self.depths = depths
        self.save_data = save_data
        self.slices = {}
        self.frame_slice = None
        self.queued_frames = []

        self.orderbook_l2_channel = OrderBookL2Emitter\
            .generate_channel_name('1m', self.symbol)

        self.freq = settings.TICK_INTERVAL
        self.frame_channel = f'{self.symbol.value}_' \
            f'{BitmexOrderBookChannels.OrderBookFrame.value}'

        if self.subscriptions_enabled:
            self.on(TimeChannels.Tick.value, self.save_frame)
            self.on(TimeChannels.Tick.value, self.queue_frame('tick'))
            self.on(TimeChannels.Tick.value, self.emit_ticker)
            self.on('5s', self.queue_frame('5s'))
            self.on('2s', self.queue_frame('2s'))
            self.on(self.symbol.value, self.message)

        if self.should_reset_orderbook:
            self.on(self.orderbook_l2_channel, self.process_orderbook_l2)

    def print_stats(self):
        alog.info(self.print(depth=4))
        alog.info(self.dataset.dims)

    def emit_ticker(self, timestamp):
        self.publish('ticker', json.dumps({
            'best_bid': self.bids.max_price(),
            'best_ask': self.asks.min_price()
        }))

    def process_orderbook_l2(self, data):

        self.reset_orderbook()

        self.message({
            'table': 'orderBookL2',
            'data': data,
            'action': 'partial',
            'symbol': self.symbol.value
        })

    def reset_orderbook(self):
        alog.info('### reset orderbook ###')
        del self.__dict__['tape']
        del self.__dict__['bids']
        del self.__dict__['asks']

        self.tape = deque(maxlen=10)
        self.bids = OrderTree()
        self.asks = OrderTree()

    def garbage_collect(self):
        gc.collect()

    def start(self, channels=[]):
        self.sub([
            '2s',
            '5s',
            self.orderbook_l2_channel,
            self.symbol,
            TimeChannels.Tick,
        ] + channels)

    def exit(self, *args):
        self.stop()
        sys.exit(0)

    def gen_bid_side(self):
        bid_levels = list(self.bids.price_tree.items())
        price = np.array(
            [level[0] for level in bid_levels]
        )

        volume = np.array(
            [level[1].volume for level in bid_levels]
        ) * -1

        return np.vstack((price, volume))

    def gen_ask_side(self):
        bid_levels = list(self.asks.price_tree.items())
        price = np.array(
            [level[0] for level in bid_levels]
        )

        volume = np.array(
            [level[1].volume for level in bid_levels]
        )

        return np.vstack((price, volume))

    def generate_frame(self) -> ndarray:
        bid_side = np.flip(self.gen_bid_side(), axis=1)
        ask_side = self.gen_ask_side()

        bid_side_shape = bid_side.shape
        ask_side_shape = ask_side.shape

        if bid_side_shape[-1] > ask_side_shape[-1]:
            new_ask_side = np.zeros(bid_side_shape)
            new_ask_side[:ask_side_shape[0], :ask_side_shape[1]] = ask_side
            ask_side = new_ask_side
        elif ask_side_shape[-1] > bid_side_shape[-1]:
            new_bid_side = np.zeros(ask_side_shape)
            new_bid_side[:bid_side_shape[0], :bid_side_shape[1]] = bid_side
            bid_side = new_bid_side

        frame = np.array((ask_side, bid_side))

        return frame

    def measurements(self, timestamp):
        frame = self.generate_frame()

        measurements = []

        if isinstance(timestamp, str):
            timestamp = self.parse_timestamp(timestamp)
        elif isinstance(timestamp, float):
            timestamp = self.parse_timestamp(timestamp)

        for depth in self.depths:
            if depth > 0:
                frame_slice = frame[:, :, :depth]
            else:
                frame_slice = frame

            self.frame_slice = frame_slice

            measurement = self.slice(depth, frame_slice, timestamp)

            self.slices[self.channel_for_depth(depth)] = measurement

            measurements.append(measurement)

        return [m.__dict__ for m in measurements]

    def slice(self, depth, frame_slice, timestamp):
        fields = dict(
            data=json.dumps(frame_slice.tolist()),
            best_bid=self.get_best_bid(),
            best_ask=self.get_best_ask()
        )

        return Measurement(measurement=self.channel_for_depth(depth),
                           tags={'symbol': self.symbol.value},
                           time=timestamp, fields=fields)

    def queue_frame(self, frame_key):
        this = self

        def _queue_frame(*args):
            this.queued_frames.append(frame_key)

        return _queue_frame

    def emit_frames(self, timestamp, frame_key):
        for depth in self.depths:
            frame_slice = self.slices.get(self.channel_for_depth(depth))
            if frame_slice is not None:
                if frame_key == 'tick':
                    channel = self.channel_for_depth(depth)
                else:
                    channel = f'{self.channel_for_depth(depth)}_{frame_key}'
                msg = channel, str(frame_slice)

                self.publish(*msg)

    def save_frame(self, timestamp):
        self.last_timestamp = DateTimeUtils.parse_timestamp(timestamp,
                                                            tz.tzutc())

        measurements = self.measurements(timestamp)

        while len(self.queued_frames) > 0:
            frame_key = self.queued_frames.pop()
            self.emit_frames(timestamp, frame_key)

        if self.save_data:
            self.write_points(measurements, time_precision='ms')

    @lru_cache()
    def channel_for_depth(self, depth):
        if depth == 0:
            return self.frame_channel

        return f'{self.frame_channel}_depth_{depth}'


@click.command()
@click.argument('symbol', type=str)
@click.option('--save-data/--no-save-data', default=True)
@click.option('--reset-orderbook/--no-reset-orderbook', default=True)
def main(**kwargs):
    recorder = BinanceOrderBookChannels(
        depths=[0, 21, 40],
        subscriptions_enabled=True,
        **kwargs
    )

    try:
        recorder.start()
    except Exception as e:
        traceback.print_exc()
        recorder.stop()
        sys.exit(-1)


if __name__ == '__main__':
    main()
