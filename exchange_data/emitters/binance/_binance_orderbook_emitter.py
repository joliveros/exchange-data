#!/usr/bin/env python

from collections import deque
from dateutil import tz
from exchange_data import settings, Database, Measurement
from exchange_data.emitters import Messenger, TimeChannels, SignalInterceptor
from exchange_data.emitters.binance._full_orderbook_emitter import FullOrderBookEmitter
from exchange_data.utils import DateTimeUtils
from functools import lru_cache

import alog
import click
import gc
import json
import numpy as np
import sys
import traceback



class BinanceOrderBookEmitter(
    Messenger,
    Database,
    SignalInterceptor,
    DateTimeUtils,
):
    def __init__(
        self, symbol: str,
        depths=None,
        database_name='binance',
        reset_orderbook: bool = True,
        save_data: bool = True,
        subscriptions_enabled: bool = True,
        **kwargs
    ):
        self.subscriptions_enabled = subscriptions_enabled
        self.should_reset_orderbook = reset_orderbook
        self.symbol = symbol

        super().__init__(
            database_name=database_name,
            database_batch_size=10,
            **kwargs
        )

        if depths is None:
            depths = [21]

        self.depths = depths
        self.last_frame = None
        self.save_data = save_data
        self.slices = {}
        self.queued_frames = []

        self.orderbook_l2_channel = FullOrderBookEmitter\
            .generate_channel_name('30s', self.symbol)

        self.freq = settings.TICK_INTERVAL
        self.frame_channel = f'{self.symbol}_OrderBookFrame'

        if self.subscriptions_enabled:
            self.on('2s', self.queue_frame('2s'))
            self.on(self.symbol, self.message)
            self.on(TimeChannels.Tick.value, self.emit_ticker)
            self.on(TimeChannels.Tick.value, self.queue_frame('tick'))
            self.on(TimeChannels.Tick.value, self.save_frame)

    def message(self, frame):
        self.last_frame = np.asarray(frame)

    def emit_ticker(self, timestamp):
        if self.last_frame is not None:
            self.publish('ticker', json.dumps({
                'best_bid': self.last_frame[1][0][0],
                'best_ask': self.last_frame[0][0][0]
            }))

    def start(self, channels=[]):
        self.sub([
            '2s',
            self.symbol,
            TimeChannels.Tick,
        ] + channels)

    def exit(self, *args):
        self.stop()
        sys.exit(0)

    def measurements(self, timestamp):
        frame = self.last_frame
        if frame is None:
            return

        measurements = []

        if isinstance(timestamp, str):
            timestamp = self.parse_timestamp(timestamp)
        elif isinstance(timestamp, float):
            timestamp = self.parse_timestamp(timestamp)

        for depth in self.depths:
            if depth > 0:
                frame_slice = frame[:, :depth, :]
            else:
                frame_slice = frame

            measurement = self.slice(depth, frame_slice, timestamp)

            self.slices[self.channel_for_depth(depth)] = measurement

            measurements.append(measurement)

        return [m.__dict__ for m in measurements]

    def slice(self, depth, frame_slice, timestamp):
        fields = dict(
            best_ask=frame_slice[0][0][0],
            best_bid=frame_slice[1][0][0],
            data=json.dumps(frame_slice.tolist())
        )

        return Measurement(measurement=self.channel_for_depth(depth),
                           tags={'symbol': self.symbol},
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

        if self.save_data and measurements:
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
    recorder = BinanceOrderBookEmitter(
        depths=[0, 40, 80],
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
