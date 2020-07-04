#!/usr/bin/env python

from collections import deque
from datetime import timedelta

from dateutil import tz
from pytimeparse.timeparse import timeparse

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
        self,
        depths=None,
        database_name='binance',
        reset_orderbook: bool = True,
        save_data: bool = True,
        subscriptions_enabled: bool = True,
        **kwargs
    ):
        self.subscriptions_enabled = subscriptions_enabled
        self.should_reset_orderbook = reset_orderbook

        super().__init__(
            database_name=database_name,
            database_batch_size=10,
            **kwargs
        )

        if depths is None:
            depths = [21]

        self.depths = depths
        self.last_frames = {}
        self.save_data = save_data
        self.slices = {}
        self.queued_frames = []

        self.freq = settings.TICK_INTERVAL

        if self.subscriptions_enabled:
            self.on('depth', self.message)
            # self.on(TimeChannels.Tick.value, self.emit_ticker)
            self.on('30s', self.queue_frame('30s'))
            self.on('30s', self.save_frame)

    def frame_channel(self, symbol):
        return f'{symbol}_OrderBookFrame'

    def message(self, frame):
        symbol = frame['symbol']
        self.last_frames[symbol] = np.asarray(frame['depth'])

    def emit_ticker(self, timestamp):
        for symbol in self.last_frames.keys():
            self.publish(f'{symbol}_ticker', json.dumps({
                'best_bid': self.last_frames[symbol][1][0][0],
                'best_ask': self.last_frames[symbol][0][0][0]
            }))

    def start(self, channels=[]):
        self.sub([
            'depth',
            '2s',
            '30s',
        ] + channels)

    def exit(self, *args):
        self.stop()
        sys.exit(0)

    def measurements(self, timestamp, symbol):
        frame = self.last_frames[symbol]
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

            measurement = self.slice(symbol, depth, frame_slice, timestamp)

            self.slices[self.channel_for_depth(symbol, depth)] = measurement

            measurements.append(measurement)

        return [m.__dict__ for m in measurements]

    def slice(self, symbol, depth, frame_slice, timestamp):
        fields = dict(
            best_ask=frame_slice[0][0][0],
            best_bid=frame_slice[1][0][0],
            data=json.dumps(frame_slice.tolist())
        )

        return Measurement(measurement=self.channel_for_depth(symbol, depth),
                           tags={'symbol': symbol},
                           time=timestamp, fields=fields)

    def queue_frame(self, frame_key):
        this = self

        def _queue_frame(*args):
            this.queued_frames.append(frame_key)

        return _queue_frame

    def emit_frames(self, timestamp, symbol, frame_key):
        for depth in self.depths:
            frame_slice = self.slices.get(self.channel_for_depth(symbol, depth))
            if frame_slice is not None:
                if frame_key == 'tick':
                    channel = self.channel_for_depth(symbol, depth)
                else:
                    channel = f'{self.channel_for_depth(symbol, depth)}' \
                              f'_{frame_key}'

                msg = channel, str(frame_slice)

                self.publish(*msg)

    def save_frame(self, timestamp):
        while len(self.queued_frames) > 0:
            frame_key = self.queued_frames.pop()

            for symbol in self.last_frames.keys():
                self.emit_frames(timestamp, symbol, frame_key)

        for symbol in self.last_frames.keys():
            self.save_frame_for_symbol(symbol, timestamp)

    def save_frame_for_symbol(self, symbol, timestamp):
        self.last_timestamp = DateTimeUtils.parse_timestamp(timestamp,
                                                            tz.tzutc())
        measurements = self.measurements(timestamp, symbol)


        if self.save_data and measurements:
            self.write_points(measurements, time_precision='ms')

    @lru_cache()
    def channel_for_depth(self, symbol, depth):
        if depth == 0:
            return self.frame_channel(symbol)

        return f'{self.frame_channel(symbol)}_depth_{depth}'


@click.command()
@click.option('--save-data/--no-save-data', default=True)
@click.option('--reset-orderbook/--no-reset-orderbook', default=True)
@click.option('--depth', '-d', type=int, default=0)
def main(depth, **kwargs):
    recorder = BinanceOrderBookEmitter(
        depths=[depth],
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
