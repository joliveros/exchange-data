#!/usr/bin/env python


from dateutil import tz
from exchange_data import settings, Database, Measurement
from exchange_data.emitters import Messenger, SignalInterceptor
from exchange_data.utils import DateTimeUtils
from functools import lru_cache

import alog
import click
import json
import sys
import traceback
import numpy as np



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
            database_batch_size=40,
            **kwargs
        )

        if depths is None:
            depths = [21]

        self.depths = depths
        self.last_frames = {}
        self.save_data = save_data
        self.slices = {}
        self.queued_frames = []
        self.last_timestamp = DateTimeUtils.now()

        self.freq = settings.TICK_INTERVAL

        if self.subscriptions_enabled:
            self.on('depth', self.message)
            self.on('1m', self.queue_frame('1m'))
            self.on('1m', self.save_frame)

    def frame_channel(self, symbol):
        return f'{symbol}_OrderBookFrame'

    def message(self, frame):
        symbol = frame['symbol']
        self.last_frames[symbol] = frame['depth']

    def start(self, channels=[]):
        self.sub([
            '1m',
            'depth',
        ] + channels)

    def exit(self, *args):
        self.stop()
        sys.exit(0)

    def measurements(self, symbol):
        frame = self.last_frames[symbol]
        if frame is None:
            return

        measurements = []

        for depth in self.depths:
            if depth > 0:
                frame = np.asarray(frame)
                frame_slice = frame[:, :depth, :].tolist()
            else:
                frame_slice = frame

            frame_slice = np.asarray(frame_slice, dtype=np.float32)
            frame_slice = frame_slice.tolist()

            measurement = self.slice(symbol, depth, frame_slice)

            self.slices[self.channel_for_depth(symbol, depth)] = measurement

            measurements.append(measurement)

        return [m.__dict__ for m in measurements]

    def slice(self, symbol, depth, frame_slice):
        fields = dict(
            best_ask=frame_slice[0][0][0],
            best_bid=frame_slice[1][0][0],
            data=json.dumps(frame_slice)
        )

        return Measurement(measurement=self.channel_for_depth(symbol, depth),
                           tags={'symbol': symbol},
                           time=self.last_timestamp, fields=fields)

    def queue_frame(self, frame_key):
        alog.info('## queue frames ##')
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
        self.last_timestamp = DateTimeUtils.parse_timestamp(timestamp,
                                                            tz.tzutc())
        while len(self.queued_frames) > 0:
            frame_key = self.queued_frames.pop()

            for symbol in self.last_frames.keys():
                self.emit_frames(timestamp, symbol, frame_key)

        measurements = []
        for symbol in self.last_frames.keys():
            measurements += self.save_frame_for_symbol(symbol)

        if self.save_data and len(measurements) > 0:
            self.write_points(measurements, time_precision='ms')

    def save_frame_for_symbol(self, symbol):
        return self.measurements(symbol)

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
