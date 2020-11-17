#!/usr/bin/env python
import gzip
import logging
from collections import deque
from datetime import timedelta
from distutils.util import strtobool

from exchange_data import settings
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger
from exchange_data.streamers._orderbook_level import OrderBookLevelStreamer
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils
from tgym.envs.orderbook.ascii_image import AsciiImage

import alog
import click
import json
import numpy as np
import requests


class TradeJob(object):
    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        self.job_name = f'trade_{self.symbol}'


class PredictionBase(object):
    frames = []

    def __init__(self, model_version=None, **kwargs):
        self.model_version = model_version

    def get_prediction(self):
        frames = np.expand_dims(np.asarray(self.frames, dtype=np.float32),
                                axis=0)

        data = json.dumps(dict(
            signature_name='serving_default',
            instances=frames.tolist()
        ))

        headers = {
            'content-type': 'application/json',
            'Content-Encoding': 'gzip'
        }

        data = gzip.compress(bytes(data, encoding='utf8'))
        url = None

        if self.model_version:
            url = f'http://{settings.MODEL_HOST}:8501/v1/models/{self.symbol}' \
                  f'/versions/{self.model_version}:predict'
        else:
            url = f'http://{settings.MODEL_HOST}:8501/v1/models/' \
                  f'{self.symbol}:predict'

        json_response = requests.post(
            url,
            data=data,
            headers=headers
        )

        data = json.loads(json_response.text)

        predictions = data['predictions']

        max_index = np.argmax(predictions[0])

        position = [
            position for position in Positions
            if position.value == max_index
        ][0]

        # alog.info((position, max_index, predictions[0][max_index]))

        return position


class PredictionEmitter(Messenger, TradeJob):
    def __init__(self, volume_max, database_name, sequence_length, depth,
                 symbol, **kwargs):
        super().__init__(symbol=symbol, **kwargs)

        self.depth = depth
        self.volume_max = volume_max
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.database_name = database_name
        self.orderbook_channel = \
            f'{symbol}_OrderBookFrame_depth_{depth}_2s'
        self.trading_enabled = True
        self.frames = deque(maxlen=sequence_length)

        self.load_previous_frames(depth)

        self.on(self.orderbook_channel, self.emit_prediction)
        self.on('should_trade', self.should_trade)

    def should_trade(self, data):
        alog.info(data)
        alog.info(bool(strtobool(data['should_trade'])))
        self.trading_enabled = strtobool(data['should_trade'])

    def load_previous_frames(self, depth):
        now = DateTimeUtils.now()
        start_date = now - timedelta(seconds=49*2)

        levels = OrderBookLevelStreamer(
            database_name=self.database_name,
            depth=depth,
            end_date=DateTimeUtils.now(),
            groupby='2s',
            sample_interval='48s',
            start_date=start_date,
            symbol=self.symbol,
            window_size='48s'
        )

        for timestamp, best_ask, best_bid, orderbook_img in levels:
            if orderbook_img:
                orderbook_img = np.asarray(json.loads(orderbook_img))
                try:
                    orderbook_img = self.normalize_frame(orderbook_img)
                    self.frames.append(orderbook_img)
                except:
                    pass

    def normalize_frame(self, orderbook_levels):
        orderbook_levels = np.concatenate((orderbook_levels[0],
                                          orderbook_levels[1]))
        orderbook_levels = np.sort(orderbook_levels, axis=0)
        orderbook_levels = np.delete(orderbook_levels, 0, 1)

        orderbook_levels = np.reshape(orderbook_levels,
                                      (orderbook_levels.shape[0], 1)) / self.volume_max
        return np.clip(orderbook_levels, a_min=0.0, a_max=self.volume_max)

    def _emit_prediction(self, data):
        frame = data['fields']['data']
        frame = json.loads(frame)

        self.frames.append(self.normalize_frame(frame))

        position = self.get_prediction()

        self.publish(self.job_name, json.dumps({'data': position.value}))

    def emit_prediction(self, data):
        if self.trading_enabled:
            try:
                self._emit_prediction(data)
            except:
                pass

        else:
            self.publish(self.job_name, json.dumps({'data': Positions.Flat.value}))

    def run(self):
        self.sub([self.orderbook_channel, 'should_trade'])


@click.command()
@click.option('--depth', '-d', default=40, type=int)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--volume-max', '-v', default=1.0e4, type=float)
@click.argument('symbol', type=str)
def main(**kwargs):

    emitter = PredictionEmitter(**kwargs)
    emitter.run()


if __name__ == '__main__':
    main()
