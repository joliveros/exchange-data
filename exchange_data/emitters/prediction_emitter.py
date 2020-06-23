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
    def __init__(self, symbol: BitmexChannels, **kwargs):
        self.symbol = symbol
        self.job_name = f'trade_{self.symbol.value}'


class PredictionEmitter(Messenger, TradeJob):
    def __init__(self, sequence_length, depth, symbol, model_name, **kwargs):
        self.symbol = symbol
        self.sequence_length = sequence_length

        super().__init__(symbol=symbol, **kwargs)

        self.model_name = model_name
        self.orderbook_channel = f'XBTUSD_OrderBookFrame_depth_{depth}_2s'
        self.trading_enabled = False
        self.frames = deque(maxlen=sequence_length)

        self.load_previous_frames(depth)

        self.on(self.orderbook_channel, self.emit_prediction)
        self.on('should_trade', self.should_trade)

    def should_trade(self, should_trade):
        self.trading_enabled = strtobool(should_trade.decode())

    def load_previous_frames(self, depth):
        now = DateTimeUtils.now()
        start_date = now - timedelta(seconds=49*2)

        levels = OrderBookLevelStreamer(
            start_date=start_date,
            end_date=DateTimeUtils.now(),
            database_name='bitmex',
            depth=depth,
            groupby='2s',
            window_size='48s',
            sample_interval='48s'
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
        orderbook_levels = np.delete(orderbook_levels, 0, 1)
        orderbook_levels[0][0] = np.flip(orderbook_levels[0][0])
        orderbook_levels[1] =  orderbook_levels[1] * -1
        max = 3.0e6
        orderbook_levels = np.reshape(orderbook_levels, (80, 1)) / max
        return np.clip(orderbook_levels, a_min=0.0, a_max=max)

    def get_prediction(self):
        frames = np.expand_dims(np.asarray(self.frames), axis=0)

        data = json.dumps(dict(
            signature_name='serving_default',
            instances=frames.tolist()
        ))

        headers = {
            'content-type': 'application/json',
            'Content-Encoding': 'gzip'
        }

        data = gzip.compress(bytes(data, encoding='utf8'))

        json_response = requests.post(
            f'http://{settings.RESNET_HOST}:8501/v1/models/resnet:predict',
            data=data,
            headers=headers
        )

        predictions = json.loads(json_response.text)['predictions']

        max_index = np.argmax(predictions[0])

        position = [
            position for position in Positions
            if position.value == max_index
        ][0]

        alog.info((position, max_index, predictions[0][max_index]))

        return position

    def _emit_prediction(self, data):
        frame = data['fields']['data']
        frame = json.loads(frame)

        self.frames.append(self.normalize_frame(frame))

        position = self.get_prediction()

        self.publish(self.job_name, json.dumps({'data': position.value}))

    def emit_prediction(self, data):
        self._emit_prediction(data)

        return

        if self.trading_enabled:
            try:
                self._emit_prediction(data)
            except:
                pass

        else:
            self.publish(self.job_name, json.dumps({'data': Positions.Flat.value}))

    def run(self):
        self.sub([self.orderbook_channel])


@click.command()
@click.option('--model-name', '-m', default=None, type=str)
@click.option('--depth', '-d', default=40, type=int)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
def main(symbol, **kwargs):

    emitter = PredictionEmitter(
        symbol=BitmexChannels[symbol],
        **kwargs)
    emitter.run()


if __name__ == '__main__':
    main()
