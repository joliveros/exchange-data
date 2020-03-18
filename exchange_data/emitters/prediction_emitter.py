#!/usr/bin/env python
import gzip
import logging
from collections import deque

from exchange_data import settings
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger
from exchange_data.trading import Positions
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
        self.orderbook_channel = f'orderbook_img_frame_{symbol.value}_{depth}'
        self.frames = deque(maxlen=sequence_length)

        self.on(self.orderbook_channel, self.emit_prediction)

    def emit_prediction(self, data):
        frame_list = [json.loads(data['frame'])]
        self.frames.append(frame_list)

        if len(self.frames) < self.sequence_length:
            return

        data = json.dumps(dict(
            signature_name='serving_default',
            instances=list(self.frames)
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

        if settings.LOG_LEVEL == logging.DEBUG:
            frame = np.array(frame_list, dtype=np.uint8)
            alog.info(AsciiImage(frame, new_width=12))
            alog.info(alog.pformat(json.loads(json_response.text)))
            alog.info((position, max_index, predictions[0][max_index]))

        self.publish(self.job_name, json.dumps({'data': position.value}))

    def run(self):
        self.sub([self.orderbook_channel])


@click.command()
@click.option('--model-name', '-m', default=None, type=str)
@click.option('--depth', '-d', default=21, type=int)
@click.option('--sequence-length', '-l', default=4, type=int)
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
def main(symbol, **kwargs):

    emitter = PredictionEmitter(
        symbol=BitmexChannels[symbol],
        **kwargs)
    emitter.run()


if __name__ == '__main__':
    main()
