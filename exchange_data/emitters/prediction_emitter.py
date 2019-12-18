#!/usr/bin/env python

from exchange_data import settings
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger

import alog
import click
import json
import numpy as np
import requests

from exchange_data.trading import Positions
from tgym.envs.orderbook.ascii_image import AsciiImage


class PredictionEmitter(Messenger):
    def __init__(self, symbol, **kwargs):
        super().__init__(**kwargs)
        self.orderbook_channel = f'orderbook_img_frame_{symbol}'
        self.on(self.orderbook_channel, self.emit_prediction)

    def emit_prediction(self, data):
        frame = np.array(json.loads(data['frame']), dtype=np.uint8)

        data = json.dumps(dict(
            signature_name='serving_default',
            instances=[json.loads(data['frame'])]
        ))

        headers = {'content-type': 'application/json'}

        json_response = requests.post(
            f'http://{settings.RESNET_HOST}:8501/v1/models/resnet:predict',
            data=data,
            headers=headers
        )

        predictions = json.loads(json_response.text)['predictions']
        max_index = np.argmax(predictions[0])
        position = [position for position in Positions if position.value == max_index][0]

        if max_index != 0:
            alog.info(AsciiImage(frame, new_width=12))
            alog.info(alog.pformat(json.loads(json_response.text)))
            alog.info((position, max_index, predictions[0][max_index]))

    def run(self):
        self.sub([self.orderbook_channel])


@click.command()
@click.option('--min-std-dev', '-std', default=2.0, type=float)
@click.option('--print-ascii-chart', '-a', is_flag=True)
@click.option('--summary-interval', '-si', default=6, type=int)
@click.option('--use-volatile-ranges', '-v', is_flag=True)
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__))
def main(**kwargs):
    emitter = PredictionEmitter(**kwargs)
    emitter.run()


if __name__ == '__main__':
    main()
