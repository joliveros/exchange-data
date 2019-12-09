from abc import ABC
from exchange_data import settings
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger
from exchange_data.emitters.bitmex._bitmex_position_emitter import BitmexPositionEmitter
from exchange_data.utils import DateTimeUtils
from prometheus_client import Gauge
from pytimeparse.timeparse import timeparse

import alog
import click
import json
import numpy as np


# pos_summary = Gauge('emit_position_resnet', 'Trading Position')


class ResnetPositionEmitter(Messenger):
    def __init__(self, frame_width, **kwargs):
        super().__init__()

        self.orderbook_channel = 'XBTUSD_OrderBookFrame_depth_21'
        self.obs_stream = []
        self.frame_width = frame_width
        self.on(self.orderbook_channel, self.predict)

    def start(self):
        self.sub([self.orderbook_channel])

    def predict(self, data):
        frame = data['fields']['data']
        alog.info(data)
        # alog.info(frame)


@click.command()
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__),
                default=BitmexChannels.XBTUSD.value)
@click.argument(
    'job_name',
    type=str,
    default=None
)
@click.option('--frame-width', '-f', type=int, default=224)
def main(**kwargs):
    env = ResnetPositionEmitter(
        **kwargs
    )

    env.start()


if __name__ == '__main__':
    main()
