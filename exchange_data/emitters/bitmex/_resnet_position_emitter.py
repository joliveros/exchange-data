from abc import ABC

from tensorflow.python.data import Dataset

from exchange_data.channels import BitmexChannels
from exchange_data.emitters.bitmex._bitmex_position_emitter import BitmexPositionEmitter
from exchange_data.models.resnet.model import Model
from exchange_data.tfrecord.dataset import dataset
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils
from pathlib import Path
from pytimeparse.timeparse import timeparse
from tensorflow.python.keras.estimator import model_to_estimator
from tgym.envs import OrderBookTradingEnv

import alog
import click
import numpy as np
import tensorflow as tf


class ResnetPositionEmitter(OrderBookTradingEnv, BitmexPositionEmitter, ABC):
    def __init__(self, frame_width, checkpoint_name, **kwargs):
        super().__init__(
            frame_width=frame_width,
            env='orderbook-trading-v0',
            is_training=False,
            **kwargs
        )
        BitmexPositionEmitter.__init__(self, **kwargs)
        self.frame_width = frame_width
        return
        for prediction in self.predictions(checkpoint_name):
            action_value = np.argmax(prediction['dense_1'], 0)
            # self.publish_position()

    def dataset(self):
        return Dataset.from_generator(generator=)

    def predictions(self, checkpoint_name):
        model = Model(0.0004, self.frame_width)
        model_dir = f'{Path.home()}/.exchange-data/models/resnet'
        checkpoint_path = f'{model_dir}/model.ckpt-{checkpoint_name}'

        def input_fn():
            for record in dataset(batch_size=1):
                yield record['frame']

        resnet_estimator = model_to_estimator(
            keras_model=model, model_dir=model_dir,
            checkpoint_format='checkpoint',
        )

        predictions = resnet_estimator.predict(
            checkpoint_path=checkpoint_path,
            input_fn=lambda: dataset(batch_size=1)
        )

        return predictions
        # for pred in predictions:
        #     alog.info(pred)
        #     alog.info(np.argmax(pred['dense_1'], 0))

    def publish_position(self, action):
        _action = None

        if Positions.Flat.value == action:
            _action = Positions.Flat
        elif Positions.Long.value == action:
            _action = Positions.Long

        if _action:
            self.publish(self.job_name, dict(data=_action.name))


@click.command()
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__),
                default=BitmexChannels.XBTUSD.value)
@click.argument(
    'job_name',
    type=str,
    default=None
)
@click.option('--frame-width', '-f', type=int, default=224)
@click.option('--checkpoint-name', '-c', type=str, default='checkpoint')
def main(**kwargs):
    env = ResnetPositionEmitter(
        start_date=DateTimeUtils.now(),
        window_size='1m',
        max_frames=5,
        summary_interval=timeparse('1m'),
        **kwargs
    )

    env.reset()

    env.start()


if __name__ == '__main__':
    main()
