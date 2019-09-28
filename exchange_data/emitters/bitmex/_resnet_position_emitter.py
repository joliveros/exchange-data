from abc import ABC

from tensorflow import TensorShape

from exchange_data import Measurement, settings
from exchange_data.channels import BitmexChannels
from exchange_data.emitters.bitmex._bitmex_position_emitter import BitmexPositionEmitter
from exchange_data.models.resnet.model import Model
from exchange_data.tfrecord.dataset import dataset
from exchange_data.trading import Positions
from exchange_data.utils import DateTimeUtils
from pathlib import Path
from prometheus_client import Gauge
from pytimeparse.timeparse import timeparse
from tensorflow.python.data import Dataset
from tensorflow.python.keras.estimator import model_to_estimator
from tgym.envs import OrderBookTradingEnv
from time import sleep

import alog
import json
import click
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(settings.LOG_LEVEL)

pos_summary = Gauge('emit_position_resnet', 'Trading Position')


class ResnetPositionEmitter(OrderBookTradingEnv, BitmexPositionEmitter, ABC):
    def __init__(self, frame_width, checkpoint_name, **kwargs):
        super().__init__(
            frame_width=frame_width,
            env='orderbook-trading-v0',
            is_training=False,
            **kwargs
        )
        BitmexPositionEmitter.__init__(self, **kwargs)
        self.obs_stream = []
        self.frame_width = frame_width

        for prediction in self.predictions(checkpoint_name):
            alog.info('#### here ####')
            action_value = np.argmax(prediction['dense_1'], 0)
            alog.info(action_value)
            # self.publish_position()

    def obs_generator(self):
        while sleep(1e-3):
            if len(self.obs_stream):
                obs = self.obs_stream.pop()
                yield obs, 0

    def dataset(self, batch_size):
        dataset = Dataset.from_generator(
                generator=self.obs_generator,
                # output_shapes=(TensorShape((self.frame_width, self.frame_width, 3))),
                output_types=((tf.float32, tf.float32, tf.float32), (tf.int32,)),
                args=([1])
            ) \
            .batch(batch_size)

        iterator = dataset.make_one_shot_iterator()

    def predictions(self, checkpoint_name):
        model = Model(0.0004, self.frame_width)
        model_dir = f'{Path.home()}/.exchange-data/models/resnet'
        checkpoint_path = f'{model_dir}/model.ckpt-{checkpoint_name}'

        resnet_estimator = model_to_estimator(
            keras_model=model, model_dir=model_dir,
            checkpoint_format='checkpoint',
        )

        predictions = resnet_estimator.predict(
            checkpoint_path=checkpoint_path,
            input_fn=lambda: self.dataset(batch_size=1)
        )

        return predictions

    def _get_observation(self):
        if self.last_obs_len() < self.observation_space.shape[0]:
            return super()._get_observation()
        else:
            return self.last_timestamp, self.orderbook_frame

    def publish_position(self, action):
        _action = None

        if Positions.Flat.value == action:
            _action = Positions.Flat
        elif Positions.Long.value == action:
            _action = Positions.Long

        if _action:
            self.publish(self.job_name, dict(data=_action.name))

    @pos_summary.time()
    def _emit_position(self, data):
        meas = Measurement(**data)
        self.last_timestamp = meas.time
        self.orderbook_frame = np.asarray(json.loads(meas.fields['data']))

        obs = self.get_observation()

        self.obs_stream.append(obs[0])

        # action = self.agent.compute_action(self.last_observation)

        # self.step(action)
        #
        # self.publish_position(action)


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
        max_frames=1,
        summary_interval=timeparse('1m'),
        **kwargs
    )

    env.reset()

    env.start()


if __name__ == '__main__':
    main()
