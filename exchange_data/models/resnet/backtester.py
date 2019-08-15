from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow_estimator.python.estimator.run_config import RunConfig

from exchange_data.models.resnet.model import Model
from pathlib import Path

import alog
import click
import tensorflow as tf
import numpy as np
from exchange_data.tfrecord.dataset import dataset


@click.command()
# @click.option('--epochs', '-e', type=int, default=10)
@click.option('--frame-size', '-f', type=int, default=224)
# @click.option('--learning-rate', '-l', type=float, default=0.3e-4)
# @click.option('--clear', '-c', is_flag=True)
# @click.option('--eval-span', type=str, default='20m')
def main(frame_size):
    model = Model(0.0004, frame_size)
    model_dir = f'{Path.home()}/.exchange-data/models/resnet'
    checkpoint_path = f'{model_dir}/model.ckpt-146881'

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

    for pred in predictions:
        alog.info(pred)
        alog.info(np.argmax(pred['dense_1'], 0))


if __name__ == '__main__':
    main()
