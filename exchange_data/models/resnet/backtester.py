from abc import ABC

from exchange_data.emitters.bitmex._resnet_position_emitter import ResnetPositionEmitter
from exchange_data.models.resnet.model import Model
from exchange_data.tfrecord.dataset import dataset
from pathlib import Path
from tensorflow.python.keras.estimator import model_to_estimator

import alog
import click
import numpy as np
import tensorflow as tf


class BackTester(ResnetPositionEmitter, ABC):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)



@click.command()
# @click.option('--epochs', '-e', type=int, default=10)
@click.option('--frame-size', '-f', type=int, default=224)
# @click.option('--learning-rate', '-l', type=float, default=0.3e-4)
# @click.option('--clear', '-c', is_flag=True)
# @click.option('--eval-span', type=str, default='20m')
def main(frame_size):



if __name__ == '__main__':
    main()
