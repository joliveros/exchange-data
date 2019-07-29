import logging

import alog
import click
from pytimeparse.timeparse import timeparse
from tensorflow.python.keras import Input
from tensorflow.python.keras.api.keras import models
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras.layers import Dense, Dropout, TimeDistributed, \
    GlobalAveragePooling1D, GlobalAveragePooling3D, GlobalAveragePooling2D, Flatten
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow_estimator.python.estimator.training import TrainSpec, EvalSpec, \
    train_and_evaluate
from pathlib import Path
from exchange_data.tfrecord.dataset import dataset
import tensorflow

tensorflow.compat.v1.logging.set_verbosity(logging.DEBUG)

model = models.Sequential()
base = ResNet50(include_top=True, weights=None, classes=3)
for layer in base.layers:
    layer.trainable = True
model.add(Input(shape=(6, 229, 229, 3)))
model.add(TimeDistributed(base))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-2, decay=1e-2),
              metrics=['accuracy'])

model.summary()

model_dir = f'{Path.home()}/.exchange-data/models/resnet2/'
est_resnet = model_to_estimator(keras_model=model, model_dir=model_dir,
                                checkpoint_format='checkpoint')



@click.command()
@click.option('--epochs', '-e', type=int, default=10)
def main(epochs, **kwargs):
    eval_span = timeparse('10m')
    train_spec = TrainSpec(input_fn=lambda: dataset(epochs).skip(eval_span),
                           max_steps=epochs*6*60*60)
    eval_spec = EvalSpec(input_fn=lambda: dataset(1).take(eval_span))

    train_and_evaluate(est_resnet, train_spec, eval_spec)


if __name__ == '__main__':
    main()
