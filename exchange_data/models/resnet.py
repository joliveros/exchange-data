import alog
import click
from pytimeparse.timeparse import timeparse
from tensorflow.python.keras.api.keras import models
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, \
    Dropout, TimeDistributed
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.layers.convolutional import Conv2D, Conv1D
from tensorflow_estimator.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate
from pathlib import Path
from exchange_data.tfrecord.dataset import dataset

model = models.Sequential()
base = ResNet50(include_top=False, input_shape=(96, 192, 3), weights=None, classes=3)
base.trainable = True
model.add(base)
model.add(TimeDistributed(Conv1D(64, (3,), 3)))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, decay=0.95),
              metrics=['accuracy'])

model_dir = f'{Path.home()}/.exchange-data/models/resnet'
est_resnet = model_to_estimator(keras_model=model, model_dir=model_dir,
                                checkpoint_format='checkpoint')


@click.command()
@click.option('--epochs', '-e', type=int, default=10)
def main(epochs, **kwargs):
    eval_span = timeparse('10m')
    train_spec = TrainSpec(input_fn=lambda: dataset(epochs).skip(eval_span))
    eval_spec = EvalSpec(input_fn=lambda: dataset(1).take(eval_span))

    train_and_evaluate(est_resnet, train_spec, eval_spec)


if __name__ == '__main__':
    main()
