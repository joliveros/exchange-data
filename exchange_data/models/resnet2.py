import alog
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs
from tensorflow_estimator.python.estimator.run_config import RunConfig

from exchange_data import settings
from exchange_data.tfrecord.dataset import dataset
from pathlib import Path
from pytimeparse.timeparse import timeparse
from tensorflow.python.keras import Input
from tensorflow.python.keras.api.keras import models
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras.layers import Dense, Reshape, LSTM, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow_estimator.python.estimator.training import TrainSpec, EvalSpec, train_and_evaluate
import shutil
import click
import tensorflow


class ProfitAndLossHook(SessionRunHook):
    def __init__(self):
        super(SessionRunHook).__init__()

    def begin(self):
        pass

    def before_run(self, run_context):
        return SessionRunArgs()

    def after_run(self, run_context, run_values):
        loss_value = run_values.results
        print("loss value:", loss_value)


@click.command()
@click.option('--epochs', '-e', type=int, default=10)
@click.option('--batch-size', '-b', type=int, default=20)
@click.option('--learning-rate', '-l', type=float, default=0.1)
@click.option('--clear', '-c', is_flag=True)
def main(epochs, batch_size, clear, learning_rate, **kwargs):
    tensorflow.compat.v1.logging.set_verbosity(settings.LOG_LEVEL)

    model = models.Sequential()
    base = ResNet50(include_top=False, weights=None, classes=3)

    for layer in base.layers:
        layer.trainable = True

    model.add(Input(shape=(115, 115, 3)))
    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Reshape((64, 1)))
    model.add(LSTM(24))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=SGD(lr=learning_rate, decay=5e-3),
                  metrics=['accuracy'])

    model.summary()

    model_dir = f'{Path.home()}/.exchange-data/models/resnet'

    if clear:
        try:
            shutil.rmtree(model_dir)
        except Exception:
            pass

    run_config = RunConfig(save_checkpoints_steps=100)
    est_resnet = model_to_estimator(
        keras_model=model, model_dir=model_dir,
        checkpoint_format='checkpoint',
        config=run_config,
    )
    # alog.info(alog.pformat(est_resnet.get_variable_names()))
    eval_span = timeparse('20m')
    train_spec = TrainSpec(input_fn=lambda: dataset(batch_size, epochs).skip(eval_span),
                           max_steps=epochs * 6 * 60 * 60)
    eval_spec = EvalSpec(
        input_fn=lambda: dataset(batch_size, 1).take(eval_span),
        steps=eval_span,
        hooks=[ProfitAndLossHook()]
    )

    train_and_evaluate(est_resnet, train_spec, eval_spec)


if __name__ == '__main__':
    main()
