#!/usr/bin/env python
import time

import optuna
from optuna import Trial

from exchange_data.data.orderbook_frame import OrderBookFrame
from exchange_data.emitters.backtest import BackTest
from exchange_data.models.resnet.expected_position import \
    expected_position_frame
from exchange_data.models.resnet.model import ModelTrainer
from pathlib import Path
from tensorboard.plugins.hparams import api as hp

import alog
import click
import logging
import tensorflow as tf

HP_LRATE = hp.HParam('Learning Rate', hp.RealInterval(0.00001, 0.0001))
HP_EPSILON = hp.HParam('Epsilon', hp.RealInterval(0.09, 0.1064380))

METRIC_ACCURACY = 'accuracy'
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','RMSprop']))
SYMBOL = 'BANDBNB'
group_by = '1m'
volume_max = 30000.0
min_change = 0.0
expected_position_length = 4
sequence_length = 48

backtest = BackTest(**{
    # 'start_date': DateTimeUtils.parse_datetime_str('2020-07-01 17:34:00'),
    # 'end_date': DateTimeUtils.parse_datetime_str('2020-07-01 21:34:00'),
    'group_by': group_by,
    'database_name': 'binance',
    'depth': 40,
    'interval': '1d',
    'window_size': '3m',
    'plot': False,
    'sequence_length': 48,
    'symbol': SYMBOL,
    'volume_max': volume_max,
    'volatility_intervals': False,
})

df = OrderBookFrame(**{
    'database_name': 'binance',
    'depth': 40,
    'group_by': group_by,
    'interval': '4d',
    'plot': False,
    'sequence_length': 48,
    'symbol': SYMBOL,
    'volatility_intervals': True,
    'volume_max': volume_max,
    'window_size': '3m'
}).frame()


def run(trial: Trial):
    backtest.trial = trial

    tf.keras.backend.clear_session()

    run_dir = f'{Path.home()}/.exchange-data/models/{SYMBOL}_params/' \
              f'{trial.number}'

    hparams = dict(
        pos_change_quant=trial.suggest_float('pos_change_quant', 0.1, .9),
        # take_ratio=trial.suggest_float('take_ratio', 1.0009, 1.005),
        # expected_position_length=trial.suggest_int(
        #     'expected_position_length', 1, 6),
        # expected_position_length=
        # trial.suggest_int('expected_position_length', 1, 12),
        # epochs=trial.suggest_int('epochs', 30),
    )

    with tf.summary.create_file_writer(run_dir).as_default():
        _df = expected_position_frame(
            df,
            take_ratio=0.99,
            expected_position_length=1,
            **hparams
        )

        train_df = _df.sample(frac=0.9, random_state=0)
        eval_df = _df.sample(frac=0.1, random_state=0)

        params = {
            'epochs': 20,
            'batch_size': 20,
            'clear': True,
            'directory': trial.number,
            'export_model': True,
            'train_df': train_df,
            'eval_df': eval_df,
            'learning_rate': 1.0e-5,
            'levels': 40,
            'seed': 216,
            'sequence_length': 48,
            'symbol': SYMBOL,
        }

        hp.hparams(hparams, trial_id=str(trial.number))

        alog.info(hparams)

        hparams = {**params, **hparams}

        alog.info(alog.pformat(hparams))

        model = ModelTrainer(**hparams)
        result = model.run()
        accuracy = result.get('accuracy')
        global_step = result.get('global_step')
        exported_model_path = result.get('exported_model_path')

        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=global_step)

        time.sleep(5)

        backtest.test()

        try:
            if trial.study.best_trial.value > backtest.capital:
                alog.info('## deleting trial ###')
                alog.info(exported_model_path)
                Path(exported_model_path).rmdir()
        except ValueError:
            pass

        if backtest.capital == 1.0:
            alog.info('## deleting trial ###')
            Path(exported_model_path).rmdir()
            return 0.0

        return backtest.capital


@click.command()
def main(**kwargs):

    logging.getLogger('tensorflow').setLevel(logging.INFO)
    session_limit = 100

    study = optuna.create_study(direction='maximize')

    study.optimize(run, n_trials=session_limit)



if __name__ == '__main__':
    main()
