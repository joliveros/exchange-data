#!/usr/bin/env python
import shutil

from exchange_data.data.orderbook_frame import OrderBookFrame
from exchange_data.emitters.backtest import BackTest
from exchange_data.models.resnet.expected_position import expected_position_frame
from exchange_data.models.resnet.model import ModelTrainer
from optuna import Trial
from pathlib import Path
from tensorboard.plugins.hparams import api as hp

import alog
import click
import logging
import optuna
import tensorflow as tf
import time

logging.getLogger('tensorflow').setLevel(logging.INFO)


class SymbolTuner(OrderBookFrame):

    def __init__(self, volatility_intervals, session_limit, backtest_interval,
                 **kwargs):
        if not volatility_intervals:
            kwargs['window_size'] = '1h'
        super().__init__(volatility_intervals=volatility_intervals, **kwargs)

        self.train_df = self.label_positive_change(5)

        kwargs['interval'] = backtest_interval
        kwargs['window_size'] = '1h'
        self.backtest = BackTest(quantile=self.quantile, **kwargs)

        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.run, n_trials=session_limit)

    @property
    def run_dir(self):
        return f'{Path.home()}/.exchange-data/models/{self.symbol}_params/' \
        f'{self.trial.number}'

    def run(self, trial: Trial):
        self.trial = trial
        self.backtest.trial = trial

        tf.keras.backend.clear_session()

        hparams = dict(
            # consecutive_positive_change=trial.suggest_int(
            #     'consecutive_positive_change', 1, 6),
            take_ratio=trial.suggest_float('take_ratio', 0.99, 1.0)
        )

        with tf.summary.create_file_writer(self.run_dir).as_default():
            _df = expected_position_frame(
                df=self.train_df,
                **hparams
            )

            train_df = _df.sample(frac=0.9, random_state=0)
            eval_df = _df.sample(frac=0.1, random_state=0)

            params = {
                'epochs': 30,
                'batch_size': 20,
                'clear': True,
                'directory': trial.number,
                'export_model': True,
                'train_df': train_df,
                'eval_df': eval_df,
                'learning_rate': 1.0e-5,
                'levels': 40,
                # 'seed': 216,
                'sequence_length': 48,
                'symbol': self.symbol,
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

            tf.summary.scalar('accuracy', accuracy, step=global_step)

            time.sleep(5)

            self.backtest.test()

            try:
                if trial.study.best_trial.value > self.backtest.capital:
                    alog.info('## deleting trial ###')
                    alog.info(exported_model_path)
                    shutil.rmtree(exported_model_path, ignore_errors=True)
            except ValueError:
                pass

            if self.backtest.capital == 1.0:
                alog.info('## deleting trial ###')
                alog.info(exported_model_path)
                shutil.rmtree(exported_model_path, ignore_errors=True)
                return 0.0

            return self.backtest.capital


@click.command()
@click.option('--backtest-interval', '-b', default='4h', type=str)
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--depth', '-d', default=40, type=int)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--interval', '-i', default='4h', type=str)
@click.option('--max-volume-quantile', '-m', default=0.99, type=float)
@click.option('--plot', '-p', is_flag=True)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--session-limit', '-s', default=75, type=int)
@click.option('--volatility-intervals', '-v', is_flag=True)
@click.option('--window-size', '-w', default='3m', type=str)
@click.argument('symbol', type=str)
def main(**kwargs):
    SymbolTuner(**kwargs)


if __name__ == '__main__':
    main()
