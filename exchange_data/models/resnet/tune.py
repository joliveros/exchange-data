#!/usr/bin/env python
import shutil

from exchange_data.data.orderbook_frame import OrderBookFrame
from exchange_data.emitters.backtest import BackTest
from exchange_data.models.resnet.expected_position import expected_position_frame
from exchange_data.models.resnet.model_trainer import ModelTrainer
from optuna import Trial
from pathlib import Path
from tensorboard.plugins.hparams import api as hp
import pandas as pd
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
            learning_rate=trial.suggest_float('learning_rate', 0.03, 0.1),
            flat_ratio=trial.suggest_float('flat_ratio', 1.0, 5.0),
            # lstm_units=trial.suggest_int('lstm_units', 8, 16),
            prefix_length=trial.suggest_int('prefix_length', 1, 6),
            # filters=trial.suggest_int('epochs', 1, 5),
            # inception_units=trial.suggest_int('inception_units', 1, 4),
            # epochs=trial.suggest_int('epochs', 5, 30),
            min_consecutive_count=trial.suggest_int('min_consecutive_count',
                                                    5, 6)
        )

        with tf.summary.create_file_writer(self.run_dir).as_default():
            flat_ratio = hparams.get('flat_ratio')
            _df = self.label_positive_change(**hparams)
            flat_df = _df[_df['expected_position'] == 0]
            long_df = _df[_df['expected_position'] == 1]
            flat_count = len(long_df) * flat_ratio
            flat_df = flat_df.sample(frac=flat_count/len(flat_df),
                                     replace=True)
            _df = pd.concat([flat_df, long_df])

            train_df = _df.sample(frac=0.9, random_state=0)
            eval_df = _df.sample(frac=0.1, random_state=0)

            params = {
                'epochs': 25,
                'batch_size': 4,
                'clear': True,
                'directory': trial.number,
                'export_model': True,
                'train_df': train_df,
                'eval_df': eval_df,
                # 'learning_rate': 0.0024994,
                'levels': 40,
                'seed': 216,
                'sequence_length': 48,
                'symbol': self.symbol,
            }

            hp.hparams(hparams, trial_id=str(trial.number))

            hparams = {**params, **hparams}

            model = ModelTrainer(**hparams)
            result = model.run()
            accuracy = result.get('accuracy')
            global_step = result.get('global_step')
            exported_model_path = result.get('exported_model_path')
            trial.set_user_attr('exported_model_path', exported_model_path)

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

        if self.backtest.capital <= 0.99 or self.backtest.capital == 1.0:
            alog.info('## deleting trial ###')
            alog.info(exported_model_path)
            # shutil.rmtree(self.run_dir)
            shutil.rmtree(exported_model_path, ignore_errors=True)

        if self.backtest.capital == 1.0:
            return 0.0

        return self.backtest.capital


@click.command()
@click.option('--backtest-interval', '-b', default='15m', type=str)
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--depth', '-d', default=40, type=int)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--interval', '-i', default='1h', type=str)
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
