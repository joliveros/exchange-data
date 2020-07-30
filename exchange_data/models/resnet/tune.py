#!/usr/bin/env python
import re
import shutil

from pytimeparse.timeparse import timeparse
from redlock import RedLock, RedLockError

from exchange_data.data.orderbook_frame import OrderBookFrame
from exchange_data.emitters.backtest import BackTest
from exchange_data.models.resnet.expected_position import expected_position_frame
from exchange_data.models.resnet.model_trainer import ModelTrainer
from optuna import Trial, load_study
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
                 num_locks=2,
                 **kwargs):
        if not volatility_intervals:
            kwargs['window_size'] = '1h'
        super().__init__(volatility_intervals=volatility_intervals, **kwargs)
        self.num_locks = num_locks
        self.current_lock_ix = 0
        self.run_count = 0
        self.on('run', self.run)
        self.split_gpu()
        study_db_path = f'{Path.home()}/.exchange-data/models/{self.symbol}.db'
        study_db_path = Path(study_db_path)
        db_conn_str = f'sqlite:///{study_db_path}'

        if not study_db_path.exists():
            self.study = optuna.create_study(
                study_name=self.symbol, direction='maximize',
                storage=db_conn_str)
        else:
            self.study = load_study(study_name=self.symbol, storage=db_conn_str)

        kwargs['interval'] = backtest_interval
        kwargs['window_size'] = '1h'

        self.backtest = BackTest(quantile=self.quantile, **kwargs)

        self.study.optimize(self.run, n_trials=session_limit)

    @property
    def run_dir(self):
        return f'{Path.home()}/.exchange-data/models/{self.symbol}_params/' \
        f'{self.trial.number}'

    @property
    def train_lock(self):
        lock_name = f'train_lock_{self.current_lock_ix}'

        alog.info(f'### lock name {lock_name} ####')

        self.current_lock_ix += 1

        if self.current_lock_ix > self.num_locks - 1:
            self.current_lock_ix = 0

        return lock_name

    def run(self, *args):
        retry_relay = 40

        try:
            if self.run_count > 1:
                time.sleep(retry_relay)
            with RedLock(self.train_lock, retry_delay=timeparse('1m'),
                         retry_times=120, ttl=timeparse('1h') * 1000):
                self._run(*args)
            self.run_count += 1
            return self.run_backtest()

        except Exception as e:
            alog.info(e)
            time.sleep(retry_relay)
            return self.run(*args)

    def _run(self, trial: Trial):
        self.trial = trial
        self.backtest.trial = trial

        tf.keras.backend.clear_session()

        hparams = dict(
            # learning_rate=trial.suggest_float('learning_rate', 0.03, 0.1),
            # flat_ratio=trial.suggest_float('flat_ratio', 0.1, 10.0),
            # neg_change_ratio=trial.suggest_float('neg_change_ratio', 0.1, 2.0),
            # neg_change_quantile=trial.suggest_float('neg_change_quantile', 0.0,
            #                                         1.0),
            # lstm_units=trial.suggest_int('lstm_units', 8, 16),
            # prefix_length=trial.suggest_int('prefix_length', 1, 6),
            # filters=trial.suggest_int('epochs', 1, 6),
            # inception_units=trial.suggest_int('inception_units', 1, 6),
            # epochs=trial.suggest_int('epochs', 5, 14),
            # min_consecutive_count=trial.suggest_int('min_consecutive_count',
            #                                         3, 6)
        )

        with tf.summary.create_file_writer(self.run_dir).as_default():
            flat_ratio = hparams.get('flat_ratio', 1.0)
            neg_change_ratio = hparams.get('neg_change_ratio', 0.30394)
            # flat_ratio = 2.4
            _df = self.label_positive_change(prefix_length=2,
                                             min_consecutive_count=4,
                                             neg_change_quantile=0.15136,
                                             **hparams)
            large_change_df = _df[_df['large_negative_change'] == 1]
            flat_df = _df[_df['expected_position'] == 0]
            long_df = _df[_df['expected_position'] == 1]

            flat_count = len(long_df) * flat_ratio
            neg_change_count = len(long_df) * neg_change_ratio

            flat_df = flat_df.sample(frac=flat_count/len(flat_df),
                                     replace=True)

            large_change_df = large_change_df.sample(frac=neg_change_count / len(
                large_change_df),
                                     replace=True)

            _df = pd.concat([large_change_df, flat_df, long_df])

            train_df = _df.sample(frac=0.9, random_state=0)
            eval_df = _df.sample(frac=0.1, random_state=0)

            params = {
                'epochs': 5,
                'batch_size': 4,
                'clear': True,
                'directory': trial.number,
                'export_model': True,
                'train_df': train_df,
                'eval_df': eval_df,
                'learning_rate': 0.078735,
                'levels': 40,
                # 'seed': 216,
                'sequence_length': 48,
                'symbol': self.symbol,
            }

            hp.hparams(hparams, trial_id=str(trial.number))

            hparams = {**params, **hparams}

            model = ModelTrainer(**hparams)
            result = model.run()
            accuracy = result.get('accuracy')
            global_step = result.get('global_step')
            self.exported_model_path = result.get('exported_model_path')
            trial.set_user_attr('exported_model_path', self.exported_model_path)

            tf.summary.scalar('accuracy', accuracy, step=global_step)

            time.sleep(5)

    @property
    def model_version(self):
        return re.match(r'.+\/(\d+)$', self.exported_model_path).group(1)

    def run_backtest(self):
        with tf.summary.create_file_writer(self.run_dir).as_default():
            trial = self.trial
            exported_model_path = self.exported_model_path

            self.backtest.test(self.model_version)

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
                shutil.rmtree(self.run_dir)
                shutil.rmtree(exported_model_path, ignore_errors=True)

            if self.backtest.capital == 1.0:
                return 0.0
            return self.backtest.capital

    def split_gpu(self):
        physical_devices = tf.config.list_physical_devices('GPU')

        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=100),
             tf.config.LogicalDeviceConfiguration(memory_limit=100)])

        logical_devices = tf.config.list_logical_devices('GPU')

        assert len(logical_devices) == len(physical_devices) + 1

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
