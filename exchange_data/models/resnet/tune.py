#!/usr/bin/env python

from exchange_data import settings
from exchange_data.data.max_min_frame import MaxMinFrame
from exchange_data.emitters.backtest import BackTest
from exchange_data.models.resnet.model_trainer import ModelTrainer
from exchange_data.models.resnet.study_wrapper import StudyWrapper
from exchange_data.trading import Positions
from optuna import Trial
from pathlib import Path
from pytimeparse.timeparse import timeparse
from redlock import RedLock, RedLockError
from tensorboard.plugins.hparams import api as hp

import alog
import click
import logging
import pandas as pd
import re
import shutil
import tensorflow as tf
import time as t

logging.getLogger('tensorflow').setLevel(logging.INFO)


class SymbolTuner(MaxMinFrame, StudyWrapper):
    backtest = None

    def __init__(self,
                 volatility_intervals,
                 session_limit,
                 macd_session_limit,
                 backtest_interval,
                 min_capital,
                 memory,
                 num_locks=2,
                 **kwargs):

        self._kwargs = kwargs

        if not volatility_intervals:
            kwargs['window_size'] = '1h'
        super().__init__(volatility_intervals=volatility_intervals,
                         session_limit=macd_session_limit, **kwargs)

        StudyWrapper.__init__(self, **kwargs)
        self.min_capital = min_capital
        self.memory = memory
        self.hparams = None
        self.num_locks = num_locks
        self.current_lock_ix = 0
        self.run_count = 0
        self.export_dir.mkdir(exist_ok=True)

        self.split_gpu()

        kwargs['interval'] = backtest_interval
        kwargs['window_size'] = '1h'

        self.base_model_dir = f'{Path.home()}/.exchange-data/models' \
                             f'/{self.symbol}'

        self.study.optimize(self.run, n_trials=session_limit)

        self.clear()

    def clear(self):
        try:
            self._clear()
        except RedLockError:
            pass

    def _clear(self):
        with RedLock('clear_lock', [dict(
                        host=settings.REDIS_HOST,
                        db=0
                    )],
                     retry_delay=timeparse('15s'),
                     retry_times=12, ttl=timeparse('1h') * 1000):

            self.study_db_path.unlink()
            self.clear_dirs()

    def clear_dirs(self):
        shutil.rmtree(str(self.export_dir), ignore_errors=True)
        Path(self.export_dir).mkdir()

        shutil.rmtree(
            f'{Path.home()}/.exchange-data/models/{self.symbol}_params',
            ignore_errors=True)

        base_dir = Path(self.base_model_dir)
        shutil.rmtree(str(base_dir), ignore_errors=True)

        if not base_dir.exists():
            base_dir.mkdir()

    @property
    def export_dir(self):
        return Path(f'{Path.home()}/.exchange-data/models/' \
                 f'{self.symbol}_export')

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
        retry_relay = 10

        try:
            if self.run_count > 1:
                t.sleep(retry_relay)
            with RedLock(self.train_lock, [dict(
                    host=settings.REDIS_HOST,
                    db=0
                )], retry_delay=timeparse('15s'),
                    retry_times=12, ttl=timeparse('1h') * 1000):
                self._run(*args)
            self.run_count += 1
            return self.run_backtest()

        except RedLockError as e:
            alog.info(e)
            t.sleep(retry_relay)
            return self.run(*args)

    def _run(self, trial: Trial):
        self.trial = trial

        tf.keras.backend.clear_session()

        hparams = dict(
            flat_ratio=trial.suggest_float('flat_ratio', 1.02, 1.05),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.125),
        )

        group_by = 4

        self.group_by_min = group_by
        self.group_by = f'{group_by}m'

        self.hparams = hparams
        self._kwargs['group_by'] = self.group_by

        with tf.summary.create_file_writer(self.run_dir).as_default():
            flat_ratio = hparams.get('flat_ratio')
            _df = self.train_df = self.label_positive_change().copy()

            alog.info(_df)

            flat_df = _df[_df['expected_position'] == Positions.Flat]
            flat_df.loc[:, 'expected_position'] = 0
            long_df = _df[_df['expected_position'] == Positions.Long]
            long_df.loc[:, 'expected_position'] = 1

            flat_count = len(long_df) * flat_ratio

            flat_df = flat_df.sample(frac=flat_count/len(flat_df),
                                     replace=True)

            _df = pd.concat([flat_df, long_df])

            train_df = _df.sample(frac=0.9, random_state=0)
            eval_df = _df.sample(frac=0.1, random_state=0)

            alog.info(train_df)
            alog.info(eval_df)

            params = {
                'batch_size': 4,
                'depth': self.output_depth,
                'directory': trial.number,
                'epochs': 1,
                'eval_df': eval_df,
                'export_model': True,
                'relu_alpha': 0.294,
                'round_decimals': self.round_decimals,
                'sequence_length': self.sequence_length,
                'base_filter_size': 16,
                'symbol': self.symbol,
                'dir_name': self.symbol,
                'train_df': train_df,
                'num_conv': 3
            }

            hp.hparams(hparams, trial_id=str(trial.number))

            hparams = {**params, **hparams}

            self.hparams = hparams

            model = ModelTrainer(**hparams)
            result = model.run()

            self.model_dir = model.model_dir

            accuracy = result.get('accuracy')
            global_step = result.get('global_step')
            self.exported_model_path = result.get('exported_model_path')
            trial.set_user_attr('exported_model_path', self.exported_model_path)
            trial.set_user_attr('model_version', self.model_version)
            trial.set_user_attr('quantile', self.quantile)

            tf.summary.scalar('accuracy', accuracy, step=global_step)

            t.sleep(5)

    @property
    def model_version(self):
        return re.match(r'.+\/(\d+)$', self.exported_model_path).group(1)

    def run_backtest(self):
        self._kwargs.pop('offset_interval', None)

        self.backtest = BackTest(quantile=self.quantile, **self._kwargs)

        self.backtest.trial = self.trial

        tf.keras.backend.clear_session()

        with tf.summary.create_file_writer(self.run_dir).as_default():
            trial = self.trial
            exported_model_path = self.exported_model_path

            test_df = self.backtest.test(self.model_version)

            alog.info(test_df)

            try:
                if trial.study.best_trial.value > self.backtest.capital:
                    alog.info('## deleting trial ###')
                    alog.info(exported_model_path)
                    shutil.rmtree(exported_model_path, ignore_errors=True)
            except ValueError:
                pass

            if self.backtest.capital <= self.min_capital:
                alog.info('## deleting trial ###')
                alog.info(exported_model_path)
                shutil.rmtree(self.run_dir)
                shutil.rmtree(self.model_dir)
                shutil.rmtree(exported_model_path, ignore_errors=True)

            if self.backtest.capital == 1.0:
                return 0.0
            return self.backtest.capital

    def split_gpu(self):
        physical_devices = tf.config.list_physical_devices('GPU')

        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [
                tf.config.LogicalDeviceConfiguration(memory_limit=self.memory),
             ])

        #logical_devices = tf.config.list_logical_devices('GPU')

        #assert len(logical_devices) == len(physical_devices) + 1

@click.command()
@click.argument('symbol', type=str)
@click.option('--backtest-interval', '-b', default='15m', type=str)
@click.option('--database-name', '-d', default='binance', type=str)
@click.option('--depth', '-d', default=72, type=int)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--interval', '-i', default='1h', type=str)
@click.option('--macd-session-limit', default=200, type=int)
@click.option('--memory', '-m', default=1000, type=int)
@click.option('--min-capital', default=1.0, type=float)
@click.option('--num-locks', '-n', default=2, type=int)
@click.option('--offset-interval', '-o', default='3h', type=str)
@click.option('--plot', '-p', is_flag=True)
@click.option('--round-decimals', '-D', default=3, type=int)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--session-limit', '-s', default=None, type=int)
@click.option('--volatility-intervals', '-v', is_flag=True)
@click.option('--window-size', '-w', default='3m', type=str)
def main(**kwargs):
    SymbolTuner(**kwargs)


if __name__ == '__main__':
    main()
