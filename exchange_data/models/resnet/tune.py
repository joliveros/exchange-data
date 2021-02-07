#!/usr/bin/env python
from pip._vendor.contextlib2 import nullcontext

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
                 group_by_min,
                 volatility_intervals,
                 session_limit,
                 clear_runs,
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
        self.group_by_min = group_by_min
        self.clear_runs = clear_runs
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

        Path(self.best_model_dir).mkdir(parents=True, exist_ok=True)

        self.base_model_dir = f'{Path.home()}/.exchange-data/models' \
                             f'/{self.symbol}'

        self.backtest = BackTest(quantile=self.quantile, **kwargs)

        self.study.optimize(self.run, n_trials=session_limit)

    @property
    def best_model_dir(self):
        return f'{Path.home()}/.exchange-data/best_exported_models/' \
               f'{self.symbol}_export/1'

    def clear(self):
        alog.info('### clear runs ###')
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
        if self.num_locks == 0:
            return nullcontext()

        lock_name = f'train_lock_{self.current_lock_ix}'

        alog.info(f'### lock name {lock_name} ####')

        self.current_lock_ix += 1

        if self.current_lock_ix > self.num_locks - 1:
            self.current_lock_ix = 0

        return RedLock(lock_name, [dict(
            host=settings.REDIS_HOST,
            db=0
        )], retry_delay=timeparse('15s'),
                retry_times=12, ttl=timeparse('1h') * 1000)

    def run(self, *args):
        retry_relay = 10

        try:
            if self.run_count > 1:
                t.sleep(retry_relay)
            with self.train_lock:
                self._run(*args)
            self.run_count += 1
            return self.run_backtest()

        except RedLockError as e:
            alog.info(e)
            t.sleep(retry_relay)
            return self.run(*args)

    def _run(self, trial: Trial):
        self.trial = trial

        alog.info(f'### trial number {trial.number} ###')

        if self.clear_runs < trial.number and self.clear_runs > 0:
            self.clear()

        tf.keras.backend.clear_session()

        def multiples(m, count, min_val=0):
            results = []
            for i in range(0, count * m, m):
                if i > min_val:
                    results.append(i)
            return results

        hparams = dict(
            positive_change_quantile=trial.suggest_float(
                'positive_change_quantile', 0.84, 0.95),
            negative_change_quantile=trial.suggest_float(
                'negative_change_quantile', 0.73, 0.95),
            flat_ratio=trial.suggest_float('flat_ratio', 1.05, 1.23),
            #learning_rate=trial.suggest_float('learning_rate', 0.04, 0.06),
            #round_decimals=trial.suggest_int('round_decimals', 5, 8),
            #num_conv=trial.suggest_int('num_conv', 3, 8),
            # depth=trial.suggest_categorical('depth', multiples(2, 60, 22)),
        )

        # self.sequence_length = hparams['sequence_length']
        # self.output_depth = hparams['depth']
        self.positive_change_quantile=hparams['positive_change_quantile']
        self.negative_change_quantile=hparams['negative_change_quantile']
        #self.round_decimals = hparams['round_decimals']

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
                'batch_size': 2,
                'depth': self.output_depth,
                'directory': trial.number,
                'epochs': 1,
                'eval_df': eval_df,
                'export_model': True,
                'relu_alpha': 0.294,
                'learning_rate': 0.048254,
                'round_decimals': self.round_decimals,
                'sequence_length': self.sequence_length,
                'lstm_layers': 1,
                'base_filter_size': 16,
                'symbol': self.symbol,
                'dir_name': self.symbol,
                'train_df': train_df,
                'num_conv': 6
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
        kwargs = self._kwargs.copy()
        kwargs['sequence_length'] = self.sequence_length
        kwargs['depth'] = self.output_depth

        self.backtest.quantile = self.quantile

        self.backtest.trial = self.trial

        tf.keras.backend.clear_session()

        with tf.summary.create_file_writer(self.run_dir).as_default():
            trial = self.trial
            exported_model_path = self.exported_model_path

            test_df = self.backtest.test(self.model_version)

            alog.info(test_df)

            try:
                if trial.study.best_trial.value < self.backtest.capital:
                    shutil.rmtree(self.best_model_dir, ignore_errors=True)
                    shutil.copytree(exported_model_path, self.best_model_dir)
            except ValueError:
                pass

            if self.min_capital > self.backtest.capital:
                shutil.rmtree(self.run_dir, ignore_errors=True)
                shutil.rmtree(self.model_dir, ignore_errors=True)

            if self.backtest.capital == 1.0:
                return 0.0

            return self.backtest.capital

    def split_gpu(self):
        physical_devices = tf.config.list_physical_devices('GPU')

        if len(physical_devices) > 0:
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
@click.option('--depth', '-d', default=76, type=int)
@click.option('--clear-runs', '-c', default=0, type=int)
@click.option('--group-by', '-g', default='1m', type=str)
@click.option('--group-by-min', default='1m', type=str)
@click.option('--interval', '-i', default='1h', type=str)
@click.option('--macd-session-limit', default=200, type=int)
@click.option('--memory', '-m', default=1000, type=int)
@click.option('--min-capital', default=1.0, type=float)
@click.option('--num-locks', '-n', default=0, type=int)
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
