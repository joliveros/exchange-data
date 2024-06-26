from cached_property import cached_property_with_ttl
from optuna import Trial
from optuna import load_study
from optuna.storages import RDBStorage
from optuna.pruners import MedianPruner
from os import environ
from pathlib import Path
from redis_collections import Dict

import alog
import logging
import optuna
import pandas as pd
import sec
import sys


class StudyWrapper(object):
    def __init__(self, symbol, **kwargs):
        optuna.logging.get_logger("optuna")\
            .addHandler(logging.StreamHandler(sys.stdout))

        self.symbol = symbol
        self.base_path = f'{Path.home()}/.exchange-data/models/'

        self.base_model_dir = f'{Path.home()}/.exchange-data/models' \
                              f'/{self.symbol}'

        db_conn_str = sec.load('KERAS_DB', lowercase=False)

        storage = RDBStorage(db_conn_str,
                             engine_kwargs={
                                 "pool_pre_ping": True,
                                 "connect_args": {
                                     "keepalives": 1,
                                     "keepalives_idle": 30,
                                     "keepalives_interval": 10,
                                     "keepalives_count": 5,
                                 }
                             })

        try:
            self.study = \
                load_study(study_name=self.symbol, storage=storage)
        except KeyError:
            self.create_study(storage=storage,
                              pruner=MedianPruner())

    def create_study(self, **kwargs):
        self.study = optuna.create_study(
            study_name=self.symbol, direction='maximize',
            sampler=optuna.samplers.NSGAIISampler(population_size=5), **kwargs)

    def save_best_params(self):
        self.best_study_params = vars(self.study.best_trial)

    @property
    def best_study_params(self):
        return Dict(key=f'best_key_{self.symbol}', redis=self.redis_client)

    @best_study_params.setter
    def best_study_params(self, values):
        self._best_study_params = \
            Dict(key=f'best_key_{self.symbol}', data=values,
                 redis=self.redis_client)
    @property
    def best_trial(self):
        best_trial_id = self.best_trial_id

        return Trial(trial_id=best_trial_id, study=self.study)

    @property
    def best_tuned_trial(self):
        best_trial_id = self.best_tuned_trial_id

        return Trial(trial_id=best_trial_id, study=self.study)

    @cached_property_with_ttl(ttl=60)
    def best_trial_params(self):
        trial = self.best_trial

        params = {**trial.params, **trial.user_attrs}

        return params

    @cached_property_with_ttl(ttl=60)
    def best_tuned_trial_params(self):
        trial = Trial(trial_id=self.best_tuned_trial_id, study=self.study)

        alog.info(alog.pformat(trial.__dict__))

        params = {**trial.params, **trial.user_attrs}

        return params

    @cached_property_with_ttl(ttl=60)
    def best_trial_id(self):
        df = self.study.trials_dataframe()
        df = df[df['value'] > 0.0]

        pd.set_option('display.max_rows', len(df) + 1)

        if not df.empty:
            df = df[df['user_attrs_tuned'] == False]
            df = df[df['state'] == 'COMPLETE']

        if len(df) == 0:
            raise NotEnoughTrialsException()

        # row_id = df[['value']].idxmax()['value']

        trial_id = int(df.iloc[-1]['number']) + 1

        alog.info(df)

        return trial_id

    @cached_property_with_ttl(ttl=60)
    def best_tuned_trial_id(self):
        df = self.study.trials_dataframe()
        df = df[df['value'] > 0.0]

        pd.set_option('display.max_rows', len(df) + 1)

        if not df.empty:
            df = df[df['user_attrs_tuned'] == True]
            df = df[df['state'] == 'COMPLETE']

        if len(df) == 0:
            raise NotEnoughTrialsException()

        row_id = df[['value']].idxmax()['value']

        trial_id = int(df.loc[row_id]['number']) + 1

        alog.info(df)

        return trial_id
