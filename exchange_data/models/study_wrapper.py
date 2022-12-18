from os import environ
from pathlib import Path
from optuna import load_study
from redis_collections import Dict

import alog
import optuna
from optuna.storages import RDBStorage


class StudyWrapper(object):
    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        self.base_path = f'{Path.home()}/.exchange-data/models/'

        self.base_model_dir = f'{Path.home()}/.exchange-data/models' \
                             f'/{self.symbol}'

        db_conn_str = environ.get('KERAS_DB')
        storage = RDBStorage(db_conn_str)

        try:
            self.study = \
                load_study(study_name=self.symbol, storage=storage)
        except KeyError:
            self.create_study(storage=storage)

    def create_study(self, **kwargs):
        self.study = optuna.create_study(
            study_name=self.symbol, direction='maximize',
            sampler=optuna.samplers.CmaEsSampler, **kwargs)

    def save_best_params(self):
        self.best_study_params = vars(self.study.best_trial)

    @property
    def best_study_params(self):
        return Dict(key=f'best_key_{self.symbol}', redis=self.redis_client)

    @best_study_params.setter
    def best_study_params(self, values):
        self._best_study_params = \
            Dict(key=f'best_key_{self.symbol}', data=values, redis=self.redis_client)
