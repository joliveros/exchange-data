from pathlib import Path
from optuna import load_study
from redis_collections import Dict

import alog
import optuna





class StudyWrapper(object):
    study_db_path: Path = None

    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        self.base_path = f'{Path.home()}/.exchange-data/models/'

        self.base_model_dir = f'{Path.home()}/.exchange-data/models' \
                             f'/{self.symbol}'

        self.study_db_path = \
            f'{Path.home()}/.exchange-data/models/{self.symbol}.db'
        self.study_db_path = Path(self.study_db_path)
        db_conn_str = f'sqlite:///{self.study_db_path}'

        if not self.study_db_path.exists():
            self.create_study(db_conn_str)
        else:
            try:
                self.study = \
                    load_study(study_name=self.symbol, storage=db_conn_str)
            except KeyError:
                self.create_study(db_conn_str)

    def create_study(self, db_conn_str):
        self.study = optuna.create_study(
            study_name=self.symbol, direction='maximize',
            storage=db_conn_str, sampler=optuna.samplers.CmaEsSampler)

    def save_best_params(self):
        self.best_study_params = vars(self.study.best_trial)

    @property
    def best_study_params(self):
        return Dict(key=f'best_key_{self.symbol}', redis=self.redis_client)

    @best_study_params.setter
    def best_study_params(self, values):
        self._best_study_params = \
            Dict(key=f'best_key_{self.symbol}', data=values, redis=self.redis_client)
