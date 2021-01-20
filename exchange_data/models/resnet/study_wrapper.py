from pathlib import Path

import alog
import optuna
from optuna import load_study


class StudyWrapper(object):
    study_db_path: Path = None

    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        self.base_path = f'{Path.home()}/.exchange-data/models/'
        self.study_db_path = f'{Path.home()}/.exchange-data/models/{self.symbol}.db'
        self.study_db_path = Path(self.study_db_path)
        db_conn_str = f'sqlite:///{self.study_db_path}'

        if not self.study_db_path.exists():
            self.create_study(db_conn_str)
        else:
            try:
                self.study = load_study(study_name=self.symbol, storage=db_conn_str)
            except KeyError as e:
                self.create_study(db_conn_str)

    def create_study(self, db_conn_str):
        self.study = optuna.create_study(
            study_name=self.symbol, direction='maximize',
            storage=db_conn_str)
