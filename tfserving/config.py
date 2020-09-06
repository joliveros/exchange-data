#!/usr/bin/env python
import alog

from exchange_data.emitters.binance._trade_executor import TradeExecutor
from pathlib import Path
from tfserving.model_server_config_pb2 import ModelServerConfig

import click


@click.command()
def main(**kwargs):
    models_dir = Path().home() / '.exchange-data/models/'
    exported_model_dirs = list(models_dir.rglob('*_export'))

    exported_models = [
        (p.name, p.name.split('_')[0]) for p in exported_model_dirs
    ]

    server_config: ModelServerConfig = ModelServerConfig()

    for path, symbol in exported_models:
        config = server_config.model_config_list.config.add()
        config.name = symbol
        config.base_path = f'/models/{path}'
        config.model_platform = 'tensorflow'
        config.model_version_policy.All()

    config_file = Path('./tfserving/models.config')

    if config_file.exists():
        config_file.unlink()

    with open(config_file, 'wb') as f:
        f.write(str(server_config).encode())


if __name__ == '__main__':
    main()
