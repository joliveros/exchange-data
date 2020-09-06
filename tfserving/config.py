#!/usr/bin/env python

from exchange_data.emitters.binance._trade_executor import TradeExecutor
from pathlib import Path
from tfserving.model_server_config_pb2 import ModelServerConfig

import click


@click.command()
def main(**kwargs):
    symbols = TradeExecutor(base_asset='BNB',
                            database_name='binance').symbols

    server_config: ModelServerConfig = ModelServerConfig()

    for symbol in symbols:
        config = server_config.model_config_list.config.add()
        config.name = symbol
        config.base_path = f'/models/{symbol}_export'
        config.model_platform = 'tensorflow'
        config.model_version_policy.All()

    config_file = Path('./tfserving/models.config')

    if config_file.exists():
        config_file.unlink()

    with open(config_file, 'wb') as f:
        f.write(str(server_config).encode())


if __name__ == '__main__':
    main()
