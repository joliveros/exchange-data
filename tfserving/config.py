#!/usr/bin/env python

from pathlib import Path
from tensorflow_serving.sources.storage_path.file_system_storage_path_source_pb2 \
    import FileSystemStoragePathSourceConfig
from tfserving.model_server_config_pb2 import ModelServerConfig

import alog
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

        config.model_version_policy.all.CopyFrom(
            FileSystemStoragePathSourceConfig.ServableVersionPolicy.All())

        config = server_config.model_config_list.config.add()
        config.name = f'best_{symbol}'
        config.base_path = f'/best_exported_models/{symbol}'
        config.model_platform = 'tensorflow'

        config.model_version_policy.all.CopyFrom(
            FileSystemStoragePathSourceConfig.ServableVersionPolicy.All())

    config_file = Path.home() / Path('.exchange-data/models/models.config')

    if config_file.exists():
        config_file.unlink()

    alog.info(server_config)

    with open(config_file, 'wb') as f:
        f.write(str(server_config).encode())


if __name__ == '__main__':
    main()
