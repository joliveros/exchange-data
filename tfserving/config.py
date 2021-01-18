#!/usr/bin/env python
import os
from pathlib import Path
from tensorflow_serving.sources.storage_path.file_system_storage_path_source_pb2 \
    import FileSystemStoragePathSourceConfig
from tfserving.model_server_config_pb2 import ModelServerConfig

import alog
import click


class EDModelServerConfig(object):
    def __init__(self, symbol, path, base_path=None):
        self.name = symbol

        if 'best' in base_path:
            self.name = f'best_{symbol}'

        self.base_path = f'/models/{path}'

        if base_path:
            self.base_path = f'/{base_path}/{path}'

        self.base_path = '/root' + str(self.base_path)

        self.model_platform = 'tensorflow'

        self.model_version_policy.all.CopyFrom(
            FileSystemStoragePathSourceConfig.ServableVersionPolicy.All())


def config_for_model_dirs(exported_model_dirs, exp_dir, server_config, kwargs):
    exported_models = [
        (p.name, p.name.split('_')[0]) for p in exported_model_dirs
    ]

    exported_models = [(path, symbol) for path, symbol in exported_models
                        if len(os.listdir(Path().home() / exp_dir / path)) > 0]

    for path, symbol in exported_models:
        config = server_config.model_config_list.config.add()

        EDModelServerConfig.__init__(config, symbol, path, exp_dir,
                                     **kwargs)


@click.command()
def main(**kwargs):
    base_dir = '.exchange-data/models'
    models_dir = Path().home() / '.exchange-data/models'
    exported_model_dirs = list(models_dir.rglob('*_export'))
    best_model_dir = '.exchange-data/best_exported_models'
    best_model_dirs = list((Path().home() / best_model_dir).rglob(
        '*_export'))

    server_config: ModelServerConfig = ModelServerConfig()

    config_for_model_dirs(exported_model_dirs, base_dir, server_config, kwargs)

    config_for_model_dirs(best_model_dirs, best_model_dir, server_config,
                          kwargs)

    base_config_dir = Path().home() / Path('.exchange-data/models/')
    base_config_dir.mkdir(exist_ok=True, parents=True)

    config_file = base_config_dir / 'models.config'

    alog.info(config_file)

    if config_file.exists():
        config_file.unlink()

    alog.info(server_config)

    with open(config_file, 'wb') as f:
        f.write(str(server_config).encode())


if __name__ == '__main__':
    main()
