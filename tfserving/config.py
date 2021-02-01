#!/usr/bin/env python
import os
from pathlib import Path
from tensorflow_serving.sources.storage_path.file_system_storage_path_source_pb2 \
    import FileSystemStoragePathSourceConfig
from tfserving.model_server_config_pb2 import ModelServerConfig

import alog
import click


class EDModelServerConfig(object):
    def __init__(self, symbol, path):
        if 'best' in str(path):
            self.name = f'best_{symbol}'
        else:
            self.name = symbol

        self.base_path = str(path)
        self.base_path = self.base_path.replace('/home/joliveros', '/root')

        self.model_platform = 'tensorflow'

        self.model_version_policy.all.CopyFrom(
            FileSystemStoragePathSourceConfig.ServableVersionPolicy.All())


def config_for_model_dirs(exported_model_dirs, exp_dir, server_config, kwargs):
    exported_models = [
        (p.name, p.name.split('_')[0]) for p in exported_model_dirs
    ]
    exported_models = [(Path().home() / exp_dir / path, os.listdir(Path(

    ).home() / exp_dir / path), symbol)
                       for path, symbol in exported_models]

    expand_exported_models = []

    for base_path, paths, symbol in exported_models:
        for path in paths:
            expand_exported_models.append((base_path, base_path / path, symbol))

    exported_models = [(base_path, symbol) for base_path, path, symbol in
                       expand_exported_models
                        if len(os.listdir(Path().home() / exp_dir / path)) > 0]

    exported_models = set(exported_models)

    for path, symbol in exported_models:
        config = server_config.model_config_list.config.add()

        EDModelServerConfig.__init__(config, symbol, path, **kwargs)


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

    # alog.info(config_file)

    if config_file.exists():
        config_file.unlink()

    alog.info(server_config)

    with open(config_file, 'wb') as f:
        f.write(str(server_config).encode())


if __name__ == '__main__':
    main()
