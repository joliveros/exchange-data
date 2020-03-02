from pathlib import Path


class ModelDirectoryInfo(object):
    def __init__(
        self,
        directory_name=None,
        directory=None,
        **kwargs
    ):
        if directory_name is None:
            raise Exception()

        if directory is None:
            directory = f'{Path.home()}/.exchange-data/models/{directory_name}'

        self.directory_name = directory_name
        self.directory = Path(directory)

        if not self.directory.exists():
            self.directory.mkdir()

        super().__init__(**kwargs)

    def clear(self):
        self.directory.unlink()
