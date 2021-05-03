import yaml
from yaml.representer import SafeRepresenter


class Logging(object):
    def __init__(self):
        yaml.add_representer(float, SafeRepresenter.represent_float)

    def yaml(self, value: dict):
        return yaml.dump(value)