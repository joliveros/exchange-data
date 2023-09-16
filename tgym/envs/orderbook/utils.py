import alog
import yaml
from yaml.representer import SafeRepresenter


class Logging(object):
    def __init__(self, **kwargs):
        super().__init__()
        yaml.add_representer(float, SafeRepresenter.represent_float)

    def yaml(self, value: dict):
        trades = value['trades']

        if len(trades) > 0:
            trades = [str(t) for t in trades]

        value['trades'] = trades

        alog.info('\n' + yaml.dump(value))

def import_by_string(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
