from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.models import Model


class CriticModel(Model):
    def __init__(self, **kwargs):
        super(CriticModel, self).__init__(**kwargs)
        self.layer_c1 = Dense(64, activation='relu')
        self.layer_c2 = Dense(64, activation='relu')
        self.value = Dense(1)

    def call(self, state):
        layer_c1 = self.layer_c1(state)
        layer_c2 = self.layer_c2(layer_c1)
        value = self.value(layer_c2)
        return value
