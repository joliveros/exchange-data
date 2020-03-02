import alog
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.models import Model


class ActorModel(Model):
    def __init__(self, num_action, **kwargs):
        super(ActorModel, self).__init__(**kwargs)
        self.layer_a1 = Dense(64, activation='relu')
        self.layer_a2 = Dense(64, activation='relu')
        self.logits = Dense(num_action, activation='softmax')

    def call(self, state):
        layer_a1 = self.layer_a1(state)
        layer_a2 = self.layer_a2(layer_a1)
        alog.info(layer_a2.shape)
        logits = self.logits(layer_a2)
        alog.info(logits.shape)
        return logits
