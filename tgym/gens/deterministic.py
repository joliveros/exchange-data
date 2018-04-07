import numpy as np
from tgym.core import DataGenerator


class WavySignal(DataGenerator):
    """Modulated sine generator
    """

    def __init__(self, period_1, period_2, epsilon, ba_spread=0):
        super()
        self.ba_spread = ba_spread
        self.epsilon = epsilon
        self.period_2 = period_2
        self.period_1 = period_1

    def _generator(self):
        i = 0
        while True:
            i += 1
            bid_price = (1 - self.epsilon) * np.sin(2 * i * np.pi / self.period_1) + \
                        self.epsilon * np.sin(2 * i * np.pi / self.period_2)
            yield bid_price, bid_price + self.ba_spread
