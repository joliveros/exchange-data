import numpy as np


class Actions(list):
    Buy = np.array([0, 1, 0])
    Hold = np.array([1, 0, 0])
    Sell = np.array([0, 1, 0])

    def __init__(self):
        super().__init__()
        self.append(self.Buy)
        self.append(self.Hold)
        self.append(self.Sell)


class Positions(list):
    Flat = np.array([1, 0, 0])
    Long = np.array([0, 1, 0])
    Short = np.array([0, 1, 0])

    def __init__(self):
        super().__init__()
        self.append(self.Flat)
        self.append(self.Long)
        self.append(self.Short)
