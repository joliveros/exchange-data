from exchange_data.utils import NoValue


class Actions(NoValue):
    Buy = 0
    Hold = 1
    Sell = 2


class Positions(NoValue):
    Flat = 0
    Short = 1
    Long = 2
