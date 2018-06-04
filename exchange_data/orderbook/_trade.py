import alog

from exchange_data.orderbook import OrderBookSide, Order


class TradeParty(object):
    def __init__(self, counter_party_id: int, side: OrderBookSide):
        self.counter_party_id = counter_party_id
        self.side = side

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return alog.pformat(self.__dict__)


class Trade(object):
    def __init__(self, party1: TradeParty, party2: TradeParty,
                 quantity: float):
        self.quantity = quantity
        self.party2 = party2
        self.party1 = party1

    def __str__(self):
        return alog.pformat(self.__dict__)


class TradeSummary(object):
    def __init__(self, quantity_to_trade: float, trades: [Trade], order: Order=None):
        self.order = order
        self.quantity_to_trade = quantity_to_trade
        self.trades = trades

    def __str__(self):
        return alog.pformat(self.__dict__)

