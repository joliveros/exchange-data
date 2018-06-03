import alog

from exchange_data.orderbook import OrderBookSide


class TransactionParty(object):
    def __init__(self, counter_party_id: int, new_book_quantity: float,
                 side: OrderBookSide):
        self.counter_party_id = counter_party_id
        self.new_book_quantity = new_book_quantity
        self.side = side

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return alog.pformat(self.__dict__)
