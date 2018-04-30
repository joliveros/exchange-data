import alog

from exchange_data.orderbook._transaction_party import TransactionParty


class Transaction(object):
    def __init__(self, party1: TransactionParty, party2: TransactionParty,
                 quantity: float):
        self.quantity = quantity
        self.party2 = party2
        self.party1 = party1

    def __str__(self):
        return alog.pformat(self.__dict__)
