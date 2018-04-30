import sys
import time
from collections import deque  # a faster insert/pop queue

import alog
from six.moves import cStringIO as StringIO
from decimal import Decimal

from exchange_data.orderbook import OrderType, OrderBookSide, Order
from exchange_data.orderbook._ordertree import OrderTree
from exchange_data.orderbook._transaction_party import TransactionParty
from exchange_data.orderbook._transcation import Transaction
from ._ordertree import OrderTree


class OrderBook(object):
    asks: OrderTree
    bids: OrderTree

    def __init__(self, tick_size=0.0001):
        self.tape = deque(maxlen=0)  # Index[0] is most recent trade
        self.bids = OrderTree()
        self.asks = OrderTree()
        self.last_tick = None
        self.last_timestamp = 0
        self.tick_size = tick_size
        self.time = 0
        self._next_order_id = 0

    @property
    def next_order_id(self):
        uid = self._next_order_id
        self._next_order_id = self._next_order_id + 1

        return uid

    @next_order_id.setter
    def next_order_id(self, value):
        self._next_order_id = value

    @property
    def timestamp(self):
        self.last_timestamp = time.time()
        return self.last_timestamp

    def update_time(self):
        self.time += 1

    def process_order(self, order: object) -> object:
        order_in_book = None
        trades = []

        order.timestamp = self.timestamp

        if order.uid is not None:
            self.next_order_id = order.uid

        if order.type == OrderType.MARKET:
            trades = self.process_market_order(order)
        elif order.type == OrderType.LIMIT:
            trades, order_in_book = self.process_limit_order(order)

        return trades, order_in_book

    def process_limit_order(self, order: Order):
        order_in_book = None
        price = order.price
        quantity_to_trade = order.quantity
        side = order.side
        trades = []

        if order.uid is None:
            order.uid = self.next_order_id

        if side == OrderBookSide.BID:
            while self.asks and price >= self.asks.min_price() and \
                    quantity_to_trade > 0:
                quantity_to_trade, new_trades = \
                    self._process_order_list(
                        side,
                        quantity_to_trade,
                        order
                    )

                trades += new_trades

            # If volume remains, need to update the book with new quantity
            if quantity_to_trade > 0:
                self.bids.insert_order(order)
                order_in_book = order

        elif side == OrderBookSide.ASK:
            while self.bids and price <= self.bids.max_price() and \
                    quantity_to_trade > 0:
                quantity_to_trade, new_trades = \
                    self._process_order_list(
                        side,
                        quantity_to_trade,
                        order
                    )

                trades += new_trades

            # If volume remains, need to update the book with new quantity
            if quantity_to_trade > 0:
                self.asks.insert_order(order)
                order_in_book = order

        return trades, order_in_book

    def _process_order_list(
            self,
            side: OrderBookSide,
            quantity,
            order
    ):
        """
        Takes an OrderList (stack of orders at one price) and an incoming
        order and matches appropriate trades given the order's quantity.
        """
        if side == OrderBookSide.BID:
            order_list = self.bids.min_price_list()
        else:
            order_list = self.asks.max_price_list()

        trades = []

        for transaction in self._transactions(order, order_list, quantity):
            self.tape.append(transaction)
            trades.append(transaction)
            quantity = transaction.quantity

        return quantity, trades

    def _transactions(self,
                      order,
                      order_list,
                      quantity
                      ):
        alog.debug(order)
        alog.debug(order_list)
        while len(order_list) > 0 and quantity > 0:
            head_order = order_list.get_head_order()
            traded_price = head_order.price
            counter_party = head_order.uid
            new_book_quantity = None
            side = order.side

            if quantity < head_order.quantity:
                traded_quantity = quantity
                # Do the transaction
                new_book_quantity = head_order.quantity - quantity
                head_order.update_quantity(new_book_quantity,
                                           head_order.timestamp)
                quantity = 0

            elif quantity == head_order.quantity:
                traded_quantity = quantity

                if side == OrderBookSide.BID:
                    self.bids.remove_order_by_id(head_order.uid)
                else:
                    self.asks.remove_order_by_id(head_order.uid)

                quantity = 0

            else:  # quantity to trade is larger than the head order
                traded_quantity = head_order.quantity

                if side == OrderBookSide.BID:
                    self.bids.remove_order_by_id(head_order.uid)
                else:
                    self.asks.remove_order_by_id(head_order.uid)

                quantity -= traded_quantity

            summary = {
                'time': self.time,
                'price': traded_price,
                'quantity': traded_quantity,
                'uid': counter_party,
                'matching_uid': order.uid
            }

            alog.debug(alog.pformat(summary))

            transaction = {
                'timestamp': self.time,
                'price': traded_price,
                'quantity': traded_quantity,
                'time': self.time
            }

            # transaction_record
            if side == OrderBookSide.BID:
                party1 = TransactionParty(counter_party,
                                          new_book_quantity,
                                          OrderBookSide.ASK)

                party2 = TransactionParty(order.uid, new_book_quantity, side)

                transaction = Transaction(party1, party2, quantity)
            else:
                transaction['party1'] = [counter_party, side,
                                         head_order.uid,
                                         new_book_quantity]

                transaction['party2'] = [order.uid, OrderBookSide.BID,
                                         None,
                                         None]

            yield transaction

    def process_market_order(self, order: Order):
        trades = []
        quantity_to_trade = order.quantity
        side = order.side

        if side == OrderBookSide.BID:
            while quantity_to_trade > 0 and self.asks:
                quantity_to_trade, new_trades = \
                    self._process_order_list(
                        side,
                        quantity_to_trade,
                        order)

                trades += new_trades

        elif side == OrderBookSide.ASK:
            while quantity_to_trade > 0 and self.bids:
                quantity_to_trade, new_trades = \
                    self._process_order_list(
                        side,
                        quantity_to_trade,
                        order)

                trades += new_trades

        return trades

    def cancel_order(self, side, order_id, time=None):
        if time:
            self.time = time
        else:
            self.update_time()
        if side == 'bid':
            if self.bids.order_exists(order_id):
                self.bids.remove_order_by_id(order_id)
        elif side == 'ask':
            if self.asks.order_exists(order_id):
                self.asks.remove_order_by_id(order_id)
        else:
            sys.exit('cancel_order() given neither "bid" nor "ask"')

    def modify_order(self, order_id, order_update, time=None):
        if time:
            self.time = time
        else:
            self.update_time()
        side = order_update['side']
        order_update['order_id'] = order_id
        order_update['timestamp'] = self.time
        if side == 'bid':
            if self.bids.order_exists(order_update['order_id']):
                self.bids.update_order(order_update)
        elif side == 'ask':
            if self.asks.order_exists(order_update['order_id']):
                self.asks.update_order(order_update)
        else:
            sys.exit('modify_order() given neither "bid" nor "ask"')

    def get_volume_at_price(self, side, price):
        price = Decimal(price)
        if side == 'bid':
            volume = 0
            if self.bids.price_exists(price):
                volume = self.bids.get_price(price).volume
            return volume
        elif side == 'ask':
            volume = 0
            if self.asks.price_exists(price):
                volume = self.asks.get_price(price).volume
            return volume
        else:
            sys.exit('get_volume_at_price() given neither "bid" nor "ask"')

    def get_best_bid(self):
        return self.bids.max_price()

    def get_worst_bid(self):
        return self.bids.min_price()

    def get_best_ask(self):
        return self.asks.min_price()

    def get_worst_ask(self):
        return self.asks.max_price()

    def tape_dump(self, filename, filemode, tapemode):
        dumpfile = open(filename, filemode)
        for tapeitem in self.tape:
            dumpfile.write(
                'Time: %s, Price: %s, Quantity: %s\n' % (tapeitem['time'],
                                                         tapeitem['price'],
                                                         tapeitem['quantity']))
        dumpfile.close()
        if tapemode == 'wipe':
            self.tape = []

    def __str__(self):
        tempfile = StringIO()
        tempfile.write("***Bids***\n")
        if self.bids != None and len(self.bids) > 0:
            for key, value in self.bids.price_tree.items(reverse=True):
                tempfile.write('%s' % value)
        tempfile.write("\n***Asks***\n")
        if self.asks != None and len(self.asks) > 0:
            for key, value in list(self.asks.price_tree.items()):
                tempfile.write('%s' % value)
        tempfile.write("\n***Trades***\n")
        if self.tape != None and len(self.tape) > 0:
            num = 0
            for entry in self.tape:
                if num < 10:  # get last 5 entries
                    tempfile.write(str(entry['quantity']) + " @ " + str(
                        entry['price']) + " (" + str(
                        entry['timestamp']) + ") " + str(
                        entry['party1'][0]) + "/" + str(
                        entry['party2'][0]) + "\n")
                    num += 1
                else:
                    break
        tempfile.write("\n")
        return tempfile.getvalue()
