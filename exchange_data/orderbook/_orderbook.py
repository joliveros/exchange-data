import alog

from ._ordertree import OrderTree
from collections import deque  # a faster insert/pop queue
from exchange_data import Buffer
from exchange_data.orderbook import OrderType, OrderBookSide, Order, Trade, \
    TradeSummary, TradeParty
from exchange_data.orderbook.exceptions import OrderExistsException, \
    PriceDoesNotExistException
from typing import Callable

import datetime
import time


class OrderBook(object):
    asks: OrderTree
    bids: OrderTree

    def __init__(self, tick_size=0.0001, **kwargs):
        # Index[0] is most recent trade
        self.tape = deque(maxlen=10)
        self.bids = OrderTree()
        self.asks = OrderTree()
        self.last_tick = None
        self.last_timestamp = 0
        self.tick_size = tick_size
        self.time = 0
        self._next_order_id = 0
        self.last_trades = []

        super().__init__()

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

    @property
    def last_date(self):
        return datetime.datetime.fromtimestamp(self.last_timestamp/1000)

    def update_time(self):
        self.time += 1

    def process_order(self, order: Order) -> TradeSummary:
        self.last_timestamp = order.timestamp

        if order.uid is not None:
            self.next_order_id = order.uid + 1

        if order.type == OrderType.MARKET:
            return self._process_market_order(order)
        elif order.type == OrderType.LIMIT:
            return self._process_limit_order(order)

    def _process_limit_order(self, order: Order):
        price = order.price
        quantity_to_trade = order.quantity
        side = order.side
        trades = []

        if order.uid is None:
            order.uid = self.next_order_id

        if side == OrderBookSide.BID:
            return self.bid_limit_order(order,
                                        price,
                                        quantity_to_trade,
                                        side, trades)

        else:
            return self.ask_limit_order(order,
                                        price,
                                        quantity_to_trade,
                                        side, trades)

    def ask_limit_order(self, order, price, quantity_to_trade,
                        side, trades):
        if self.bids.max_price() is not None:
            while self.bids and price <= self.bids.max_price() and \
                    quantity_to_trade > 0:
                trade_summary = \
                    self._process_order_list(
                        side,
                        quantity_to_trade,
                        order
                    )

                quantity_to_trade = trade_summary.quantity_to_trade
                trades += trade_summary.trades

        # If volume remains, need to update the book with new quantity
        if quantity_to_trade > 0:
            self.asks.insert_order(order)

        return TradeSummary(quantity_to_trade, trades, order)

    def bid_limit_order(self, order, price, quantity_to_trade,
                        side, trades):
        if self.asks.min_price() is not None:
            while self.asks and price >= self.asks.min_price() and \
                    quantity_to_trade > 0:
                trade_summary = \
                    self._process_order_list(
                        side,
                        quantity_to_trade,
                        order
                    )

                quantity_to_trade = trade_summary.quantity_to_trade
                trades += trade_summary.trades

        # If volume remains, need to update the book with new quantity
        if quantity_to_trade > 0:
            self.bids.insert_order(order)

        return TradeSummary(quantity_to_trade, trades, order)

    def _process_order_list(self, side: OrderBookSide, quantity,
                            order) -> TradeSummary:
        """
        Takes an OrderList (stack of orders at one price) and an incoming
        order and matches appropriate trades given the order's quantity.
        """
        if side == OrderBookSide.BID:
            order_list = self.asks.min_price_list
        else:
            order_list = self.bids.max_price_list

        trades = []

        for trade in self._trades(order, order_list, quantity):
            self.tape.append(trade)
            trades.append(trade)
            quantity = trade.remaining

        self.last_trades = trades

        return TradeSummary(quantity, trades)

    def _trades(self, order: Order, order_list: Callable, quantity: int):
        while order_list() and quantity > 0:
            _order_list = order_list()
            head_order = _order_list.get_head_order()
            traded_price = head_order.price
            counter_party = head_order.uid
            new_book_quantity = None
            side = order.side

            head_price = _order_list.head_order.price

            if head_price != order.price:
                order.price = head_price
                # raise PriceDoesNotExistException()

            if quantity < head_order.quantity:
                traded_quantity = quantity
                # Do the transaction
                new_book_quantity = head_order.quantity - quantity
                head_order.update_quantity(new_book_quantity)
                quantity = 0


            elif quantity == head_order.quantity:
                traded_quantity = quantity

                if side == OrderBookSide.BID:
                    self.asks.remove_order_by_id(head_order.uid)
                else:
                    self.bids.remove_order_by_id(head_order.uid)

                quantity = 0

            # quantity to trade is larger than the head order
            else:
                traded_quantity = head_order.quantity

                if side == OrderBookSide.BID:
                    self.asks.remove_order_by_id(head_order.uid)
                else:
                    self.bids.remove_order_by_id(head_order.uid)
                quantity -= traded_quantity

            # transaction_record
            if side == OrderBookSide.BID:
                party1 = TradeParty(counter_party,
                                    OrderBookSide.ASK)

                party2 = TradeParty(order.uid, side)

                trade = Trade(party1, party2, quantity, traded_quantity,
                              traded_price)
            else:
                party1 = TradeParty(counter_party,
                                    OrderBookSide.BID)
                party2 = TradeParty(order.uid, side)

                trade = Trade(party1, party2, quantity, traded_quantity,
                              traded_price)

            yield trade

    def _process_market_order(self, order: Order):
        trades = []
        quantity_to_trade = order.quantity
        side = order.side

        if side == OrderBookSide.BID:
            return self._bid_market_order(order,
                                          quantity_to_trade,
                                          side, trades)
        else:
            return self._ask_market_order(order,
                                          quantity_to_trade,
                                          side, trades)

    def _ask_market_order(self, order, quantity_to_trade, side, trades):
        trade_summary = None
        while quantity_to_trade > 0 and self.bids:
            trade_summary = \
                self._process_order_list(
                    side,
                    quantity_to_trade,
                    order)
            trades += trade_summary.trades
            quantity_to_trade = trade_summary.quantity_to_trade
            trade_summary = TradeSummary(trade_summary.quantity_to_trade,
                                         trades)
        return trade_summary

    def _bid_market_order(self, order, quantity_to_trade, side, trades):
        trade_summary = None

        while quantity_to_trade > 0 and self.asks:
            trade_summary = \
                self._process_order_list(
                    side,
                    quantity_to_trade,
                    order)

            trades += trade_summary.trades
            quantity_to_trade = trade_summary.quantity_to_trade
            trade_summary = TradeSummary(trade_summary.quantity_to_trade,
                                         trades)

        return trade_summary

    def cancel_order(self, order_id: int):
        if self.bids.order_exists(order_id):
            self.bids.remove_order_by_id(order_id)
        elif self.asks.order_exists(order_id):
            self.asks.remove_order_by_id(order_id)
        else:
            raise OrderExistsException()

    def modify_order(self, order_id: int, price: float, quantity: float,
                     timestamp: int=None):

        if self.bids.order_exists(order_id):
            self.bids.modify_order(order_id, price, quantity, timestamp)

        elif self.asks.order_exists(order_id):
            self.asks.modify_order(order_id, price, quantity, timestamp)
        else:
            raise OrderExistsException()

    def order_exists(self, order_id):
        if self.bids.order_exists(order_id):
            return True
        elif self.asks.order_exists(order_id):
            return True
        else:
            return False

    def get_volume(self, price: float):
        if self.bids.price_exists(price):
            return self.bids.get_price_list(price).volume
        elif self.asks.price_exists(price):
            return self.asks.get_price_list(price).volume
        else:
            raise PriceDoesNotExistException()

    def get_best_bid(self):
        best_bid = self.bids.max_price()
        return best_bid if best_bid else 0.0

    def get_worst_bid(self):
        return self.bids.min_price()

    def get_best_ask(self):
        best_ask = self.asks.min_price()
        return best_ask if best_ask else 0.0

    @property
    def spread(self):
        return self.get_best_ask() - self.get_best_bid()

    def get_worst_ask(self):
        return self.asks.max_price()

    def print(self, depth: int = 0, trades: bool = False):
        bid_levels = list(self.bids.price_tree.items(reverse=True))
        ask_levels = list(self.asks.price_tree.items())

        if depth > 0:
            bid_levels = bid_levels[:depth]
            ask_levels = list(reversed(ask_levels[:depth]))

        summary = Buffer()
        summary.newline()

        summary.section('Asks')
        summary.newline()

        if self.asks is not None and len(self.asks) > 0:
            for key, value in ask_levels:
                summary.write('%s' % value)

        summary.newline()

        if self.bids is not None and len(self.bids) > 0:
            for key, value in bid_levels:
                summary.write('%s' % value)

        summary.newline()

        summary.section('Bids')

        summary.newline()

        if trades:
            summary.section('Trades')
            summary.newline()

            if len(self.tape) > 0:
                for trade in self.tape:
                    line = f'{trade.price} @ {trade.quantity} ' \
                           f'{trade.timestamp} {trade.party1} / {trade.party2}'

                    summary.newline()
                    summary.write(line)

            summary.newline()

        return str(summary)

    def __str__(self):
        return self.print(depth=25, trades=False)
