"""
HFT-Orderbook

Limit Order Book for high-frequency trading (HFT), as described by WK Selph,
implemented in Python3.

Based on WK Selph's Blogpost:
http://howtohft.wordpress.com/2011/02/15/how-to-build-a-fast-limit-order-book/

Available at Archive.org's WayBackMachine:
(https://goo.gl/KF1SRm)


    "There are three main operations that a limit order book (LOB) has to
    implement: add, cancel, and execute.  The goal is to implement these
    operations in O(1) time while making it possible for the trading model to
    efficiently ask questions like “what are the best bid and offer?”, “how much
    volume is there between prices A and B?” or “what is order X’s current
    position in the book?”.

    The vast majority of the activity in a book is usually made up of add and
    cancel operations as market makers jockey for position, with executions a
    distant third (in fact I would argue that the bulk of the useful information
    on many stocks, particularly in the morning, is in the pattern of adds and
    cancels, not executions, but that is a topic for another post).  An add
    operation places an order at the end of a list of orders to be executed at
    a particular limit price, a cancel operation removes an order from anywhere
    in the book, and an execution removes an order from the inside of the book
    (the inside of the book is defined as the oldest buy order at the highest
    buying price and the oldest sell order at the lowest selling price).  Each
    of these operations is keyed off an id number (Order.idNumber in the
    pseudo-code below), making a hash table a natural structure for tracking
    them.

    Depending on the expected sparsity of the book (sparsity being the
    average distance in cents between limits that have volume, which is
    generally positively correlated with the instrument price), there are a
    number of slightly different implementations I’ve used.  First it will help
    to define a few objects:

        Order
          int idNumber;
          bool buyOrSell;
          int shares; // order size
          int limit;
          int entryTime;
          int eventTime;
          Order *nextOrder;
          Order *prevOrder;
          Limit *parentLimit;

        Limit  // representing a single limit price
          int limitPrice;
          int size;
          int totalVolume;
          Limit *parent;
          Limit *leftChild;
          Limit *rightChild;
          Order *headOrder;
          Order *tailOrder;

        Book
          Limit *buyTree;
          Limit *sellTree;
          Limit *lowestSell;
          Limit *highestBuy;

    The idea is to have a binary tree of Limit objects sorted by limitPrice,
    each of which is itself a doubly linked list of Order objects.  Each side
    of the book, the buy Limits and the sell Limits, should be in separate trees
    so that the inside of the book corresponds to the end and beginning of the
    buy Limit tree and sell Limit tree, respectively.  Each order is also an
    entry in a map keyed off idNumber, and each Limit is also an entry in a
    map keyed off limitPrice.

    With this structure you can easily implement these key operations with
    good performance:

    Add – O(log M) for the first order at a limit, O(1) for all others
    Cancel – O(1)
    Execute – O(1)
    GetVolumeAtLimit – O(1)
    GetBestBid/Offer – O(1)

    where M is the number of price Limits (generally << N the number of orders).
    Some strategy for keeping the limit tree balanced should be used because the
    nature of markets is such that orders will be being removed from one side
    of the tree as they’re being added to the other.  Keep in mind, though,
    that it is important to be able to update Book.lowestSell/highestBuy
    in O(1) time when a limit is deleted (which is why each Limit has a Limit
    *parent) so that GetBestBid/Offer can remain O(1)."

"""

# Import Built-Ins
from itertools import islice

import alog

from exchange_data.limit_orderbook import Order
from exchange_data.limit_orderbook._limit_level import LimitLevel
from exchange_data.limit_orderbook._limit_level_tree import LimitLevelTree
from exchange_data.utils import roundup_to_nearest


class LimitOrderBook:
    """Limit Order Book (LOB) implementation for High Frequency Trading

    Implementation as described by WK Selph (see header doc string for link).

    """

    def __init__(self):
        self.bids = LimitLevelTree()
        self.asks = LimitLevelTree()
        self.best_bid = None
        self.best_ask = None
        self._price_levels = {}
        self._orders = {}

    @property
    def top_level(self):
        """Returns the best available bid and ask.

        :return:
        """
        return self.best_bid, self.best_ask

    def process(self, order):
        """Processes the given order.

        If the order's size is 0, it is removed from the book.

        If its size isn't zero and it exists within the book, the order is updated.

        If it doesn't exist, it will be added.

        :param order:
        :return:
        """
        if order.size == 0:
            self.remove(order)
        else:
            try:
                self.update(order)
            except KeyError:
                self.add(order)

    def update(self, order):
        """Updates an existing order in the book.

        It also updates the order's related LimitLevel's size, accordingly.

        :param order:
        :return:
        """
        size_diff = self._orders[order.uid].size - order.size
        self._orders[order.uid].size = order.size
        self._orders[order.uid].parent_limit.size -= size_diff

    def remove(self, order):
        """Removes an order from the book.

        If the Limit Level is then empty, it is also removed from the book's
        relevant tree.

        If the removed LimitLevel was either the top bid or ask, it is replaced
        by the next best value (which is the LimitLevel's parent in an
        AVL tree).

        :param order:
        :return:
        """
        # Remove Order from self._orders
        try:
            popped_item = self._orders.pop(order.uid)
        except KeyError:
            return False

        # Remove order from its doubly linked list
        popped_item.pop_from_list()

        # Remove Limit Level from self._price_levels and tree, if no orders are
        # left within that limit level
        try:
            if len(self._price_levels[order.price]) == 0:
                popped_limit_level = self._price_levels.pop(order.price)
                # Remove Limit Level from LimitLevelTree
                if order.is_bid:
                    if popped_limit_level == self.best_bid:
                        if not isinstance(popped_limit_level.parent,
                                          LimitLevelTree):
                            self.best_bid = popped_limit_level.parent
                        else:
                            self.best_bid = None

                    popped_limit_level.remove()
                else:
                    if popped_limit_level == self.best_ask:
                        if not isinstance(popped_limit_level.parent,
                                          LimitLevelTree):
                            self.best_ask = popped_limit_level.parent
                        else:
                            self.best_ask = None
                    popped_limit_level.remove()
        except KeyError:
            pass

        return popped_item

    def add(self, order: Order):
        """
        Adds a new LimitLevel to the book and appends the given order to it.

        :param order: Order() Instance
        :return:
        """
        if order.price not in self._price_levels:
            limit_level = LimitLevel(order)
            self._orders[order.uid] = order
            self._price_levels[limit_level.price] = limit_level

            self.process_trades(limit_level, order)

            if order.is_bid:
                self.bids.insert(limit_level)
                self.update_best_bid(limit_level)

            else:
                self.asks.insert(limit_level)
                self.update_best_ask(limit_level)
        else:
            # The price level already exists, hence we need to append the order
            # to that price level
            self._orders[order.uid] = order
            self._price_levels[order.price].append(order)

    def process_trades(self, limit_level, order):
        if order.is_bid and self.best_ask:
            if limit_level.price > self.best_ask.price:
                # trade_size = order.size
                alog.debug(order)
        elif not order.is_bid and self.best_bid:
            if limit_level.price < self.best_bid.price:
                alog.debug(order)


    def update_best_ask(self, limit_level):
        if self.best_ask is None or limit_level.price < self.best_ask.price:
            self.best_ask = limit_level

    def update_best_bid(self, limit_level):
        if self.best_bid is None or limit_level.price > self.best_bid.price:
            self.best_bid = limit_level

    def levels(self, depth=None):
        """Returns the price levels as a dict {'bids': [bid1, ...], 'asks': [ask1, ...]}
        
        :param depth: Desired number of levels on each side to return.
        :return:
        """
        levels_sorted = sorted(self._price_levels.keys())

        bids_all = reversed([price_level for price_level in levels_sorted
                             if price_level < self.best_ask.price])

        bids = list(islice(bids_all, depth)) if depth else list(bids_all)

        asks_all = (price_level for price_level in levels_sorted if
                    price_level > self.best_bid.price)

        asks = list(islice(asks_all, depth)) if depth else list(asks_all)

        levels_dict = {
            'bids': [self._price_levels[price] for price in bids],
            'asks': [self._price_levels[price] for price in asks],
        }

        return levels_dict

    def ask_levels_by_price(self, group_size=10):
        result = {}

        best_price = roundup_to_nearest(self.best_ask.price, group_size)

        levels = sorted(self._price_levels.keys())

        next_level = True

        while next_level:
            bids = reversed(
                [level for level in levels if best_price <= level < best_price +
                 group_size])
            bids = list(bids)

            if len(bids) > 0:
                result[best_price] = sum(
                    [len(self._price_levels[price]) for price in bids])

            best_price = best_price + group_size
            next_level = best_price <= levels[-1]

        return result

    def bid_levels_by_price(self, group_size=10):
        result = {}

        best_price = roundup_to_nearest(self.best_bid.price, group_size)

        levels = sorted(self._price_levels.keys())

        next_level = True

        while next_level:
            bids = reversed(
                [level for level in levels if best_price >= level > best_price -
                 group_size])
            bids = list(bids)

            if len(bids) > 0:
                result[best_price] = sum(
                    [len(self._price_levels[price]) for price in bids])

            best_price = best_price - group_size
            next_level = best_price >= levels[0]

        return result

    def levels_by_price(self, group_size=10):
        return {
            'bids': self.bid_levels_by_price(group_size),
            'asks': self.ask_levels_by_price(group_size)
        }
