import traceback

import alog
from bintrees import RBTree

from exchange_data.orderbook.exceptions import PriceDoesNotExistException
from ._orderlist import OrderList
from ._order import Order


class OrderTree(object):
    """
    A red-black tree used to store OrderLists in price order

    The exchange will be using the OrderTree to hold bid and ask data (one
    OrderTree for each side). Keeping the information in a red black tree
    makes it easier/faster to detect a match.
    """

    def __init__(self):
        self.price_tree = RBTree()
        self.price_map = {}  # Dictionary containing price : OrderList object
        self.order_map = {}  # Dictionary containing order_id : Order object
        self.volume = 0  # Contains total quantity from all Orders in tree
        self.num_orders = 0  # Contains count of Orders in tree

        # Number of different prices in tree (
        # http://en.wikipedia.org/wiki/Order_book_(trading)#Book_depth)
        self.depth = 0

    def __len__(self):
        return len(self.order_map)

    def get_price_list(self, price) -> OrderList:
        return self.price_map[price]

    def get_order(self, order_id):
        return self.order_map[order_id]

    def create_price(self, price):
        self.depth += 1  # Add a price depth level to the tree
        new_list = OrderList()
        self.price_tree.insert(price,
                               new_list)  # Insert a new price into the tree

        # Can i just get this by using self.price_tree.get_value(price)?
        # Maybe this is faster though.
        self.price_map[price] = new_list

    def remove_price(self, price):
        try:
            self.price_tree.remove(price)
            del self.price_map[price]
            self.depth -= 1  # Remove a price depth level
        except KeyError as e:
            raise PriceDoesNotExistException()

    def price_exists(self, price):
        return price in self.price_map

    def order_exists(self, order):
        return order in self.order_map

    def insert_order(self, order: Order):
        uid = order.uid
        price = order.price
        if self.order_exists(order.uid):
            self.remove_order_by_id(order.uid)

        self.num_orders += 1

        if price not in self.price_map:
            self.create_price(price)  # If price not in Price Map,
            # create a node
            # in RBtree

        order.order_list = self.price_map[price]

        self.price_map[price].append_order(order)

        self.order_map[uid] = order

        self.volume += order.quantity

    def modify_order(self, order_id: int, price: float, quantity: float,
                     timestamp: int=None):

        order = self.order_map[order_id]
        order.timestamp = timestamp

        if order.price != price:
            order_list = self.price_map[order.price]
            self.remove_order_by_id(order.uid)

            if len(order_list) == 0:
                self.remove_price(order.price)

            order.price = price
            self.insert_order(order)

        if order.quantity != quantity:
            old_quantity = order.quantity
            order.update_quantity(quantity)
            self.volume += quantity - old_quantity

    def remove_order_by_id(self, order_id):
        order = self.order_map[order_id]
        self.num_orders -= 1
        self.volume -= order.quantity

        order.order_list.remove_order(order)

        if len(order.order_list) == 0:
            self.remove_price(order.price)
        del self.order_map[order_id]

    def max_price(self):
        if self.depth > 0:
            return self.price_tree.max_key()
        else:
            return None

    def min_price(self):
        if self.depth > 0:
            return self.price_tree.min_key()
        else:
            return None

    def max_price_list(self):
        if self.depth > 0:
            return self.get_price_list(self.max_price())
        else:
            return None

    def min_price_list(self):
        if self.depth > 0:
            return self.get_price_list(self.min_price())
        else:
            return None
