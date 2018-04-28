import alog

from exchange_data.limit_orderbook import Order


class OrderList:
    """Doubly-Linked List Container Class.

    Stores head and tail orders, as well as count.

    Keeps a reference to its parent LimitLevel Instance.

    This container was added because it makes deleting the LimitLevels easier.

    Has no other functionality.

    """
    head: Order
    parent_limit: 'LimitLevel'
    tail: Order

    __slots__ = ['head', 'tail', 'parent_limit', 'count']

    def __init__(self, parent_limit):
        self.head = None
        self.tail = None
        self.count = 0
        self.parent_limit = parent_limit

    def __len__(self):
        return self.count

    def append(self, order):
        """Appends an order to this List.

        Same as LimitLevel append, except it automatically updates head and tail
        if it's the first order in this list.

        :param order:
        :return:
        """
        if not self.tail:
            order.root = self
            self.tail = order
            self.head = order
            self.count += 1
        else:
            self.tail.append(order)
