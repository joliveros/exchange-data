from exchange_data import Buffer


class OrderList(object):
    """
    A doubly linked list of Orders. Used to iterate through Orders when
    a price match is found. Each OrderList is associated with a single
    price. Since a single price match can have more quantity than a single
    Order, we may need multiple Orders to fullfill a transaction. The
    OrderList makes this easy to do. OrderList is naturally arranged by time.
    Orders at the front of the list have priority.
    """

    def __init__(self):
        # first order in the list
        self.head_order: 'Order' = None

        # last order in the list
        self.tail_order: 'Order' = None

        # number of Orders in the list
        self.length = 0

        # sum of Order quantity in the list AKA share volume
        self.volume = 0

        # helper for iterating
        self.last = None

    def __len__(self):
        return self.length

    def __iter__(self):
        self.last = self.head_order
        return self

    def next(self):
        """
        Get the next order in the list.

        Set self.last as the next order. If there is no next order, stop
        iterating through list.
        """
        if self.last is None:
            raise StopIteration
        else:
            return_value = self.last
            self.last = self.last.next_order
            return return_value

    __next__ = next

    def get_head_order(self):
        return self.head_order

    def append_order(self, order):
        if len(self) == 0:
            order.next_order = None
            order.prev_order = None
            self.head_order = order
            self.tail_order = order
        else:
            order.prev_order = self.tail_order
            order.next_order = None
            self.tail_order.next_order = order
            self.tail_order = order
        self.length += 1
        self.volume += order.quantity

    def remove_order(self, order):
        self.volume -= order.quantity
        self.length -= 1

        # if there are no more Orders, stop/return
        if len(self) == 0:
            return

        # Remove an Order from the OrderList. First grab next / prev order
        # from the Order we are removing. Then relink everything. Finally
        # remove the Order.
        next_order = order.next_order
        prev_order = order.prev_order

        if next_order is not None and prev_order is not None:
            next_order.prev_order = prev_order
            prev_order.next_order = next_order

        # There is no previous order
        elif next_order is not None:
            next_order.prev_order = None

            # The next order becomes the first order in the OrderList after
            # this Order is removed
            self.head_order = next_order

        # There is no next order
        elif prev_order is not None:
            prev_order.next_order = None

            # The previous order becomes the last order in the OrderList
            # after this Order is removed
            self.tail_order = prev_order

    def move_to_tail(self, order):
        """
        After updating the quantity of an existing Order, move it to the tail
        of the OrderList

        Check to see that the quantity is larger than existing, update the
        quantities, then move to tail.
        """

        # This Order is not the first Order in the OrderList
        if order.prev_order is not None:
            # Link the previous Order to the next Order, then move the Order
            # to tail
            order.prev_order.next_order = order.next_order
        else:
            # This Order is the first Order in the OrderList
            # Make next order the first
            self.head_order = order.next_order

        order.next_order.prev_order = order.prev_order

        # Move Order to the last position. Link up the previous last position
        #  Order.
        self.tail_order.next_order = order
        self.tail_order = order

    def __str__(self):
        buffer = Buffer()

        for order in self:
            buffer.write("%s\n" % str(order))

        return str(buffer)
