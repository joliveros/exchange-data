import time


class Order:
    """Doubly-Linked List Order item.

    Keeps a reference to root, as well as previous and next order in line.

    It also performs any and all updates to the root's tail, head and count
    references, as well as updating the related LimitLevel's size, whenever
    a method is called on this instance.

    Offers append() and pop() methods. Prepending isn't implemented.

    """
    __slots__ = ['uid', 'is_bid', 'size', 'price', 'timestamp',
                 'next_item', 'previous_item', 'root']

    def __init__(self, uid, is_bid, size, price, root=None,
                 timestamp=None, next_item=None, previous_item=None):
        # Data Values
        self.uid = uid
        self.is_bid = is_bid
        self.price = price
        self.size = size
        self.timestamp = timestamp if timestamp else time.time()

        # DLL Attributes
        self.next_item = next_item
        self.previous_item = previous_item
        self.root = root

    @property
    def parent_limit(self):
        return self.root.parent_limit

    def append(self, order):
        """Append an order.

        :param order: Order() instance
        :return:
        """
        if self.next_item is None:
            self.next_item = order
            self.next_item.previous_item = self
            self.next_item.root = self.root

            # Update Root Statistics in OrderList root obj
            self.root.count += 1
            self.root.tail = order

            self.parent_limit.size += order.size

        else:
            self.next_item.append(order)

    def pop_from_list(self):
        """Pops this item from the DoublyLinkedList it belongs to.

        :return: Order() instance values as tuple
        """
        if self.previous_item is None:
            # We're head
            self.root.head = self.next_item
            if self.next_item:
                self.next_item.previous_item = None

        if self.next_item is None:
            # We're tail
            self.root.tail = self.previous_item
            if self.previous_item:
                self.previous_item.next_item = None

        # Update the Limit Level and root
        self.root.count -= 1
        self.parent_limit.size -= self.size

        return self.__repr__()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str((self.uid, self.is_bid, self.price, self.size, self.timestamp))