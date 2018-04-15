from exchange_data.limit_orderbook._order_list import OrderList
from exchange_data.limit_orderbook._limit_level_tree import LimitLevelTree


class LimitLevelBalanceError(BaseException):
    pass


class LimitLevel:
    """AVL BST node.

    This Binary Tree implementation balances on each insert.

    If performance is of concern to you, implementing a bulk-balance
    method may be of interest (c-based implementations aside).

    Attributes:
        parent: Parent node of this Node
        is_root: Boolean, to determine if this Node is root
        left_child: Left child of this Node; Values smaller than price
        right_child: Right child of this Node; Values greater than price

    Properties:
        height: Height of this Node
        balance: Balance factor of this Node
    """
    __slots__ = ['price', 'size', 'parent', 'left_child',
                 'right_child', 'head', 'tail', 'count', 'orders']

    def __init__(self, order):
        """Initialize a Node() instance.

        :param order:
        """
        # Data Values
        self.price = order.price
        self.size = order.size

        # BST Attributes
        self.parent = None
        self.left_child = None
        self.right_child = None

        # Doubly-Linked-list attributes
        self.orders = OrderList(self)
        self.append(order)

    @property
    def is_root(self):
        return isinstance(self.parent, LimitLevelTree)

    @property
    def volume(self):
        return self.price * self.size

    @property
    def balance_factor(self):
        """Calculate and return the balance of this Node.

        Calculate balance by dividing the right child's height from
        the left child's height. Children which evaluate to False (None)
        are treated as zeros.
        :return:
        """
        right_height = self.right_child.height if self.right_child else 0
        left_height = self.left_child.height if self.left_child else 0

        return right_height - left_height

    @property
    def grandpa(self):
        try:
            if self.parent:
                return self.parent.parent
            else:
                return None
        except AttributeError:
            return None

    @property
    def height(self):
        """Calculates the height of the tree up to this Node.

        :return: int, max height among children.
        """
        left_height = self.left_child.height if self.left_child else 0
        right_height = self.right_child.height if self.right_child else 0
        if left_height > right_height:
            return left_height + 1
        else:
            return right_height + 1

    @property
    def min(self):
        """Returns the smallest node under this node.

        :return:
        """
        minimum = self
        while minimum.left_child:
            minimum = minimum.left_child
        return minimum

    def append(self, order):
        """Wrapper function to make appending to Order List simpler.

        :param order: Order() Instance
        :return:
        """
        return self.orders.append(order)

    def _replace_node_in_parent(self, new_value=None):
        """Replaces Node in parent on a delete() call.

        :param new_value: LimitLevel() instance
        :return:
        """
        if not self.is_root:
            if self == self.parent.left_child:
                self.parent.left_child = new_value
            else:
                self.parent.right_child = new_value
        if new_value:
            new_value.parent = self.parent

    def remove(self):
        """Deletes this limit level.

        :return:
        """

        if self.left_child and self.right_child:
            # We have two kids
            succ = self.right_child.min

            # Swap Successor and current node
            self.left_child, succ.left_child = succ.left_child, self.left_child
            self.right_child, succ.right_child = succ.right_child, self.right_child
            self.parent, succ.parent = succ.parent, self.parent
            self.remove()
            self.balance_grandpa()
        elif self.left_child:
            # Only left child
            self._replace_node_in_parent(self.left_child)
        elif self.right_child:
            # Only right child
            self._replace_node_in_parent(self.right_child)
        else:
            # No children
            self._replace_node_in_parent(None)

    def balance_grandpa(self):
        """Checks if our grandparent needs rebalancing.

        :return:
        """
        if self.grandpa and self.grandpa.is_root:
            # If our grandpa is root, we do nothing.
            pass
        elif self.grandpa and not self.grandpa.is_root:
            # Tell the grandpa to check his balance.
            self.grandpa.balance()
        elif self.grandpa is None:
            # We don't have a grandpa!
            pass
        else:
            # Unforeseen things have happened. D:
            raise LimitLevelBalanceError()

        return

    def balance(self):
        """Call the rotation method relevant to this Node's balance factor.

        This call works itself up the tree recursively.

        :return:
        """
        if self.balance_factor > 1:
            # right is heavier
            if self.right_child.balance_factor< 0:
                # right_child.left is heavier, RL case
                self._rl_case()
            elif self.right_child.balance_factor> 0:
                # right_child.right is heavier, RR case
                self._rr_case()
        elif self.balance_factor < -1:
            # left is heavier
            if self.left_child.balance_factor< 0:
                # left_child.left is heavier, LL case
                self._ll_case()
            elif self.left_child.balance_factor> 0:
                # left_child.right is heavier, LR case
                self._lr_case()
        else:
            # Everything's fine.
            pass

        # Now check upwards
        if not self.is_root and not self.parent.is_root:
            self.parent.balance()

    def _ll_case(self):
        """Rotate Nodes for LL Case.

        Reference:
            https://en.wikipedia.org/wiki/File:Tree_Rebalancing.gif
        :return:
        """
        child = self.left_child

        if self.parent.is_root or self.price > self.parent.price:
            self.parent.right_child = child
        else:
            self.parent.left_child = child

        child.parent, self.parent = self.parent, child
        child.right_child, self.left_child = self, child.right_child

    def _rr_case(self):
        """Rotate Nodes for RR Case.

        Reference:
            https://en.wikipedia.org/wiki/File:Tree_Rebalancing.gif
        :return:
        """
        child = self.right_child

        if self.parent.is_root or self.price > self.parent.price:
            self.parent.right_child = child
        else:
            self.parent.left_child = child

        child.parent, self.parent = self.parent, child
        child.left_child, self.right_child = self, child.left_child

    def _lr_case(self):
        """Rotate Nodes for LR Case.

        Reference:
            https://en.wikipedia.org/wiki/File:Tree_Rebalancing.gif
        :return:
        """
        child, grand_child = self.left_child, self.left_child.right_child
        child.parent, grand_child.parent = grand_child, self
        child.right_child = grand_child.left_child
        self.left_child, grand_child.left_child = grand_child, child
        self._ll_case()

    def _rl_case(self):
        """Rotate Nodes for RL case.

        Reference:
            https://en.wikipedia.org/wiki/File:Tree_Rebalancing.gif
        :return:
        """
        child, grand_child = self.right_child, self.right_child.left_child
        child.parent, grand_child.parent = grand_child, self
        child.left_child = grand_child.right_child
        self.right_child, grand_child.right_child = grand_child, child
        self._rr_case()

    def __str__(self):
        if not self.is_root:
            s = 'Node Value: %s\n' % self.price
            s += 'Node left_child value: %s\n' % (self.left_child.price if self.left_child else 'None')
            s += 'Node right_child value: %s\n\n' % (self.right_child.price if self.right_child else 'None')
        else:
            s = ''

        left_side_print = self.left_child.__str__() if self.left_child else ''
        right_side_print = self.right_child.__str__() if self.right_child else ''
        return s + left_side_print + right_side_print

    def __len__(self):
        return len(self.orders)