class LimitLevelTree:
    """AVL BST Root Node.

    """
    __slots__ = ['right_child', 'is_root']

    def __init__(self):
        # BST Attributes
        self.right_child = None
        self.is_root = True

    def insert(self, limit_level):
        """Iterative AVL Insert method to insert a new Node.

        Inserts a new node and calls the grand-parent's balance() method -
        but only if it isn't root.

        :param value:
        :return:
        """
        current_node = self
        while True:
            if current_node.is_root or limit_level.price > current_node.price:
                if current_node.right_child is None:
                    current_node.right_child = limit_level
                    current_node.right_child.parent = current_node
                    current_node.right_child.balance_grandpa()
                    break
                else:
                    current_node = current_node.right_child
                    continue
            elif limit_level.price < current_node.price:
                if current_node.left_child is None:
                    current_node.left_child = limit_level
                    current_node.left_child.parent = current_node
                    current_node.left_child.balance_grandpa()
                    break
                else:
                    current_node = current_node.left_child
                    continue
            else:
                # The level already exists
                break