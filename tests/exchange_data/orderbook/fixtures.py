import pytest

from exchange_data.orderbook import SellOrder, BuyOrder, OrderBook


@pytest.fixture
def orders():
    return [
        SellOrder(price=120.00, quantity=5),
        SellOrder(price=110.00, quantity=5),
        SellOrder(price=100.00, quantity=5),
        BuyOrder(price=90.00, quantity=5),
        BuyOrder(price=80.00, quantity=5),
        BuyOrder(price=70.00, quantity=5)
    ]


@pytest.fixture
def orderbook(orders):
    _orderbook = OrderBook()

    for order in orders:
        _orderbook.process_order(order)

    return _orderbook