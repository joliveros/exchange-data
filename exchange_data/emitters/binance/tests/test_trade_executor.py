from exchange_data.emitters.binance._trade_executor import TradeExecutor
import pytest

from exchange_data.trading import Positions


class TestTradeExecutor(object):

    @pytest.mark.vcr()
    def test(self, mocker):
        mocker.patch('exchange_data.emitters.messenger.Redis')

        kwargs = {
            'base_asset': 'USDT',
            'futures': True,
            'leverage': 10,
            'log_requests': False,
            'quantity': 5.0,
            'symbol': 'BAKE',
            'trade_min': True,
            'trading_enabled': False
        }

        tex = TradeExecutor(**kwargs)
        tex.bid_price = 1.9412

        tex.trade(Positions.Flat)
        tex.trade(Positions.Short)
        tex.trade(Positions.Flat)

