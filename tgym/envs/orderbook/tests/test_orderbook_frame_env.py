from exchange_data.trading import Positions
from tgym.envs import OrderBookFrameEnv
import pytest


class TestTradeExecutor(object):

    @pytest.mark.vcr()
    def test(self, mocker):
        # mocker.patch('exchange_data.emitters.messenger.Redis')

        kwargs = {
            'cache': True,
            'database_name': 'binance_futures',
            'depth': 12,
            'group_by': '30s',
            'interval': '2h',
            'leverage': 20.0,
            'max_volume_quantile': 0.99,
            'offset_interval': '0h',
            'round_decimals': 4,
            'sequence_length': 12,
            'summary_interval': 1,
            'symbol': 'LINAUSDT',
            'window_size': '2m'
        }

        env = OrderBookFrameEnv(**kwargs)

        env.reset()

        env.step(0)
