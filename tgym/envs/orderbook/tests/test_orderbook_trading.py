import pytest

from dateutil import parser, tz
from tgym.envs import OrderBookTradingEnv
from tgym.envs.orderbook import Actions


class TestOrderBookTradingEnv(object):

    @pytest.mark.vcr(record_mode='all')
    def test_init(self):
        start_date = parser.parse('2019-03-07 01:31:48.315491+00:00') \
            .replace(tzinfo=tz.tzutc())

        env = OrderBookTradingEnv(
            window_size='1s',
            start_date=start_date
        )

        env.step(Actions.Buy)
