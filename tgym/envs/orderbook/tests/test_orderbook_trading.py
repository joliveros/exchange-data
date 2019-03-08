from dateutil import parser, tz
from pytimeparse import timeparse
from tgym.envs import OrderBookTradingEnv
from tgym.envs.orderbook.utils import Positions

from time import sleep
import alog
import pytest


class TestOrderBookTradingEnv(object):

    @pytest.mark.vcr()
    def test_orderbook_env_reset(self):
        start_date = parser.parse('2019-03-07 01:31:48.315491+00:00') \
            .replace(tzinfo=tz.tzutc())

        env = OrderBookTradingEnv(
            window_size='1s',
            start_date=start_date,
            max_frames='1s'
        )

        env.total_reward += 10

        env.reset()

        assert env.total_reward == 0

    # @pytest.mark.vcr()
    def test_orderbook_env_step(self):
        start_date = parser.parse('2019-03-07 01:31:48.315491+00:00') \
            .replace(tzinfo=tz.tzutc())

        env = OrderBookTradingEnv(
            window_size='1s',
            start_date=start_date,
            max_frames='5s',
            orderbook_depth=21
        )

        for i in range(timeparse.timeparse('5s')):
            env.step(Positions.Long.value)

        env.step(Positions.Flat.value)

