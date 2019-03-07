from time import sleep

import alog
import pytest

from dateutil import parser, tz
from pytimeparse import timeparse

from tgym.envs import OrderBookTradingEnv
from tgym.envs.orderbook import Actions, Positions


class TestOrderBookTradingEnv(object):

    @pytest.mark.vcr()
    def test_orderbook_env_reset(self):
        start_date = parser.parse('2019-03-07 01:31:48.315491+00:00') \
            .replace(tzinfo=tz.tzutc())

        env = OrderBookTradingEnv(
            window_size='1s',
            start_date=start_date
        )

        env.total_reward += 10

        env.reset()

        assert env.total_reward == 0

    # @pytest.mark.vcr(record_mode='once')
    def test_orderbook_env_step(self):
        start_date = parser.parse('2019-03-07 01:31:48.315491+00:00') \
            .replace(tzinfo=tz.tzutc())

        env = OrderBookTradingEnv(
            window_size='1s',
            random_start_date=True
            # start_date=start_date
        )

        for i in range(timeparse.timeparse('35m')):
            env.step(Positions.Long.value)
            sleep(0.2)

        env.step(Positions.Flat.value)

        # alog.info(env.frames)

