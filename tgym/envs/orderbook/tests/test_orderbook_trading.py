import mock
from dateutil import parser, tz
from pytimeparse import timeparse
from tgym.envs import OrderBookTradingEnv
from tgym.envs.orderbook.utils import Positions

from time import sleep
import alog
import pytest


class TestOrderBookTradingEnv(object):

    @pytest.mark.vcr()
    @mock.patch(
        'exchange_data.streamers._bitmex.SignalInterceptor'
    )
    def test_orderbook_env_reset(self, sig_mock):
        start_date = parser.parse('2019-03-07 01:31:48.315491+00:00') \
            .replace(tzinfo=tz.tzutc())

        env = OrderBookTradingEnv(
            window_size='1s',
            start_date=start_date,
            max_frames='1s',
            random_start_date=False
        )

        env.total_reward += 10

        env.reset()

        assert env.total_reward == 0

    @pytest.mark.vcr()
    @mock.patch(
        'exchange_data.streamers._bitmex.SignalInterceptor'
    )
    def test_orderbook_env_step(self, sig_mock):
        start_date = parser.parse('2019-03-07 01:31:48.315491+00:00') \
            .replace(tzinfo=tz.tzutc())

        env = OrderBookTradingEnv(
            max_frames='5s',
            orderbook_depth=21,
            random_start_date=False,
            start_date=start_date,
            window_size='30s'
        )

        env.reset()

        for i in range(timeparse.timeparse('5s')):
            env.step(Positions.Long.value)

        env.step(Positions.Flat.value)

        alog.info(alog.pformat(env.summary()))

