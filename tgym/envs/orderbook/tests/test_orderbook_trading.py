import random

from dateutil import parser, tz
from pytimeparse import timeparse
import matplotlib.pyplot as plt

import alog
import mock
import pytest

from exchange_data.trading import Positions
from tgym.envs.orderbook import OrderBookTradingEnv


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
    def test_get_volatile_ranges(self, sig_mock):
        start_date = parser.parse('2019-03-07 01:31:48.315491+00:00') \
            .replace(tzinfo=tz.tzutc())

        env = OrderBookTradingEnv(
            max_frames=10,
            max_summary=10,
            orderbook_depth=21,
            random_start_date=False,
            start_date=start_date,
            window_size='10m'
        )

        assert env.start_date == parser.parse('2019-03-07 00:50:00-06:00')

    # @pytest.mark.vcr(record_mode='all')
    @mock.patch(
        'exchange_data.streamers._bitmex.SignalInterceptor'
    )
    def test_orderbook_env_step(self, sig_mock):
        start_date = parser.parse('2019-03-15 01:31:00+00:00') \
            .replace(tzinfo=tz.tzutc())

        env = OrderBookTradingEnv(
            max_frames=60,
            max_summary=10,
            orderbook_depth=21,
            random_start_date=False,
            start_date=start_date,
            window_size='1m',
            summary_interval=60,
            is_training=False
        )

        env.reset()

        trade_length = 10
        test_length = timeparse.timeparse('2m')

        while test_length > 0:
            for i in range(trade_length):
                env.step(random.randint(0, 2))
                test_length -= 1

            env.step(Positions.Flat.value)

        alog.info(alog.pformat(env.summary()))
