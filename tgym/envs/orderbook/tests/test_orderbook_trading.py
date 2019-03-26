from dateutil import parser, tz
from pytimeparse import timeparse
from tgym.envs import OrderBookTradingEnv
from tgym.envs.orderbook.utils import Positions

import alog
import mock
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
            max_frames=10,
            max_summary=10,
            orderbook_depth=21,
            random_start_date=False,
            start_date=start_date,
            window_size='1m'
        )

        env.reset()

        for i in range(timeparse.timeparse('1m')):
            if i % 2 == 0:
                env.step(Positions.Flat.value)
            else:
                env.step(Positions.Long.value)

        env.step(Positions.Flat.value)

        alog.debug(alog.pformat(env.summary()))

    @pytest.mark.vcr()
    @mock.patch(
        'exchange_data.streamers._bitmex.SignalInterceptor'
    )
    def test_trades(self, sig_mock):
        start_date = parser.parse('2019-03-14 01:31:48.315491+00:00') \
            .replace(tzinfo=tz.tzutc())

        env = OrderBookTradingEnv(
            max_frames=10,
            max_summary=10,
            orderbook_depth=21,
            random_start_date=False,
            start_date=start_date,
            window_size='1m',
            use_volatile_ranges=False
        )

        env.reset()

        alog.info(dict(best_bid=env.best_bid, best_ask=env.best_ask))
        alog.info(alog.pformat(env.summary()))
        env.step(Positions.Long.value)
        alog.info(alog.pformat(env.summary()))

        env.last_orderbook[0][0][0] = 3862.0
        env.last_orderbook[1][0][0] = 3862.5

        env.step(Positions.Short.value)

        env.last_orderbook[0][0][0] = 3862.0
        env.last_orderbook[1][0][0] = 3862.5

        env.step(Positions.Flat.value)

        env.last_orderbook[0][0][0] = 3865.0
        env.last_orderbook[1][0][0] = 3865.5

        alog.info(alog.pformat(env.summary()))
        alog.info(dict(best_bid=env.best_bid, best_ask=env.best_ask))


