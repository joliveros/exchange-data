from dateutil import parser, tz
from tgym.envs import OrderBookTradingEnv

import alog


class TestOrderBookTradingEnv(object):
    def test_init(self):
        start_date = parser.parse('2019-02-25T03:36:15.751397')\
            .replace(tzinfo=tz.tzlocal())

        env = OrderBookTradingEnv(
            start_date=start_date,
            influxdb='http://jose:jade121415@0.0.0.0:28953/'
        )
