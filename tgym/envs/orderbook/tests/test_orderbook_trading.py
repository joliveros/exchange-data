from dateutil import parser, tz
from tgym.envs import OrderBookTradingEnv
from tgym.envs.orderbook import Actions


class TestOrderBookTradingEnv(object):
    def test_init(self):
        start_date = parser.parse('2019-03-06 21:49:07.807158+00:00') \
            .replace(tzinfo=tz.tzutc())

        env = OrderBookTradingEnv(
            window_size='1s',
            start_date=start_date,
            influxdb='http://jose:jade121415@0.0.0.0:28953/'
        )

        env.step(Actions.Buy)
