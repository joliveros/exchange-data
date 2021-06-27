from dateutil import parser
from dateutil.tz import tz
from tgym.envs.orderbook import LongOrderBookTradingEnv
import alog


class TestLongOrderBookTradingEnv(object):
    start_date = parser.parse('2019-03-07 01:31:48.315491+00:00') \
            .replace(tzinfo=tz.tzutc())

    def test_init(self):

        env = LongOrderBookTradingEnv(
            start_date=self.start_date,
            use_volatile_ranges=False,
            window_size='1m',
            max_frame=5
        )

        alog.info(env.action_space)
