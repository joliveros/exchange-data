from abc import ABC
from exchange_data.utils import NoValue
from gym.spaces import Discrete
from pytimeparse.timeparse import timeparse
from tgym.envs.orderbook._orderbook import OrderBookTradingEnv

import alog
import click


class Positions(NoValue):
    Flat = 0
    Long = 1


class LongOrderBookTradingEnv(OrderBookTradingEnv, ABC):
    def __init__(self, **kwargs):
        action_space = Discrete(2)
        OrderBookTradingEnv.__init__(self, action_space=action_space, **kwargs)

    def change_position(self, action):
        self.last_position = self.position

        if action == Positions.Long.value:
            self.long()
        elif action == Positions.Flat.value:
            self.reward += self.step_reward
            self.flat()

    def pnl(self):
        # alog.info(self.position.value)
        if self.position.value != Positions.Flat.value:
            return self.long_pnl
        else:
            return 0.0

    def should_change_position(self, action):
        return self.position.value != action


@click.command()
@click.option('--test-span', default='2m')
def main(test_span, **kwargs):
    env = LongOrderBookTradingEnv(
        random_start_date=True,
        use_volatile_ranges=True,
        window_size='1m',
        max_frames=5,
        **kwargs
    )

    env.reset()

    alog.info(env.action_space)

    for i in range(timeparse(test_span) - 10):
        env.step(Positions.Long.value)
        # alog.info(alog.pformat(env.summary()))
        # if env.step_count % 5 == 0:
        #     alog.info(env.best_bid)

    env.step(Positions.Flat.value)

    alog.info(alog.pformat(env.summary()))


if __name__ == '__main__':
    main()
