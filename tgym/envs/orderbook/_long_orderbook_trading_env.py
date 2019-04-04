from abc import ABC
from exchange_data.utils import NoValue
from gym.spaces import Discrete
from pytimeparse.timeparse import timeparse
from tgym.envs.orderbook._orderbook import OrderBookTradingEnv, Trade, \
    AlreadyFlatException

import alog
import click


class Positions(NoValue):
    Flat = 0
    Long = 1


class LongOrderBookTradingEnv(OrderBookTradingEnv, ABC):
    def __init__(self, **kwargs):
        action_space = Discrete(2)
        OrderBookTradingEnv.__init__(self, action_space=action_space, **kwargs)

    def long(self):
        if self.is_long:
            raise Exception('Already long.')
        self.position = Positions.Long

        self.entry_price = self.best_ask

    def flat(self):

        if self.position.value == Positions.Flat.value:
            raise AlreadyFlatException()

        if self.is_long:
            self.close_long()

        self.position = Positions.Flat

    def close_long(self):
        if self.position != Positions.Long:
            raise Exception('Not long.')

        pnl = self.long_pnl

        trade = Trade(self.position.name[0], pnl, self.entry_price,
                      self.best_bid)

        self.trades.append(trade)

        self.total_pnl += pnl
        self.capital += pnl
        self.reward += pnl * 2 if pnl < 0.0 else pnl
        self.entry_price = 0.0

    def change_position(self, action):
        # alog.info(action)
        if action == Positions.Long.value:
            self.long()
        elif action == Positions.Flat.value:
            # self.reward += self.step_reward * 2
            self.flat()

    def _pnl(self, exit_price):
        diff = exit_price - self.entry_price

        if self.entry_price == 0.0:
            change = 0.0
        else:
            change = diff / self.entry_price

        pnl = (self.capital * change) + (-1 * self.capital * self.trading_fee)

        return pnl

    def close_short(self):
        raise Exception()

    def short(self):
        raise Exception()


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
