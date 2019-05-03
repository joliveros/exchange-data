from abc import ABC
from exchange_data.utils import NoValue
from gym.spaces import Discrete
from pytimeparse.timeparse import timeparse
from tgym.envs.orderbook._orderbook import OrderBookTradingEnv, Trade, \
    AlreadyFlatException

import alog
import click
import numpy as np


class Positions(NoValue):
    Flat = 0
    Long = 1


REWARD_BASE = 1 / 10
NEGATIVE_FACTOR = 30/100

class LongOrderBookTradingEnv(OrderBookTradingEnv, ABC):
    positive_pnl_reward = REWARD_BASE
    negative_pnl_reward = REWARD_BASE * -1 * NEGATIVE_FACTOR
    flat_reward = REWARD_BASE * NEGATIVE_FACTOR
    step_reward = REWARD_BASE
    max_position_pnl = None

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
                      self.best_bid, self.position_repeat)

        self.trades.append(trade)

        if self.pnl >= self.min_profit:
            self.reward += self.positive_pnl_reward
        else:
            self.reward += self.negative_pnl_reward

        self.total_pnl += pnl
        self.capital += pnl

        self.long_pnl_history = []

        self.entry_price = 0.0
        self.position_pnl_history = np.array([])
        self.position_pnl_diff_history = np.array([])
        self.max_position_pnl = None

    def change_position(self, action):
        # alog.info(action)
        if action == Positions.Long.value:
            self.long()
        elif action == Positions.Flat.value:
            self.flat()

    def _pnl(self, exit_price, entry_price=None):
        if entry_price is None:
            entry_price = self.entry_price

        entry_price = entry_price if entry_price > 0.0 \
            else self.entry_price

        if entry_price < 1:
            entry_price = exit_price

        if exit_price == 0.0:
            return 0.0

        diff = exit_price - entry_price

        change = diff / entry_price

        pnl = (self.capital * change) + (-1 * self.capital * self.trading_fee)

        return pnl

    def reset_reward(self):
        if self.is_long:
            pnl = self.long_pnl

            if self.position_repeat > 0 and self.position_pnl_history.shape[0] > 0:
                # if self.position_repeat == 2:
                #     self.reward += self.negative_pnl_reward

                last_pnl = self.position_pnl_history[-1]
                pnl_diff = pnl - last_pnl
                max_pnl = np.amax(self.position_pnl_history)

                if self.max_position_pnl is None:
                    self.max_position_pnl = max_pnl
                else:
                    if self.max_position_pnl == max_pnl and self.max_position_pnl > self.min_profit:
                        self.reward += self.negative_pnl_reward * 10

                    if max_pnl > self.max_position_pnl:
                        self.reward += self.positive_pnl_reward

                    # if pnl_diff > 0.0:
                    #     self.reward += self.positive_pnl_reward
                    # else:
                    #     self.reward += self.negative_pnl_reward

                    self.max_position_pnl = max_pnl

                self.position_pnl_diff_history = np.append(self.position_pnl_diff_history, [pnl_diff])

            # else:
            #     self.reward += self.positive_pnl_reward / 100

            self.position_pnl_history = np.append(self.position_pnl_history, [pnl])

        if self.is_flat:
            last_best_ask = self.position_data_history[0][1]
            pnl = self._pnl(self.best_bid, last_best_ask)

            if pnl < self.min_profit:
                self.reward += self.flat_reward * -1
            else:
                self.reward += self.flat_reward

        reward = self.reward
        self.total_reward += reward
        self.reward = 0.0
        return reward

    @property
    def is_long(self):
        return self.position.value == Positions.Long.value

    @property
    def is_flat(self):
        return self.position.value == Positions.Flat.value

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
