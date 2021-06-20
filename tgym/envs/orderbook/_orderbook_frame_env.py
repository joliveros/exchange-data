#!/usr/bin/env python

from cached_property import cached_property
from exchange_data.data.orderbook_frame import OrderBookFrame
from gym.spaces import Discrete
from pytimeparse.timeparse import timeparse
from tgym.envs.orderbook import OrderBookTradingEnv

import click
import random
import traceback


class OrderBookFrameEnv(OrderBookFrame, OrderBookTradingEnv):
    def __init__(
        self,
        trial=None,
        num_env=1,
        **kwargs
    ):
        super().__init__(
            action_space=Discrete(2),
            **kwargs
        )
        OrderBookTradingEnv.__init__(
            self,
            action_space=Discrete(2),
            **kwargs
        )
        self.trial = trial
        self.num_env = num_env
        kwargs['batch_size'] = 1

        self.observations = None
        self.prune_capital = 1.01
        self.total_steps = 0
        self.was_reset = False

    def reset_dataset(self):
        self.was_reset = True

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, value):
        if value:
            traceback.print_stack()

        self._done = value

    @property
    def best_bid(self):
        return self._best_bid

    @property
    def best_ask(self):
        return self._best_ask

    @cached_property
    def frame(self):
        return super().frame

    def _get_observation(self):
        self.max_steps = len(self.frame)

        for i in range(len(self.frame)):
            if self.was_reset:
                self.was_reset = False
                break

            row = self.frame.iloc[i]
            best_ask = row.best_ask
            best_bid = row.best_bid
            frame = row.orderbook_img
            timestamp = row.name.to_pydatetime()

            yield timestamp, best_ask, best_bid, frame

    def get_observation(self):
        try:
            timestamp, best_ask, best_bid, frame = next(self.observations)
        except StopIteration:
            self.observations = self._get_observation()
            timestamp, best_ask, best_bid, frame = next(self.observations)

        self._best_ask = best_ask
        self._best_bid = best_bid

        self.position_history.append(self.position.name[0])

        self.last_datetime = str(timestamp)
        self._last_datetime = timestamp

        self.last_observation = frame

        return self.last_observation

    def step(self, action):
        done = False
        assert self.action_space.contains(action)

        self.step_position(action)

        self.reward += self.current_trade.reward

        # if type(self.current_trade) == FlatTrade:
        #     self.reward = self.reward * 2

        self.step_count += 1

        if self.trial:
            self.trial.report(self.capital, self.step_count)

        if not self.eval_mode:
            if self.capital < self.min_capital and not self.eval_mode:
                done = True

            if self.current_trade:
                if self.current_trade.pnl <= self.max_negative_pnl:
                    done = True

        observation = self.get_observation()

        if self.step_count > self.max_steps:
            done = True

        reward = self.reset_reward()

        self.print_summary()

        # if done:
        #     traceback.print_stack()

        return observation, reward, done, {'capital': self.capital}


@click.command()
@click.option('--database_name', '-d', default='binance', type=str)
@click.option('--depth', default=72, type=int)
@click.option('--group-by', '-g', default='30s', type=str)
@click.option('--interval', '-i', default='10m', type=str)
@click.option('--max-volume-quantile', '-m', default=0.99, type=float)
@click.option('--offset-interval', '-o', default='0h', type=str)
@click.option('--round-decimals', '-D', default=4, type=int)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--summary-interval', '-s', default=1, type=int)
@click.option('--test-span', default='20s')
@click.option('--window-size', '-w', default='2m', type=str)
@click.argument('symbol', type=str)
def main(test_span, **kwargs):
    env = OrderBookFrameEnv(
        is_training=False,
        **kwargs
    )

    for t in range(1):
        env.reset()
        for i in range(timeparse(test_span)):
            env.step(random.randint(0, 1))


if __name__ == '__main__':
    main()
