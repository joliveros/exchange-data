#!/usr/bin/env python
import json

from skimage import color
from matplotlib import pyplot as plt
from cached_property import cached_property
from exchange_data.data.orderbook_frame import OrderBookFrame
from gym.spaces import Discrete
from pytimeparse.timeparse import timeparse
from tgym.envs.orderbook import OrderBookTradingEnv
from PIL import Image
import cv2

import alog
import click
import random
import traceback
import numpy as np

from tgym.envs.orderbook.ascii_image import AsciiImage


class OrderBookFrameEnv(OrderBookFrame, OrderBookTradingEnv):
    random_frame_start: bool = False

    def __init__(
        self,
        frame_width=96,
        macd_diff_enabled=True,
        random_frame_start=False,
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
            frame_width=frame_width,
            action_space=Discrete(2),
            **kwargs
        )

        self.plot_count = 0

        if random_frame_start:
            self.random_frame_start = random_frame_start

        self.trial = trial
        self.num_env = num_env
        kwargs['batch_size'] = 1
        self.macd_diff_enabled = macd_diff_enabled
        self.observations = None
        self.prune_capital = 1.01
        self.total_steps = 0
        self.was_reset = False
        self.macd_diff = None

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, value):
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

    @property
    def frame_start(self):
        if self.random_frame_start:
            return random.randint(0, len(self.frame))
        else:
            return 0

    def _get_observation(self):
        self.max_steps = len(self.frame)

        for i in range(self.frame_start, len(self.frame)):
            row = self.frame.iloc[i]
            best_ask = row.best_ask
            best_bid = row.best_bid
            frame = row.orderbook_img
            macd_diff = row.macd_diff
            timestamp = row.name.to_pydatetime()

            yield timestamp, best_ask, best_bid, frame, macd_diff

    def get_observation(self):
        if self.observations is None:
            self.observations = self._get_observation()

        try:
            timestamp, best_ask, best_bid, frame, macd_diff = \
                next(self.observations)
        except StopIteration:
            self.observations = None
            self.done = True
            return self.last_observation

        self._best_ask = best_ask
        self._best_bid = best_bid

        self.macd_diff = macd_diff

        self.position_history.append(self.position.name[0])

        self.last_datetime = str(timestamp)

        self._last_datetime = timestamp

        if self.current_trade:
            self.position_pnl_history.append(self.current_trade.pnl)

        ob_img = self.plot_orderbook(frame)
        pnl_img = self.plot_pnl()

        ob_pnl = np.concatenate([ob_img, pnl_img]) / 255

        self.last_observation = np.expand_dims(ob_pnl, axis=2)

        return self.last_observation

    def plot_orderbook(self, data):
        fig, frame = plt.subplots(1, 1, figsize=(2, 1),
                                        dpi=self.frame_width)

        # frame.imshow(np.rot90(np.fliplr(self.last_observation)))

        plt.autoscale(tight=True)
        frame.axis('off')
        fig.patch.set_visible(False)
        frame.imshow(data)
        fig.canvas.draw()
        img = fig.canvas.renderer._renderer

        plt.close(fig)

        img = np.array(img)
        img = Image.fromarray(np.uint8(img * 255)).convert('L')

        return np.asarray(img)

    def plot_pnl(self):
        pnl = np.asarray(self.position_pnl_history)

        if pnl.shape[0] > 0:
            fig, price_frame = plt.subplots(1, 1, figsize=(2, 1),
                                            dpi=self.frame_width)

            min = abs(pnl.min())
            pnl = pnl + min

            pnl_frame = price_frame.twinx()
            pnl_frame.plot(pnl, color='black')
            # pnl_frame = price_frame.twinx()
            # pnl_frame.plot(self.pnl_history, color='green')

            plt.fill_between(range(pnl.shape[0]), pnl, color='black')

            plt.autoscale(tight=True)
            pnl_frame.axis('off')
            fig.patch.set_visible(False)
            fig.canvas.draw()

            img = fig.canvas.renderer._renderer
            img = np.array(img)

            plt.close(fig)
            img = Image.fromarray(np.uint8(img * 255)).convert('L')

            return img
        else:
            return np.zeros([self.frame_width, self.frame_width * 2])

        # return str(AsciiImage(img, new_width=21)) + '\n'

    def step(self, action):
        # if macd is negative then assume position should be flat otherwise
        # use provided by prediction

        done = self.done

        # assert self.action_space.contains(action)
        action_before = action

        if self.macd_diff_enabled:
            if self.macd_diff > 0:
                action = 0

        self.step_position(action)

        self.reward += self.current_trade.reward

        self.step_count += 1

        if self.trial:
            self.trial.report(self.capital, self.step_count)

        # if self.current_trade and not self.is_test:
        #     if self.current_trade.pnl <= self.max_negative_pnl:
        #       done = True

        observation = self.get_observation()

        if not done:
            done = self.done

        reward = self.reset_reward()

        self.print_summary()

        return observation, reward, done, {
            'capital': self.capital,
            'trades': self.trades,
            'action': action
        }


@click.command()
@click.option('--cache', is_flag=True)
@click.option('--database_name', '-d', default='binance', type=str)
@click.option('--depth', default=72, type=int)
@click.option('--group-by', '-g', default='30s', type=str)
@click.option('--interval', '-i', default='10m', type=str)
@click.option('--leverage', default=1.0, type=float)
@click.option('--max-volume-quantile', '-m', default=0.99, type=float)
@click.option('--offset-interval', '-o', default='0h', type=str)
@click.option('--round-decimals', '-D', default=4, type=int)
@click.option('--sequence-length', '-l', default=48, type=int)
@click.option('--summary-interval', '-s', default=1, type=int)
@click.option('--test-span', default='20s')
@click.option('--window-size', '-w', default='2m', type=str)
@click.argument('symbol', type=str)
def main(test_span, **kwargs):
    for t in range(1):
        # kwargs['sequence_length'] = random.randrange(10, 100)
        env = OrderBookFrameEnv(
            short_class_str='ShortRewardPnlDiffTrade',
            flat_class_str='NoRewardFlatTrade',
            random_frame_start=False,
            short_reward_enabled=True,
            is_training=False,
            max_short_position_length=0,
            **kwargs
        )

        obs = env.reset()

        test_length = timeparse(test_span)
        step_reset = int(test_length / 2)

        for i in range(test_length):
            if i == step_reset:
                env.reset()

            actions = [1] * 39 + [0] * 10

            env.step(random.choice(actions))


if __name__ == '__main__':
    main()
