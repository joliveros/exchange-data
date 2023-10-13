#!/usr/bin/env python

from PIL import Image
from cached_property import cached_property
from exchange_data.data.orderbook_frame import OrderBookFrame
from gym.spaces import Discrete
from matplotlib import pyplot as plt
from tgym.envs.orderbook import OrderBookTradingEnv

import click
import matplotlib
import numpy as np
import random
import cv2
import alog




class OrderBookFrameEnv(OrderBookFrame, OrderBookTradingEnv):
    random_frame_start: bool = False

    def __init__(
        self,
        show_img=False,
        frame_width=224,
        macd_diff_enabled=False,
        random_frame_start=False,
        trial=None,
        num_env=1,
        **kwargs
    ):
        super().__init__(action_space=Discrete(2), **kwargs)
        OrderBookTradingEnv.__init__(
            self, frame_width=frame_width, action_space=Discrete(2), **kwargs
        )
        self.plot_count = 0

        if random_frame_start:
            self.random_frame_start = random_frame_start
        self._show_img = show_img

        if not self._show_img:
            matplotlib.use("agg")

        self.trial = trial
        self.num_env = num_env
        kwargs["batch_size"] = 1
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
            # macd_diff = row.macd_diff
            timestamp = row.name.to_pydatetime()

            yield timestamp, best_ask, best_bid, frame

    def get_observation(self):
        if self.observations is None:
            self.observations = self._get_observation()

        try:
            timestamp, best_ask, best_bid, frame = next(self.observations)
        except StopIteration:
            self.observations = None
            self.done = True
            return self.last_observation

        self._best_ask = best_ask
        self._best_bid = best_bid

        self.position_history.append(self.position.name[0])

        self.last_datetime = str(timestamp)

        self._last_datetime = timestamp

        if self.current_trade:
            self.position_pnl_history.append(self.current_trade.pnl)

        ob_img = self.plot_orderbook(frame)

        if self._show_img:
            self.show_img(ob_img)

        ob_img = ob_img[:, :, :3]
        ob_img = np.expand_dims(ob_img, axis=0)

        self.last_observation = ob_img

        return self.last_observation

    def plot_orderbook(self, data):
        fig, frame = plt.subplots(1, 1, figsize=(1, 1), dpi=self.frame_width)
        # frame.axis('off')
        frame = frame.twinx()
        plt.autoscale(tight=True)
        frame.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        fig.patch.set_visible(False)
        frame.imshow(data)
        fig.canvas.draw()
        img = fig.canvas.renderer._renderer

        plt.close()

        return np.array(img)

    def plot_pnl(self):
        pnl = np.asarray(self.position_pnl_history)

        if pnl.shape[0] > 0:
            fig, price_frame = plt.subplots(1, 1, figsize=(2, 1), dpi=self.frame_width)

            min = abs(pnl.min())
            pnl = pnl + min

            # pnl_frame = price_frame.twinx()
            pnl_frame = price_frame
            pnl_frame.plot(pnl, color="black")

            plt.fill_between(range(pnl.shape[0]), pnl, color="black")

            plt.autoscale(tight=True)
            pnl_frame.axis("off")
            fig.patch.set_visible(False)
            fig.canvas.draw()

            _img = fig.canvas.renderer._renderer

            plt.close()

            img = np.array(_img)
            img = Image.fromarray(np.uint8(img * 255)).convert("L")

            return np.array(img)
        else:
            return np.zeros([self.frame_width, self.frame_width * 2])

    def show_img(self, img):
        # img = np.array(Image.fromarray(np.uint8(np.array(img) * 255))
        #     .convert("RGB"))

        cv2.imshow("image", img)
        cv2.waitKey(1)

    def step(self, action):
        done = self.done

        if self.macd_diff_enabled:
            if self.macd_diff > 0:
                action = 0

        self.step_position(action)

        self.reward += self.current_trade.reward

        self.step_count += 1

        if self.trial:
            self.trial.report(self.capital, self.step_count)

        observation = self.get_observation()

        if not done:
            done = self.done

        reward = self.reset_reward()

        # alog.info((reward, self.reward, self.current_trade.reward))

        self.print_summary()

        return (
            observation,
            reward,
            done,
            {"capital": self.capital, "trades": self.trades, "action": action},
        )


@click.command()
@click.option("--cache", is_flag=True)
@click.option("--database_name", "-d", default="binance", type=str)
@click.option("--depth", default=72, type=int)
@click.option("--group-by", "-g", default="30s", type=str)
@click.option("--interval", "-i", default="10m", type=str)
@click.option("--leverage", default=1.0, type=float)
@click.option("--max-volume-quantile", "-m", default=0.99, type=float)
@click.option("--offset-interval", "-o", default="0h", type=str)
@click.option("--round-decimals", "-D", default=4, type=int)
@click.option("--sequence-length", "-l", default=48, type=int)
@click.option("--summary-interval", "-s", default=1, type=int)
@click.option("--window-size", "-w", default="2m", type=str)
@click.argument("symbol", type=str)
def main(**kwargs):
    env = OrderBookFrameEnv(
                show_img=True,
        short_class_str="ShortRewardPnlDiffTrade",
        flat_class_str="FlatRewardPnlDiffTrade",
        random_frame_start=False,
        short_reward_enabled=True,
        is_training=False,
        max_short_position_length=0,
        min_change=-0.5,
        **kwargs
    )

    env.reset()

    choice_args = dict(a=[0, 1], p=[0.9, 0.1])

    _done = False
    step = 0

    while not _done:
        state, reward, done, summary = env.step(np.random.choice(**choice_args))
        step += 1
        _done = done


if __name__ == "__main__":
    main()
