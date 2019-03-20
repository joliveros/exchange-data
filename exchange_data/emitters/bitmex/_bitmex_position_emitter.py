import logging
from abc import ABC
from datetime import timedelta
from exchange_data import Measurement, settings
from exchange_data.agents._apex_agent_check_point import ApexAgentCheckPoint
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger, SignalInterceptor, TimeChannels
from exchange_data.emitters.bitmex import BitmexEmitterBase
from exchange_data.utils import DateTimeUtils
from prometheus_client import Gauge, push_to_gateway, REGISTRY
from stringcase import pascalcase
from tgym.envs import OrderBookTradingEnv
from tgym.envs.orderbook.utils import Positions

import alog
import click
import inspect
import json
import numpy as np
import ray
import sys

pos_summary = Gauge('emit_position', 'Trading Position')
profit_gauge = Gauge('profit', 'Profit', unit='BTC')


class BitmexPositionEmitter(
    OrderBookTradingEnv,
    BitmexEmitterBase,
    Messenger,
    DateTimeUtils,
    ABC
):
    def __init__(
        self,
        checkpoint_id,
        result_path,
        job_name: str,
        agent_cls: str,
        start_date=None,
        end_date=None,
        env='orderbook-trading-v0',
        **kwargs
    ):
        checkpoint = result_path + \
            f'/checkpoint_{checkpoint_id}' \
            f'/checkpoint-{checkpoint_id}/'

        kwargs['checkpoint'] = checkpoint
        kwargs['env'] = env

        DateTimeUtils.__init__(self)
        Messenger.__init__(self)
        SignalInterceptor.__init__(self, self.stop)
        DateTimeUtils.__init__(self)
        OrderBookTradingEnv.__init__(
            self,
            should_penalize_even_trade=False,
            trading_fee=0.0,
            time_fee=0.0,
            max_summary=10,
            **kwargs
        )

        if isinstance(agent_cls, str):
            classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)

            agent_cls = pascalcase(agent_cls)
            agent_cls = [
                cls[1] for cls in classes
                if agent_cls in cls[0]
            ][0]

        self.checkpoint_id = checkpoint_id
        self.default_action = Positions.Flat.value
        self.prev_action = None
        self.prev_reward = None
        self.agent = agent_cls(**kwargs)
        self.job_name = f'{job_name}_{checkpoint_id}'
        self._database = 'bitmex'
        self._index = 0.0
        self.channel = 'XBTUSD_OrderBookFrame_depth_21'
        obs_len = self.observation_space.shape[0]

        now = self.now()

        if start_date:
            self.start_date = start_date
        else:
            self.start_date = now - timedelta(seconds=obs_len)

        if end_date:
            self.end_date = end_date
        else:
            self.end_date = now

        while self.last_observation.shape[0] < obs_len:
            self.prev_action = self.default_action
            self.prev_reward = 0.0
            self.get_observation()

        self.on(self.channel, self.emit_position)

    def exit(self, *args):
        super().stop(*args)

    def set_index(self, value):
        self._index = value

    def emit_position(self, *args):
        self._emit_position(*args)
        self._push_metrics()

    def _get_observation(self):
        if self.last_observation.shape[0] < self.observation_space.shape[0]:
            return super()._get_observation()
        else:
            return self.last_timestamp, self.orderbook_frame

    @pos_summary.time()
    def _emit_position(self, data):
        dt = self.now()
        meas = Measurement(**data)
        self.last_timestamp = meas.time
        self.orderbook_frame = np.asarray(json.loads(meas.fields['data']))

        action = self.agent.compute_action(
            self.last_observation,
            prev_action=self.prev_action,
            prev_reward=self.prev_reward
        )

        # TODO: publish ???
        self.step(action)

    def step(self, action):
        self.prev_action = action
        obs, reward, done, info = super().step(action)
        self.prev_reward = reward
        profit_gauge.set(self.capital)

        if settings.LOG_LEVEL == logging.DEBUG:
            alog.info(alog.pformat(self.summary()))

    def _push_metrics(self):
        push_to_gateway(
            settings.PROMETHEUS_HOST,
            job=self.job_name,
            registry=REGISTRY
        )

    def start(self):
        self.sub([self.channel])


class OrderBookTradingEvaluator(OrderBookTradingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@click.command()
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__),
                default=BitmexChannels.XBTUSD.value)
@click.option('--job-name', '-n', default=None)
@click.option('--agent-cls', '-a', default=None)
@click.option('--checkpoint_id', '-c')
@click.option('--result-path', '-r')
def main(**kwargs):
    emitter = BitmexPositionEmitter(**kwargs)
    emitter.start()


if __name__ == '__main__':
    main()
