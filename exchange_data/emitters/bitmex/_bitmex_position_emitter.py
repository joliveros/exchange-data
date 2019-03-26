from abc import ABC
from datetime import timedelta

from gym import Env

from exchange_data import Measurement, settings
from exchange_data.agents._apex_agent_check_point import ApexAgentCheckPoint
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger, SignalInterceptor, TimeChannels
from exchange_data.emitters.bitmex import BitmexEmitterBase
from exchange_data.utils import DateTimeUtils
from prometheus_client import Gauge, push_to_gateway, REGISTRY
from pyee import EventEmitter
from stringcase import pascalcase
from tgym.envs import OrderBookTradingEnv, LongOrderBookTradingEnv
from tgym.envs.orderbook.utils import Positions

import alog
import click
import inspect
import json
import logging
import numpy as np
import sys

pos_summary = Gauge('emit_position', 'Trading Position')
profit_gauge = Gauge('profit', 'Profit', unit='BTC')


class BitmexPositionEmitter(
    Env,
    BitmexEmitterBase,
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

        BitmexEmitterBase.__init__(self, **kwargs)

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

        self.on(self.channel_name, self.emit_position)

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
        meas = Measurement(**data)
        self.last_timestamp = meas.time
        self.orderbook_frame = np.asarray(json.loads(meas.fields['data']))

        # alog.info(alog.pformat(self.orderbook_frame))
        # alog.info(self.last_observation.tolist())

        action = self.agent.compute_action(self.last_observation)

        self.step(action)

        self.publish_position(action)

    def publish_position(self, action):
        _action = None

        if Positions.Flat.value == action:
            _action = Positions.Flat
        elif Positions.Long.value == action:
            _action = Positions.Long
        elif Positions.Short.value == action:
            _action = Positions.Short

        if _action:
            self.publish(self.job_name, dict(data=_action.name))

    def publish(self, channel, data):
        super().publish(channel, json.dumps(data))

    def step(self, action):
        self.prev_action = action

        obs, reward, done, info = super().step(action)

        self.prev_reward = reward
        profit_gauge.set(self.capital)

    def _push_metrics(self):
        push_to_gateway(
            settings.PROMETHEUS_HOST,
            job=self.job_name,
            registry=REGISTRY
        )

    def start(self):
        self.sub([self.channel_name])


class OrderBookTradingEnvAbs(object):
    def __init__(self, **kwargs):
        pass


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
