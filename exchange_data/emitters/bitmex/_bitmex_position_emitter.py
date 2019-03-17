import ray
from abc import ABC

from stringcase import pascalcase

from exchange_data import Measurement, settings
from exchange_data.agents._apex_agent_check_point import ApexAgentCheckPoint
from exchange_data.channels import BitmexChannels
from exchange_data.emitters import Messenger, SignalInterceptor, TimeChannels
from exchange_data.emitters.bitmex import BitmexEmitterBase
from exchange_data.utils import DateTimeUtils
from prometheus_client import Gauge, push_to_gateway, REGISTRY
from tgym.envs import OrderBookTradingEnv

import alog
import click
import inspect
import json
import numpy as np
import sys

from tgym.envs.orderbook.utils import Positions

pos_summary = Gauge('emit_position', 'Trading Position')
profit_gauge = Gauge('profit', 'Profit', unit='BTC')


class BitmexPositionEmitter(
    OrderBookTradingEnv,
    BitmexEmitterBase,
    Messenger,
    ABC
):
    def __init__(
        self,
        job_name: str,
        agent_cls: str,
        env='orderbook-trading-v0',
        checkpoint='/ray_results/orderbook-apex-v3/APEX_orderbook-trading'
                   '-v0_0_2019-03-15_09-43-04v2xyo9_t/checkpoint_88/'
                   'checkpoint-88',
        **kwargs
    ):
        kwargs['checkpoint'] = checkpoint
        kwargs['env'] = env

        DateTimeUtils.__init__(self)
        Messenger.__init__(self)
        SignalInterceptor.__init__(self, self.stop)
        OrderBookTradingEnv.__init__(
            self,
            should_penalize_even_trade=False,
            trading_fee=0.075/100.00,
            time_fee=0.0,
            **kwargs
        )

        classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
        agent_cls = pascalcase(agent_cls)
        agent_cls = [
            cls[1] for cls in classes
            if agent_cls in cls[0]
        ][0]

        self.default_action = Positions.Flat.value
        self.prev_action = None
        self.prev_reward = None
        self.agent = agent_cls(**kwargs)
        self.job_name = job_name
        self._database = 'bitmex'
        self._index = 0.0
        self.channel = 'XBTUSD_OrderBookFrame_depth_21_5s'

        self.on(self.channel, self.emit_position)
        self.on(BitmexChannels.BXBT_s.value, self.set_index)

    def exit(self, *args):
        super().stop(*args)

    def set_index(self, value):
        self._index = value

    def emit_position(self, *args):
        self._emit_position(*args)
        self._push_metrics()

    def _get_observation(self):
        return self._index, self.orderbook_frame, \
               self.last_timestamp.timestamp()

    @pos_summary.time()
    def _emit_position(self, data):
        dt = self.now()
        meas = Measurement(**data)
        self.last_timestamp = meas.time
        self.orderbook_frame = np.asarray(json.loads(meas.fields['data']))

        if len(self.frames) < self.max_frames:
            self.prev_action = self.default_action
            self.prev_reward = 0.0
            self.get_observation()
        else:
            action = self.agent.compute_action(
                self.last_observation,
                prev_action=self.prev_action,
                prev_reward=self.prev_reward
            )
            # publish ???
            self.step(action)

        alog.info((self.now() - dt).total_seconds() )

    def step(self, action):
        self.prev_action = action
        obs, reward, done, info = super().step(action)
        self.prev_reward = reward
        profit_gauge.set(self.total_pnl)
        alog.info(alog.pformat(self.summary()))

    def _push_metrics(self):
        push_to_gateway(
            settings.PROMETHEUS_HOST,
            job=self.job_name,
            registry=REGISTRY
        )

    def start(self):
        self.sub([self.channel, BitmexChannels.BXBT_s])


class OrderBookTradingEvaluator(OrderBookTradingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@click.command()
@click.argument('symbol', type=click.Choice(BitmexChannels.__members__),
                default=BitmexChannels.XBTUSD.value)
@click.option('--job-name', '-n', default=None)
@click.option('--agent-cls', '-a', default=None)
def main(**kwargs):
    ray.init()
    emitter = BitmexPositionEmitter(**kwargs)
    emitter.start()


if __name__ == '__main__':
    main()
