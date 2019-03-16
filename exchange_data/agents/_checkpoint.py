from pathlib import Path

import alog
import gym
from numpy.core.multiarray import ndarray
from ray.rllib.agents.dqn import ApexAgent


class AgentCheckPoint(ApexAgent):
    def __init__(
        self,
        env_name='orderbook-trading-v0',
        config=None,
        checkpoint='/ray_results/orderbook-apex-v3/APEX_orderbook-trading'
                   '-v0_0_2019-03-15_09-43-04v2xyo9_t/checkpoint_88/'
                   'checkpoint-88'
    ):
        super().__init__(env=env_name, config=config)
        if config is None:
            config = dict()

        self.checkpoint = None
        self.use_lstm = False
        self.config = config
        self.set_checkpoint_file(checkpoint)

        if hasattr(self, "local_evaluator"):
            state_init = self.local_evaluator.policy_map[
                "default"].get_initial_state()
        else:
            state_init = []

        self.state_init = state_init

        if state_init:
            self.use_lstm = True

    def set_checkpoint_file(self, checkpoint):
        checkpoint = Path(f'{Path.home()}{checkpoint}')
        alog.debug(checkpoint.exists())
        assert checkpoint.exists()
        self.checkpoint = checkpoint.resolve()

    def compute_action(self, observation: ndarray, **kwargs):
        if self.use_lstm:
            return super().compute_action(
                observation, state=self.state_init, **kwargs)
        else:
            return super().compute_action(observation, **kwargs)

    def restore(self):
        super().restore(self.checkpoint)


