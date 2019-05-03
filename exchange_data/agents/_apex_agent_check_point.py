import json
import os

import ray
from pathlib import Path

import alog
import gym
from numpy.core.multiarray import ndarray
from ray.rllib.agents.dqn import DQNAgent


class ApexAgentCheckPoint(DQNAgent):
    def __init__(
        self,
        env,
        checkpoint,
        config=None,
        **kwargs
    ):
        self.checkpoint = None
        self.set_checkpoint_file(checkpoint)

        config = dict(model=self.config()['model'])

        DQNAgent.__init__(self, env=env, config=config)

        if hasattr(self, "local_evaluator"):
            state_init = self.local_evaluator.policy_map[
                "default"].get_initial_state()

    def config(self):
        config_dir = os.path.dirname(self.checkpoint)
        config_path = os.path.join(config_dir, "params.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.json")
        if not os.path.exists(config_path):
            raise ValueError(
                "Could not find params.json in either the checkpoint dir or "
                "its parent directory.")
        with open(config_path) as f:
            config = json.load(f)
        if "num_workers" in config:
            config["num_workers"] = min(2, config["num_workers"])
        alog.info(config)
        return config

    def set_checkpoint_file(self, checkpoint):
        checkpoint = Path(f'{Path.home()}{checkpoint}')
        assert checkpoint.exists()
        self.checkpoint = checkpoint.resolve()

    def compute_action(self, observation: ndarray, **kwargs):
        return super().compute_action(observation, **kwargs)

    def restore(self):
        super().restore(self.checkpoint)


