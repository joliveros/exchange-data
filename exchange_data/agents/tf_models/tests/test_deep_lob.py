from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
from tensorflow import Dimension

from exchange_data.agents.tf_models.deep_lob import DeepLOBModel
from gym.spaces import Discrete, Box, Dict
from ray.rllib.models import ModelCatalog

import alog
import pytest
import tensorflow as tf
import numpy as np

@pytest.fixture()
def model():
    framestack = 99
    options = {
        'conv_activation': 'relu',
        'custom_model': None,
        'custom_options': {},
        'custom_preprocessor': None,
        'fcnet_activation': 'leaky_relu',
        'free_log_std': False,
        'grayscale': False,
        'framestack': framestack,
        'lstm_cell_size': 64,
        'lstm_use_prev_action_reward': False,
        'max_seq_len': framestack,
        'squash_to_range': False,
        'use_lstm': True,
        'zero_mean': False
    }

    high = np.full(
        (framestack, 21, 4),
        np.inf
    )

    obs_space = Box(-high, high, dtype=np.float32)

    action_space = Discrete(2)
    prev_rewards = tf.placeholder(tf.float32, [None], name="prev_reward")
    obs = tf.placeholder(tf.float32, [None, framestack, 21, 4], name="obs")

    return DeepLOBModel(
        action_space=action_space,
        input_dict=dict(
            obs=obs,
            is_training=tf.placeholder_with_default(False, ()),
            prev_actions=ModelCatalog.get_action_placeholder(action_space),
            prev_rewards=prev_rewards
        ),
        num_outputs=2,
        obs_space=obs_space,
        options=options,
        seq_lens=None,
        state_in=None
    )


class TestDeepLOBModel(object):
    def test_init_deep_lob_model(self, model):
        model._validate_output_shape()
        alog.info(model)
