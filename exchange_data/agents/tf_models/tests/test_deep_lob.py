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
    orderbook_depth = 21
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
        (framestack, orderbook_depth, 2),
        1.0,
        dtype=np.float32
    )
    low = np.full(
        (framestack, orderbook_depth, 2),
        0.0,
        dtype=np.float32
    )
    position_data_high = np.full(
        (framestack, 5),
        np.inf,
        dtype=np.float32
    )

    obs_space = Dict(dict(
        levels=Box(low, high, dtype=np.float32),
        posistion_data=Box(-position_data_high,
                           position_data_high,
                           dtype=np.float32)
    ))
    # obs_space = DictFlatteningPreprocessor(obs_space).observation_space
    action_space = Discrete(2)
    prev_rewards = tf.placeholder(tf.float32, [None], name="prev_reward")
    obs = dict(
        levels=tf.placeholder(tf.float32,
                              [None, framestack, 21, 4],
                              name="levels"),
        position_data=tf.placeholder(tf.float32,
                                     [None, framestack, 5],
                                     name="position_data"))

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
