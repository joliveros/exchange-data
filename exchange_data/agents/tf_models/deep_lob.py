from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.spaces import Box
from ray.rllib.models import LSTM
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.lstm import add_time_dimension
from ray.rllib.models.misc import normc_initializer
from ray.rllib.models.model import Model
from ray.rllib.utils.annotations import override
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.layers.pooling import max_pooling1d
import alog
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.slim as slim


class DeepLOBModel(Model):
    def __init__(self,
                 input_dict,
                 obs_space,
                 action_space,
                 num_outputs,
                 options,
                 state_in=None,
                 seq_lens=None):

        # seq_lens = tf.placeholder(
        #     dtype=tf.int32,
        #     shape=(options.get('max_seq_len'),),
        #     name="seq_lens"
        # )
        # seq_lens = (options.get('max_seq_len'),)

        Model.__init__(
            self,
            input_dict,
            obs_space,
            action_space,
            num_outputs,
            options,
            state_in,
            seq_lens
        )

    @override(Model)
    def _build_layers_v2(self, input_dict, num_outputs, options):
        is_training = input_dict['is_training']
        # prev_actions = input_dict['prev_actions']
        # prev_rewards = input_dict['prev_rewards']
        obs = input_dict['obs']
        alog.debug(obs)
        orderbook_in = obs['levels']
        position_data_in = obs['position_data']

        alog.info(orderbook_in)
        alog.info(position_data_in)

        convs = [
            [16, [1, 2], 1],
            [16, [1, 4], 1],
            [16, [1, 4], 1],
            [16, [1, 2], 1],
            [16, [1, 4], 1],
            [16, [1, 4], 1],
            [16, [1, 7], 1],
        ]

        fcnet_activation = options.get("fcnet_activation", "leaky_relu")

        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu
        elif fcnet_activation == "leaky_relu":
            activation = lambda features, name=None: \
                tf.nn.leaky_relu(features, 0.1, name)

        with tf.name_scope("orderbook"):
            for i, (out_size, kernel, stride) in enumerate(convs[:-1], 1):
                orderbook_in = slim.conv2d(
                    orderbook_in,
                    out_size,
                    kernel,
                    stride,
                    padding='VALID',
                    activation_fn=activation,
                    scope="conv{}".format(i))

                alog.debug(orderbook_in)

            out_size, kernel, stride = convs[-1]
            orderbook_in = slim.conv2d(
                orderbook_in,
                out_size,
                kernel,
                stride,
                padding='VALID',
                activation_fn=activation,
                scope="conv_out")

            alog.debug(orderbook_in)

        with tf.name_scope('inception'):
            ob_1, ob_2, ob_3 = tf.split(orderbook_in,
                                        num_or_size_splits=3,
                                        axis=1)

            inception_conv = [
                [32, 1, 1],
                [32, [3, 1], 1]
            ]
            for i, (out_size, kernel, stride) in enumerate(inception_conv, 1):
                 ob_1 = slim.conv2d(
                    ob_1,
                    out_size,
                    kernel,
                    stride,
                    activation_fn=activation,
                    scope="inception_conv_1{}".format(i))

            inception_conv_2 = [
                [32, 1, 1],
                [32, [5, 1], 1]
            ]
            for i, (out_size, kernel, stride) in enumerate(inception_conv_2, 1):
                ob_2 = slim.conv2d(
                ob_2,
                out_size,
                kernel,
                stride,
                activation_fn=activation,
                scope="inception_conv_2{}".format(i))

            ob_3 = slim.max_pool2d(
                ob_3,
                kernel_size=[3, 1],
                padding='SAME',
                stride=1
            )

            ob_3 = slim.conv2d(
                ob_3,
                32,
                1,
                1,
                activation_fn=activation,
                scope='inception_conv_3')

            orderbook_in = tf.concat([ob_1, ob_2, ob_3], axis=1)
            orderbook_in = tf.squeeze(orderbook_in, axis=[2])
            alog.debug(orderbook_in)
            copy = dict(input_dict)
            copy['obs'] = orderbook_in

            last_layer = slim.flatten(orderbook_in)

            # last_layer = self.lstm_layers(copy, num_outputs, options)
            # alog.info(last_layer)

        with tf.name_scope("position_data"):
            position_data_in = slim.fully_connected(
                position_data_in,
                16,
                weights_initializer=normc_initializer(0.01),
                activation_fn=tf.nn.softmax,
                scope="position_data_out")
            position_data_in = slim.flatten(position_data_in)

        alog.debug(position_data_in)
        alog.debug(last_layer)

        with tf.name_scope("orderbook_out"):
            last_layer = tf.concat([last_layer, position_data_in], axis=1)
            alog.debug(last_layer)
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=tf.nn.softmax,
                scope="fc_out")

        return output, last_layer

    def lstm_layers(self, input_dict, num_outputs, options):
        cell_size = options.get("lstm_cell_size")
        if options.get("lstm_use_prev_action_reward"):
            action_dim = int(
                np.product(
                    input_dict["prev_actions"].get_shape().as_list()[1:]))
            features = tf.concat(
                [
                    input_dict["obs"],
                    tf.reshape(
                        tf.cast(input_dict["prev_actions"], tf.float32),
                        [-1, action_dim]),
                    tf.reshape(input_dict["prev_rewards"], [-1, 1]),
                ],
                axis=1)
        else:
            features = input_dict["obs"]

        last_layer = features
        alog.info(last_layer)
        # Setup the LSTM cell
        lstm = rnn.BasicLSTMCell(cell_size, state_is_tuple=True)
        self.state_init = [
            np.zeros(lstm.state_size.c, np.float32),
            np.zeros(lstm.state_size.h, np.float32)
        ]

        # Setup LSTM inputs
        if self.state_in:
            c_in, h_in = self.state_in
        else:
            c_in = tf.placeholder(
                tf.float32, [None, lstm.state_size.c], name="c")
            h_in = tf.placeholder(
                tf.float32, [None, lstm.state_size.h], name="h")
            self.state_in = [c_in, h_in]

        # Setup LSTM outputs
        state_in = rnn.LSTMStateTuple(c_in, h_in)
        lstm_out, lstm_state = tf.nn.dynamic_rnn(
            lstm,
            last_layer,
            initial_state=state_in,
            sequence_length=self.seq_lens,
            time_major=False,
            dtype=tf.float32)

        self.state_out = list(lstm_state)

        last_layer = tf.reshape(lstm_out, [-1, cell_size])

        return last_layer
