from baselines import logger
from baselines.a2c.model import Model
from baselines.a2c.runner import Runner
from baselines.common import set_global_seeds, explained_variance
from baselines.common.models import get_network_builder
from baselines.ppo2.ppo2 import safemean
from collections import deque
from pathlib import Path
from tensorboard.plugins.hparams import api as hp

import alog
import os.path as osp
import tensorflow as tf
import time


# tf.debugging.set_log_device_placement(True)

tf.get_logger().setLevel('DEBUG')


def learn(
    env,
    nsteps,
    gamma=0.99,
    lr=7e-4,
    log_interval=100,
    load_path=None,
    run_name='default',
    hparams={},
    **network_kwargs):

    # alog.info(alog.pformat(network_kwargs))
    # raise Exception()

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''
    run_dir = f'{Path.home()}/.exchange-data/models/a2c/{run_name}'
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams, trial_id=run_name)

        # Instantiate the model object (that creates step_model and train_model)
        model = Model(nsteps=nsteps, env=env, lr=lr, run_name=run_name, **network_kwargs)

        if load_path is not None:
            load_path = osp.expanduser(load_path)
            alog.info(load_path)
            ckpt = tf.train.Checkpoint(model=model)
            manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
            ckpt.restore(manager.latest_checkpoint)

        # Instantiate the runner object
        runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
        epinfobuf = deque(maxlen=100)

        # Start total timer
        tstart = time.time()

        total_updates = model.nupdates + 1

        train_updates = total_updates * 0.8
        reset_for_eval = False

        for update in range(1, total_updates):
            # Get mini batch of experiences
            obs, states, rewards, masks, actions, values, epinfos = runner.run()

            capital = runner.env.envs[0].env.capital

            if capital != 1.0 and update < train_updates:
                model.capital = capital
                tf.summary.scalar('capital', capital, step=update)

            if update >= train_updates:
                if not reset_for_eval:
                    reset_for_eval = True
                    runner.env.envs[0].env.reset()

                model.capital = runner.env.envs[0].env.capital

                tf.summary.scalar('eval_capital', model.capital, step=update)

            epinfobuf.extend(epinfos)

            obs = tf.constant(obs)
            if states is not None:
                states = tf.constant(states)
            rewards = tf.constant(rewards)
            masks = tf.constant(masks)
            actions = tf.constant(actions)
            values = tf.constant(values)

            if update < train_updates:
                model.train(obs, states, rewards, masks, actions, values)

        return model

