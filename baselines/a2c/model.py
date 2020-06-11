from baselines.common import set_global_seeds
from baselines.common.models import register, get_network_builder
from baselines.common.policies import PolicyWithValue
from exchange_data.models.resnet.model import Model as ED_Model
from pathlib import Path

import alog
import tensorflow as tf


@register("nasnet")
def network(*args, **net_kwargs):
    net_kwargs['sequence_length'] = net_kwargs.get('max_frames')

    def network_fn(input_shape):
        return ED_Model(input_shape=input_shape, num_categories=2, include_last=False, **net_kwargs)

    return network_fn



class Model(tf.keras.Model):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """
    def __init__(self, *, env, network, seed, nsteps, run_name='default',
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), max_frames, **kwargs):

        super(Model, self).__init__(name='A2CModel')
        nenvs = None

        try:
            nenvs = env.num_env
        except:
            nenvs = env.num_envs

        set_global_seeds(seed)

        total_timesteps = int(total_timesteps)
        alpha = alpha

        total_timesteps = total_timesteps  # Calculate the batch_size
        self.nbatch = nenvs * nsteps
        self.nupdates = total_timesteps // self.nbatch
        # Get state_space and action_space
        ob_space = env.observation_space
        ac_space = env.action_space

        kwargs['batch_size'] = nsteps

        if type(network) == str:
            policy_network_fn = get_network_builder(network)(**kwargs)
            policy_network = policy_network_fn(ob_space.shape)

            policy_network.summary()
        else:
            policy_network = network

        self.run_name = run_name
        self.max_frames = max_frames
        self.ac_space = ac_space
        self.policy_network = policy_network
        self.capital = 0.0
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.train_model = PolicyWithValue(ac_space, policy_network, value_network=None, estimate_q=False)
        self.step = self.train_model.step
        self.value = self.train_model.value
        self.initial_state = self.train_model.initial_state
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon)
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)

    @tf.function
    def train(self, obs, states, rewards, masks, actions, values):
        advs = rewards - values
        with tf.GradientTape() as tape:
            policy_latent = self.train_model.policy_network(obs)
            pd, _ = self.train_model.pdtype.pdfromlatent(policy_latent)
            neglogpac = pd.neglogp(actions)
            entropy = tf.reduce_mean(pd.entropy())
            vpred = self.train_model.value(obs)
            vf_loss = tf.reduce_mean(tf.square(vpred - rewards))
            pg_loss = tf.reduce_mean(advs * neglogpac)
            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        var_list = tape.watched_variables()
        grads = tape.gradient(loss, var_list)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        grads_and_vars = list(zip(grads, var_list))
        self.optimizer.apply_gradients(grads_and_vars)

        return pg_loss, vf_loss, entropy

    def save(self):
        _export_dir = f'{Path.home()}/.exchange-data/models/a2c/model_export'
        export_dir = Path(_export_dir).mkdir(exist_ok=True)

        model: tf.keras.Model = self.train_model.policy_network

        model.save(f'{_export_dir}/{self.run_name}.h5', overwrite=True, save_format='h5')


