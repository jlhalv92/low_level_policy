import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from src.algorithm.deep_actor_critic.ppo import PPO
from src.algorithm.policies.torch_policy import GaussianTorchPolicy

from mushroom_rl.core import Logger
from src.core.core_quadruped import Core
from src.envs.go1_env import Go1
import torch.nn as nn
from mushroom_rl.core import Agent

from mushroom_rl.utils.dataset import compute_J

from tqdm import trange


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.orthogonal_(self._h1.weight, np.sqrt(2))
        nn.init.constant_(self._h1.bias, 0.0)
        nn.init.orthogonal_(self._h2.weight, np.sqrt(2))
        nn.init.constant_(self._h2.bias, 0.0)
        nn.init.orthogonal_(self._h3.weight, 0.01)
        nn.init.constant_(self._h3.bias, 0.0)

    def forward(self, state, **kwargs):
        features1 = F.elu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.elu(self._h2(features1))
        a = self._h3(features2)

        return a


class ValueNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.orthogonal_(self._h1.weight, np.sqrt(2))
        nn.init.constant_(self._h1.bias, 0.0)
        nn.init.orthogonal_(self._h2.weight, np.sqrt(2))
        nn.init.constant_(self._h2.bias, 0.0)
        nn.init.orthogonal_(self._h3.weight, 1.0)
        nn.init.constant_(self._h3.bias, 0.0)

    def forward(self, state, **kwargs):
        features1 = F.elu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.elu(self._h2(features1))
        a = self._h3(features2)

        return a


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate


def experiment(alg, n_epochs, n_steps, n_steps_per_fit, n_episodes_test):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    task_name = "nominal"
    agent_task = "{}_fast_locomotion".format(task_name)
    logger.info('Experiment Algorithm: ' + alg.__name__ + " task " + agent_task )

    # MDP
    horizon = 1000
    gamma = 0.99
    n_features = 256
    mini_batch_size = int((21*4096)/4) #1000 rollout 32000
    lr = 3e-4
    n_epochs_policy = 5
    eps_ppo = 0.2
    lam = .95
    ent_coeff = 0
    retrain = False




    policy_params = dict(
        std_0=1.,
        n_features=n_features,
        use_cuda=torch.cuda.is_available()

    )

    alg_params = dict(actor_optimizer={'class': optim.Adam,
                                       'params': {'lr':lr}},
                      n_epochs_policy=n_epochs_policy,
                      batch_size=mini_batch_size,
                      eps_ppo=eps_ppo,
                      lam=lam,
                      ent_coeff=ent_coeff)

    actor = PolicyNetwork
    critic = ValueNetwork
    #kp=40
    #kd=1
    cfg_rewards = dict(curriculum_steps=12e6)
    cfg = dict(kp=20.0,
               kd=0.5,
               antagonist=False,
               max_level=False,
               curriculum=False,
               joystick=False,
               resample=True,
               domain_randomization=False,
               rewards=cfg_rewards)

    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name)

    critic_params = dict(network=critic,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr':lr}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         batch_size=mini_batch_size,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(actor,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    if retrain:
        path = "../src/nominal_models/go1/epoch_last"

        agent = Agent.load(path)
    else:
        agent = alg(mdp.info, policy, **alg_params)

    agent.set_logger(logger)

    core = Core(agent, mdp)

    dataset = core.evaluate(n_episodes=5, render=True)


    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy()

    logger.epoch_info(0, J=J, R=R, entropy=E)

    for it in trange(n_epochs, leave=False):
        mdp.train=True
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        mdp.train = False

        if it % 100==0:
            render=True
        else:
            render=False


        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy()

        if (it+1) % 10 == 0:
            agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1',alg.__name__, agent_task, it+1))

        logger.epoch_info(it + 1, J=J, R=R, entropy=E)
    agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1', alg.__name__, agent_task, "last"))
    logger.info('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':


    experiment(alg=PPO,n_epochs=300, n_steps=int(10*21*4096), n_steps_per_fit=21*4096,
               n_episodes_test=5)


