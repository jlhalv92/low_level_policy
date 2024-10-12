import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.algorithm.deep_actor_critic.sac import  SAC
from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.core import Agent

from mushroom_rl.core import Logger
from src.core.core_quadruped_navigation import Core
from src.envs.go1_env import Go1
from src.networks.nets import ActorNetwork, CriticNetwork,ActorGO1Network,CriticGO1Network
import torch.nn as nn

from mushroom_rl.utils.dataset import compute_J, parse_dataset

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


def experiment(alg, n_epochs, n_steps, n_steps_per_fit, n_episodes_test, control_agent=None):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"
    # task_name = "navigation"
    agent_task = "{}_obstacle_avoidance_min_max".format(task_name)
    logger.info('Experiment Algorithm: ' + alg.__name__ + " task " + agent_task )

    # MDP
    horizon = 1000
    gamma = 0.99
    n_features = 256
    lr = 3e-4
    warmup_transitions = 3000
    batch_size = 256
    initial_replay_size = 3000

    max_replay_size = int(1e6)
    tau = 0.005
    lr_alpha = 3e-4



    cfg_rewards=dict(tracking_goal_vel = 2,
                    tracking_yaw = 1.5,
                     lin_vel_z = -1.0,
    ang_vel_xy = -0.05,
    orientation = -1.,
    dof_acc = -2.5e-7,
    collision = -1.,
    action_rate = -0.1 ,
    delta_torques = -1.0e-7 ,
    torques = -0.00001 ,
    hip_pos = -0.5,
    dof_error = -0.04,
    feet_stumble = -1,
    feet_edge = -1,
    airtime = 1,
    clip=True)

    log = False
    all_obs =False



    cfg = dict(kp=40,
               kd=1,
               interpolate=0,
               use_all_obs=all_obs,
               rewards=cfg_rewards)

    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name)

    # actor_input_shape = mdp.info.observation_space.shape
    actor_input_shape = (5*18,)
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=(3,),
                           use_cuda=True)

    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=(3,),
                              use_cuda=True)


    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}

    critic_input_shape = (actor_input_shape[0] + 3,)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=True)

    # Agent
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)

    antagonist_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=(1,),
                           use_cuda=True)

    antagonist_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=(1,),
                              use_cuda=True)

    antagonist_critic_input_shape = (actor_input_shape[0] + 1,)
    antagonist_critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=antagonist_critic_input_shape,
                         output_shape=(1,),
                         use_cuda=True)

    antagonist = alg(mdp.info, antagonist_mu_params, antagonist_sigma_params,
                actor_optimizer, antagonist_critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None, range=[-1, 1])

    # Algorithm
    core = Core([agent, antagonist], mdp, low_level_agent=control_agent)
    dataset = core.evaluate(n_episodes=5, render=True)
    s, *_ = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)

    logger.epoch_info(0, J=J, R=R, entropy=E)
    mdp.train = True
    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions)
    mdp.train = False

    for it in trange(n_epochs, leave=False):
        mdp.train=True
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        mdp.train = False

        if it % 10==0:
            render=True
        else:
            render=False



        dataset = core.evaluate(n_episodes=n_episodes_test, render=render)
        s, *_ = parse_dataset(dataset)
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)

        if it % 10 == 0:
            agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1',alg.__name__, agent_task, it))

        logger.epoch_info(it + 1, J=J, R=R, entropy=E)

    logger.info('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':

    path ="checkpoint/GO1/PPO/nominal_robust_locomotion/epoch_last"
    control_agent = Agent.load(path)
    experiment(alg=SAC,
               n_epochs=200,
               n_steps=5000,
               n_steps_per_fit=1,
               n_episodes_test=5,
               control_agent=control_agent)


