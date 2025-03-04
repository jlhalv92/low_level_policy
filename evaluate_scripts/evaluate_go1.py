import numpy as np
from mushroom_rl.utils.replay_memory import ReplayMemory

from src.networks.nets import ActorParkourNetworkGO1, ActorNetwork,CriticNetwork, CriticParkourNetworkGO1

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from mushroom_rl.core import Logger
from src.core.core_quadruped import Core
from src.envs.go1_env import Go1
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.core import Agent


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
def experiment():
    np.random.seed()

    logger = Logger(results_dir=None)


    # MDP

    horizon = 10000
    gamma = 0.99
    cfg_rewards=dict(curriculum_steps=1)


    cfg = dict(kp=40,
               kd=1,
               antagonist=False,
               max_level=False,
               curriculum=False,
               joystick=True,
               resample=False,
               domain_randomization=False,
               rewards=cfg_rewards)

    mdp = Go1(horizon, gamma,
              timestep=0.005,
              cfg=cfg,
              task_name="nominal")



    # path = "/home/jose-luis/Project/residual_rl/checkpoint/GO1/PPO/nominal_joystick_reward_param_1/epoch_280"
    # path = "checkpoint/GO1/PPO/nominal_joystick/epoch_99"
    # path = "checkpoint/GO1/PPO/nominal_joystick/epoch_200"
    # path = "checkpoint/GO1/PPO/nominal_joystick_tuning_2/epoch_150"
    # path = "checkpoint/GO1/PPO/nominal_joystick_reward_param_3/epoch_280"
    # path = "checkpoint/GO1/PPO/nominal_joystick_tuning/epoch_last"


    path = "../checkpoint/GO1/PPO/nominal_robust_locomotion_domain_randomization_new/epoch_450"
    path = "../src/nominal_models/go1/epoch_last"
    path = "../src/nominal_models/go1/epoch_last"
   # path = "checkpoint/GO1/PPO/nominal_robust_locomotion_domain_randomization/epoch_1400"
    #path = "src/nominal_models/GO_1/PPO_low_level/epoch_last"
    # path = "checkpoint/GO1/PPO/nominal_robust_locomotion_1000_epochs/epoch_160"
   # path = "checkpoint/GO1/PPO/nominal_robust_locomotion_1000_epochs/epoch_40"


    agent = Agent.load(path)

    # agent = Agent.load("src/nominal_models/walker/nominal_walker")

    # print(
    # agent._replay_memory.size)
    # exit()
    # Algorithm
    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=5, render=True, record=False)


    s,a,s_1,reward,absorbing,info,last = parse_dataset(dataset)

    E = agent.policy.entropy(s)
    # plt.plot(s[:,2] - s[:,-3])
    # plt.show()

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))

    # success_rate = calculate_success_rate(info, last)

    logger.epoch_info(0, J=J, R=R, entropy=E)

    #


if __name__ == '__main__':

    for _ in range(1):
        experiment()
