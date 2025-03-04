import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from mushroom_rl.core import Logger
from src.core.core_quadruped_navigation import Core
from src.envs.go1_env_rays import Go1
# from src.envs.go1_env_privilege_rays import Go1

from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.core import Agent
from src.networks.nets import ActorNetwork, CriticNetwork, PolicyNetwork, ValueNetwork

class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        # Intrinsic Processing (Expand 4 → 10)
        self._h_intrinsic = nn.Linear(4, 10)
        nn.init.xavier_uniform_(self._h_intrinsic.weight, gain=nn.init.calculate_gain('relu'))

        # Extrinsic Processing (54 → n_features → 20)
        self._h_extrinsic_1 = nn.Linear(54, n_features)
        self._h_extrinsic_2 = nn.Linear(n_features, 20)  # Increased from 10 → 20

        nn.init.xavier_uniform_(self._h_extrinsic_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h_extrinsic_2.weight, gain=nn.init.calculate_gain('relu'))

        # State Embedding (Intrinsic: 10 + Extrinsic: 20 = 30)
        self._h1 = nn.Linear(30, n_features)
        self._h2 = nn.Linear(n_features, n_features)

        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain('relu'))

        # Output Layer
        self._out = nn.Linear(n_features, n_output)
        nn.init.xavier_uniform_(self._out.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        state_extrinsic_privilege = state[:,4:]
        state_intrinsic = state[:,:4]

        # Intrinsic Processing
        intrinsic_transformed = F.relu(self._h_intrinsic(state_intrinsic.float()))

        # Extrinsic Processing
        features_extrinsic_1 = F.relu(self._h_extrinsic_1(torch.squeeze(state_extrinsic_privilege, 1).float()))
        features_extrinsic_2 = F.relu(self._h_extrinsic_2(features_extrinsic_1))

        # Concatenate Intrinsic and Extrinsic Features
        state_embedding = torch.cat((intrinsic_transformed, features_extrinsic_2.float()), dim=1)

        # Fully Connected Layers
        features1 = F.relu(self._h1(torch.squeeze(state_embedding, 1).float()))
        features2 = F.relu(self._h2(features1))

        # Output Action
        a = self._out(features2)

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

    horizon = 500
    gamma = 0.99
    cfg_rewards=dict()
    joystick = False


    cfg = dict(kp=40,
               kd=1,
               antagonist=False,
               max_level=True,
               curriculum=False,
               joystick=joystick,
               resample=False,
               domain_randomization=False,
               rewards=cfg_rewards,
               static_obstacles=True)

    mdp = Go1(horizon, gamma,
              timestep=0.005,
              cfg=cfg,
              task_name="obstacles_navigation") # multi_obstacles_navigation


    path = "../checkpoint/GO1/SAC/obstacles_navigation_random_move/epoch_70"
    path = "../checkpoint/GO1/SAC/obstacles_navigation_history_rays/epoch_last"
    path = "../checkpoint/GO1/SAC/obstacles_navigation_50_rays/epoch_last"
    path = "../checkpoint/GO1/Dagger/obstacles_navigation/epoch_last"
    path = "../checkpoint/GO1/SAC/obstacles_navigation_50_rays/epoch_last"
    path = "../checkpoint/GO1/SAC/obstacles_navigation_50_rays_new_privilege_state_network_no_walls_retrain/epoch_150"


    path_c = "../src/nominal_models/go1/epoch_last"


    control_agent = Agent.load(path_c)
    agent = Agent.load(path)

    core = Core(agent, mdp,obs_dim=58, low_level_agent=control_agent, joystick=joystick)
    dataset = core.evaluate(n_episodes=5, render=True,headless=False, record=False)

    s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
    success_rate = calculate_success_rate(info, last)

    E = agent.policy.entropy(s)

    # vx = a[:,0]
    # vy = a[:,1]
    # wz = a[:,2]
    # time = np.arange(len( s[:,56]))*0.02
    # # Create a figure and set of subplots
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
    #
    # # Plot vx in the first subplot
    # ax1.plot(time, vx, color='b')
    # ax1.set_title("Horizontal Velocity (vx)")
    # ax1.set_xlabel("Time")
    # ax1.set_ylabel("Velocity (vx)")
    #
    # # Plot vy in the second subplot
    # ax2.plot(time, vy, color='g')
    # ax2.set_title("Vertical Velocity (vy)")
    # ax2.set_xlabel("Time")
    # ax2.set_ylabel("Velocity (vy)")
    #
    # # Plot wz in the third subplot
    # ax3.plot(time, wz, color='r')
    # ax3.set_title("Angular Velocity (wz)")
    # ax3.set_xlabel("Time")
    # ax3.set_ylabel("Velocity (wz)")



    # Adjust layout to prevent overlap
    # plt.tight_layout()
    # plt.show()
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))

    # success_rate = calculate_success_rate(info, last)

    logger.epoch_info(0, J=J, R=R, entropy=E, success_rate=success_rate)

    #


if __name__ == '__main__':


    experiment()
