import numpy as np
from mushroom_rl.utils.replay_memory import ReplayMemory


from src.networks.nets import ActorParkourNetworkGO1, ActorNetwork,CriticNetwork, CriticParkourNetworkGO1




import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from mushroom_rl.core import Logger
from src.core.core_quadruped_navigation import Core
from src.envs.go1_env import Go1
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.core import Agent
from src.networks.nets import ActorNetwork, CriticNetwork

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

    horizon = 1000
    gamma = 0.99
    cfg_rewards=dict(tracking_goal_vel = 1.5,
                    tracking_yaw = 0.5,
                     lin_vel_z = -1.0,
    ang_vel_xy = -0.05,
    orientation = -1.,
    dof_acc = -2.5e-7,
    collision = -10.,
    action_rate = -0.1 ,
    delta_torques = -1.0e-7 ,
    torques = -0.00001 ,
    hip_pos = -0.5,
    dof_error = -0.04,
    feet_stumble = -1,
    feet_edge = -1,
    airtime = 1,
                     clip=True)

    cfg = dict(kp=40,
               kd=1,
               interpolate=1,
               use_all_obs=False,
               rewards=cfg_rewards)

    mdp = Go1(horizon, gamma,
              n_substep=4,
              timestep=0.005,
              cfg=cfg,
              task_name="obstacles_navigation")



    # path_c = "/home/jose-luis/Project/residual_rl/checkpoint/GO1/PPO/nominal_joystick_reward_param_3/epoch_280"
    # path = "checkpoint/GO1/SAC/obstacles_navigation_obstacles_navigation_r_2/epoch_190"
    path = "checkpoint/GO1/SAC/obstacles_navigation_avoid_obstacles/epoch_80"
    path = "checkpoint/GO1/SAC/obstacles_navigation_history_avoid_moving_obstacles_terminal/epoch_190"
    path_c ="checkpoint/GO1/PPO/nominal_robust_locomotion/epoch_last"

    control_agent = Agent.load(path_c)
    agent = Agent.load(path)

    core = Core(agent, mdp, low_level_agent=control_agent)


    mdp.resample=False
    mdp.level = 5
    dataset = core.evaluate(n_episodes=5, render=True)


    s,a,_,_,info,last = parse_dataset(dataset)

    E = agent.policy.entropy(s)
    # plt.plot(s[:,2] - s[:,-3])
    # plt.show()
    # plt.plot(s[:,54])
    # plt.plot(s[:,55])
    # plt.plot(s[:,56])
    # plt.show()
    vx = a[:,0]
    vy = a[:,1]
    wz = a[:,2]
    time = np.arange(len( s[:,56]))*0.02
    # Create a figure and set of subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10))

    # Plot vx in the first subplot
    ax1.plot(time, vx, color='b')
    ax1.set_title("Horizontal Velocity (vx)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Velocity (vx)")

    # Plot vy in the second subplot
    ax2.plot(time, vy, color='g')
    ax2.set_title("Vertical Velocity (vy)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Velocity (vy)")

    # Plot wz in the third subplot
    ax3.plot(time, wz, color='r')
    ax3.set_title("Angular Velocity (wz)")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Velocity (wz)")

    # Plot wz in the third subplot
    # print(s[:,-311:][:,9])
    # ax4.scatter(time, s[:,-311:][:,-300:], color='r')
    # ax4.set_title("distance to target")
    # ax4.set_xlabel("Time")
    # ax4.set_ylabel("distance to target")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))

    # success_rate = calculate_success_rate(info, last)

    logger.epoch_info(0, J=J, R=R, entropy=E)

    #


if __name__ == '__main__':

    for _ in range(1):
        experiment()
