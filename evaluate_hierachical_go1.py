import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from mushroom_rl.core import Logger
from src.core.core_quadruped_navigation import Core
from src.envs.go1_env import Go1
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.core import Agent
from src.networks.nets import ActorNetwork, CriticNetwork, PolicyNetwork, ValueNetwork


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
    cfg_rewards=dict()

    cfg = dict(kp=40,
               kd=1,
               antagonist=False,
               max_level=True,
               curriculum=False,
               joystick=False,
               resample=False,
               rewards=cfg_rewards)

    mdp = Go1(horizon, gamma,
              timestep=0.005,
              cfg=cfg,
              task_name="multi_obstacles_navigation") # multi_obstacles_navigation


    path ="checkpoint/GO1/Curriculum_SAC/obstacles_navigation_Curriculum_velocity/epoch_last"
    path_c ="checkpoint/GO1/PPO/nominal_robust_locomotion/epoch_last"

    control_agent = Agent.load(path_c)
    agent = Agent.load(path)

    core = Core(agent, mdp, low_level_agent=control_agent, joystick=False)


    mdp.resample=False
    mdp.level = 5
    dataset = core.evaluate(n_episodes=5, render=True, record=False)

    s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
    E = agent.policy.entropy(s)

    vx = a[:,0]
    vy = a[:,1]
    wz = a[:,2]
    time = np.arange(len( s[:,56]))*0.02
    # Create a figure and set of subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

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
