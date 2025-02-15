import os
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.utils.callbacks import CollectDataset

from src.algorithm.deep_actor_critic.sac import SAC
from src.algorithm.deep_actor_critic.adaptative_temp_sac import AdaptiveTempSAC
from pathlib import Path
from mushroom_rl.core import Agent
from mushroom_rl.core import Logger
from src.core.core_adversary_eval import Core as Core_adversary
from src.envs.go1_env_privilege_rays import Go1
from src.networks.nets import ActorNetwork, CriticNetwork, PolicyNetwork, ValueNetwork, ActorBiNetwork
from src.teachers.self_paced_gamma_teacher_fixed_beta import (
    SelfPacedGammaTemperatureTeacherFixedBeta,
)
from src.utils.buffer import Buffer
from src.utils.state_preprocessor import StateVariableConcatenation
import wandb
from mushroom_rl.utils.dataset import compute_J, parse_dataset



def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate



def experiment(
        method_name="Hi-QARL",
        agents=[None, None, None],
        n_episodes_test=10,rationality=0.,
        add_temperature_antagonist=False,
        add_temperature_protagonist=False,
        joystick =False,

):

    logger = Logger(method_name, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: SAC vs QARL Obstacle")
    task_name = "obstacles_navigation"

    # MDP
    horizon = 500
    gamma = 0.99


    cfg_rewards = dict()



    cfg = dict(kp=40,
               kd=1,
               antagonist=True,
               max_level=False,
               curriculum=False,
               joystick=joystick,
               resample=False,
               domain_randomization=False,
               rewards=cfg_rewards,
               static_obstacles=True)

    mdp = Go1(horizon, gamma, timestep=0.005, cfg=cfg, task_name=task_name)
    protagonist_agent, antagonist_agent, control_agent = agents
    # Control agent
    control_agent = control_agent

    core = Core_adversary(
        [protagonist_agent, antagonist_agent],
        mdp,
        low_level_agent=control_agent,
        obs_dim=60,
        joystick=joystick
    )

    dataset_online = core.evaluate(
        n_episodes=n_episodes_test, render=True, id_agent=0, return_antagonist_state=True
    ,temperature=rationality, add_temp_antagonist=add_temperature_antagonist,
        add_temp_protagonist=add_temperature_protagonist)


    s, a, r, s1, absorbing, info, last = parse_dataset(dataset_online)
    J_online = np.mean(compute_J(dataset_online, mdp.info.gamma))
    R_online = np.mean(compute_J(dataset_online))
    success_rate_online = calculate_success_rate(info, last)
    E_online = antagonist_agent.policy.entropy(s)

    logger.epoch_info(
        0,
        J_online=J_online,
        R_online=R_online,
        entropy=E_online,
        success_rate_online=success_rate_online
    )




if __name__ == "__main__":
    path = "src/nominal_models/go1/epoch_last"
    # path_protagonist = "checkpoint/temperature_comparison/Uniform_adversary_1/agent_last_0"
    # path_antagonist =  "checkpoint/temperature_comparison/Uniform_adversary_1/antagonist_last_0"
    # path_protagonist = "checkpoint/GO1/obstacles_navigation/RARL/0/Protagonist/epoch_190"
    # path_antagonist = "checkpoint/GO1/obstacles_navigation/RARL/0/ANTAGONIST/epoch_190"
    path_antagonist = "src/nominal_models/GO1/obstacles_navigation/RARL/0/ANTAGONIST/epoch_last"
    path_protagonist = "checkpoint/temperature_comparison/QARL_1/agent_last_0"
    # path_antagonist = "checkpoint/temperature_comparison/QARL_1/antagonist_last_0"

    control_agent = Agent.load(path)
    protagonist_agent = Agent.load(path_protagonist)
    antagonist_agent = Agent.load(path_antagonist)

    experiment(
        method_name="RARL",
        agents=[protagonist_agent,
                antagonist_agent,
                control_agent],
        n_episodes_test=10,
        rationality=0.001,
        add_temperature_antagonist=False,
        add_temperature_protagonist=True,
        joystick=True
    )
