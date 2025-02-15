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
from src.core.core_quadruped_adversary import Core as Core_adversary
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



def experiment_qarl(
        method_name="Hi-QARL",
        agents=[None, None, None],
        n_episodes_test=10,

):

    logger = Logger(method_name, results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: SAC vs QARL Obstacle")
    task_name = "obstacles_navigation"

    # MDP
    horizon = 500
    gamma = 0.99


    cfg_rewards = dict()

    joystick = False

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



    # Algorithm
    preprocessor = StateVariableConcatenation()

    core = Core_adversary(
        [protagonist_agent, antagonist_agent],
        mdp,
        low_level_agent=control_agent,
        callback_step=CollectDataset(),
        preprocessors=[preprocessor],
        obs_dim=60,
        joystick=joystick
    )


    evaluation_temp = 0.5
    core.set_variables_for_training(evaluation_temp * np.ones(10))

    core.callback_step.clean()
    dataset_online = core.evaluate(
        n_episodes=n_episodes_test, render=True, id_agent=0, return_antagonist_state=True
    )


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
    path_protagonist = "checkpoint/temperature_comparison/Uniform_adversary_1/agent_last_0"
    path_antagonist =  "checkpoint/temperature_comparison/Uniform_adversary_1/antagonist_last_0"
    control_agent = Agent.load(path)
    protagonist_agent = Agent.load(path_protagonist)
    antagonist_agent = Agent.load(path_antagonist)

    experiment_qarl(
        method_name="QARL",
        agents=[protagonist_agent, antagonist_agent, control_agent],
        n_episodes_test=10
    )
