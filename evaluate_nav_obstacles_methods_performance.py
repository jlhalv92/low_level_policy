import argparse

import numpy as np
from src.teachers.self_paced_gamma_teacher_fixed_beta import (
    SelfPaced,
)
from src.utils.buffer import Buffer
from src.utils.state_preprocessor import StateVariableConcatenation
import torch
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.utils.callbacks import CollectDataset
from src.core.core_quadruped_adversary import Core
import json


from src.algorithm.deep_actor_critic.sac import  SAC
from src.algorithm.deep_actor_critic.adaptative_temp_sac import AdaptiveTempSAC

from mushroom_rl.core import Agent
from mushroom_rl.core import Logger

from src.envs.go1_nav_multi_obstacles import Go1MultiObs

from src.networks.nets import  ActorNetwork ,CriticNetwork, ValueNetwork, ActorBiNetwork, PolicyNetwork, PolicyNetworkStudent
import wandb
from src.mushroom.dataset.dataset import compute_J,parse_dataset

from tqdm import trange


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate

def extract_temp_and_sample_return(data, gamma):
    temperatures = list()
    for i in range(len(data)):
        if data[i][-1] or i == len(data) - 1:
            temperatures.append(data[i][0][-1])

    sample_returns = compute_J(data, gamma)

    return temperatures, sample_returns

def benchmark(
        protagonist=None,
        antagonist=None,
        control_agent=None,
        aware_rationality=False,
        max_v_antagonist=1.0,
        n_episodes_test=10,
        alpha=0.5,
        protagonist_id = "model",
        seed=0,
        file_path="data.json",
        render=True,
        record=True,
        save_file=False,
    joystick=False,
horizon=500
):

    np.random.seed()

    logger = Logger("BENCHMARK", results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"

    agent_task = f"{task_name}"
    logger.info('Experiment Algorithm: ' +"BENCHMARK"+ " task " + agent_task )

    # MDP

    gamma = 0.99


    joystick = False
    cfg = dict(antagonist=True,
               max_level=False,
               joystick=joystick,
               static_obstacles=False,
               activate_arrow=False,
               max_antagonist_v=max_v_antagonist,
               n_static_obstacles=0,
               obstacle_position_x=8,
               obstacle_position_y=-1,
                corridor_size = 4)


    multiple_obstacles = True


    mdp = Go1MultiObs(horizon,
                  gamma,
                  timestep=0.005,
                  cfg=cfg,
                  task_name=task_name)


    # Algorithm
    preprocessor = StateVariableConcatenation()

    core = Core(
        [protagonist, antagonist],
        mdp,
        low_level_agent=control_agent,
        callback_step=CollectDataset(),
        preprocessors=[preprocessor],
        joystick=joystick,
        aware_rationality=aware_rationality,
        rationality_antagonist=True)



    temps = alpha*np.ones(n_episodes_test)

    core.set_variables_for_training(temps)
    dataset_online = core.evaluate(
        n_episodes=n_episodes_test, render=render, id_agent=0, return_antagonist_state=True, record=False
    )

    s, a, r, s1, absorbing, info, last = parse_dataset(dataset_online)
    J_online = np.mean(compute_J(dataset_online, mdp.info.gamma))
    R_online = np.mean(compute_J(dataset_online))
    success_rate = calculate_success_rate(info, last)
    E_online = antagonist.policy.entropy(s)

    logger.epoch_info(
        0,
        J_online=J_online,
        R_online=R_online,
        entropy=E_online,
        success_rate=success_rate,
        alfa=np.mean(temps)
    )


    return success_rate


if __name__ == "__main__":
    # args = parse_args()
    # print(args)
    path = "src/nominal_models/low_level/policy"
    control_agent = Agent.load(path)

    velocity = 1.5

    path_antagonist = "checkpoint/GO1/obstacles_navigation/UNIFORM_mv=1.5_support=0.5/run_0/ANTAGONIST/epoch_last"
    antagonist = Agent.load(path_antagonist)

    performance = []

    model_id = "MAX_ENTROPY"
    success_rates = []
    path_protagonist =  "checkpoint/GO1/obstacles_navigation/UNIFORM_mv=1.5_support=0.5/run_0/PROTAGONIST/epoch_last"

            # path_protagonist = f"/home/jose-luis/CORL_final_results/SAC_heuristic_velocity_antagonist_0.0/obstacles_navigation_velocity_antagonist_0.0/epoch_last_{i}"
    protagonist = Agent.load(path_protagonist)
    success_rate = benchmark(
                protagonist=protagonist,
                antagonist=antagonist,
                control_agent=control_agent,
                aware_rationality=False,
                max_v_antagonist=velocity,
                n_episodes_test=50,
                alpha=1.,
                protagonist_id = 0,
                # file_path=file_path,
                render=True,
                record=False,
                joystick=False,
                horizon=500,

            )





