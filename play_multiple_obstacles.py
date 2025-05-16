import argparse

import numpy as np
from src.teachers.self_paced_gamma_teacher_fixed_beta import (
    SelfPaced,
)
from src.utils.buffer import Buffer
import os
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

from src.envs.go1_nav_multi_obstacles import Go1MultiObs as Go1
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
        render=True,
        record=False,
    joystick=False,
horizon=500,
        n_static_obstacles=1,
use_antagonist =False
):

    np.random.seed()

    logger = Logger("BENCHMARK", results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"

    agent_task = f"{task_name}"
    logger.info('Experiment Algorithm: ' +"BENCHMARK"+ " task " + agent_task )

    # MDP

    gamma = 0.99


    cfg = dict(max_antagonist_v=max_v_antagonist,
               antagonist=use_antagonist,
               activate_arrow=False,
               joystick=joystick,
               use_heuristic=False,
               static_obstacle=False,
               n_static_obstacles=n_static_obstacles)

    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name)


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

    temps = alpha * np.ones(n_episodes_test)
    core.set_variables_for_training(temps)
    dataset = core.evaluate(n_episodes=n_episodes_test, render=render, record=record, id_agent=0)


    s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
    J_online = np.mean(compute_J(dataset, mdp.info.gamma))
    R_online = np.mean(compute_J(dataset))
    success_rate = calculate_success_rate(info, last)

    logger.epoch_info(
        0,
        J_online=J_online,
        R_online=R_online,
        success_rate=success_rate
    )


    return success_rate


if __name__ == "__main__":

    path = "src/nominal_models/low_level/policy"
    control_agent = Agent.load(path)
    max_v_antagonist = 1.5
    n_static_obstacles =1
    save_results =True

    model_ids = ["RAMEN", "HEURISTIC", "STATIC", "RARL",  "UNIFORM"]

    # path_protagonist = "/home/jose-luis/experiments_different_supports/final_result_corl_not_safety_penalties/GO1/obstacles_navigation/DORAEMON_velocity_antagonist_1.5_support=0.5/run_{}/PROTAGONIST/epoch_last"
    #
    path_protagonists = ["/home/jose-luis/data_server/checkpoint_MULTI_OBJECTS/GO1/obstacles_navigation/RAMEN_velocity_antagonist_1.5_multi_obstacles/run_{}/PROTAGONIST/epoch_last",
                         "/home/jose-luis/data_server/checkpoint_MULTI_OBJECTS/GO1/HEURISTIC_SAC_heuristic_velocity_antagonist_1.5_multi_obstacles/obstacles_navigation/epoch_last_{}",
                         "/home/jose-luis/data_server/checkpoint_MULTI_OBJECTS/GO1/STATIC_SAC_heuristic_velocity_antagonist_0.0_multi_obstacles/obstacles_navigation/epoch_last_{}",
                         "/home/jose-luis/data_server/checkpoint_MULTI_OBJECTS/GO1/obstacles_navigation/RARL_antagonist_velocity=1.5/run_{}/PROTAGONIST/epoch_last",
                         "/home/jose-luis/data_server/checkpoint_MULTI_OBJECTS/GO1/obstacles_navigation/UNIFORM_mv=1.5_support=0.5/run_{}/PROTAGONIST/epoch_last"
                         ]

    # path_antagonist = "checkpoint/GO1/obstacles_navigation/Antagonist_FIX_Temp_uniform_alpha_1.0/run_0/ANTAGONIST/epoch_last"
    #path_antagonist = "/home/jose-luis/Project/adversarial_locomotion/checkpoint/GO1/obstacles_navigation/SAC_ANTAGONIST_EV_alpha_1.0/run_0/ANTAGONIST/epoch_last"
    #path_antagonist = "/home/jose-luis/data_server/checkpoint_MULTI_OBJECTS/GO1/obstacles_navigation/RARL_antagonist_velocity=1.5/run_0/ANTAGONIST/epoch_last"
    path_antagonist = "/home/jose-luis/data_server/checkpoint_MULTI_OBJECTS/GO1/obstacles_navigation/UNIFORM_mv=1.5_support=0.5/run_0/ANTAGONIST/epoch_last"
    alphas = [1, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

    for alpha in alphas:
        for model_idx, model_name in enumerate(model_ids):
            performance = []
            print("EVALUATION MODEL: {}".format(model_name))
            for i in range(5):
                path_protagonist = path_protagonists[model_idx].format(i)
                # print(path_protagonist)
                protagonist = Agent.load(path_protagonist)
                antagonist = Agent.load(path_antagonist)
                success_rate = benchmark(
                    protagonist=protagonist,
                    control_agent=control_agent,
                    aware_rationality=False,
                    max_v_antagonist=max_v_antagonist,
                    n_episodes_test=50,
                    antagonist= antagonist,
                    # file_path=file_path,
                    render=True,
                    record=False,
                    joystick=False,
                    horizon=500,
                    alpha = alpha,
                n_static_obstacles=n_static_obstacles,
                use_antagonist=True,)

                performance.append(success_rate)

            performance_np = np.array(np.mean(performance))
            results_dir = f"results_multi_obstacles_alpha_{alpha}_velocity_{max_v_antagonist}_number_static_obstacles_{n_static_obstacles}"

            if save_results:
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)

                np.save(f'{results_dir}/performance_{model_name}.npy', performance_np)




