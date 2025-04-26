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
from src.core.core_quadruped_navigation import Core
import json


from src.algorithm.deep_actor_critic.sac import  SAC
from src.algorithm.deep_actor_critic.adaptative_temp_sac import AdaptiveTempSAC

from mushroom_rl.core import Agent
from mushroom_rl.core import Logger

from src.envs.go1_nav_environment_multiple_obstacles import Go1

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






    cfg = dict(max_antagonist_v=velocity,
               antagonist=False,
               activate_arrow=False,
               joystick=joystick,
               use_heuristic=False,)

    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name)




    core = Core(protagonist,
                mdp,
                low_level_agent=control_agent, joystick=joystick)


    dataset = core.evaluate(n_episodes=n_episodes_test, render=render)


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
    # args = parse_args()
    # print(args)
    path = "src/nominal_models/low_level/policy"
    control_agent = Agent.load(path)

    velocity = 0

    # level_a =0.4

    # protagonist_id =f"{model_id}_{temperature}"
    # msg =f"RUNNING_EXPERIMENT {protagonist_id}, seed {run} with antagonist temperature {temperature} max velocity {velocity}"
    # print(f"-----------------------------------------{msg}--------------------------------------------")
    performance = []
    # file_path = f"results_exp/{protagonist_id}_seed_{run}_antagonist_level_{alpha}_velocity_{velocity}.json"
    # path_protagonist = f"checkpoint/GO1/obstacles_navigation/FIX_Temp_linear_curriculum_alpha_1.0/run_{i}/PROTAGONIST/epoch_last"
    # model_id = "SAC_HEURISTIC"
    # model_id = "SAC_STATIC"
    # model_id = "MAX_ENTROPY"
    # model_id = "RARL"
    # model_id = "UNIFORM"
    model_ids = ["SAC_HEURISTIC",  "SAC_STATIC", "MAX_ENTROPY", "RARL", "UNIFORM"]
    i=0
    protagonist_paths = ["/home/jose-luis/CORL_final_results/SAC_heuristic_velocity_antagonist_1.5/obstacles_navigation_velocity_antagonist_1.5/epoch_last_{}",
                         "/home/jose-luis/CORL_final_results/SAC_heuristic_velocity_antagonist_0.0/obstacles_navigation_velocity_antagonist_0.0/epoch_last_{}",
                         "/home/jose-luis/CORL_final_results/obstacles_navigation/DORAEMON_velocity_antagonist_1.5/run_{}/PROTAGONIST/epoch_last",
                         "/home/jose-luis/CORL_final_results/obstacles_navigation/RARL_antagonist_velocity=1.5/run_{}/PROTAGONIST/epoch_last",
                         "/home/jose-luis/CORL_final_results/obstacles_navigation/UNIFORM_mv=1.5/run_{}/PROTAGONIST/epoch_last",
                         ]
    for idx,model_id in enumerate(model_ids):
        success_rates = []
        for i in range(5):
            path_protagonist = protagonist_paths[idx].format(i)
            print(path_protagonist)
            protagonist = Agent.load(path_protagonist)
            success_rate = benchmark(
                protagonist=protagonist,
                control_agent=control_agent,
                aware_rationality=False,
                max_v_antagonist=velocity,
                n_episodes_test=50,
                protagonist_id = str(i),
                # file_path=file_path,
                render=False,
                record=False,
                joystick=False,
                horizon=700,)
            success_rates.append(success_rate)

        performance.append(success_rates)

        performance_np = np.array(np.mean(performance))
        np.save(f'results_multi_obstacles/performance_{model_id}.npy', performance_np)




