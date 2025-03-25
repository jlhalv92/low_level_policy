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

from src.envs.go1_nav_environment import Go1

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
        max_v_antagonist=1,
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




    cfg_rewards=dict()


    cfg = dict(
               antagonist=True,
               max_level=True,
               curriculum=False,
               joystick=joystick,
               resample=False,
               domain_randomization=False,
               rewards=cfg_rewards,
               static_obstacles=False,
                activate_arrow=True,
                max_antagonist_v=max_v_antagonist)


    mdp = Go1(horizon,
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

    # results_dict = dict(
    #     protagonist_id=protagonist_id,
    #     J_online=J_online,
    #     R_online=R_online,
    #     success_rate=success_rate,
    #     alfa=np.mean(temps),
    #     antagonist_velocity=max_v_antagonist,
    #     seed=seed,
    #     level_antagonist=level_antagonist)
    #
    # if save_file:
    #     print("writing results...")
    #     with open(file_path, 'w') as f:
    #         json.dump(results_dict, f, indent=4)

    return success_rate


if __name__ == "__main__":
    # args = parse_args()
    # print(args)
    path = "src/nominal_models/low_level/policy"
    control_agent = Agent.load(path)

    velocity = 1.
    model_id = "RARL_alpha"
    temperatures = np.linspace(0.5, 0, num=11)
    # level_a =0.4
    run = 4
    alpha_antagonist = 0.
    # path_protagonist = f"experiments_server/GO1/obstacles_navigation/{model_id}_{alpha}/run_{run}/PROTAGONIST/epoch_last"
    # path_protagonist = "checkpoint/GO1/obstacles_navigation/DORAEMON_aware_rationality_True_temperature_support_0_0.1/run_1/PROTAGONIST/epoch_last"
    # path_protagonist = "checkpoint/GO1/obstacles_navigation/DORAEMON_aware_rationality_True_temperature_support_0_0.4_partial_obs/run_0/PROTAGONIST/epoch_160"

    # path_antagonist = 'checkpoint/GO1/obstacles_navigation/DORAEMON_aware_rationality_True_temperature_support_0_0.1/run_0/ANTAGONIST/epoch_last'
    # path_antagonist = "checkpoint/GO1/obstacles_navigation/DORAEMON_aware_rationality_True_temperature_support_0_0.4_partial_obs/run_0/ANTAGONIST/epoch_160"
    # path_antagonist = 'checkpoint/GO1/obstacles_navigation/RARL_alpha_0.5/run_0/ANTAGONIST/epoch_20'
    path_antagonist = "checkpoint/GO1/obstacles_navigation/Antagonist_FIX_Temp_uniform_alpha_1.0/run_0/ANTAGONIST/epoch_last"

    antagonist = Agent.load(path_antagonist)

    # protagonist_id =f"{model_id}_{temperature}"
    # msg =f"RUNNING_EXPERIMENT {protagonist_id}, seed {run} with antagonist temperature {temperature} max velocity {velocity}"
    # print(f"-----------------------------------------{msg}--------------------------------------------")
    performance = []
    # file_path = f"results_exp/{protagonist_id}_seed_{run}_antagonist_level_{alpha}_velocity_{velocity}.json"

    for temperature in temperatures:
        success_rates = []
        for i in range(5):
            path_protagonist = f"checkpoint/GO1/obstacles_navigation/FIX_Temp_linear_curriculum_alpha_1.0/run_{i}/PROTAGONIST/epoch_last"
            protagonist = Agent.load(path_protagonist)
            success_rate = benchmark(
                protagonist=protagonist,
                antagonist=antagonist,
                control_agent=control_agent,
                aware_rationality=False,
                max_v_antagonist=velocity,
                n_episodes_test=50,
                alpha=temperature,
                protagonist_id = str(i),
                seed=run,
                # file_path=file_path,
                render=False,
                record=False,
                joystick=False,
                horizon=500,

            )
            success_rates.append(success_rate)
        mean_success_rate = np.mean(success_rates)
        performance.append(mean_success_rate)


    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.bar(temperatures, performance, width=0.008, color='blue', alpha=0.7)

    # Labels and title
    plt.xlabel("Temperature")
    plt.ylabel("Success Rate")
    plt.title(f"Obstacle Navigation Performance, Antagonist max velocity = {velocity}")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Show the plot
    plt.show()



