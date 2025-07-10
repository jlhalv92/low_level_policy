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
from src.core.core_quadruped_adversary2d import Core
import json


from src.algorithm.deep_actor_critic.sac import  SAC
from src.algorithm.deep_actor_critic.adaptative_temp_sac import AdaptiveTempSAC

from mushroom_rl.core import Agent
from mushroom_rl.core import Logger

from src.envs.go1_nav_environment_hunter_antagonist_obstacle import Go1

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
        use_antagonist=False,
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



    cfg = dict(
               antagonist=False,
               joystick=joystick,
               domain_randomization=False,
               static_obstacles=False,
                activate_arrow=False,
                max_antagonist_v=max_v_antagonist,)

    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name)


    # Algorithm
    preprocessor = StateVariableConcatenation()

    core = Core(
        [protagonist, None],
        mdp,
        low_level_agent=control_agent,
        callback_step=CollectDataset(),
        preprocessors=[preprocessor],
        joystick=joystick,
        aware_rationality=False,
        rationality_antagonist=False,
    use_antagonist=False)



    # temps = alpha*np.ones(n_episodes_test)
    temps = alpha*np.ones(n_episodes_test)

    core.set_variables_for_training(temps)
    dataset_online = core.evaluate(
        n_episodes=n_episodes_test,
        render=render,
        id_agent=0,
        return_antagonist_state=True,
        record=False
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
    model_ids = ["UNIFORM", "QARL", "RARL", "RAMEN"]
    velocity = 1.
    temperature = 0
    performance = []

    path_antagonist = "checkpoint_strong_antagonist/GO1/obstacles_navigation/SAC_ANTAGONIST-cd0316c4/run_0/ANTAGONIST/epoch_150"



    success_rates_list = []
    antagonist = Agent.load(path_antagonist)
    path_protagonist = "checkpoint/GO1/obstacles_navigation/Heuristic_hard_antagonist_heurisitic_antagonist_mv=1.0/run_0/PROTAGONIST/epoch_last"

    protagonist = Agent.load(path_protagonist)
    success_rate = benchmark(
                protagonist=protagonist,
                antagonist=antagonist,
                control_agent=control_agent,
                aware_rationality=False,
                max_v_antagonist=velocity,
                n_episodes_test=10,
                alpha=temperature,
                render=True,
                use_antagonist=False,
                joystick=False,
                horizon=500

            )
    success_rates_list.append(success_rate)






