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
        max_v_antagonist=1.,
        view=False,
        n_episodes_test=10,
        alpha=0.5,
        use_cuda=True,
        protagonist_id = "model",
        seed=0,
        level_antagonist=0,
        file_path="data.json"):

    np.random.seed()

    logger = Logger("BENCHMARK", results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"

    agent_task = f"{task_name}"
    logger.info('Experiment Algorithm: ' +"BENCHMARK"+ " task " + agent_task )

    # MDP
    horizon = 500
    gamma = 0.99




    cfg_rewards=dict()

    joystick = False
    cfg = dict(
               antagonist=True,
               joystick=joystick,
               rewards=cfg_rewards,
                max_antagonist_v=max_v_antagonist)

    vel_controller = dict(max_vx=1.5, max_vy=1, max_wz=1.5)

    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name, **vel_controller)


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
        rationality_antagonist=False)



    temps = alpha*np.ones(n_episodes_test)

    core.set_variables_for_training(temps)
    dataset_online = core.evaluate(
        n_episodes=n_episodes_test, render=False, id_agent=0, return_antagonist_state=True
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

    results_dict = dict(
        protagonist_id=protagonist_id,
        J_online=J_online,
        R_online=R_online,
        success_rate=success_rate,
        alfa=np.mean(temps),
        antagonist_velocity=max_v_antagonist,
        seed=seed,
        level_antagonist=level_antagonist)

    print("writing results...")
    with open(file_path, 'w') as f:
        json.dump(results_dict, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Parse command-line arguments for the script.")
    parser.add_argument("--method", type=str, default="DORAEMON",choices=["DORAEMON", "UNIFORM"],help="DORAEMON or UNIFORM")
    parser.add_argument("--n_steps", type=int, default=5000, help="Number of steps")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of steps")
    parser.add_argument("--n_steps_per_fit", type=int, default=1, help="Number of steps per fit")
    parser.add_argument("--n_episodes_test", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--run_id", type=int, default=0, help="Run ID")
    parser.add_argument("--log",default=False,  help="Enable logging")
    parser.add_argument("--use_cuda", type=bool, default=True, help="Use CUDA for computation")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value")
    parser.add_argument("--kl_max",type=float,  default=0.05,  help="kl_max")
    parser.add_argument("--aware_rationality", default=False,  help="Enable logging")


    return parser.parse_args()

if __name__ == "__main__":


    path = "src/nominal_models/low_level/policy"
    control_agent = Agent.load(path)
    rationalities = [False]
    runs = [0, 1, 2, 3, 4]
    antagonist_level = [0.5 , 0.2, 0.0] # [0.5 , 0.2, 0.0]
    velocity = 0.5
    model_id = "UNIFORM_aware_rationality"
    for level_a, alpha_antagonist in enumerate(antagonist_level):
        for rationality in rationalities:
            for run in runs:

                path_protagonist = f"experiments_server/GO1/obstacles_navigation/{model_id}_{rationality}/run_{run+1}/PROTAGONIST/epoch_last"
                protagonist = Agent.load(path_protagonist)


                path_antagonist = f"experiments_server/GO1/obstacles_navigation/RARL_alpha_{alpha_antagonist}/run_0/ANTAGONIST/epoch_last"
                antagonist = Agent.load(path_antagonist)

                protagonist_id =f"{model_id}_{rationality}"
                msg =f"RUNNING_EXPERIMENT {protagonist_id}, seed {run} with antagonist level {level_a} max velocity {velocity}"
                print(f"-----------------------------------------{msg}--------------------------------------------")
                file_path = f"results_exp/{protagonist_id}_seed_{run}_antagonist_level_{level_a}_velocity_{velocity}.json"
                benchmark(
                    protagonist=protagonist,
                    antagonist=antagonist,
                    control_agent=control_agent,
                    aware_rationality=rationality,
                    max_v_antagonist=velocity,
                    n_episodes_test=50,
                    alpha=alpha_antagonist,
                    use_cuda=True,
                    protagonist_id = protagonist_id,
                    seed=run,
                    level_antagonist=level_a,
                    file_path=file_path

                )



