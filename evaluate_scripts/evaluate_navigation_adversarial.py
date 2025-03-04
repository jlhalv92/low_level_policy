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
from src.core.core_quadruped_adversary import Core as Core_adversary


from mushroom_rl.core import Agent
from mushroom_rl.core import Logger

from src.envs.nav_environment import Go1

from src.networks.nets import  ActorNetwork ,CriticNetwork, ValueNetwork, ActorBiNetwork, PolicyNetwork, PolicyNetworkStudent
import wandb
import torch.nn as nn
from mushroom_rl.utils.dataset import compute_J, parse_dataset

def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate


def experiment_qarl(
            method_name="ADVERSARIAL",
            agents=None,
            n_episodes_test=10, evaluation_temp=0.1):

    np.random.seed()

    logger = Logger(method_name, results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"

    agent_task = f"{task_name}_adversarial_static"
    logger.info('Experiment Algorithm: ' +method_name+ " task " + agent_task )

    view = False
    # MDP
    horizon = 500
    gamma = 0.99


    cfg_rewards=dict()


    joystick = False
    cfg = dict(kp=20,
               kd=0.5,
               antagonist=True,
               max_level=True,
               curriculum=False,
               joystick=joystick,
               resample=False,
               domain_randomization=False,
               rewards=cfg_rewards,
               static_obstacles=False)
    vel_controller = dict(max_vx=1.5, max_vy=1, max_wz=1.5)
    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name, **vel_controller)


    # Algorithm
    preprocessor = StateVariableConcatenation()
    protagonist, antagonist, control_agent = agents
    core = Core_adversary(
        [protagonist, antagonist],
        mdp,
        low_level_agent=control_agent,
        callback_step=CollectDataset(),
        preprocessors=[preprocessor],
    )


    render = True
    temps = evaluation_temp * np.ones(n_episodes_test)
    core.set_variables_for_training(temps)
    core.callback_step.clean()
    logger.info("Evaluate Protagonist")
    dataset_online = core.evaluate(
        n_episodes=n_episodes_test, render=render, id_agent=0, return_antagonist_state=True
    )

    s, a, r, s1, absorbing, info, last = parse_dataset(dataset_online)
    J_online = np.mean(compute_J(dataset_online, mdp.info.gamma))
    R_online = np.mean(compute_J(dataset_online))
    success_rate_online = calculate_success_rate(info, last)
    E_online = antagonist.policy.entropy(s)



    core.set_variables_for_training(temps)
    core.callback_step.clean()
    logger.info("Evaluate Antagonist")
    dataset_antagonist = core.evaluate(n_episodes=n_episodes_test, render=False, id_agent=1)
    R_antagonist = np.mean(compute_J(dataset_antagonist))

    logger.epoch_info(
        0,
        J_online=J_online,
        R_online=R_online,
        success_rate_online=success_rate_online,
        E_online=E_online,
        mean_temperature=np.mean(temps),
        r_antagonist=R_antagonist
    )


if __name__ == "__main__":

    path = "../checkpoint/GO1/DaggerSP/nominal_RAPID_LOCOMOTION_STUDENT_vx=2_wz=2/epoch_15"
    control_agent = Agent.load(path)

    path = "../checkpoint/temperature_comparison/DORAEMON/agent_epoch_10_0"
    protagonist_agent = Agent.load(path)

    path = "../checkpoint/temperature_comparison/DORAEMON/antagonist_epoch_10_0"
    antagonist_agent = Agent.load(path)

    for i in range(4):

        method = "DORAEMON"
        if method == "DORAEMON":
            kl_max =0.05
        else:
            kl_max =0.05
        method_name = f"{method}"
        experiment_qarl(
            method_name=method_name,
            agents=[protagonist_agent, antagonist_agent, control_agent],
            n_episodes_test=10,
            evaluation_temp=0.1

        )



