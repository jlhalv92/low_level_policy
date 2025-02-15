import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.algorithm.deep_actor_critic.imitation.dagger import Dagger

from mushroom_rl.core import Agent
from mushroom_rl.core import Logger
from src.core.core_imitation_quadruped_navigation import Core
from src.envs.go1_env_rays import Go1
from src.networks.nets import ActorNetwork, CriticNetwork,PolicyNetwork, ValueNetwork
import wandb



from mushroom_rl.utils.dataset import compute_J, parse_dataset

from tqdm import trange


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate

def experiment(n_episodes_test=1,
                   control_agent=None,
                   student_agent=None):

    np.random.seed()

    logger = Logger("Evaluate", results_dir=None)
    logger.strong_line()
    task_name = "multi_obstacles_navigation"

    agent_task = "{}".format(task_name)
    logger.info('Experiment Algorithm: ' +"Dagger"+ " task " + agent_task )

    view = False
    # MDP
    horizon = 2000
    gamma = 0.99
    n_features = 256


    log = False
    joystick = False

    cfg = dict(kp=40,
               kd=1,
               antagonist=False,
               max_level=True,
               curriculum=False,
               joystick=joystick,
               resample=False,
               domain_randomization=False,
               rewards={},
               static_obstacles=False)

    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name)

    # actor_input_shape = mdp.info.observation_space.shape
    actor_input_shape = (14,)


    # Define Learner
    agent = student_agent

    # Load Expert

    core = Core(agent,
                mdp,
                low_level_agent=control_agent,
                obs_dim=14,
                joystick=joystick)


    dataset = core.evaluate(n_episodes=n_episodes_test, render=True)

    s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    # E = agent.policy.entropy(s, img)
    success_rate = calculate_success_rate(info, last)
    logger.epoch_info(0, J=J, R=R, success_rate=success_rate)


if __name__ == '__main__':

    path = "src/nominal_models/go1/epoch_last"
    student_path = "checkpoint/GO1/Dagger/obstacles_navigation/epoch_last"

    control_agent = Agent.load(path)
    student_agent = Agent.load(student_path)

    experiment(n_episodes_test=10,
                control_agent=control_agent,
                student_agent=student_agent)


