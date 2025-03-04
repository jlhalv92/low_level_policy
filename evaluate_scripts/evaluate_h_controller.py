import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.algorithm.deep_actor_critic.sac import  SAC

from mushroom_rl.core import Agent
from mushroom_rl.core import Logger
from src.core.core_quadruped_navigation import Core
from src.envs.nav_environment import Go1

from src.networks.nets import  CriticNetwork, ValueNetwork, ActorBiNetwork, PolicyNetwork
import wandb
import torch.nn as nn

class PolicyNetworkStudent(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, 2*n_features)
        self._h2 = nn.Linear(2*n_features, n_features)
        self._h3 = nn.Linear(n_features, int(n_features/2))
        self._h4 = nn.Linear(int(n_features/2), n_output)

        nn.init.orthogonal_(self._h1.weight, np.sqrt(2))
        nn.init.constant_(self._h1.bias, 0.0)
        nn.init.orthogonal_(self._h2.weight, np.sqrt(2))
        nn.init.constant_(self._h2.bias, 0.0)
        nn.init.orthogonal_(self._h3.weight, np.sqrt(2))
        nn.init.constant_(self._h3.bias, 0.0)
        nn.init.orthogonal_(self._h4.weight, 0.01)
        nn.init.constant_(self._h4.bias, 0.0)

    def forward(self, state, **kwargs):
        features1 = F.elu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.elu(self._h2(features1))
        features3 = F.elu(self._h3(features2))
        a = self._h4(features3)

        return a

from mushroom_rl.utils.dataset import compute_J, parse_dataset

from tqdm import trange


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate

def experiment(method_name ,alg, n_episodes_test, control_agent=None):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"

    agent_task = f"{task_name}_adversarial_static"
    logger.info('Experiment Algorithm: ' +method_name+ " task " + agent_task )

    # MDP
    horizon = 500
    gamma = 0.99

    initial_replay_size = 10_000
    max_replay_size = int(1e6)

    history = False

    cfg_rewards=dict()

    log = False
    joystick = False
    cfg = dict(kp=20,
               kd=0.5,
               antagonist=False,
               max_level=True,
               curriculum=False,
               joystick=joystick,
               resample=False,
               domain_randomization=False,
               rewards=cfg_rewards,
               static_obstacles=True)
    vel_controller = dict(max_vx=1.5, max_vy=1, max_wz=1.5)
    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name, **vel_controller)

    # Agent


    path = "../checkpoint/GO1/SAC/obstacles_navigation_adversarial_static/epoch_last"
    agent = Agent.load(path)
    agent.reset_replay_memory(initial_replay_size,
            max_replay_size)

    core = Core(agent,
                mdp,
                low_level_agent=control_agent, joystick=joystick,use_history=history)

    dataset = core.evaluate(n_episodes=10, render=True, record=False)

    s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)
    success_rate = calculate_success_rate(info, last)
    logger.epoch_info(0, J=J, R=R, entropy=E, success_rate=success_rate, success_rate_adversarial=0)



if __name__ == '__main__':

    path = "../src/nominal_models/nominal_student_ppo/policy"
    control_agent = Agent.load(path)
    experiment(method_name="SAC",
               alg=SAC,
               n_episodes_test=10,
               control_agent=control_agent)


