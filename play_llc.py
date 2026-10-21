import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.algorithm.deep_actor_critic.imitation.dagger import DaggerSP
from src.algorithm.policies.torch_policy import GaussianTorchPolicy

from mushroom_rl.core import Logger
from src.core.core_gym import Core

from src.envs.gymnasium_env import GymnasiumGO1
import torch.nn as nn
from mushroom_rl.core import Agent

from mushroom_rl.utils.dataset import compute_J

from tqdm import trange


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.fc1 = nn.Linear(n_input, 2*n_features)
        self.ln1 = nn.LayerNorm(2*n_features)

        self.fc2 = nn.Linear(2*n_features, n_features)
        self.ln2 = nn.LayerNorm(n_features)

        self.fc3 = nn.Linear(n_features, n_features//2)
        self.ln3 = nn.LayerNorm(n_features//2)

        self.fc4 = nn.Linear(n_features//2, n_output)

    def forward(self, x):
        x = torch.squeeze(x, 1).float()

        x = F.silu(self.ln1(self.fc1(x)))
        x = F.silu(self.ln2(self.fc2(x)))
        x = F.silu(self.ln3(self.fc3(x)))

        return self.fc4(x)
class ValueNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, 2*n_features)
        self.ln1 = nn.LayerNorm(2*n_features)

        self._h2 = nn.Linear(2*n_features, n_features)
        self.ln2 = nn.LayerNorm(n_features)

        self._h3 = nn.Linear(n_features, n_features//2)
        self.ln3 = nn.LayerNorm(n_features//2)

        self._h4 = nn.Linear(n_features//2, n_output)

        nn.init.orthogonal_(self._h1.weight, np.sqrt(2))
        nn.init.constant_(self._h1.bias, 0.0)
        nn.init.orthogonal_(self._h2.weight, np.sqrt(2))
        nn.init.constant_(self._h2.bias, 0.0)
        nn.init.orthogonal_(self._h3.weight, np.sqrt(2))
        nn.init.constant_(self._h3.bias, 0.0)
        nn.init.orthogonal_(self._h4.weight, 0.01)
        nn.init.constant_(self._h4.bias, 0.0)

    def forward(self, state, **kwargs):
        features1 = F.silu(self.ln1(self._h1(torch.squeeze(state, 1).float())))
        features2 = F.silu(self.ln2(self._h2(features1)))
        features3 = F.silu(self.ln3(self._h3(features2)))
        V = self._h4(features3)

        return V


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate


def experiment(agent=None,
               joystick=False, name_exp ="", n_episodes=10, render=False, vel=[1.0,1.0,1.0]):

    np.random.seed()

    logger = Logger("PPO", results_dir=None)
    logger.strong_line()
    task_name = "nominal"
    # agent_task = f"{task_name}_{name_exp}"
    # logger.info('Experiment Algorithm: ' + alg.__name__ + " task " + agent_task )

    # MDP
    batch_size = 1024 #
    gamma = 0.99
    n_features = 256

    env_args = dict(vx_limit=vel[0], wz_limit=vel[2], vy_limit=vel[1])
    env_args["task"] = "flat"
    env_args["current_level"] = 1
    mdp = GymnasiumGO1(gamma=gamma,  n_envs=1, joystick=joystick, domain_randomization=False,  **env_args)
    mdp.env.eval =True
    mdp.env.track_velocities = True
    mdp.env.resampling_command_steps = 500




    agent.set_logger(logger)

    core = Core(agent, mdp, asymetric=False)

    dataset = core.evaluate(n_episodes=n_episodes, render=render)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy()

    logger.epoch_info(0, J=J, R=R, entropy=E)

    # from mushroom_rl.utils.dataset import parse_dataset
    # import matplotlib.pyplot as plt
    # s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
    # velocities = np.array(mdp.env.tracking_velocities_array)
    # vx = velocities[:, 0]
    # vy = velocities[:, 1]
    # wz = velocities[:, 2]
    #
    # commands = s[:, [45, 46, 47]]
    # cvx = commands[:, 0]
    # cvy = commands[:, 1]
    # cwz = commands[:, 2]
    #
    # time = np.arange(len(vx))

    # # Create subplots
    # fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    # colors = plt.get_cmap('tab10')

    # # Plot Vx and Command Vx
    # axs[0].plot(time, cvx, color=colors(0), linestyle='-', label='Command Vx')
    # axs[0].plot(time, vx, color=colors(0), label='Vx')
    # axs[0].set_title('Vx and Command Vx over Time')
    # axs[0].set_ylabel('Vx')
    # axs[0].grid(True)
    # axs[0].legend()

    # # Plot Vy and Command Vy
    # axs[1].plot(time, cvy, color=colors(1), linestyle='-', label='Command Vy')
    # axs[1].plot(time, vy, color=colors(1), label='Vy')
    # axs[1].set_title('Vy and Command Vy over Time')
    # axs[1].set_ylabel('Vy')
    # axs[1].grid(True)
    # axs[1].legend()
    #
    # # Plot Wz and Command Wz
    # axs[2].plot(time, cwz, color=colors(2), linestyle='-', label='Command Wz')
    # axs[2].plot(time, wz, color=colors(2), label='Wz')
    # axs[2].set_title('Wz and Command Wz over Time')
    # axs[2].set_xlabel('Time')
    # axs[2].set_ylabel('Wz')
    # axs[2].grid(True)
    # axs[2].legend()
    #
    # # Adjust layout
    # plt.tight_layout()
    # plt.show()



if __name__ == '__main__':



    student_path = "nominal_policy/stairs/epoch_last"
    low_level_control = Agent.load(student_path)
    experiment(
               name_exp="RAPID_LOCOMOTION_",
               agent=low_level_control, joystick=True, n_episodes=10, render=True, vel=[1.5,0.5,1.5])


