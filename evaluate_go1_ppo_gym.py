import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from src.algorithm.deep_actor_critic.ppo import PPO
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


class ValueNetwork(nn.Module):
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
        V = self._h4(features3)

        return V


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate


def experiment(alg, n_episodes_test=5, joystick=False):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    task_name = "nominal"
    agent_task = "{}_DR_robust_locomotion_gym".format(task_name)
    logger.info('Experiment Algorithm: ' + alg.__name__ + " task " + agent_task )

    gamma = 0.99



    env_args = dict(vx_limit=2.5, vy_limit=1, wz_limit=3)
    mdp_eval = GymnasiumGO1(gamma=gamma,
                            n_envs=1,
                            joystick=joystick,
                            domain_randomization=False,
                            horizon=1000, **env_args)
    mdp_eval.env.eval=True
    mdp_eval.env.track_velocities = True
    mdp_eval.env.resampling_command_steps = 100

    # path = "../checkpoint/GO1/PPO/nominal_DR_robust_locomotion_gym/epoch_20"
    # path = "../checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_JOYSTICK_NR/epoch_135"
    # # path = "checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_150M_domain_randomization/epoch_100"
    # # path = "checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_150M_domain_randomization/epoch_70"
    # # path = "checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_150M_domain_randomization_vmax_rt2/epoch_last"
    # # path = "checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_150M_Priv/epoch_last"
    # path = "../checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_200M_VC/epoch_120"
    # path = "../checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_200M_VC_MAX_B/epoch_last"
    # path = "../checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_100M_VC_S_2/epoch_last"
    # # path = "checkpoint/GO1/DaggerSP/nominal_RAPID_LOCOMOTION_student/epoch_90"
    # # path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_200_VC/epoch_150"
    path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_100_VC_V2/epoch_last"
    path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_100_STABLE/epoch_last"
    path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_400_STABLE/epoch_last"
    path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_100_STABLE/epoch_60"
    path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_400_STABLE_RR/epoch_120"


    # path = "../checkpoint/GO1/PPO/nominal_LOCOMOTION_400_VC_R/epoch_last"
    # path = "../checkpoint/GO1/PPO/nominal_LOCOMOTION_100_VC_V2/epoch_last"
    # path = "../checkpoint/GO1/PPO/nominal_LOCOMOTION_200_VC_NR/epoch_100"
    # path = "../checkpoint/GO1/PPO/nominal_LOCOMOTION_50_VC_NR_RT/epoch_20"
    # path = "../checkpoint/GO1/PPO/nominal_LOCOMOTION_100_VC_AGILE/epoch_last"
    # path = "../checkpoint/GO1/PPO/nominal_LOCOMOTION_500_VC_AGILE_vx=4_vy=2_wz=5_hip/epoch_400"
    # path = "../checkpoint/GO1/PPO/nominal_LOCOMOTION_100_VC_AGILE_vx=4_vy=2_wz=5_hip_KP=30_KD=0.65_regularization_torques_qvel/epoch_60"
    # path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_100_VC_AGILE/epoch_last"



    agent = Agent.load(path)
    agent.set_logger(logger)

    core_eval = Core(agent, mdp_eval, student=False)


    dataset = core_eval.evaluate(n_episodes=n_episodes_test, render=True)
    from mushroom_rl.utils.dataset import compute_J, parse_dataset
    import matplotlib.pyplot as plt
    s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
    velocities = np.array(mdp_eval.env.tracking_velocities_array)
    vx = velocities[:, 0]
    vy = velocities[:, 1]
    wz = velocities[:, 2]

    commands = s[:, [50, 51, 52]]
    cvx = commands[:, 0]
    cvy = commands[:, 1]
    cwz = commands[:, 2]

    time = np.arange(len(vx))

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    colors = plt.get_cmap('tab10')

    # Plot Vx and Command Vx
    axs[0].plot(time, cvx, color=colors(0), linestyle='-', label='Command Vx')
    axs[0].plot(time, vx, color=colors(0), label='Vx')
    axs[0].set_title('Vx and Command Vx over Time')
    axs[0].set_ylabel('Vx')
    axs[0].grid(True)
    axs[0].legend()

    # Plot Vy and Command Vy
    axs[1].plot(time, cvy, color=colors(1), linestyle='-', label='Command Vy')
    axs[1].plot(time, vy, color=colors(1) ,label='Vy')
    axs[1].set_title('Vy and Command Vy over Time')
    axs[1].set_ylabel('Vy')
    axs[1].grid(True)
    axs[1].legend()

    # Plot Wz and Command Wz
    axs[2].plot(time, cwz, color=colors(2), linestyle='-', label='Command Wz')
    axs[2].plot(time, wz, color=colors(2), label='Wz')
    axs[2].set_title('Wz and Command Wz over Time')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Wz')
    axs[2].grid(True)
    axs[2].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

    J = np.mean(compute_J(dataset, mdp_eval.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy()

    logger.epoch_info(0, J=J, R=R, entropy=E)




if __name__ == '__main__':
    experiment(alg=PPO,
                   n_episodes_test=5, joystick=False)


