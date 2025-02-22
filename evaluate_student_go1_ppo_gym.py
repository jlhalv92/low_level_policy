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

class PolicyNetworkStudent(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, dropout_rate=0.2,  **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, 2 * n_features)
        # self._dropout1 = nn.Dropout(p=dropout_rate)  # Dropout added

        self._h2 = nn.Linear(2 * n_features, n_features)
        # self._dropout2 = nn.Dropout(p=dropout_rate)  # Dropout added

        self._h3 = nn.Linear(n_features, int(n_features / 2))
        # self._dropout3 = nn.Dropout(p=dropout_rate)  # Dropout added

        self._h4 = nn.Linear(int(n_features / 2), n_output)

        # Orthogonal initialization (recommended for policy networks)
        nn.init.orthogonal_(self._h1.weight, np.sqrt(2))
        nn.init.constant_(self._h1.bias, 0.0)
        nn.init.orthogonal_(self._h2.weight, np.sqrt(2))
        nn.init.constant_(self._h2.bias, 0.0)
        nn.init.orthogonal_(self._h3.weight, np.sqrt(2))
        nn.init.constant_(self._h3.bias, 0.0)
        nn.init.orthogonal_(self._h4.weight, 0.01)
        nn.init.constant_(self._h4.bias, 0.0)

    def forward(self, state):
        features1 = F.elu(self._h1(torch.squeeze(state, 1).float()))
        # features1 = self._dropout1(features1)

        features2 = F.elu(self._h2(features1))
        # features2 = self._dropout2(features2)

        features3 = F.elu(self._h3(features2))
        # features3 = self._dropout3(features3)

        a = self._h4(features3)
        return a


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


def experiment(alg, n_episodes_test=5, joystick=False):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    task_name = "nominal"
    agent_task = "{}_DR_robust_locomotion_gym".format(task_name)
    logger.info('Experiment Algorithm: ' + alg.__name__ + " task " + agent_task )

    gamma = 0.99



    env_args = dict(vx_limit=1.5,vy_limit=1, wz_limit=1.5)
    mdp_eval = GymnasiumGO1(gamma=gamma,
                            n_envs=1,
                            joystick=joystick,
                            domain_randomization=False,
                            horizon=1000, **env_args)
    mdp_eval.env.eval=True


    # path = "checkpoint/GO1/DaggerSP/nominal_RAPID_LOCOMOTION_student_v2/epoch_last"
    path = "checkpoint/GO1/DaggerSP/nominal_RAPID_LOCOMOTION_STUDENT_vx=2_wz=2/epoch_15"


    agent = Agent.load(path)
    agent.set_logger(logger)

    core_eval = Core(agent, mdp_eval, student=True)


    dataset = core_eval.evaluate(n_episodes=n_episodes_test, render=True)


    J = np.mean(compute_J(dataset, mdp_eval.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy()

    logger.epoch_info(0, J=J, R=R, entropy=E)




if __name__ == '__main__':
    experiment(alg=PPO,
                   n_episodes_test=5, joystick=True)


