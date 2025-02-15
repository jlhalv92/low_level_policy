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



    velocity_limits=dict(limit_vx=2, limit_vy=1, limit_wz=2)
    domain_parameters = dict(tangencial_friction=1,
                             friction_torsional_ground=1,
                             friction_rolling_ground=1,
                             friction_torsional_feet=1,
                             friction_rolling_feet=1,
                             damping=1,
                             stiffnes=1,
                             gravity=9.81,
                             payload=2)
    commands = dict(vx=1, vy=0, wz=0)
    env_parameters =dict(velocity_limits=velocity_limits,
                         domain_parameters=domain_parameters,
                         commands=commands)
    # mdp = GymnasiumGO1(gamma=gamma,  n_envs=n_envs, joystick=True)
    mdp_eval = GymnasiumGO1(gamma=gamma,
                            n_envs=1,
                            joystick=joystick,
                            domain_randomization=False,
                            horizon=2000,
                            env_parameters=env_parameters)
    mdp_eval.env.eval=True

    path = "checkpoint/GO1/PPO/nominal_DR_robust_locomotion_gym/epoch_20"
    path = "checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_JOYSTICK_NR/epoch_135"
    # path = "checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_150M_domain_randomization/epoch_100"
    # path = "checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_150M_domain_randomization/epoch_70"
    # path = "checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_150M_domain_randomization_vmax_rt2/epoch_last"
    # path = "checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_150M_Priv/epoch_last"
    path = "checkpoint/GO1/PPO/nominal_RAPID_LOCOMOTION_300M_vc_rt/epoch_30"
    agent = Agent.load(path)
    agent.set_logger(logger)

    core_eval = Core(agent, mdp_eval)


    dataset = core_eval.evaluate(n_episodes=n_episodes_test, render=True)


    J = np.mean(compute_J(dataset, mdp_eval.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy()

    logger.epoch_info(0, J=J, R=R, entropy=E)




if __name__ == '__main__':
    experiment(alg=PPO,
               n_episodes_test=5, joystick=True)


