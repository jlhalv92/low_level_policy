import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from src.algorithm.deep_actor_critic.imitation.dagger import DaggerSP
from src.algorithm.policies.torch_policy import GaussianTorchPolicy

from mushroom_rl.core import Logger
from src.core.core_imitation_ppo import Core

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


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                return True
        return False


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate


def experiment(alg, n_epochs,
               n_steps,
               n_steps_per_fit,
               teacher=None, name_exp =""):

    np.random.seed()
    early_stopping = EarlyStopping(patience=3)

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    task_name = "nominal"
    agent_task = f"{task_name}_{name_exp}"
    logger.info('Experiment Algorithm: ' + alg.__name__ + " task " + agent_task )

    # MDP
    batch_size = 1024 #
    gamma = 0.99
    n_features = 256

    policy_params = dict(
        std_0=1.,
        n_features=n_features,
        use_cuda=torch.cuda.is_available()

    )

    actor = PolicyNetworkStudent
    env_args = dict(vx_limit=2, wz_limit=2)

    mdp = GymnasiumGO1(gamma=gamma,  n_envs=1, domain_randomization=True,  **env_args)
    mdp.env.eval =True

    obs_shape = (48,)
    action_shape = (mdp.info.action_space.shape[0],)



    policy = GaussianTorchPolicy(actor,
                                 obs_shape,
                                 action_shape,
                                 **policy_params)

    # initial_replay_size = 10_000

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 1e-3}}
    # Define Learner
    agent = alg(mdp.info, policy,
                actor_optimizer, batch_size, iterations=20)


    agent.set_logger(logger)

    core = Core(agent, mdp, expert_agent=teacher)

    dataset = core.evaluate(n_episodes=5, render=False)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy()

    logger.epoch_info(0, J=J, R=R, entropy=E)
    # print("warm-up")
    # core.learn(n_steps=10_000, n_steps_per_fit=10_000, render=False)

    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        render = False
        if it % 10==0:
            render = False
        dataset = core.evaluate(n_episodes=10, render=render)
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy()
        val_loss = np.mean(agent._loss_list)
        if early_stopping(val_loss):
            break  # Stop training if early stopping condition is met
        if (it+1) % 1 == 0:
            agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1',alg.__name__, agent_task, it+1))

        logger.epoch_info(it + 1, J=J, R=R, loss= val_loss, entropy= E, beta=core.beta)
        agent._loss_list = []
    agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1', alg.__name__, agent_task, "last"))
    logger.info('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':

    n_envs= 1 #100
    n_epochs = 40 # 150

    # teacher_path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_200_VC_NR/epoch_100"
    teacher_path = "../checkpoint/GO1/PPO/nominal_LOCOMOTION_100_VC_AGILE/epoch_last"



    teacher = Agent.load(teacher_path)
    experiment(alg=DaggerSP,
               n_epochs=n_epochs,
               n_steps=20_000,
               n_steps_per_fit=20_000,
               name_exp="RAPID_LOCOMOTION_STUDENT_vx=2_wz=2",
               teacher=teacher)


