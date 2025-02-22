import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from src.algorithm.deep_actor_critic.ppo import PPO
from src.algorithm.policies.torch_policy import GaussianTorchPolicy

from mushroom_rl.core import Logger
from src.core.core_vectorize import Core
from src.core.core_gym import Core as CoreEval

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


def experiment(alg, n_epochs,
               n_steps, n_steps_per_fit,
               n_episodes_test, n_envs = 10, n_minibatches=4, retrain=False, name_exp =""):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    task_name = "nominal"
    agent_task = f"{task_name}_{name_exp}"
    logger.info('Experiment Algorithm: ' + alg.__name__ + " task " + agent_task )

    # MDP

    gamma = 0.99
    n_features = 256

    mini_batch_size = int((n_envs*n_steps_per_fit)/n_minibatches) #1000 rollout 32000

    lr = 1e-3
    n_epochs_policy = 5
    eps_ppo = 0.2
    lam = .95
    ent_coeff = 0.01 # 0.01


    retrain = retrain




    policy_params = dict(
        std_0=1.,
        n_features=n_features,
        use_cuda=torch.cuda.is_available()

    )

    alg_params = dict(actor_optimizer={'class': optim.Adam,
                                       'params': {'lr':lr}},
                      n_epochs_policy=n_epochs_policy,
                      batch_size=mini_batch_size,
                      eps_ppo=eps_ppo,
                      lam=lam,
                      ent_coeff=ent_coeff)

    actor = PolicyNetwork
    critic = ValueNetwork
    # env_args = dict(vx_limit=1.2, wz_limit=1.5)
    env_args = dict(vx_limit=3, vy_limit=2, wz_limit=3)
    mdp = GymnasiumGO1(gamma=gamma,  n_envs=n_envs, domain_randomization=True, **env_args)
    mdp_eval = GymnasiumGO1(gamma=gamma, n_envs=1, domain_randomization=True, **env_args)

    obs_shape = (mdp_eval.info.observation_space.shape[0],)
    action_shape = (mdp_eval.info.action_space.shape[0],)


    critic_params = dict(network=critic,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr':lr}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         batch_size=mini_batch_size,
                         input_shape=obs_shape,
                         output_shape=(1,),
                         use_cuda=torch.cuda.is_available())

    policy = GaussianTorchPolicy(actor,
                                 obs_shape,
                                 action_shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    if retrain:
        #BASE POLICY
        path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_200_VC_NR/epoch_120"

        #SAFE POLICY
        path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_50_VC_NR_RT/epoch_20"

        #AGILE POLICY
        path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_100_VC_AGILE/epoch_last"
        # AGILE SAFE
        path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_400_VC_AGILE_vx=4_vy=2_wz=5_hip_KP=30_KD=0.65/epoch_220"

        agent = Agent.load(path)
    else:
        agent = alg(mdp_eval.info, policy, **alg_params)

    agent.set_logger(logger)

    core = Core(agent, mdp,  vectorize=True)

    mdp_eval.env.eval=True
    core_eval = CoreEval(agent, mdp_eval)

    dataset = core_eval.evaluate(n_episodes=n_episodes_test, render=False)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy()

    logger.epoch_info(0, J=J, R=R, entropy=E)



    for it in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)


        if (it + 1) % 5 == 0:
            render = False
        else:
            render = False

        dataset = core_eval.evaluate(n_episodes=n_episodes_test, render=render)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy()

        if (it+1) % 10 == 0:
            agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1',alg.__name__, agent_task, it+1))

        logger.epoch_info(it + 1, J=J, R=R, entropy=E)
    agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1', alg.__name__, agent_task, "last"))
    logger.info('Press a button to visualize')
    input()
    core_eval.evaluate(n_episodes=n_episodes_test, render=True)


if __name__ == '__main__':

    n_envs= 100 #100
    n_steps = 10
    n_steps_per_fit= 2000 #1000
    n_epochs = 100 # 150
    total_data = n_epochs*n_steps*n_steps_per_fit
    print('Total number of steps per env: {}'.format(total_data), "total number of steps: {}".format(total_data*n_envs))
    experiment(alg=PPO,
               n_epochs=n_epochs,
               n_steps=int(20_000),
               n_steps_per_fit=n_steps_per_fit,
               n_episodes_test=10,
               n_envs=n_envs,
               n_minibatches=20,
               retrain=False,
               name_exp=f"LOCOMOTION_{n_epochs}_VC_AGILE_vx=4_vy=2_wz=5_hip_KP=30_KD=0.65_regularization_torques_qvel")


