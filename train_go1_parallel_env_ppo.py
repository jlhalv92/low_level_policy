import argparse

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

from src.mushroom.dataset.dataset import compute_J
from tqdm import trange
import wandb


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
               n_episodes_test, n_envs = 10,
               n_minibatches=4,
               retrain=False,
               name_exp ="",
               log=False,
                wandb_project="LocomotionPolicy",
               config=None):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    task_name = "nominal"
    agent_task = f"{task_name}_{name_exp}"
    logger.info('Experiment Algorithm: ' + alg.__name__ + " task " + agent_task )

    # MDP

    gamma = config.gamma
    n_features = config.n_features

    mini_batch_size = int((n_envs*n_steps_per_fit)/n_minibatches) #1000 rollout 32000

    lr = config.lr
    n_epochs_policy =config.n_epochs_policy
    eps_ppo = config.eps_ppo
    lam = config.lam
    ent_coeff = config.ent_coeff # 0.01

    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_project,
            group=task_name,
            job_type="Locomotion_policy_ppo",
            name="seed_{}".format(0)
        )


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
    env_args = dict(vx_limit=2, vy_limit=1, wz_limit=2)
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
        # #BASE POLICY
        # path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_200_VC_NR/epoch_120"
        #
        # #SAFE POLICY
        # path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_50_VC_NR_RT/epoch_20"
        #
        # #AGILE POLICY
        # path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_100_VC_AGILE/epoch_last"
        # # AGILE SAFE
        # path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_400_VC_AGILE_vx=4_vy=2_wz=5_hip_KP=30_KD=0.65/epoch_220"

        # path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_100_STABLE/epoch_last"
        path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_200_STABLE/epoch_last"

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
        if log:
            log_dict = dict(J=J,
                            R=R,
                            entropy=E)

            wandb.log(log_dict, step=it)

        if (it+1) % 10 == 0:
            agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1',alg.__name__, agent_task, it+1))

        logger.epoch_info(it + 1, J=J, R=R, entropy=E)
    agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1', alg.__name__, agent_task, "last"))

    if log:
        wandb.finish()

def parse_args():

    parser = argparse.ArgumentParser(description="Parse command-line arguments for the script.")
    parser.add_argument("--n_envs", type=int, default=100, help="Number of steps")
    parser.add_argument("--n_steps", type=int, default=10_000, help="Number of steps")
    parser.add_argument("--n_epochs", type=int, default=400, help="Number of steps")
    parser.add_argument("--n_steps_per_fit", type=int, default=1000, help="Number of steps per fit")
    parser.add_argument("--n_episodes_test", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--n_minibatches", type=int, default=4, help="Number of test episodes")
    parser.add_argument("--name_exp", type=str, default="LOCOMOTION_{}_STABLE")
    parser.add_argument("--log", default=False,  help="Enable logging")
    parser.add_argument("--wandb_project",type=str,default="ADVERSARIAL_LOCOMOTION_RATIONALITY_ANALYSIS")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps_ppo", type=float, default=0.2)
    parser.add_argument("--n_epochs_policy", type=int, default=5)
    parser.add_argument("--lam", type=float, default=.95)
    parser.add_argument("--ent_coeff", type=float, default= 0.01)
    parser.add_argument("--n_features", type=int, default= 256)
    parser.add_argument("--gamma", type=float, default= 0.99)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    n_envs= args.n_envs #100
    n_steps = args.n_steps
    n_steps_per_fit= args.n_steps_per_fit #1000
    n_epochs = args.n_epochs # 150
    total_data = n_epochs*n_steps*n_steps_per_fit
    print('Total number of steps per env: {}'.format(total_data), "total number of steps: {}".format(total_data*n_envs))
    experiment(alg=PPO,
               n_epochs=n_epochs,
               n_steps=int(n_steps),
               n_steps_per_fit=n_steps_per_fit,
               n_episodes_test=args.n_episodes_test,
               n_envs=n_envs,
               n_minibatches=args.n_minibatches,
               retrain=False,
               name_exp=args.name_exp.format(n_epochs),
               log=args.log,
               config=args)


