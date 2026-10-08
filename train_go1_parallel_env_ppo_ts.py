import argparse

import numpy as np

import torch

import torch.optim as optim
import torch.nn.functional as F
from src.algorithm.deep_actor_critic.ppo_ts import PPO
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

class PolicyNetworkStudent(nn.Module):
    def __init__(self,  input_shape, output_shape, n_features, **kwargs):
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

# class ValueNetwork(nn.Module):
#     def __init__(self, input_shape, output_shape, n_features, **kwargs):
#         super().__init__()
#
#         n_input = input_shape[-1]
#         n_output = output_shape[0]
#
#         self._h1 = nn.Linear(n_input, 2*n_features)
#         self._h2 = nn.Linear(2*n_features, n_features)
#         self._h3 = nn.Linear(n_features, int(n_features/2))
#         self._h4 = nn.Linear(int(n_features/2), n_output)
#
#         nn.init.orthogonal_(self._h1.weight, np.sqrt(2))
#         nn.init.constant_(self._h1.bias, 0.0)
#         nn.init.orthogonal_(self._h2.weight, np.sqrt(2))
#         nn.init.constant_(self._h2.bias, 0.0)
#         nn.init.orthogonal_(self._h3.weight, np.sqrt(2))
#         nn.init.constant_(self._h3.bias, 0.0)
#         nn.init.orthogonal_(self._h4.weight, 0.01)
#         nn.init.constant_(self._h4.bias, 0.0)
#
#     def forward(self, state, **kwargs):
#         features1 = F.elu(self._h1(torch.squeeze(state, 1).float()))
#         features2 = F.elu(self._h2(features1))
#         features3 = F.elu(self._h3(features2))
#         V = self._h4(features3)
#
#         return V


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate


def experiment(alg, n_epochs,
               n_steps, n_steps_per_fit,
               n_episodes_test,
               n_envs = 10,
               n_minibatches=4,
               retrain=False,
               name_exp ="",
               log=False,
                wandb_project="LocomotionPolicy",
               config=None,
               init_epoch=0,
               target_v=None,):

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
            job_type="Locomotion_policy_ppo_" + config.name_exp,
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
    print(f"Target velocity vx: {target_v[0]} vy: {target_v[1]}  wz: {target_v[2]}")
    env_args = dict(vx_limit=target_v[0], vy_limit=target_v[1], wz_limit=target_v[2])
    # env_args = dict(vx_limit=1.5, vy_limit=0.8, wz_limit=1.5)

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

    student_actor = PolicyNetworkStudent

    obs_shape = (48,)

    student_policy = GaussianTorchPolicy(student_actor,
                                 obs_shape,
                                 action_shape,
                                 **policy_params)

    alg_params['critic_params'] = critic_params

    if retrain:

        path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_200_STABLE/epoch_last"
        path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_400_STABLE_RR/epoch_150"
        path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_100_T/epoch_last"
        path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_100_T2/epoch_last"
        path = "checkpoint/GO1/PPO/nominal_R_LOCOMOTION/epoch_last"
        path = "checkpoint/GO1/PPO/nominal_R_LOCOMOTION_S/epoch_last"
        path = "checkpoint/GO1/PPO/nominal_LOCOMOTION_RAPID_CURRICULUM/epoch_last"
        path = f"checkpoint/GO1/PPO/nominal_LOCOMOTION_ST/epoch_{config.init_epoch}"

        agent = Agent.load(path)
    else:
        agent = alg(mdp_eval.info, policy, student_policy, **alg_params)

    agent.set_logger(logger)

    mdp.env.eval = config.use_uniform_commands
    core = Core(agent, mdp,  vectorize=True)

    mdp_eval.env.eval=True
    core_eval = CoreEval(agent, mdp_eval)
    core_evalstudent = CoreEval(agent, mdp_eval)


    dataset = core_eval.evaluate(n_episodes=n_episodes_test, render=False)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy()

    dataset_student = core_evalstudent.evaluate(n_episodes=n_episodes_test, render=False, student=True)
    J_s = np.mean(compute_J(dataset_student, mdp.info.gamma))
    R_s = np.mean(compute_J(dataset_student))
    E_s = agent.student_policy.entropy()

    logger.epoch_info(0, J=J, R=R, entropy=E, J_s=J_s, R_s=R_s, entropy_S=E_s)


    for it in trange(init_epoch,n_epochs + init_epoch, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)


        if (it + 1) % 5 == 0:
            render = False
        else:
            render = False

        dataset = core_eval.evaluate(n_episodes=n_episodes_test, render=render, student=False)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy()

        dataset_student = core_evalstudent.evaluate(n_episodes=n_episodes_test, render=False, student=True)
        J_s = np.mean(compute_J(dataset_student, mdp.info.gamma))
        R_s = np.mean(compute_J(dataset_student))

        logger.epoch_info(it, J=J, R=R, entropy=E, J_s=J_s, R_s=R_s)

        if log:
            log_dict = dict(J=J,
                            R=R,
                            entropy=E)

            wandb.log(log_dict, step=it*n_envs*n_steps)

        if (it+1) % 10 == 0:
            agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1',alg.__name__, agent_task, it+1))

        # logger.epoch_info(it + 1, J=J, R=R, entropy=E)
    agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1', alg.__name__, agent_task, "last"))

    if log:
        wandb.finish()


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', '1', 'True'):
        return True
    elif value.lower() in ('no', 'false', 'f', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():

    parser = argparse.ArgumentParser(description="Parse command-line arguments for the script.")
    parser.add_argument("--n_envs", type=int, default=100, help="Number of steps")
    parser.add_argument("--n_steps", type=int, default=10_000, help="Number of steps")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of steps")
    parser.add_argument("--n_steps_per_fit", type=int, default=1000, help="Number of steps per fit")
    parser.add_argument("--n_episodes_test", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--n_minibatches", type=int, default=20, help="Number of test episodes")
    parser.add_argument("--name_exp", type=str, default="LOCOMOTION_C_TS")
    parser.add_argument("--log", type=str2bool, nargs='?', const=True, default=False,  help="Enable logging")
    parser.add_argument("--wandb_project",type=str,default="POLICY_TRAINING")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps_ppo", type=float, default=0.2)
    parser.add_argument("--n_epochs_policy", type=int, default=10)
    parser.add_argument("--init_epoch", type=int, default=0)
    parser.add_argument("--lam", type=float, default=.95)
    parser.add_argument("--ent_coeff", type=float, default= 0.01)
    parser.add_argument("--n_features", type=int, default= 256)
    parser.add_argument("--gamma", type=float, default= 0.99)
    parser.add_argument("--retrain", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--use_uniform_commands", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--target_vx", type=float, default= 1.5)
    parser.add_argument("--target_vy", type=float, default= 0.8)
    parser.add_argument("--target_wz", type=float, default= 1.5)



    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    n_envs= args.n_envs #100
    n_steps = args.n_steps
    n_steps_per_fit= args.n_steps_per_fit #1000
    n_epochs = args.n_epochs # 150
    total_data = n_epochs*n_steps
    print('Total number of steps per env: {}'.format(total_data), "total number of steps: {}".format(total_data*n_envs))
    init_epoch = args.init_epoch
    target_v = np.array([args.target_vx, args.target_vy, args.target_wz])
    experiment(alg=PPO,
               n_epochs=n_epochs,
               n_steps=int(n_steps),
               n_steps_per_fit=n_steps_per_fit,
               n_episodes_test=args.n_episodes_test,
               n_envs=n_envs,
               n_minibatches=args.n_minibatches,
               retrain=args.retrain,
               name_exp=args.name_exp.format(n_epochs),
               log=args.log,
               config=args,
               init_epoch=init_epoch,
               target_v=target_v)


