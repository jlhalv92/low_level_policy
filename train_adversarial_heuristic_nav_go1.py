import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.algorithm.deep_actor_critic.sac import SAC

from mushroom_rl.core import Agent
from mushroom_rl.core import Logger
from src.core.core_quadruped_navigation import Core

from src.envs.go1_nav_environment import Go1
from src.envs.go1_nav_multi_obstacles import Go1MultiObs

from src.networks.nets import  CriticNetwork, ValueNetwork, ActorBiNetwork, PolicyNetwork
import wandb
import torch.nn as nn


from src.utils.dataset.dataset import parse_dataset, compute_J
from tqdm import trange


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


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate

def experiment(method_name ,alg,
               n_epochs,
               n_steps,
               n_steps_per_fit,
               n_episodes_test,
               control_agent=None,
               run_id=0,
               velocity=1,
               log=False,
               use_heuristic = False,
               config=None):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"

    agent_task = f"{task_name}"
    logger.info('Experiment Algorithm: ' +method_name+ " task " + agent_task )

    view = False
    # MDP
    horizon = 500
    gamma = 0.99
    n_features = 256
    warmup_transitions = 5000
    batch_size = 256 #256
    initial_replay_size = 5000
    retrain = False
    max_replay_size = int(1e6)
    tau = 0.005
    lr_alpha = 3e-4



    joystick = False
    cfg = dict(max_antagonist_v=velocity,
               antagonist=False,
               joystick=joystick,
               use_heuristic=use_heuristic,)

    multi_obstacle = True
    if multi_obstacle:
        mdp = Go1MultiObs(horizon,
                  gamma,
                  timestep=0.005,
                  cfg=cfg,
                  task_name=task_name)
    else:
        mdp = Go1(horizon,
                  gamma,
                  timestep=0.005,
                  cfg=cfg,
                  task_name=task_name)

    job_id =method_name + f"_heuristic_velocity_antagonist_{velocity}_multi_obstacles"
    print(job_id)
    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project=config.wandb_project,
            group=task_name,
            job_type=job_id,
            name="seed_{}".format(run_id),
            config=vars(config),
        )
    actor_input_shape = mdp.info.observation_space.shape

    actor_input_shape = actor_input_shape

    actor_mu_params = dict(network=ActorBiNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=(3,),
                           size_propioception=6,
                           use_cuda=True)

    actor_sigma_params = dict(network=ActorBiNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=(3,),
                              size_propioception=6,
                              use_cuda=True)


    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}

    critic_input_shape = (actor_input_shape[0] + 3,)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=True)


    min_a = -np.array([1.5,1,1.5])
    max_a = np.array([1.5, 1, 1.5])
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None, range=[min_a, max_a])


    core = Core(agent,
                mdp,
                low_level_agent=control_agent, joystick=joystick)



    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)


    s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
    J = np.mean(compute_J(dataset, mdp.info.gamma))


    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)
    success_rate = calculate_success_rate(info, last)



    logger.epoch_info(0, J=J, R=R, entropy=E, success_rate=success_rate)

    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions)



    for it in trange(n_epochs, leave=False):


        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        loss_mean = np.mean((agent._critic_approximator[0]._last_loss, agent._critic_approximator[1]._last_loss))


        if (it+1) % 10==0 and view:
            render=True
        else:
            render=False


        dataset = core.evaluate(n_episodes=n_episodes_test, render=render)

        s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
        Q = np.mean(agent._critic_approximator.predict(s, a, prediction='min', output_tensor=False))
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)
        success_rate = calculate_success_rate(info, last)




        if log:
            logs_dict = dict(J=J , R=R, entropy=E,
                          success_rate=success_rate, critic_loss =loss_mean, Q=Q)

            wandb.log(logs_dict, step=it)

        if it % 10 == 0:
            agent.save("{}/{}/{}/{}/epoch_{}_{}".format(args.checkpoint_dir,'GO1',job_id, agent_task, it, run_id))


        logger.epoch_info(it + 1, J=J , R=R, entropy=E,
                          success_rate=success_rate, critic_loss =loss_mean, Q=Q)

    agent.save("{}/{}/{}/{}/epoch_{}_{}".format(args.checkpoint_dir,'GO1', job_id, agent_task, "last", run_id))


    core.evaluate(n_episodes=n_episodes_test, render=False)
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
    parser.add_argument("--n_steps", type=int, default=5000, help="Number of steps")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of steps")
    parser.add_argument("--n_steps_per_fit", type=int, default=1, help="Number of steps per fit")
    parser.add_argument("--n_episodes_test", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--run_id", type=int, default=0, help="Run ID")
    parser.add_argument("--log", type=str2bool, nargs='?', const=True, default=False, help="Enable logging")
    parser.add_argument("--use_heuristic", type=str2bool, nargs='?', const=True, default=False, help="Enable Heuristic")
    parser.add_argument("--use_cuda", type=str2bool, nargs='?', const=True, default=True,
                        help="Use CUDA for computation")
    parser.add_argument("--velocity", type=float, default=1.5)
    parser.add_argument("--wandb_project", type=str, default="ADVERSARIAL_LOCOMOTION")
    parser.add_argument("--method", type=str, default="SAC")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint")

    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()

    # path = "src/nominal_models/go1/epoch_last"
    # path = "checkpoint/GO1/DaggerSP/nominal_RAPID_LOCOMOTION_student/epoch_50"
    # path = "checkpoint/GO1/DaggerSP/nominal_RAPID_LOCOMOTION_STUDENT_vx=2_wz=2/epoch_15"
    path = "src/nominal_models/low_level/policy"

    # path = "checkpoint/GO1/DaggerSP/nominal_RAPID_LOCOMOTION_student/epoch_70"

    control_agent = Agent.load(path)

    experiment(method_name=args.method,
               alg=SAC,
               n_epochs=args.n_epochs,
               n_steps=args.n_steps,
               n_steps_per_fit=args.n_steps_per_fit,
               n_episodes_test=args.n_episodes_test,
               control_agent=control_agent,
               run_id=args.run_id,
               velocity=args.velocity,
               log=args.log,
               use_heuristic=args.use_heuristic,
               config=args)


