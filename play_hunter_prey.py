import argparse

import numpy as np

from src.teachers.doraemon import Doraemon
from src.utils.buffer import Buffer
from src.utils.state_preprocessor import StateVariableConcatenation
import torch
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.utils.callbacks import CollectDataset
from src.core.core_quadruped_adversary2d import Core


from src.algorithm.deep_actor_critic.sac import  SAC
from src.algorithm.deep_actor_critic.adaptative_temp_sac import AdaptiveTempSAC

from mushroom_rl.core import Agent
from mushroom_rl.core import Logger

from src.envs.go1_multi_hunter_prey import Go1

from src.networks.nets import  ActorNetwork ,CriticNetwork, ValueNetwork, ActorBiNetwork, PolicyNetwork, PolicyNetworkStudent
import wandb
from src.mushroom.dataset.dataset import compute_J,parse_dataset

from tqdm import trange



def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate



def experiment(
        method_name="",
        alg=[None, None],
        n_epochs=200,
        n_steps=5000,
        n_steps_per_fit=1,
        n_episodes_test=10,
        control_agent=None,
        run_id=0,
        log=False,
        use_cuda=False,
        view=False,
        wandb_project="",
        checkpoint_dir="",
        antagonist_velocity=1.,
        conf=None):

    np.random.seed()

    logger = Logger(method_name, results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"

    agent_task = f"{task_name}"
    logger.info('Experiment Algorithm: ' +method_name+ " task " + agent_task )

    # MDP
    horizon = 500
    gamma = 0.99
    n_features = 256
    warmup_transitions = 5000
    batch_size = 256 #256
    initial_replay_size = 5000
    max_replay_size = int(1e6)
    tau = 0.005
    lr_alpha = 3e-4
    job_name = method_name + f"_heurisitic_antagonist_mv={antagonist_velocity}"
    print(job_name)
    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_project,
            group=task_name,
            job_type=job_name,
            name="seed_{}".format(run_id)
        )




    cfg_rewards=dict()

    joystick = False
    cfg = dict(
               antagonist=False,
               joystick=joystick,
               domain_randomization=False,
               rewards=cfg_rewards,
               static_obstacles=False,
                activate_arrow=False,
                max_antagonist_v=antagonist_velocity,
                number_obstacles=5,
                social_movement=False,
                antagonist_temperature=1)

    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name)

    actor_input_shape = mdp.info.observation_space.shape


    actor_input_shape = (actor_input_shape[0],)

    # actor_mu_params = dict(network=ActorBiNetwork,
    #                        n_features=n_features,
    #                        input_shape=actor_input_shape,
    #                        output_shape=(3,),
    #                        size_propioception=9,
    #                        use_cuda=use_cuda)
    #
    # actor_sigma_params = dict(network=ActorBiNetwork,
    #                           n_features=n_features,
    #                           input_shape=actor_input_shape,
    #                           output_shape=(3,),
    #                           size_propioception=9,
    #                           use_cuda=use_cuda)
    #
    #
    # actor_optimizer = {'class': optim.Adam,
    #                    'params': {'lr': 3e-4}}
    #
    # critic_input_shape = (actor_input_shape[0] + 3,)
    # critic_params = dict(network=CriticNetwork,
    #                      optimizer={'class': optim.Adam,
    #                                 'params': {'lr': 3e-4}},
    #                      loss=F.mse_loss,
    #                      n_features=n_features,
    #                      input_shape=critic_input_shape,
    #                      output_shape=(1,),
    #                      use_cuda=use_cuda)

    # Agent

    min_a = -np.array([1.5,1,1.5])
    max_a = np.array([1.5, 1, 1.5])
    agent = alg



    # Algorithm
    preprocessor = StateVariableConcatenation()

    core = Core(
        [agent, None],
        mdp,
        low_level_agent=control_agent,
        callback_step=CollectDataset(),
        preprocessors=[preprocessor],
        joystick=joystick,
        aware_rationality=False,
        rationality_antagonist=False,
    use_antagonist=False)


    dataset_online = core.evaluate(
        n_episodes=10, render=True, id_agent=0, return_antagonist_state=True
    )
    s, a, r, s1, absorbing, info, last = parse_dataset(dataset_online)
    J_online = np.mean(compute_J(dataset_online, mdp.info.gamma))
    R_online = np.mean(compute_J(dataset_online))
    success_rate = calculate_success_rate(info, last)

    logger.epoch_info(
        0,
        J_online=J_online,
        R_online=R_online,
        success_rate=success_rate
    )





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
    parser.add_argument("--log", type=str2bool, nargs='?', const=True, default=False,  help="Enable logging")
    parser.add_argument("--use_cuda", type=str2bool, nargs='?', const=True, default=True, help="Use CUDA for computation")
    parser.add_argument("--wandb_project",type=str,default="2D_Adversary")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint")
    parser.add_argument("--velocity", type=float, default=1., help=" antagonist vel value")



    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()


    path = "src/nominal_models/low_level/policy"
    control_agent = Agent.load(path)
    run_id = args.run_id
    # path_nav = "checkpoint/GO1/obstacles_navigation/Heuristic_hard_antagonist_heurisitic_antagonist_mv=1.0/run_0/PROTAGONIST/epoch_last"
    # path_nav = "checkpoint/GO1/obstacles_navigation/Heuristic_prey_predator_heurisitic_antagonist_mv=1.0/run_0/PROTAGONIST/epoch_110"
    path_nav = "checkpoint/GO1/obstacles_navigation/Heuristic_prey_predator_heurisitic_antagonist_mv=0.0/run_0/PROTAGONIST/epoch_last"
    nav_agent = Agent.load(path_nav)

    experiment(
        method_name="Heuristic_hard_antagonist",
        alg=nav_agent,
        n_epochs=args.n_epochs,
        n_steps=args.n_steps,
        n_steps_per_fit=args.n_steps_per_fit,
        n_episodes_test=args.n_episodes_test,
        control_agent=control_agent,
        run_id=run_id,
        antagonist_velocity=args.velocity,
        log=args.log,
        use_cuda=True,
        view=True,
        wandb_project=args.wandb_project,
        checkpoint_dir = args.checkpoint_dir,
        conf=vars(args))



