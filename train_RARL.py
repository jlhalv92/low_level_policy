import numpy as np

import torch.optim as optim
import torch.nn.functional as F
import argparse

from src.algorithm.deep_actor_critic.asymetric_sac_v2 import  ASAC
from src.algorithm.deep_actor_critic.sac import  SAC


from mushroom_rl.core import Agent
import wandb

from mushroom_rl.core import Logger
from src.core.core_quadruped_adversary import Core
from src.envs.go1_nav_environment import Go1
from src.envs.go1_nav_multi_obstacles import Go1MultiObs

from src.networks.nets import  PolicyNavHistoryNetwork,ActorNetwork,PolicyNavNetwork, CriticNetwork,PolicyNetwork, ValueNetwork, ActorBiNetwork, PolicyNetworkStudent
from src.mushroom.dataset.dataset import compute_J,parse_dataset

from tqdm import trange



def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate


def experiment(method_name,
               alg,
               n_epochs,
               n_steps,
               n_steps_per_fit,
               n_episodes_test,
               control_agent=None,
               run_id=0,
               log=False,
               use_cuda=False,
               wandb_project="",
               antagonist_velocity=1.5,
               conf=None,
               checkpoint_dir = ""):


    np.random.seed()

    logger = Logger(conf.method, results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"
    agent_task = "{}".format(task_name)
    logger.info('Experiment Algorithm: ' + conf.method + " task " + agent_task )

    view = True
    # MDP
    horizon = 500
    gamma = 0.99
    n_features = 256
    warmup_transitions = 5000
    batch_size = 256
    initial_replay_size = 5000

    max_replay_size = int(1e6)
    tau = 0.005
    lr_alpha = 3e-4



    job_name = method_name + f"_antagonist_velocity={antagonist_velocity}"
    print(job_name)
    print(vars(conf))
    if log:
        wandb.init(
            project=wandb_project,
            group=task_name,
            job_type=job_name,
            name="seed_{}".format(run_id),
            config=vars(conf)
        )
    joystick = False
    cfg = dict(antagonist=not conf.heuristic,
               max_level=False,
               joystick=joystick,
               static_obstacles=False,
               activate_arrow=False,
               max_antagonist_v=antagonist_velocity,
               n_static_obstacles=conf.n_static_obstacles,
               obstacle_position_x=conf.obstacle_position_x,
               obstacle_position_y=conf.obstacle_position_y,
               max_pos_target=conf.max_pos_target,
                corridor_size = conf.corridor_size,
               heuristic=conf.heuristic,

               )




    mdp = Go1MultiObs(horizon,
                  gamma,
                  timestep=0.005,
                  cfg=cfg,
                  task_name=task_name)


    if conf.heuristic:
        mdp.antagonist_temperature = 0



    if conf.use_history:
        actor_obs_shape = mdp.info.observation_space.shape
        history_size = 5
        propioception = 9

        actor_input_shape = (history_size * (actor_obs_shape[0] + 3),)
        actor_mu_params = dict(network=PolicyNavHistoryNetwork,
                               n_features=n_features,
                               input_shape=actor_input_shape,
                               output_shape=(3,),
                               propioception=propioception,
                               history=history_size,
                               # size_propioception=6,
                               use_cuda=use_cuda)

        actor_sigma_params = dict(network=PolicyNavHistoryNetwork,
                                  n_features=n_features,
                                  input_shape=actor_input_shape,
                                  output_shape=(3,),
                                  history=history_size,
                                  propioception=propioception,
                                  # size_propioception=6,
                                  use_cuda=use_cuda)
    else:

        propioception = 6
        actor_input_shape  = mdp.info.observation_space.shape
        actor_mu_params = dict(network=PolicyNavNetwork,
                               n_features=n_features,
                               input_shape=actor_input_shape,
                               output_shape=(3,),
                               propioception=propioception,
                               use_cuda=use_cuda)

        actor_sigma_params = dict(network=PolicyNavNetwork,
                                  n_features=n_features,
                                  input_shape=actor_input_shape,
                                  output_shape=(3,),
                                  propioception=propioception,
                                  use_cuda=use_cuda)


    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}

    if conf.use_asymetric_ac:
        critic_obs_size = mdp.info.observation_space.shape[0]

    else:
        critic_obs_size = actor_input_shape[0]

    critic_input_shape = (critic_obs_size + 3,)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    # Agent
    min_a = -np.array([1.5,1,1.5])
    max_a = np.array([1.5, 1, 1.5])
    if conf.use_asymetric_ac:
        agent = alg[0](mdp.info, actor_mu_params, actor_sigma_params,
                    actor_optimizer, critic_params, batch_size, initial_replay_size,
                    max_replay_size, warmup_transitions, tau, lr_alpha,
                    critic_fit_params=None, range=[min_a, max_a],
                       critic_obs_size=critic_obs_size,)
    else:
        agent = alg[0](mdp.info, actor_mu_params, actor_sigma_params,
                    actor_optimizer, critic_params, batch_size, initial_replay_size,
                    max_replay_size, warmup_transitions, tau, lr_alpha,
                    critic_fit_params=None, range=[min_a, max_a])

    # Antagonist
    antagonist_input_shape = (6,)
    antagonist_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=antagonist_input_shape,
                           output_shape=(1,),
                           use_cuda=use_cuda)

    antagonist_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=antagonist_input_shape,
                              output_shape=(1,),
                              use_cuda=use_cuda)

    antagonist_critic_input_shape = (antagonist_input_shape[0] + 1,)
    antagonist_critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=antagonist_critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    antagonist = alg[1](mdp.info, antagonist_mu_params, antagonist_sigma_params,
                actor_optimizer, antagonist_critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None, range=[-antagonist_velocity, antagonist_velocity])



    core = Core([agent, antagonist],
                mdp,
                low_level_agent=control_agent,
                joystick=joystick,
                use_history=conf.use_history,
                asymetric_ac=conf.use_asymetric_ac)


    dataset = core.evaluate(n_episodes=10, render=False, id_agent=0)
    s,a, r, s1, absorbing, info, last = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))

    if conf.use_asymetric_ac:
        E = agent.policy.entropy(s[:,:-critic_obs_size])
    else:
        E = agent.policy.entropy(s)

    success_rate = calculate_success_rate(info, last)

    logger.epoch_info(0, J=J, R=R, entropy=E, succes_rate=success_rate)


    if not conf.heuristic:

        core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=1)

    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=0)


    for it in trange(n_epochs, leave=False):
        #
        #
        if not conf.heuristic:
            logger.info("Training Antagonist")
            core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit,  id_agent=1)

        logger.info("Training Protagonist")
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, id_agent=0)

        if it % 10==0 and view:
            render=True
        else:
            render=False


        dataset = core.evaluate(n_episodes=n_episodes_test, render=render, id_agent=0)

        s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        if conf.use_asymetric_ac:
            E = agent.policy.entropy(s[:,:-critic_obs_size])
        else:
            E = agent.policy.entropy(s)
        success_rate = calculate_success_rate(info, last)

        dataset_antagonist = core.evaluate(n_episodes=n_episodes_test, id_agent=1)
        s, a, r, s1, absorbing, info, last = parse_dataset(dataset_antagonist)
        R_antagonist = np.mean(compute_J(dataset_antagonist))


        if not conf.heuristic:
            E_a=antagonist.policy.entropy(s)
            alfa = antagonist._alpha_np

        else:
            E_a = mdp.heuristic_policy.mean_entropy(np.zeros(n_episodes_test))
            alfa = 0

        logger.epoch_info(it+1,
                          J=J,
                          R=R,
                          entropy=E,
                          success_rate=success_rate,
                          temperature=alfa,
                          r_antagonist=R_antagonist,
                          entropy_antagonist=E_a)



        if log:
            log_dict = dict(J=J,
                            R=R,
                            entropy=E,
                            success_rate=success_rate,
                            temperature=alfa,
                            r_antagonist=R_antagonist,
                            entropy_antagonist=E_a)
            wandb.log(log_dict, step=it)

        if it % 5 == 0:
            agent.save(f"{checkpoint_dir}/{'GO1'}/{agent_task}/{job_name}/run_{run_id}/{'PROTAGONIST'}/epoch_{it}")
            if not conf.heuristic:
                antagonist.save(f"{checkpoint_dir}/{'GO1'}/{agent_task}/{job_name}/run_{run_id}/{'ANTAGONIST'}/epoch_{it}")

    agent.save(f"{checkpoint_dir}/{'GO1'}/{agent_task}/{job_name}/run_{run_id}/{'PROTAGONIST'}/epoch_{'last'}")
    if not conf.heuristic:
        antagonist.save(f"{checkpoint_dir}/{'GO1'}/{agent_task}/{job_name}/run_{run_id}/{'ANTAGONIST'}/epoch_{'last'}")

    if log:
        wandb.finish()

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', '1', "True"):
        return True
    elif value.lower() in ('no', 'false', 'f', '0', "False"):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Parse command-line arguments for the script.")
    parser.add_argument("--n_epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--n_steps", type=int, default=5000, help="Number of steps")
    parser.add_argument("--n_steps_per_fit", type=int, default=1, help="Number of steps per fit")
    parser.add_argument("--n_episodes_test", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--run_id", type=int, default=5, help="Run ID")
    parser.add_argument("--log",type=str2bool, nargs='?', const=True, default=False,  help="Enable logging")
    parser.add_argument("--use_cuda", type=str2bool, nargs='?', const=True, default=True, help="Use CUDA for computation")
    parser.add_argument("--velocity", type=float, default=1.5, help="velocity")
    parser.add_argument("--obstacle_position_x",type=float,  default=5)
    parser.add_argument("--obstacle_position_y",type=float,  default=-1)
    parser.add_argument("--wandb_project",type=str,default="ADVERSARIAL_LOCOMOTION_DOMAIN_RANDOMIZATION")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint")
    parser.add_argument("--n_static_obstacles",type=int,  default=0)
    parser.add_argument("--corridor_size",type=float,  default=3)
    parser.add_argument("--max_pos_target",type=float,  default=3)
    parser.add_argument("--heuristic", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--method", type=str, default="RARL")
    parser.add_argument("--use_history", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--use_asymetric_ac", type=str2bool, nargs='?', const=True, default=False)


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    path = "src/nominal_models/low_level/policy"
    control_agent = Agent.load(path)

    if args.use_asymetric_ac:
        model_protagonist = ASAC
    else:
        model_protagonist = SAC

    experiment(method_name=args.method,
               alg=[model_protagonist, SAC],
               n_epochs=args.n_epochs,
               n_steps=args.n_steps,
               n_steps_per_fit=args.n_steps_per_fit,
               n_episodes_test=args.n_episodes_test,
               control_agent=control_agent,
               run_id=args.run_id,
               log=args.log,
               use_cuda=args.use_cuda,
               wandb_project=args.wandb_project,
               antagonist_velocity = args.velocity,
               checkpoint_dir=args.checkpoint_dir,
               conf=args)



