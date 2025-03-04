import numpy as np

import torch.optim as optim
import torch.nn.functional as F
import argparse

from src.algorithm.deep_actor_critic.sac import  SAC
from src.algorithm.deep_actor_critic.fix_temp_sac import fixTempSAC

from mushroom_rl.core import Agent
import wandb

from mushroom_rl.core import Logger
from src.core.core_quadruped_adversary import Core
from src.envs.nav_environment import Go1
from src.networks.nets import ActorNetwork, CriticNetwork,PolicyNetwork, ValueNetwork, ActorBiNetwork, PolicyNetworkStudent
from mushroom_rl.utils.dataset import compute_J, parse_dataset

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
               alpha=1.):

    np.random.seed()

    logger = Logger("RARL", results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"
    agent_task = "{}".format(task_name)
    logger.info('Experiment Algorithm: ' + "RARL" + " task " + agent_task )


    view = False
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


    cfg_rewards=dict()



    if log:
        wandb.init(
            project="RATIONALITY_COMPARISON",
            group=task_name,
            job_type=method_name,
            name="seed_{}".format(run_id)
        )
    joystick = False
    cfg = dict(kp=20,
               kd=0.5,
               antagonist=True,
               max_level=False,
               curriculum=False,
               joystick=joystick,
               resample=False,
               domain_randomization=False,
               rewards=cfg_rewards,
               static_obstacles=False, activate_arrow=False)



    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name)

    actor_input_shape  = mdp.info.observation_space.shape


    actor_mu_params = dict(network=ActorBiNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=(3,),
                           size_propioception=9,
                           use_cuda=use_cuda)

    actor_sigma_params = dict(network=ActorBiNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=(3,),
                              size_propioception=9,
                              use_cuda=use_cuda)


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
                         use_cuda=use_cuda)

    # Agent
    antagonist_input_shape = (6,)
    agent = alg[0](mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)

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
                critic_fit_params=None, range=[-1, 1], alpha=alpha)


    core = Core([agent, antagonist],
                mdp,
                low_level_agent=control_agent, joystick=joystick)


    dataset = core.evaluate(n_episodes=5, render=False, id_agent=0)

    s,a, r, s1, absorbing, info, last = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)
    success_rate = calculate_success_rate(info, last)

    logger.epoch_info(0, J=J, R=R, entropy=E, succes_rate=success_rate)

    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=1)
    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=0)


    for it in trange(n_epochs, leave=False):
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
        E = agent.policy.entropy(s)
        success_rate = calculate_success_rate(info, last)

        dataset_antagonist = core.evaluate(n_episodes=n_episodes_test, render=render, id_agent=1)
        s, a, r, s1, absorbing, info, last = parse_dataset(dataset_antagonist)
        R_antagonist = np.mean(compute_J(dataset_antagonist))
        alfa = antagonist._alpha.cpu().numpy()
        E_a=antagonist.policy.entropy(s)
        logger.epoch_info(it+1, J=J, R=R, entropy=E, succes_rate=success_rate, alfa=alfa, r_antagonist=R_antagonist, entropy_antagonist=E_a)


        if log:
            logs_dict = {"RETURN": J, "REWARD": R, "succes_rate":success_rate, "alfa":alfa, "REWARD_antagonist":R_antagonist}
            wandb.log(logs_dict, step=it)

        if it % 10 == 0:
            agent.save("checkpoint/{}/{}/{}/{}/{}/epoch_{}".format('GO1',agent_task,method_name,run_id,'Protagonist', it))
            antagonist.save("checkpoint/{}/{}/{}/{}/{}/epoch_{}".format('GO1',agent_task,method_name,run_id,"ANTAGONIST", it))



    agent.save("src/nominal_models/{}/{}/{}/{}/{}/epoch_{}".format('GO1',agent_task,method_name,run_id, 'Protagonist', "last"))
    antagonist.save("src/nominal_models/{}/{}/{}/{}/{}/epoch_{}".format('GO1',agent_task,method_name,run_id, "ANTAGONIST", "last"))

    if log:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Parse command-line arguments for the script.")

    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--n_steps", type=int, default=5000, help="Number of steps")
    parser.add_argument("--n_steps_per_fit", type=int, default=1, help="Number of steps per fit")
    parser.add_argument("--n_episodes_test", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--run_id", type=int, default=0, help="Run ID")
    parser.add_argument("--log", action="store_true",default=False,  help="Enable logging")
    parser.add_argument("--use_cuda", type=bool, default=True, help="Use CUDA for computation")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    path = "src/nominal_models/low_level/policy"
    control_agent = Agent.load(path)
    experiment(method_name="RARL",
               alg=[SAC, fixTempSAC],
               n_epochs=args.n_epochs,
               n_steps=args.n_steps,
               n_steps_per_fit=args.n_steps_per_fit,
               n_episodes_test=args.n_episodes_test,
               control_agent=control_agent,
               run_id=args.run_id,
               log=args.log,
               use_cuda=args.use_cuda,
               alpha=args.alpha)



