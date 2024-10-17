import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.algorithm.deep_actor_critic.sac import  SAC

from mushroom_rl.core import Agent
import wandb

from mushroom_rl.core import Logger
from src.core.core_quadruped_adversary import Core
from src.envs.go1_env import Go1
from src.networks.nets import ActorNetwork, CriticNetwork,PolicyNetwork, ValueNetwork
from mushroom_rl.utils.dataset import compute_J, parse_dataset

from tqdm import trange



def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate


def experiment(method_name, alg, n_epochs,episode, n_steps_per_fit, n_episodes_test, control_agent=None, run_id=0):

    np.random.seed()
    use_antagonist = True
    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"
    agent_task = "{}".format(task_name)
    logger.info('Experiment Algorithm: ' + alg.__name__ + " task " + agent_task )


    view = False
    # MDP
    horizon = 1000
    gamma = 0.99
    n_features = 256
    warmup_transitions = 5000
    batch_size = 256
    initial_replay_size = 5000

    max_replay_size = int(1e6)
    tau = 0.005
    lr_alpha = 3e-4

    obs_size = 18

    cfg_rewards=dict()

    log = True

    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="adversarial_obstacle_avoidance_locomotion",
            group=task_name,
            job_type=method_name,
            name="seed_{}".format(run_id)
        )

    cfg = dict(kp=40,
               kd=1,
               antagonist=False,
               max_level=False,
               curriculum=False,
               joystick=False,
               resample=False,
               rewards=cfg_rewards)



    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name)

    actor_input_shape = (5*obs_size,)
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=(3,),
                           use_cuda=True)

    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=(3,),
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

    # Agent
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)

    antagonist_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=(1,),
                           use_cuda=True)

    antagonist_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=(1,),
                              use_cuda=True)

    antagonist_critic_input_shape = (actor_input_shape[0] + 1,)
    antagonist_critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=antagonist_critic_input_shape,
                         output_shape=(1,),
                         use_cuda=True)

    antagonist = alg(mdp.info, antagonist_mu_params, antagonist_sigma_params,
                actor_optimizer, antagonist_critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None, range=[-2, 2])


    core = Core([agent, antagonist],
                mdp,
                low_level_agent=control_agent,
                obs_dim=obs_size)

    mdp_eval = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name="multi_obstacles_navigation")

    core_eval = Core([agent, antagonist],
                mdp_eval,
                low_level_agent=control_agent,
                obs_dim=obs_size)



    dataset = core.evaluate(n_episodes=episode, render=False, id_agent=0)

    s, *_ = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp_eval.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)

    dataset = core_eval.evaluate(n_episodes=episode, render=False, id_agent=0)
    s, *_ = parse_dataset(dataset)
    J_r = np.mean(compute_J(dataset, mdp_eval.info.gamma))

    logger.epoch_info(0, J=J, J_multi_agent=J_r, R=R, entropy=E)

    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=1)
    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=0)


    for it in trange(n_epochs, leave=False):
        logger.info("Training Antagonist")
        core.learn(n_episodes=episode, n_steps_per_fit=n_steps_per_fit,  id_agent=1)
        logger.info("Training Protagonist")
        core.learn(n_episodes=episode, n_steps_per_fit=n_steps_per_fit, id_agent=0)

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

        dataset = core_eval.evaluate(n_episodes=episode, render=False, id_agent=0)
        s, *_ = parse_dataset(dataset)
        J_r = np.mean(compute_J(dataset, mdp_eval.info.gamma))
        logger.epoch_info(it + 1, J=J, J_multi_agent=J_r, R=R, entropy=E)

        if log:
            logs_dict = {"RETURN": J, "Multi_obstacle_RETURN":J_r, "REWARD": R, "succes_rate":success_rate}
            wandb.log(logs_dict, step=it)

        if it % 10 == 0:
            agent.save("checkpoint/{}/{}/{}/{}/{}/epoch_{}".format('GO1',agent_task,method_name,run_id,'Protagonist', it))
            antagonist.save("checkpoint/{}/{}/{}/{}/{}/epoch_{}".format('GO1',agent_task,method_name,run_id,"ANTAGONIST", it))



    agent.save("src/nominal_models/{}/{}/{}/{}/{}/epoch_{}".format('GO1',agent_task,method_name,run_id, 'Protagonist', "last"))
    antagonist.save("src/nominal_models/{}/{}/{}/{}/{}/epoch_{}".format('GO1',agent_task,method_name,run_id, "ANTAGONIST", "last"))

    if log:
        wandb.finish()

if __name__ == '__main__':

    path ="checkpoint/GO1/PPO/nominal_robust_locomotion/epoch_last"
    control_agent = Agent.load(path)
    for i in range(5):
        experiment(method_name="Adversarial_SAC",
                   alg=SAC,
                   n_epochs=200,
                   episode=5,
                   n_steps_per_fit=1,
                   n_episodes_test=5,
                   control_agent=control_agent,
                   run_id=i)



