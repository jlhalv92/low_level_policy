import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.algorithm.deep_actor_critic.sac import  SAC

from mushroom_rl.core import Agent
from mushroom_rl.core import Logger
from src.core.core_quadruped_navigation import Core
from src.envs.go1_env_privilege_rays import Go1
from src.networks.nets import  CriticNetwork,PolicyNetwork, ValueNetwork, ActorBiNetwork
import wandb



from mushroom_rl.utils.dataset import compute_J, parse_dataset

from tqdm import trange


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate

def experiment(method_name ,alg, n_epochs, n_steps, n_steps_per_fit, n_episodes_test, control_agent=None, run_id=0):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"

    agent_task = "{}_50_rays_new_privilege_state_network_no_walls".format(task_name)
    logger.info('Experiment Algorithm: ' +method_name+ " task " + agent_task )

    view = False
    # MDP
    horizon = 500
    gamma = 0.99
    n_features = 256
    lr = 3e-4
    warmup_transitions = 3000
    batch_size = 256
    initial_replay_size = 3000
    retrain = False
    max_replay_size = int(1e6)
    tau = 0.005
    lr_alpha = 3e-4
    obs_size = 60 #History all: 39, No History: 39, History Rays: 79, no_pos_obstacle 71 # 205
    state_size = 60 #History all: 39*5, No History: 39 History Rays: 79, no_pos_obstacle 71 #205
    history = False

    cfg_rewards=dict()

    log = False
    joystick = False
    cfg = dict(kp=40,
               kd=1,
               antagonist=False,
               max_level=True,
               curriculum=False,
               joystick=joystick,
               resample=False,
               domain_randomization=False,
               rewards=cfg_rewards,
               static_obstacles=False)

    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name)
    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="adversarial_locomotion",
            group=task_name,
            job_type=method_name,
            name="seed_{}".format(run_id)
        )
    # actor_input_shape = mdp.info.observation_space.shape
    actor_input_shape = (state_size,)
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

    # Agent
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)
    if retrain:
        path = "checkpoint/GO1/SAC/obstacles_navigation_50_rays_new_privilege_state_network/epoch_last"
        agent = Agent.load(path)
        agent.reset_replay_memory(initial_replay_size,
                max_replay_size)

    core = Core(agent,
                mdp,
                low_level_agent=control_agent,
                obs_dim=obs_size, joystick=joystick,use_history=history)


    dataset = core.evaluate(n_episodes=10, render=True)

    s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)
    success_rate = calculate_success_rate(info, last)

    logger.epoch_info(0, J=J, R=R, entropy=E, success_rate=success_rate)

    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions)


    for it in trange(n_epochs, leave=False):


        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)


        if it % 10==0 and view:
            render=True
        else:
            render=False


        dataset = core.evaluate(n_episodes=n_episodes_test, render=render)
        s, a, r, s1, absorbing, info, last  = parse_dataset(dataset)
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)
        success_rate = calculate_success_rate(info, last)

        if log:
            logs_dict = {"RETURN": J, "REWARD": R, "succes_rate":success_rate}
            wandb.log(logs_dict, step=it)

        if it % 10 == 0:
            agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1',method_name, agent_task, it))

        logger.epoch_info(it + 1, J=J, R=R, entropy=E, success_rate=success_rate)

    agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1', method_name, agent_task, "last"))


    core.evaluate(n_episodes=10, render=True)
    if log:
        wandb.finish()

if __name__ == '__main__':

    path = "src/nominal_models/go1/epoch_last"

    control_agent = Agent.load(path)
    for i in range(1):
        experiment(method_name="SAC",
                   alg=SAC,
                   n_epochs=200,
                   n_steps=5000,
                   n_steps_per_fit=1,
                   n_episodes_test=5,
                   control_agent=control_agent, run_id=i)


