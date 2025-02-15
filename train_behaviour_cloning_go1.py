import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.algorithm.deep_actor_critic.imitation.dagger import Dagger

from mushroom_rl.core import Agent
from mushroom_rl.core import Logger
from src.core.core_imitation_quadruped_navigation import Core
from src.envs.go1_env_rays import Go1
from src.networks.nets import ActorNetwork, CriticNetwork,PolicyNetwork, ValueNetwork
import wandb



from mushroom_rl.utils.dataset import compute_J, parse_dataset

from tqdm import trange


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate

def experiment(method_name ,alg, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               control_agent=None,
               path_expert=None, run_id=0):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"

    agent_task = "{}".format(task_name)
    logger.info('Experiment Algorithm: ' +method_name+ " task " + agent_task )

    view = False
    # MDP
    horizon = 1000
    gamma = 0.99
    n_features = 256
    warmup_transitions = 10000
    batch_size = 2048
    initial_replay_size = 10000

    max_replay_size = int(1e6)
    obs_size = 55

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
               rewards=cfg_rewards)

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
    actor_input_shape = (54,)
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
                       'params': {'lr': 1e-4}}



    # Define Learner
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, batch_size, initial_replay_size,
                max_replay_size, iterations=20)

    # Load Expert
    expert_agent = Agent.load(path_expert)
    core = Core(agent,
                mdp,
                low_level_agent=control_agent,
                expert_agent=expert_agent,
                obs_dim=obs_size, joystick=joystick)


    dataset = core.evaluate(n_episodes=5, render=False)

    s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    # E = agent.policy.entropy(s, img)
    success_rate = calculate_success_rate(info, last)
    loss = core.agent._loss_list
    logger.epoch_info(0, J=J, R=R, success_rate=success_rate)
    core.agent._loss_list=[]
    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions,  render=False)


    for it in trange(n_epochs, leave=False):


        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)

        #
        # if it % 10==0 and view:
        #     core.headless = False
        # else:
        #     core.headless = True


        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)
        s, a, r, s1, absorbing, info, last  = parse_dataset(dataset)
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        # E = agent.policy.entropy(s)
        success_rate = calculate_success_rate(info, last)

        if log:
            logs_dict = {"RETURN": J, "REWARD": R, "succes_rate":success_rate}
            wandb.log(logs_dict, step=it)

        if it % 10 == 0:
            agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1',method_name, agent_task, it))

        loss = core.agent._loss_list
        logger.epoch_info(it+1, J=J, R=R, success_rate=success_rate, loss=np.mean(loss))
        core.agent._loss_list = []

    agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1', method_name, agent_task, "last"))


    core.evaluate(n_episodes=5, render=False)
    if log:
        wandb.finish()

if __name__ == '__main__':

    path = "src/nominal_models/go1/epoch_last"
    path_expert = "checkpoint/GO1/SAC/obstacles_navigation_50_rays/epoch_last"
    control_agent = Agent.load(path)
    for i in range(1,5):
        experiment(method_name="Dagger",
                   alg=Dagger,
                   n_epochs=200,
                   n_steps=10_000,
                   n_steps_per_fit=1000,
                   n_episodes_test=10,
                   control_agent=control_agent,
                   run_id=i,
                   path_expert=path_expert)


