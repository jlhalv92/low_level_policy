import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.algorithm.deep_actor_critic.sac import  SAC

from mushroom_rl.core import Agent
from mushroom_rl.core import Logger
from src.core.core_quadruped_navigation import Core
# from src.core.core_quadruped_navigation_old_policy import Core
# from src.envs.go1_env_obstacles_rays import Go1
# from src.envs.go1_env_obstacles_rays_old_policy import Go1
from src.envs.nav_environment import Go1

from src.networks.nets import  CriticNetwork, ValueNetwork, ActorBiNetwork, PolicyNetwork
import wandb
import torch.nn as nn

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

    agent_task = f"{task_name}_adversarial_static"
    logger.info('Experiment Algorithm: ' +method_name+ " task " + agent_task )

    view = False
    # MDP
    horizon = 500
    gamma = 0.99
    n_features = 256
    warmup_transitions = 10_000
    batch_size = 256 #256
    initial_replay_size = 10_000
    retrain = False
    max_replay_size = int(1e6)
    tau = 0.005
    lr_alpha = 3e-4
    history = False

    cfg_rewards=dict()

    log = True
    joystick = False
    cfg = dict(kp=20,
               kd=0.5,
               antagonist=False,
               max_level=True,
               curriculum=False,
               joystick=joystick,
               resample=False,
               domain_randomization=False,
               rewards=cfg_rewards,
               static_obstacles=True)
    vel_controller = dict(max_vx=1.5, max_vy=1, max_wz=1.5)
    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name, **vel_controller)
    cfg["static_obstacles"] = False
    mdp_eval = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name, **vel_controller)
    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="test_rewards",
            group=task_name,
            job_type=method_name,
            name="seed_{}".format(run_id)
        )
    actor_input_shape = mdp.info.observation_space.shape

    actor_input_shape = (actor_input_shape[0],)
    actor_mu_params = dict(network=ActorBiNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=(3,),
                           size_propioception=9,
                           use_cuda=True)

    actor_sigma_params = dict(network=ActorBiNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=(3,),
                              size_propioception=9,
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

    min_a = -np.array([1.5,1,1.5])
    max_a = np.array([1.5, 1, 1.5])
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None, range=[min_a, max_a])
    if retrain:
        path = "checkpoint/GO1/SAC/obstacles_navigation_50_rays_new_privilege_state_network/epoch_last"
        agent = Agent.load(path)
        agent.reset_replay_memory(initial_replay_size,
                max_replay_size)

    core = Core(agent,
                mdp,
                low_level_agent=control_agent, joystick=joystick,use_history=history)
    core_eval = Core(agent,
                mdp_eval,
                low_level_agent=control_agent, joystick=False,use_history=history)



    dataset = core.evaluate(n_episodes=n_episodes_test, render=True)



    s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
    J = np.mean(compute_J(dataset, mdp.info.gamma))


    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)
    success_rate = calculate_success_rate(info, last)

    dataset_e = core_eval.evaluate(n_episodes=n_episodes_test, render=False)
    s, a, r, s1, absorbing, info, last = parse_dataset(dataset_e)
    success_rate_adversarial = calculate_success_rate(info, last)

    logger.epoch_info(0, J=J, R=R, entropy=E, success_rate=success_rate, success_rate_adversarial=success_rate_adversarial)


    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions)



    for it in trange(n_epochs, leave=False):


        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        loss_mean = np.mean((agent._critic_approximator[0]._last_loss, agent._critic_approximator[1]._last_loss))


        if (it+1) % 10==0 and view:
            render=True
        else:
            render=False


        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        s, a, r, s1, absorbing, info, last  = parse_dataset(dataset)
        Q = np.mean(agent._critic_approximator.predict(s, a, prediction='min', output_tensor=False))
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)
        success_rate = calculate_success_rate(info, last)

        dataset_e = core_eval.evaluate(n_episodes=n_episodes_test, render=render)
        s, a, r, s1, absorbing, info, last  = parse_dataset(dataset_e)
        success_rate_adversarial = calculate_success_rate(info, last)



        if log:
            logs_dict = dict(J=J , R=R, entropy=E,
                          success_rate=success_rate,
                          success_rate_adversarial=success_rate_adversarial, critic_loss =loss_mean, Q=Q)

            wandb.log(logs_dict, step=it)

        if it % 10 == 0:
            agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1',method_name, agent_task, it))


        logger.epoch_info(it + 1, J=J , R=R, entropy=E,
                          success_rate=success_rate,
                          success_rate_adversarial=success_rate_adversarial, critic_loss =loss_mean, Q=Q)

    agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1', method_name, agent_task, "last"))


    core.evaluate(n_episodes=n_episodes_test, render=False)
    if log:
        wandb.finish()

if __name__ == '__main__':

    # path = "src/nominal_models/go1/epoch_last"
    # path = "checkpoint/GO1/DaggerSP/nominal_RAPID_LOCOMOTION_student/epoch_50"
    path = "checkpoint/GO1/DaggerSP/nominal_RAPID_LOCOMOTION_STUDENT_vx=2_wz=2/epoch_15"

    # path = "checkpoint/GO1/DaggerSP/nominal_RAPID_LOCOMOTION_student/epoch_70"
    #
    control_agent = Agent.load(path)
    for i in range(5):
        experiment(method_name="SAC",
                   alg=SAC,
                   n_epochs=200,
                   n_steps=5000,
                   n_steps_per_fit=1,
                   n_episodes_test=10,
                   control_agent=control_agent, run_id=i)


