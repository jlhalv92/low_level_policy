import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.algorithm.deep_actor_critic.sac import  SAC

from mushroom_rl.core import Agent

from mushroom_rl.core import Logger
from src.core.core_quadruped_navigation import Core
from src.envs.go1_env import Go1
from src.networks.nets import ActorNetwork, CriticNetwork,PolicyNetwork, ValueNetwork
from src.teachers.self_paced_gamma_teacher_fixed_beta import SelfPacedGammaTemperatureTeacherFixedBeta


from src.utils.buffer import Buffer
from src.utils.state_preprocessor import StateVariableConcatenation

from mushroom_rl.utils.dataset import compute_J, parse_dataset

from tqdm import trange

class Teacher:
    def __init__(self,
                 type="gamma_fixed_beta",
                 initial_conc: np.ndarray = np.array([50]),
                 target_conc: np.ndarray = np.array([1]),
                    fixed_rate: np.ndarray = np.array([1000]),
                    conc_lower_bound: float = 1e-8,
                    max_kl: float = 0.5,
                    performance_lower_bound: int = 10,  # cheetah: 10, walker: 10, hopper: 1, cartpole: 10, quad:100
                    target_kl_threshold: float = 0.01,
                    temperature_bounds: list = [-np.inf, np.inf],
                    n_samples_for_dist_update: int = 30):

        self.teacher = SelfPacedGammaTemperatureTeacherFixedBeta(
            initial_conc,
            target_conc,
            fixed_rate,
            max_kl,
            performance_lower_bound,
            conc_lower_bound,
            target_kl_threshold,
            temperature_bounds,
        )
    def provide_teacher(self):
        return self.teacher

    def sample_n_temperatures(self, n):
        sampled_temperatures = np.zeros(n)
        for i in range(n):
            sampled_temperatures[i] = self.teacher.sample()

        return sampled_temperatures



def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate


def experiment(alg, n_epochs, n_steps, n_steps_per_fit, n_episodes_test, control_agent=None):

    np.random.seed()
    use_antagonist = True
    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"
    # task_name = "navigation"
    agent_task = "{}_obstacle_avoidance_min_max".format(task_name)
    logger.info('Experiment Algorithm: ' + alg.__name__ + " task " + agent_task )

    use_quarl=None
    view = False
    # MDP
    horizon = 1000
    gamma = 0.99
    n_features = 256
    lr = 3e-4
    warmup_transitions = 3000
    batch_size = 256
    initial_replay_size = 3000

    max_replay_size = int(1e6)
    tau = 0.005
    lr_alpha = 3e-4
    teacher = Teacher()
    obs_size = 18 + 1

    log = False
    all_obs =False

    cfg_rewards = dict()

    cfg = dict(
        kp=40,
        kd=1,
        antagonist=True, # True if you want and intelligent obstacle
        max_level=False,
        curriculum=False,
        joystick=False,
        resample=False,
        random_obstacle=False, # True if you want a random obstacle
        static_obstacle=False, # True if you want a static ostacle
        rewards=cfg_rewards,
    )

    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name)

    # actor_input_shape = mdp.info.observation_space.shape
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
                critic_fit_params=None, range=[-1, 1])

    # Algorithm
    if use_quarl:
        preprocessor = StateVariableConcatenation()
    else:
        preprocessor = None

    core = Core([agent, antagonist],
                mdp,
                low_level_agent=control_agent,
                preprocessors=[preprocessor],
                obs_dim=obs_size)

    temp_perf_buffer = Buffer(2, 100, reset_on_query=True)

    dataset = core.evaluate(n_episodes=5, render=True)
    s, *_ = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)

    logger.epoch_info(0, J=J, R=R, entropy=E)
    mdp.train = True
    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions)
    mdp.train = False

    for it in trange(n_epochs, leave=False):
        sampled_temperatures = teacher.sample_n_temperatures(n=5)

        mdp.train=True
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        mdp.train = False

        if it % 10==0 and view:
            render=True
        else:
            render=False



        dataset = core.evaluate(n_episodes=n_episodes_test, render=render)
        s, *_ = parse_dataset(dataset)
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)

        if it % 10 == 0:
            agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1',alg.__name__, agent_task, it))
            if use_antagonist:
                antagonist.save("checkpoint/{}/{}/{}/{}/epoch_{}".format('GO1',"ANTAGONIST_minmax",alg.__name__, agent_task, it))
        logger.epoch_info(it + 1, J=J, R=R, entropy=E)

    agent.save("checkpoint/{}/{}/{}/epoch_{}".format('GO1', alg.__name__, agent_task, "last"))

    if use_antagonist:
        antagonist.save("checkpoint/{}/{}/{}/{}/epoch_{}".format('GO1', "ANTAGONIST_minmax", alg.__name__, agent_task, "last"))

    logger.info('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':

    path = "../checkpoint/GO1/PPO/nominal_robust_locomotion/epoch_last"
    path = "../src/nominal_models/go1/epoch_last"

    control_agent = Agent.load(path)
    experiment(alg=SAC,
               n_epochs=200,
               n_steps=5000,
               n_steps_per_fit=1,
               n_episodes_test=5,
               control_agent=control_agent)


