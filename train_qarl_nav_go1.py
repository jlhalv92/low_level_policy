import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.utils.callbacks import CollectDataset

from src.algorithm.deep_actor_critic.sac import  SAC
from src.algorithm.deep_actor_critic.adaptative_temp_sac import AdaptiveTempSAC

from mushroom_rl.core import Agent
import wandb

from mushroom_rl.core import Logger
from src.core.core_quadruped_adversary import Core
from src.envs.go1_env import Go1
from src.networks.nets import ActorNetwork, CriticNetwork,PolicyNetwork, ValueNetwork
from src.teachers.self_paced_gamma_teacher_fixed_beta import SelfPacedGammaTemperatureTeacherFixedBeta


from src.utils.buffer import Buffer
from src.utils.state_preprocessor import StateVariableConcatenation

from mushroom_rl.utils.dataset import compute_J, parse_dataset

from tqdm import trange


def extract_temp_and_sample_return(data, gamma):
    temperatures = list()
    for i in range(len(data)):
        if data[i][-1] or i == len(data) - 1:
            temperatures.append(data[i][0][-1])

    sample_returns = compute_J(data, gamma)

    return temperatures, sample_returns


class Teacher:
    def __init__(self,
                 initial_conc: np.ndarray = np.array([50]),
                 target_conc: np.ndarray = np.array([1]),
                    fixed_rate: np.ndarray = np.array([1000]),
                    conc_lower_bound: float = 1e-8,
                    max_kl: float = 0.5,
                    performance_lower_bound: int = 10,  # cheetah: 10, walker: 10, hopper: 1, cartpole: 10, quad:100
                    target_kl_threshold: float = 0.01,
                    temperature_bounds: list = [-np.inf, np.inf]):

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
        self.temperature_list = []

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


def experiment(method_name, alg, n_epochs, n_episodes, n_steps_per_fit, n_episodes_test, control_agent=None, run_id=0):

    np.random.seed()
    use_antagonist = True
    logger = Logger(method_name, results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"
    # task_name = "navigation"
    agent_task = "{}".format(task_name)

    logger.info('Experiment Algorithm: ' + method_name + " task " + agent_task )

    view = False
    # MDP
    horizon = 1000
    gamma = 0.99
    n_features = 256
    warmup_transitions = 3000
    batch_size = 256
    initial_replay_size = 3000
    n_samples_for_dist_update = 30
    max_replay_size = int(1e6)
    tau = 0.005
    lr_alpha = 3e-4
    teacher = Teacher()
    cfg_rewards=dict()

    log = True
    obs_dim = 18 + 1

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

    mdp_eval = Go1(horizon,
                   gamma,
                   timestep=0.005,
                   cfg=cfg,
                   task_name="multi_obstacles_navigation")



    # actor_input_shape = mdp.info.observation_space.shape
    actor_input_shape = (5*obs_dim,)
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

    # Protagonist
    agent = alg[0](mdp.info, actor_mu_params, actor_sigma_params,
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

    antagonist = alg[1](mdp.info, antagonist_mu_params, antagonist_sigma_params,
                actor_optimizer, antagonist_critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None, range=[-2, 2])

    # Algorithm

    preprocessor = StateVariableConcatenation()


    collect_dataset = CollectDataset()
    callbacks = collect_dataset

    core = Core([agent, antagonist],
                mdp,
                callback_step=callbacks,
                low_level_agent=control_agent,
                preprocessors=[preprocessor],
                obs_dim=obs_dim)

    core_eval = Core([agent, antagonist],
                     mdp_eval,
                     callback_step=callbacks,
                     low_level_agent=control_agent,
                     preprocessors=[preprocessor],
                     obs_dim=obs_dim)

    temp_perf_buffer = Buffer(2, 100, reset_on_query=True)

    evaluation_temp = teacher.teacher.temp_dist.mean()
    core.set_variables_for_training(evaluation_temp*np.ones(5))

    dataset = core.evaluate(n_episodes=n_episodes, render=False, id_agent=0)
    s, *_ = parse_dataset(dataset)
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)
    temperature = evaluation_temp

    dataset = core_eval.evaluate(n_episodes=n_episodes, render=False, id_agent=0)
    s, *_ = parse_dataset(dataset)
    J_r = np.mean(compute_J(dataset, mdp_eval.info.gamma))
    logger.epoch_info(0, J=J, J_multi_agent=J_r, R=R, entropy=E, temperature= temperature)


    sampled_temperatures = teacher.sample_n_temperatures(n=50)
    core.set_variables_for_training(sampled_temperatures)
    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=1)
    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=0)
    core.callback_step.clean()


    for it in trange(n_epochs, leave=False):


        sampled_temperatures = teacher.sample_n_temperatures(n=5)
        core.set_variables_for_training(sampled_temperatures)
        teacher.temperature_list.append(sampled_temperatures)
        logger.info("Training Antagonist")
        core.learn(n_episodes=n_episodes, n_steps_per_fit=n_steps_per_fit, id_agent=1)
        logger.info("Training Protagonist")
        core.learn(n_episodes=n_episodes, n_steps_per_fit=n_steps_per_fit, id_agent=0)

        temperatures, sample_returns = extract_temp_and_sample_return(
            core.callback_step.get(), core.mdp.info.gamma
        )

        temp_perf_buffer.update_buffer([temperatures, sample_returns])

        core.callback_step.clean()

        if temp_perf_buffer.__len__() >= n_samples_for_dist_update:
            temps_perfs = temp_perf_buffer.read_buffer(reset=True)
            teacher.teacher.update_temperature_distribution(
                temps_perfs[0], temps_perfs[1], it
            )

        if it % 10==0 and view:
            render=True
        else:
            render=False

        evaluation_temp = teacher.teacher.temp_dist.mean()
        core.set_variables_for_training(evaluation_temp*np.ones(n_episodes_test))
        logger.info("Evaluate with Best Antagonist")

        dataset = core.evaluate(n_episodes=n_episodes_test, render=render,id_agent=0)
        s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)
        E_a = antagonist.policy.entropy(s)
        success_rate = calculate_success_rate(info, last)
        evaluation_temp = teacher.teacher.temp_dist.mean()

        dataset = core_eval.evaluate(n_episodes=n_episodes, render=render, id_agent=0)
        s, *_ = parse_dataset(dataset)
        J_r = np.mean(compute_J(dataset, mdp_eval.info.gamma))

        logger.epoch_info(it + 1,
                          J=J,
                          J_multi_agent=J_r,
                          R=R,
                          entropy=E,
                          entropy_antagonist=E_a,
                          current_t_mean=evaluation_temp,
                          target_t_mean=teacher.teacher.target_dist.mean(),
                          current_conc=teacher.teacher.new_conc,
                          target_conc=1,
                          success_rate=success_rate)

        if log:
            logs_dict = {"RETURN": J, "Multi_obstacle_RETURN":J_r, "REWARD": R, "succes_rate": success_rate, "current_t_mean": evaluation_temp}
            wandb.log(logs_dict, step=it)

        if it % 10 == 0:
            agent.save(
                "checkpoint/{}/{}/{}/{}/{}/epoch_{}".format('GO1', agent_task, method_name, run_id, 'Protagonist', it))
            antagonist.save(
                "checkpoint/{}/{}/{}/{}/{}/epoch_{}".format('GO1', agent_task, method_name, run_id, "ANTAGONIST", it))



    agent.save(
        "src/nominal_models/{}/{}/{}/{}/{}/epoch_{}".format('GO1', agent_task, method_name, run_id, 'Protagonist', "last"))
    antagonist.save(
        "src/nominal_models/{}/{}/{}/{}/{}/epoch_{}".format('GO1', agent_task, method_name, run_id, "ANTAGONIST", "last"))

    if log:
        wandb.finish()



if __name__ == '__main__':

    path ="checkpoint/GO1/PPO/nominal_robust_locomotion/epoch_last"
    control_agent = Agent.load(path)
    for i in range(5):
        experiment(method_name="QARL",
            alg=[SAC, AdaptiveTempSAC],
                   n_epochs=200,
                   n_episodes=5,
                   n_steps_per_fit=1,
                   n_episodes_test=5,
                   control_agent=control_agent,
                   run_id=i)


