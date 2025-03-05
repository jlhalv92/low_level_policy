import argparse

import numpy as np
from src.teachers.self_paced_gamma_teacher_fixed_beta import (
    SelfPaced,
)
from src.utils.buffer import Buffer
from src.utils.state_preprocessor import StateVariableConcatenation
import torch
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.utils.callbacks import CollectDataset
from src.core.core_quadruped_adversary import Core


from src.algorithm.deep_actor_critic.sac import  SAC
from src.algorithm.deep_actor_critic.adaptative_temp_sac import AdaptiveTempSAC

from mushroom_rl.core import Agent
from mushroom_rl.core import Logger

from src.envs.go1_nav_environment import Go1

from src.networks.nets import  ActorNetwork ,CriticNetwork, ValueNetwork, ActorBiNetwork, PolicyNetwork, PolicyNetworkStudent
import wandb
from mushroom_rl.utils.dataset import compute_J, parse_dataset

from tqdm import trange

class Teacher:
    def __init__(
            self,
            initial_conc: np.ndarray = np.array([int(50)]),
            target_conc: np.ndarray = np.array([0.1]),
            fixed_rate: np.ndarray = np.array([1000]),
            conc_lower_bound: float = 1e-8,
            max_kl: float = 0.5,
            performance_lower_bound: int = 0.6,
            target_kl_threshold: float = 0.01,
            temperature_bounds: list = [-np.inf, np.inf],
            doraemon: bool = False,
    ):
        self.teacher = SelfPaced(
            initial_conc,
            target_conc,
            fixed_rate,
            max_kl,
            performance_lower_bound,
            conc_lower_bound,
            target_kl_threshold,
            temperature_bounds,
            doraemon,
        )
        self.temperature_list = []

    def provide_teacher(self):
        return self.teacher

    def sample_n_temperatures(self, n):
        sampled_temperatures = np.zeros(n)
        for i in range(n):
            sampled_temperatures[i] = self.teacher.sample()[0]

        return sampled_temperatures

def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate

def extract_temp_and_sample_return(data, gamma):
    temperatures = list()
    for i in range(len(data)):
        if data[i][-1] or i == len(data) - 1:
            temperatures.append(data[i][0][-1])

    sample_returns = compute_J(data, gamma)

    return temperatures, sample_returns

def experiment_qarl(
        method_name="",
        alg=[SAC, AdaptiveTempSAC],
        n_epochs=200,
        n_steps=5000,
        n_steps_per_fit=1,
        n_episodes_test=10,
        control_agent=None,
        run_id=0,
        log=False, Teacher=None, alfa=1000, use_cuda=False,
        update=10,
        kl=0.5, curriculum_type="", conc_lower_bound=1e-8,
        aware_rationality=False,
        view=False):

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
    results_dir = f"temperature_comparison/{method_name}"

    if curriculum_type == "DORAEMON":
        teacher = Teacher(performance_lower_bound=0.5, max_kl=kl, doraemon=True, conc_lower_bound=conc_lower_bound)
    else:
        teacher = Teacher(performance_lower_bound=0.5, max_kl=kl, initial_conc=np.array([int(alfa)]))

    n_samples_for_dist_update = update

    cfg_rewards=dict()

    log = log
    joystick = False
    cfg = dict(
               antagonist=True,
               max_level=True,
               curriculum=False,
               joystick=joystick,
               resample=False,
               domain_randomization=False,
               rewards=cfg_rewards,
               static_obstacles=False,
                activate_arrow=False)

    vel_controller = dict(max_vx=1.5, max_vy=1, max_wz=1.5)
    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name, **vel_controller)
    job_type = method_name + f"_aware_rationality_{aware_rationality}"
    print(job_type)
    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="RATIONALITY_COMPARISON",
            group=task_name,
            job_type=job_type,
            name="seed_{}".format(run_id)
        )
    actor_input_shape = mdp.info.observation_space.shape
    if aware_rationality:
        actor_input_shape = (actor_input_shape[0] +1,)
    else:
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
    agent = alg[0](mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None, range=[min_a, max_a])

    # Adversary
    obs_antagonist =6 + 1
    adversary_actor_input_shape = (obs_antagonist,)
    antagonist_mu_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=adversary_actor_input_shape,
        output_shape=(1,),
        use_cuda=use_cuda,
    )

    antagonist_sigma_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=adversary_actor_input_shape,
        output_shape=(1,),
        use_cuda=use_cuda,
    )

    antagonist_critic_input_shape = (adversary_actor_input_shape[0] +1,)
    antagonist_critic_params = dict(
        network=CriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": 3e-4}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=antagonist_critic_input_shape,
        output_shape=(1,),
        use_cuda=use_cuda,
    )

    antagonist = alg[1](
        mdp.info,
        antagonist_mu_params,
        antagonist_sigma_params,
        actor_optimizer,
        antagonist_critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        warmup_transitions,
        tau,
        lr_alpha,
        critic_fit_params=None,
        range=[-1, 1],
    )

    # Algorithm
    preprocessor = StateVariableConcatenation()

    core = Core(
        [agent, antagonist],
        mdp,
        low_level_agent=control_agent,
        callback_step=CollectDataset(),
        preprocessors=[preprocessor],
        joystick=joystick,
        aware_rationality=aware_rationality)

    temp_perf_buffer = Buffer(2, 100, reset_on_query=True)

    if method_name == "DORAEMON":
        evaluation_temp = teacher.teacher.temp_dist.mean()
        temps = evaluation_temp * np.ones(100)
    else:
        temps = np.random.uniform(0, 0.5, 100)

    core.set_variables_for_training(temps)
    dataset_online = core.evaluate(
        n_episodes=10, render=False, id_agent=0, return_antagonist_state=True
    )

    s, a, r, s1, absorbing, info, last = parse_dataset(dataset_online)
    J_online = np.mean(compute_J(dataset_online, mdp.info.gamma))
    R_online = np.mean(compute_J(dataset_online))
    success_rate = calculate_success_rate(info, last)
    E_online = antagonist.policy.entropy(s)

    logger.epoch_info(
        0,
        J_online=J_online,
        R_online=R_online,
        entropy=E_online,
        success_rate=success_rate,
        alfa=np.mean(temps)
    )
    if method_name=="DORAEMON":
        sampled_temperatures = teacher.sample_n_temperatures(n=100)
    else:
        sampled_temperatures = np.random.uniform(0, 0.5, 100)


    core.set_variables_for_training(sampled_temperatures)

    logger.info("warm-up transitions  Antagonist")
    core.learn(
        n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=1
    )

    logger.info("warm-up transitions  Protagonist")
    core.learn(
        n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=0
    )


    for it in trange(n_epochs, leave=False):

        if method_name == "DORAEMON":
            sampled_temperatures = teacher.sample_n_temperatures(n=100)
            sampled_temperatures /= 2
        else:
            sampled_temperatures = np.random.uniform(0, 0.5, 100)

        core.set_variables_for_training(sampled_temperatures)

        logger.info("Training Antagonist")
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, id_agent=1)

        core.set_variables_for_training(sampled_temperatures)

        logger.info("Training Protagonist")
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, id_agent=0)

        ##################EVALUATION#########################################

        if method_name=="DORAEMON":
            temps = teacher.sample_n_temperatures(n=n_episodes_test)
            temps /=2
        else:
            temps = np.random.uniform(0, 0.5, n_episodes_test)

        core.set_variables_for_training(temps)
        core.callback_step.clean()

        if it % 10 == 0 and view:
            render = True
        else:
            render = False

        logger.info("Evaluate Protagonist")
        dataset_online = core.evaluate(
            n_episodes=n_episodes_test, render=render, id_agent=0, return_antagonist_state=True
        )

        s, a, r, s1, absorbing, info, last = parse_dataset(dataset_online)
        J_online = np.mean(compute_J(dataset_online, mdp.info.gamma))
        R_online = np.mean(compute_J(dataset_online))
        success_rate = calculate_success_rate(info, last)
        E_online = antagonist.policy.entropy(s)

        # Update temperature distribution
        temperatures, _ = extract_temp_and_sample_return(
            core.callback_step.get(), core.mdp.info.gamma
        )

        if method_name == "DORAEMON":
            temp_perf_buffer.update_buffer([(2*np.array(temperatures)).tolist(), (success_rate*np.ones(n_episodes_test)).tolist()])

        else:
            temp_perf_buffer.update_buffer([temperatures, (success_rate*np.ones(n_episodes_test)).tolist()])

        ##################UPDATE distribution#########################################
        if temp_perf_buffer.__len__() >= n_samples_for_dist_update:
            temps_perfs = temp_perf_buffer.read_buffer(reset=True)

            logger.info("Update Teacher Distribution")
            teacher.teacher.update_temperature_distribution(
                temps_perfs[0], temps_perfs[1], it
            )
        ##################Evaluation 2#########################################
        core.set_variables_for_training(temps)
        core.callback_step.clean()
        logger.info("Evaluate Antagonist")
        dataset_antagonist = core.evaluate(n_episodes=n_episodes_test, render=False, id_agent=1)
        R_antagonist = np.mean(compute_J(dataset_antagonist))

        logger.epoch_info(
            it + 1,
            J_online=J_online,
            R_online=R_online,
            success_rate=success_rate,
            E_antagonist=E_online,
            mean_temperature=np.mean(temps),
            r_antagonist=R_antagonist
        )

        if log:
            log_dict = dict(J=J_online,
                            R=R_online,
                            success_rate=success_rate,
                            alfa=np.mean(temps),
                            r_antagonist=R_antagonist,
                            entropy_antagonist=E_online)

            wandb.log(log_dict, step=it)

        if it % 10 == 0:
            agent.save(f"checkpoint/{results_dir}/agent_epoch_{it}_{run_id}")
            antagonist.save(f"checkpoint/{results_dir}/antagonist_epoch_{it}_{run_id}")

    agent.save(f"checkpoint/{results_dir}/agent_last_{run_id}")
    antagonist.save(f"checkpoint/{results_dir}/antagonist_last_{run_id}")

    if log:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Parse command-line arguments for the script.")
    parser.add_argument("--method", type=str, default="DORAEMON",choices=["DORAEMON", "UNIFORM"],help="DORAEMON or UNIFORM")
    parser.add_argument("--n_steps", type=int, default=5000, help="Number of steps")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of steps")
    parser.add_argument("--n_steps_per_fit", type=int, default=1, help="Number of steps per fit")
    parser.add_argument("--n_episodes_test", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--run_id", type=int, default=0, help="Run ID")
    parser.add_argument("--log",default=False,  help="Enable logging")
    parser.add_argument("--use_cuda", type=bool, default=True, help="Use CUDA for computation")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value")
    parser.add_argument("--kl_max",type=float,  default=0.05,  help="kl_max")
    parser.add_argument("--aware_rationality",type=bool, default=False,  help="Enable logging")


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    path = "src/nominal_models/low_level/policy"
    control_agent = Agent.load(path)

    alfa = args.alpha
    run_id = args.run_id
    method = args.method
    kl_max = args.kl_max

    method_name = method
    experiment_qarl(
        method_name=method_name,
        alg=[SAC, AdaptiveTempSAC],
        n_epochs=args.n_epochs,
        n_steps=args.n_steps,
        n_steps_per_fit=args.n_steps_per_fit,
        n_episodes_test=args.n_episodes_test,
        control_agent=control_agent,
        run_id=run_id,
        log=args.log,
        Teacher=Teacher,
        alfa=alfa * 1000,
        use_cuda=True,
        update=10,
        kl=kl_max,
        curriculum_type=method,
        conc_lower_bound = 1e-8,
        aware_rationality=args.aware_rationality,
        view=False,

    )



