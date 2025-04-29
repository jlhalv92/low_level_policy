import argparse

import numpy as np
# from src.teachers.self_paced_gamma_teacher_fixed_beta import (
#     SelfPaced,
# )
from src.teachers.doraemon3 import Doraemon
from src.utils.buffer import Buffer
from src.utils.state_preprocessor import StateVariableConcatenation
import torch
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.utils.callbacks import CollectDataset
from src.core.core_quadruped_adversary import Core

from src.algorithm.deep_actor_critic.a_sac import SAC
from src.algorithm.deep_actor_critic.adaptative_temp_sac import AdaptiveTempSAC

from mushroom_rl.core import Agent
from mushroom_rl.core import Logger

from src.envs.go1_nav_environment import Go1
from src.envs.go1_nav_multi_obstacles import Go1MultiObs


from src.networks.nets import ActorNetwork, CriticNetwork, ValueNetwork, ActorBiNetwork, PolicyNetwork, \
    PolicyNetworkStudent
import wandb
from src.mushroom.dataset.dataset import compute_J, parse_dataset

from tqdm import trange

def calculate_success_rate_array(info, last):
    last_idxs = [i for i, x in enumerate(last) if x]
    success = np.array([element['success'] for element in info])
    return success[last_idxs]

class Teacher:
    def __init__(
            self,

            initial_alpha: np.ndarray = 5,  # 5.
            initial_beta: np.ndarray = 0.1,  # 0.1
            target_alpha: np.ndarray = 1,
            target_beta: np.ndarray = 1,
            alpha_lower_bound: float = 1e-8,
            max_kl: float = 0.05,
            performance_lower_bound: int = 0.1,
            target_kl_threshold: float = 0.01,
            temperature_bounds: list = [0., 1.],
            temperature: float = 1,
    ):
        self.teacher = Doraemon(
            initia_alpha=initial_alpha,
            initia_beta=initial_beta,
            target_alpha=target_alpha,
            target_beta=target_beta,
            max_kl=max_kl,
            performance_lower_bound=performance_lower_bound,
            alpha_lower_bound=alpha_lower_bound,
            target_kl_threshold=target_kl_threshold,
            temperature_bounds=temperature_bounds
        )
        self.temperature = temperature
        self.temperature_list = []

    def provide_teacher(self):
        return self.teacher

    def sample_n_temperatures(self, n):
        samples = self.teacher.sample((n,))
        a = 0
        b = self.temperature
        transformed_samples = a + (b - a) * samples

        return transformed_samples


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate


def extract_temp_and_sample_return(data, gamma, temperature_support_a=0, temperature_support_b=1):
    temperatures = list()
    for i in range(len(data)):
        if data[i][-1] or i == len(data) - 1:
            temperatures.append(
                (data[i][0][-1] - temperature_support_a) / (temperature_support_b - temperature_support_a))

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
        log=False,
        Teacher=None,
        temperature_support=1,
        use_cuda=False,
        update=10,
        kl=0.5,
        aware_rationality=False,
        view=False,
        wandb_project="",
        checkpoint_dir="",
        antagonist_velocity=1.5,
        lower_bound=0.6,
        conf=None,
        pre_train_adversary=None,):

    np.random.seed()

    logger = Logger(method_name, results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"

    agent_task = f"{task_name}"
    logger.info('Experiment Algorithm: ' + method_name + " task " + agent_task)

    # MDP
    horizon = 500
    gamma = 0.99
    n_features = 256
    warmup_transitions = 5000
    batch_size = 256  # 256
    initial_replay_size = 5000
    max_replay_size = int(1e6)
    tau = 0.005
    lr_alpha = 3e-4
    job_name = method_name + f"_velocity_antagonist_{antagonist_velocity}_multi_obstacles"

    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_project,
            job_type=job_name,
            name="seed_{}".format(run_id),
            group="DORAEMON_asymetric",
            config=vars(conf)
        )

    teacher = Teacher(max_kl=kl,
                      performance_lower_bound=lower_bound,
                      temperature=temperature_support)

    n_samples_for_dist_update = update

    cfg_rewards = dict()

    joystick = False
    cfg = dict(
        antagonist=True,
        joystick=joystick,
        rewards=cfg_rewards,
        static_obstacles=False,
        activate_arrow=False,
        max_antagonist_v=antagonist_velocity,
        safety_penalties=args.safety_penalties,)

    multiple_obstacles = True
    if multiple_obstacles:

        mdp = Go1MultiObs(horizon,
                  gamma,
                  timestep=0.005,
                  cfg=cfg,
                  task_name=task_name)
    else:

        mdp = Go1(horizon,
                  gamma,
                  timestep=0.005,
                  cfg=cfg,
                  task_name=task_name
                  )

    actor_input_shape = mdp.info.observation_space.shape

    # if aware_rationality:
    #     actor_input_shape = (actor_input_shape[0] + 1,)
    # else:
    actor_input_shape = (actor_input_shape[0],)


    actor_mu_params = dict(network=ActorBiNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=(3,),
                           size_propioception=6,
                           use_cuda=use_cuda)

    actor_sigma_params = dict(network=ActorBiNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=(3,),
                              size_propioception=6,
                              use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}

    critic_input_shape = (actor_input_shape[0] +1 + 3,)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    # Agent

    min_a = -np.array([1.5, 1, 1.5])
    max_a = np.array([1.5, 1, 1.5])
    agent = alg[0](mdp.info, actor_mu_params, actor_sigma_params,
                   actor_optimizer, critic_params, batch_size, initial_replay_size,
                   max_replay_size, warmup_transitions, tau, lr_alpha,
                   critic_fit_params=None, range=[min_a, max_a])

    # Adversary
    obs_antagonist = 6 + 1
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

    antagonist_critic_input_shape = (adversary_actor_input_shape[0] + 1,)
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
        range=[-antagonist_velocity, antagonist_velocity],
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
        aware_rationality=aware_rationality,
        rationality_antagonist=True)

    # if pre_train_adversary:
    #     core_eval = Core(
    #         [agent, pre_train_adversary],
    #         mdp,
    #         low_level_agent=control_agent,
    #         callback_step=CollectDataset(),
    #         preprocessors=[preprocessor],
    #         joystick=joystick,
    #         aware_rationality=aware_rationality,
    #         rationality_antagonist=True)

    temp_perf_buffer = Buffer(2, 100, reset_on_query=True)

    # evaluation_temp = teacher.teacher.temp_dist.mean()

    temps = teacher.sample_n_temperatures(n=100)

    core.set_variables_for_training(temps)
    dataset_online = core.evaluate(
        n_episodes=10, render=view, id_agent=0, return_antagonist_state=True
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
        entropy_antagonist=E_online,
        success_rate=success_rate,
        alfa=np.mean(temps)
    )

    sampled_temperatures = teacher.sample_n_temperatures(n=100)

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

        sampled_temperatures = teacher.sample_n_temperatures(n=100)

        core.set_variables_for_training(sampled_temperatures)

        logger.info("Training Antagonist")
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, id_agent=1)

        core.set_variables_for_training(sampled_temperatures)

        logger.info("Training Protagonist")
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, id_agent=0)

        ##################EVALUATION#########################################
        if it % 10 == 0 and view:
            render = True
        else:
            render = False

        logger.info("Evaluate Protagonist")
        if method_name == "DORAEMON":
            temps = teacher.sample_n_temperatures(n=n_episodes_test)

        else:
            temps = np.random.uniform(0, temperature_support, n_episodes_test)

        core.set_variables_for_training(temps)
        core.callback_step.clean()

        dataset_online = core.evaluate(
            n_episodes=n_episodes_test, render=render, id_agent=0, return_antagonist_state=True
        )

        s, a, r, s1, absorbing, info, last = parse_dataset(dataset_online)
        J_online = np.mean(compute_J(dataset_online, mdp.info.gamma))
        R_online = np.mean(compute_J(dataset_online))
        success_rate = calculate_success_rate(info, last)
        E_online = antagonist.policy.entropy(s)

        # Update temperature distribution
        temperatures, _ = extract_temp_and_sample_return(core.callback_step.get(),
                                                         core.mdp.info.gamma,
                                                         temperature_support_b=temperature_support)

        ##################UPDATE distribution#########################################
        success_rate_array = calculate_success_rate_array(info, last)


        temp_perf_buffer.update_buffer([temperatures, success_rate_array])

        if temp_perf_buffer.__len__() >= n_samples_for_dist_update:
            temps_perfs = temp_perf_buffer.read_buffer(reset=True)

            logger.info("Update Teacher Distribution")
            teacher.teacher.update_temperature_distribution(
                temps_perfs[0], temps_perfs[1], it
            )
        ##################Evaluation 2#########################################

        logger.info("Evaluate Antagonist")
        core.set_variables_for_training(temps)
        core.callback_step.clean()
        dataset_antagonist = core.evaluate(n_episodes=n_episodes_test, render=False, id_agent=1)
        R_antagonist = np.mean(compute_J(dataset_antagonist))



        # if pre_train_adversary:
        #     core_eval.set_variables_for_training(temps)
        #     core_eval.callback_step.clean()
        #     dataset_eval = core_eval.evaluate(
        #         n_episodes=n_episodes_test, render=render, id_agent=0, return_antagonist_state=True
        #     )
        #
        #     s, a, r, s1, absorbing, info, last = parse_dataset(dataset_eval)
        #     success_rate_eval = calculate_success_rate(info, last)
        # else:
        #     success_rate_eval = 0

        logger.epoch_info(
            it + 1,
            J_online=J_online,
            R_online=R_online,
            success_rate=success_rate,
            E_antagonist=E_online,
            mean_temperature=np.mean(temps),
            r_antagonist=R_antagonist,
            # success_rate_antagonist=success_rate_eval,
        )

        if log:
            entropy_distribution = teacher.teacher.temp_dist.entropy()[0]
            a_dist = teacher.teacher.temp_dist.parameters()[0].data.detach().cpu().numpy()[0]
            b_dist = teacher.teacher.temp_dist.parameters()[1].data.detach().cpu().numpy()[0]

            log_dict = dict(J=J_online,
                            R=R_online,
                            success_rate=success_rate,
                            temperature=np.mean(temps),
                            r_antagonist=R_antagonist,
                            entropy_antagonist=E_online,
                            entropy_distribution=entropy_distribution,
                            a_dist=a_dist,
                            b_dist=b_dist)

            wandb.log(log_dict, step=it)

        if it % 10 == 0:
            agent.save(
                f"{checkpoint_dir}/{'GO1'}/{agent_task}/{job_name}/run_{run_id}/{'PROTAGONIST'}/epoch_{it}")
            antagonist.save(
                f"{checkpoint_dir}/{'GO1'}/{agent_task}/{job_name}/run_{run_id}/{'ANTAGONIST'}/epoch_{it}")

    agent.save(
        f"{checkpoint_dir}/{'GO1'}/{agent_task}/{job_name}/run_{run_id}/{'PROTAGONIST'}/epoch_{'last'}")
    antagonist.save(
        f"{checkpoint_dir}/{'GO1'}/{agent_task}/{job_name}/run_{run_id}/{'ANTAGONIST'}/epoch_{'last'}")

    if log:
        wandb.finish()


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
    parser.add_argument("--n_episodes_test", type=int, default=20, help="Number of test episodes")
    parser.add_argument("--run_id", type=int, default=0, help="Run ID")
    parser.add_argument("--log", type=str2bool, nargs='?', const=True, default=False, help="Enable logging")
    parser.add_argument("--use_cuda", type=str2bool, nargs='?', const=True, default=True,
                        help="Use CUDA for computation")
    parser.add_argument("--temperature", type=float, default=0.5, help="Alpha value")
    parser.add_argument("--kl_max", type=float, default=0.001, help="kl_max")
    parser.add_argument("--velocity", type=float, default=1.5)
    parser.add_argument("--lower_bound", type=float, default=0.5)
    parser.add_argument("--aware_rationality", type=str2bool, nargs='?', const=True, default=True,
                        help="Enable logging")
    parser.add_argument("--wandb_project", type=str, default="ADVERSARIAL_LOCOMOTION")
    parser.add_argument("--method", type=str, default="DORAEMON")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint")
    parser.add_argument("--safety_penalties", type=str2bool, nargs='?', const=False)



    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    path = "src/nominal_models/low_level/policy"
    control_agent = Agent.load(path)

    # path_pre_train_adversary = "src/nominal_models/antagonist_1d/epoch_last"
    # pre_train_adversary = Agent.load(path_pre_train_adversary)
    temperature = args.temperature
    run_id = args.run_id
    kl_max = args.kl_max

    experiment_qarl(
        method_name=args.method,
        alg=[SAC, AdaptiveTempSAC],
        n_epochs=args.n_epochs,
        n_steps=args.n_steps,
        n_steps_per_fit=args.n_steps_per_fit,
        n_episodes_test=args.n_episodes_test,
        control_agent=control_agent,
        run_id=run_id,
        log=args.log,
        Teacher=Teacher,
        temperature_support=temperature,
        use_cuda=args.use_cuda,
        update=args.n_episodes_test,
        kl=kl_max,
        aware_rationality=args.aware_rationality,
        view=False,
        wandb_project=args.wandb_project,
        checkpoint_dir=args.checkpoint_dir,
        antagonist_velocity=args.velocity,
        lower_bound=args.lower_bound,
        conf=args,
        pre_train_adversary=None)
