import os
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.utils.callbacks import CollectDataset

from src.algorithm.deep_actor_critic.sac import SAC
from src.algorithm.deep_actor_critic.adaptative_temp_sac import AdaptiveTempSAC
from pathlib import Path
from mushroom_rl.core import Agent
from mushroom_rl.core import Logger
from src.core.core_quadruped_adversary import Core as Core_adversary
from src.envs.go1_env_privilege_rays import Go1
from src.networks.nets import ActorNetwork, CriticNetwork, PolicyNetwork, ValueNetwork, ActorBiNetwork
from src.teachers.self_paced_gamma_teacher_fixed_beta import (
    SelfPacedGammaTemperatureTeacherFixedBeta,
)
from src.utils.buffer import Buffer
from src.utils.state_preprocessor import StateVariableConcatenation
import wandb
from mushroom_rl.utils.dataset import compute_J, parse_dataset

from tqdm import trange


def extract_temp_and_sample_return(data, gamma):
    temperatures = list()
    for i in range(len(data)):
        if data[i][-1] or i == len(data) - 1:
            temperatures.append(data[i][0][-1])

    sample_returns = compute_J(data, gamma)

    return temperatures, sample_returns


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate


def experiment_qarl(
        method_name="",
        alg=[SAC, AdaptiveTempSAC],
        n_epochs=200,
        episode=5,
        n_steps_per_fit=1,
        n_episodes_test=10,
        control_agent=None,
        run_id=0,
        log=False,
        alfa=1000):

    results_dir = f"temperature_comparison/{method_name}"
    logger = Logger("HI-QARL", results_dir=None)
    logger.strong_line()
    logger.info("Experiment Algorithm: SAC vs QARL Obstacle")
    task_name = "obstacles_navigation"

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
    obs_dim = 61
    obs_agent = 60
    obs_antagonist = 5
    cfg_rewards = dict()

    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="RATIONALITY_COMPARISON",
            group=task_name,
            job_type=method_name,
            name="seed_{}".format(run_id),
        )

    cfg = dict(kp=40,
               kd=1,
               antagonist=True,
               max_level=False,
               curriculum=False,
               joystick=False,
               resample=False,
               domain_randomization=False,
               rewards=cfg_rewards,
               static_obstacles=False)

    mdp = Go1(horizon, gamma, timestep=0.005, cfg=cfg, task_name=task_name)

    # Control agent
    control_agent = control_agent

    agent_actor_input_shape = (obs_dim,)
    actor_mu_params = dict(
        network=ActorBiNetwork,
        n_features=n_features,
        input_shape=agent_actor_input_shape,
        output_shape=(3,),
        size_propioception=6,
        size_exteroception=55,
        use_cuda=True,
    )

    actor_sigma_params = dict(
        network=ActorBiNetwork,
        n_features=n_features,
        input_shape=agent_actor_input_shape,
        output_shape=(3,),
        size_propioception=6,
        size_exteroception=55,
        use_cuda=True,
    )


    actor_optimizer = {"class": optim.Adam, "params": {"lr": 1e-4}}

    critic_input_shape = (agent_actor_input_shape[0] + 3,)
    critic_params = dict(
        network=CriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": 3e-4}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=critic_input_shape,
        output_shape=(1,),
        use_cuda=True,
    )

    # Agent
    agent = alg[0](
        mdp.info,
        actor_mu_params,
        actor_sigma_params,
        actor_optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        warmup_transitions,
        tau,
        lr_alpha,
        critic_fit_params=None,
    )

    # Adversary

    adversary_actor_input_shape = (obs_antagonist,)
    antagonist_mu_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=adversary_actor_input_shape,
        output_shape=(1,),
        use_cuda=True,
    )

    antagonist_sigma_params = dict(
        network=ActorNetwork,
        n_features=n_features,
        input_shape=adversary_actor_input_shape,
        output_shape=(1,),
        use_cuda=True,
    )

    antagonist_critic_input_shape = (adversary_actor_input_shape[0] + 1,)
    antagonist_critic_params = dict(
        network=CriticNetwork,
        optimizer={"class": optim.Adam, "params": {"lr": 3e-4}},
        loss=F.mse_loss,
        n_features=n_features,
        input_shape=antagonist_critic_input_shape,
        output_shape=(1,),
        use_cuda=True,
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

    core = Core_adversary(
        [agent, antagonist],
        mdp,
        low_level_agent=control_agent,
        callback_step=CollectDataset(),
        preprocessors=[preprocessor],
        obs_dim=obs_agent,
    )


    evaluation_temp = np.random.uniform(low=0, high=alfa, size=50).mean()
    core.set_variables_for_training(evaluation_temp * np.ones(n_episodes_test))
    core.callback_step.clean()

    logger.info("Evaluate Protagonist")
    dataset_protagonist = core.evaluate(
        n_episodes=n_episodes_test, id_agent=0, return_antagonist_state=True
    )

    s, a, r, s1, absorbing, info, last = parse_dataset(dataset_protagonist)
    J_online = np.mean(compute_J(dataset_protagonist, mdp.info.gamma))
    R_online = np.mean(compute_J(dataset_protagonist))
    success_rate_online = calculate_success_rate(info, last)

    logger.info("Evaluate Antagonist")
    core.set_variables_for_training(evaluation_temp * np.ones(n_episodes_test))
    core.callback_step.clean()
    dataset_antagonist = core.evaluate(
        n_episodes=n_episodes_test, id_agent=1)
    s, _, _, _, _, _, _ = parse_dataset(dataset_antagonist)
    E_online = antagonist.policy.entropy(s)
    R_antagonist = np.mean(compute_J(dataset_antagonist))
    logger.epoch_info(
        0,
        J_online=J_online,
        R_online=R_online,
        success_rate_online=success_rate_online,
        E_online=E_online,
        mean_temperature=evaluation_temp,
        r_antagonist=R_antagonist
    )

    sampled_temperatures = np.random.uniform(low=0, high=alfa, size=50)
    core.set_variables_for_training(sampled_temperatures)

    logger.info("warm-up transitions  Antagonist")

    core.learn(
        n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=1
    )
    logger.info("warm-up transitions  Protagonist")

    core.learn(
        n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=0
    )

    #
    for it in trange(n_epochs, leave=False):


        sampled_temperatures = np.random.uniform(low=0, high=alfa, size=5)
        core.set_variables_for_training(sampled_temperatures)

        logger.info("Training Antagonist")
        core.learn(n_episodes=episode, n_steps_per_fit=n_steps_per_fit, id_agent=1)

        logger.info("Training Protagonist")
        core.learn(n_episodes=episode, n_steps_per_fit=n_steps_per_fit, id_agent=0)

        evaluation_temp = np.mean(sampled_temperatures)
        core.set_variables_for_training(evaluation_temp * np.ones(n_episodes_test))
        core.callback_step.clean()

        logger.info("Evaluate Protagonist")
        dataset_protagonist = core.evaluate(
            n_episodes=n_episodes_test, id_agent=0, return_antagonist_state=True
        )

        s, a, r, s1, absorbing, info, last = parse_dataset(dataset_protagonist)
        J_online = np.mean(compute_J(dataset_protagonist, mdp.info.gamma))
        R_online = np.mean(compute_J(dataset_protagonist))
        success_rate_online = calculate_success_rate(info, last)

        logger.info("Evaluate Antagonist")
        core.set_variables_for_training(evaluation_temp * np.ones(n_episodes_test))
        core.callback_step.clean()
        dataset_antagonist = core.evaluate(
            n_episodes=n_episodes_test, id_agent=1)
        s, _, _, _, _, _, _ = parse_dataset(dataset_antagonist)
        E_online = antagonist.policy.entropy(s)
        R_antagonist = np.mean(compute_J(dataset_antagonist))
        logger.epoch_info(
            0,
            J_online=J_online,
            R_online=R_online,
            success_rate_online=success_rate_online,
            E_online=E_online,
            mean_temperature=evaluation_temp,
            r_antagonist=R_antagonist
        )


        if log:
            logs_dict = {"RETURN": J_online, "REWARD": R_online, "REWARD_antagonist": R_antagonist,
                         "succes_rate": success_rate_online, "alfa": evaluation_temp}
            wandb.log(logs_dict, step=it)


        if it % 10 == 0:
            agent.save(f"checkpoint/{results_dir}/agent_epoch_{it}_{run_id}")
            antagonist.save(f"checkpoint/{results_dir}/antagonist_epoch_{it}_{run_id}")

    agent.save(f"checkpoint/{results_dir}/agent_last_{run_id}")
    antagonist.save(f"checkpoint/{results_dir}/antagonist_last_{run_id}")

    if log:
        wandb.finish()


if __name__ == "__main__":
    path = "src/nominal_models/go1/epoch_last"
    control_agent = Agent.load(path)
    alfa = 1
    for i in range(4):
        method = "Uniform_adversary"
        method_name = "Uniform_adversary_{}".format(alfa)
        experiment_qarl(
            method_name=method_name,
            alg=[SAC, AdaptiveTempSAC],
            n_epochs=200,
            episode=5,
            n_steps_per_fit=1,
            n_episodes_test=10,
            control_agent=control_agent,
            run_id=i,
            log=True,
            alfa=1,
        )
