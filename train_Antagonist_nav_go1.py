import numpy as np

import torch.optim as optim
import torch.nn.functional as F
import argparse

from src.algorithm.deep_actor_critic.sac import  SAC
from src.algorithm.deep_actor_critic.adaptative_temp_sac import AdaptiveTempSAC

from src.algorithm.deep_actor_critic.fix_temp_sac import fixTempSAC

from mushroom_rl.core import Agent
import wandb

from mushroom_rl.core import Logger
from src.core.core_quadruped_adversary import Core
from src.envs.go1_nav_environment import Go1
from src.envs.go1_nav_multi_obstacles import Go1MultiObs
from src.networks.nets import ActorNetwork, CriticNetwork,PolicyNetwork, ValueNetwork, ActorBiNetwork, PolicyNetworkStudent
from src.mushroom.dataset.dataset import compute_J,parse_dataset
from src.utils.state_preprocessor import StateVariableConcatenation

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
               temperature=1.,
               velocity=1.5,
               wandb_project=""):

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



    job_name = method_name + f"_alpha_{temperature}"
    if log:
        wandb.init(
            project=wandb_project,
            group=task_name,
            job_type=job_name,
            name="seed_{}".format(run_id)
        )
    joystick = False
    cfg = dict(
        antagonist=True,
        joystick=joystick,
        static_obstacles=False,
        activate_arrow=False,
        max_antagonist_v=1.5,
        n_static_obstacles=1)

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

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}

    antagonist_input_shape = (7,)
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
                critic_fit_params=None, range=[-velocity, velocity])

    preprocessor = StateVariableConcatenation()

    path_protagonist = "/home/jose-luis/CORL_final_results/SAC_heuristic_velocity_antagonist_0.0/obstacles_navigation_velocity_antagonist_0.0/epoch_last_0"

    protagonist = Agent.load(path_protagonist)

    core = Core([protagonist, antagonist],
                mdp,
                low_level_agent=control_agent, joystick=joystick,
                preprocessors = [preprocessor],
                rationality_antagonist = True)

    temps = np.random.uniform(low=0, high=temperature, size=100)
    # temps = temperature*np.ones(100)
    core.set_variables_for_training(temps)
    dataset = core.evaluate(n_episodes=5, render=True, id_agent=0)

    s,a, r, s1, absorbing, info, last = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = protagonist.policy.entropy(s)
    success_rate = calculate_success_rate(info, last)

    logger.epoch_info(0, J=J, R=R, entropy=E, succes_rate=success_rate)
    # temps = temperature * np.ones(100)
    temps = np.random.uniform(low=0, high=temperature, size=100)
    core.set_variables_for_training(temps)

    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=1)
    # core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=0)


    for it in trange(n_epochs, leave=False):
        temps = np.random.uniform(low=0, high=temperature, size=100)
        # temps = temperature * np.ones(100)
        core.set_variables_for_training(temps)

        logger.info("Training Antagonist")
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit,  id_agent=1)
        # logger.info("Training Protagonist")
        # core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, id_agent=0)

        if it % 10==0 and view:
            render=False
        else:
            render=False

        # temps = np.random.uniform(low=1, high=0.8, size=(100,))
        core.set_variables_for_training(temps)


        dataset_antagonist = core.evaluate(n_episodes=n_episodes_test, render=render, id_agent=1)
        s, a, r, s1, absorbing, info, last = parse_dataset(dataset_antagonist)
        R_antagonist = np.mean(compute_J(dataset_antagonist))
        E_a=antagonist.policy.entropy(s)
        success_rate = calculate_success_rate(info, last)
        logger.epoch_info(it+1,success_rate=success_rate, alfa=temps[:20].mean(),
                          r_antagonist=R_antagonist, entropy_antagonist=E_a)



        if log:
            log_dict = dict(J=J, R=R, entropy=E,
                            success_rate=success_rate,
                            alfa=temps.mean(), r_antagonist=R_antagonist,
                            entropy_antagonist=E_a)
            wandb.log(log_dict, step=it)

        if it % 10 == 0:
            # agent.save(f"checkpoint/{'GO1'}/{agent_task}/{job_name}/run_{run_id}/{'PROTAGONIST'}/epoch_{it}")
            antagonist.save(f"checkpoint/{'GO1'}/{agent_task}/{job_name}/run_{run_id}/{'ANTAGONIST'}/epoch_{it}")

        # if it % 20 ==0:
        # alpha = np.random.uniform(0, 1)
        # antagonist.update_alpha(alpha)


    # agent.save(f"checkpoint/{'GO1'}/{agent_task}/{job_name}/run_{run_id}/{'PROTAGONIST'}/epoch_{'last'}")
    antagonist.save(f"checkpoint/{'GO1'}/{agent_task}/{job_name}/run_{run_id}/{'ANTAGONIST'}/epoch_{'last'}")

    if log:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Parse command-line arguments for the script.")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--n_steps", type=int, default=5000, help="Number of steps")
    parser.add_argument("--n_steps_per_fit", type=int, default=1, help="Number of steps per fit")
    parser.add_argument("--n_episodes_test", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--runs", type=int, default=5, help="Run ID")
    parser.add_argument("--log",default=False,  help="Enable logging")
    parser.add_argument("--use_cuda", default=True, help="Use CUDA for computation")
    parser.add_argument("--alpha", type=float, default=5., help="Alpha value")
    parser.add_argument("--wandb_project",type=str,default="ADVERSARIAL_LOCOMOTION_RATIONALITY_ANALYSIS")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--velocity", type=float, default=2., help=" antagonist vel value")





    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    path = "src/nominal_models/low_level/policy"
    control_agent = Agent.load(path)

    experiment(method_name="SAC_ANTAGONIST_EV",
               alg=[SAC, AdaptiveTempSAC],
               n_epochs=args.n_epochs,
               n_steps=args.n_steps,
               n_steps_per_fit=args.n_steps_per_fit,
               n_episodes_test=args.n_episodes_test,
               control_agent=control_agent,
               run_id=args.run_id,
               log=args.log,
               use_cuda=args.use_cuda,
               temperature=args.alpha,
               wandb_project=args.wandb_project,
               velocity=args.velocity)



