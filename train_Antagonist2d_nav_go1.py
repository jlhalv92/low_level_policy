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
from src.core.core_quadruped_adversary2d import Core
from src.envs.go1_nav_environment_hunter_antagonist_obstacle import Go1
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
               wandb_project="",
               max_v=2):

    np.random.seed()

    logger = Logger("EXP", results_dir=None)
    logger.strong_line()
    task_name = "obstacles_navigation"
    agent_task = "{}".format(task_name)
    logger.info('Experiment Algorithm: ' + " task " + agent_task )


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



    job_name = method_name + f"_temperature_{temperature}_2d_max_v={max_v}"
    print(job_name)
    if log:
        wandb.init(
            project=wandb_project,
            group=task_name,
            job_type=job_name,
            name="seed_{}".format(run_id)
        )
    joystick = False
    cfg = dict(antagonist=True,
               joystick=joystick,
               static_obstacles=False,
               activate_arrow=False,
               max_antagonist_v=max_v)



    mdp = Go1(horizon,
              gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name, )

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}


    antagonist_input_shape = (7,)
    antagonist_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=antagonist_input_shape,
                           output_shape=(2,),
                           use_cuda=use_cuda)

    antagonist_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=antagonist_input_shape,
                              output_shape=(2,),
                              use_cuda=use_cuda)

    antagonist_critic_input_shape = (antagonist_input_shape[0] + 2,)
    antagonist_critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=antagonist_critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)
    min_a = -np.array([max_v, max_v])
    max_a = np.array([0, max_v])
    antagonist = alg[1](mdp.info, antagonist_mu_params, antagonist_sigma_params,
                actor_optimizer, antagonist_critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None, range=[min_a, max_a])

    preprocessor = StateVariableConcatenation()

    path_protagonist = "src/nominal_models/nominal_test_protagonist/model"
    protagonist = Agent.load(path_protagonist)

    core = Core([protagonist, antagonist],
                mdp,
                low_level_agent=control_agent, joystick=joystick,
                preprocessors = [preprocessor],
                rationality_antagonist = True)

    temps = np.random.uniform(low=0, high=temperature, size=(100,))
    core.set_variables_for_training(temps)
    dataset = core.evaluate(n_episodes=5, render=False, id_agent=0)

    s,a, r, s1, absorbing, info, last = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = protagonist.policy.entropy(s)
    success_rate = calculate_success_rate(info, last)

    logger.epoch_info(0, J=J, R=R, entropy=E, succes_rate=success_rate)
    temps = temperature*np.ones(100)
    core.set_variables_for_training(temps)

    core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=1)
    # core.learn(n_steps=warmup_transitions, n_steps_per_fit=warmup_transitions, id_agent=0)


    for it in trange(n_epochs, leave=False):
        temps = temperature*np.ones(100)
        core.set_variables_for_training(temps)

        logger.info("Training Antagonist")
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit,  id_agent=1)


        if it % 10==0 and view:
            render=True
        else:
            render=False

        temps_eval = temperature*np.ones(n_episodes_test)
        core.set_variables_for_training(temps_eval)

        dataset = core.evaluate(n_episodes=n_episodes_test, render=render, id_agent=0)

        s, a, r, s1, absorbing, info, last = parse_dataset(dataset)
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = protagonist.policy.entropy(s)
        success_rate = calculate_success_rate(info, last)

        core.set_variables_for_training(temps_eval)
        dataset_antagonist = core.evaluate(n_episodes=n_episodes_test, render=render, id_agent=1)
        s, a, r, s1, absorbing, info, last = parse_dataset(dataset_antagonist)
        R_antagonist = np.mean(compute_J(dataset_antagonist))
        E_a=antagonist.policy.entropy(s)

        logger.epoch_info(it+1, J=J, R=R, entropy=E,
                          success_rate=success_rate, temperature=temps_eval.mean(),
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


    antagonist.save(f"checkpoint/{'GO1'}/{agent_task}/{job_name}/run_{run_id}/{'ANTAGONIST'}/epoch_{'last'}")

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
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--n_steps", type=int, default=5000, help="Number of steps")
    parser.add_argument("--n_steps_per_fit", type=int, default=1, help="Number of steps per fit")
    parser.add_argument("--n_episodes_test", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--runs", type=int, default=5, help="Run ID")
    parser.add_argument("--log", type=str2bool, nargs='?', const=True, default=False,  help="Enable logging")
    parser.add_argument("--use_cuda", type=str2bool, nargs='?', const=True, default=False, help="Use CUDA for computation")
    parser.add_argument("--temperature", type=float, default=2., help="Alpha value")
    parser.add_argument("--wandb_project",type=str,default="Max Temperature Analysis Antagonist 2D")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--velocity", type=float, default=2.)




    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    path = "src/nominal_models/low_level/policy"
    control_agent = Agent.load(path)
    experiment(method_name="Antagonist_2d",
               alg=[SAC, AdaptiveTempSAC],
               n_epochs=args.n_epochs,
               n_steps=args.n_steps,
               n_steps_per_fit=args.n_steps_per_fit,
               n_episodes_test=args.n_episodes_test,
               control_agent=control_agent,
               run_id=args.run_id,
               log=args.log,
               use_cuda=args.use_cuda,
               temperature=args.temperature,
               wandb_project=args.wandb_project,
               max_v=args.velocity)



