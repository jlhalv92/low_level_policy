import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.algorithm.deep_actor_critic.imitation.dagger import Dagger

from mushroom_rl.core import Agent
from mushroom_rl.core import Logger
from src.core.core_imitation_vision import Core
from src.envs.go1_env_rays import Go1
from src.networks.nets import ActorNetwork, CriticNetwork,PolicyNetwork, ValueNetwork, VisualActorNetwork
import wandb



from mushroom_rl.utils.dataset import compute_J, parse_dataset_img

from tqdm import trange


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate

def experiment(method_name ,alg, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               control_agent=None,
               visual_policy=None, run_id=0):

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
    batch_size = 256
    initial_replay_size = 10000

    max_replay_size = int(1e6)
    obs_size = 49
    v_obs_size = 4

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

    # Load Expert
    agent = Agent.load(visual_policy)
    core = Core(agent,
                mdp,
                low_level_agent=control_agent,
                obs_dim=obs_size,
                joystick=joystick)

    core.headless = False
    dataset = core.evaluate(n_episodes=1, render=True)
    img, s, a, r, s1, absorbing, info, last = parse_dataset_img(dataset)
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    # E = agent.policy.entropy(s, img)
    success_rate = calculate_success_rate(info, last)

    logger.epoch_info(0, J=J, R=R, success_rate=success_rate)


if __name__ == '__main__':

    path = "../src/nominal_models/go1/epoch_last"
    path_visual_policy = "../checkpoint/GO1/Dagger/obstacles_navigation/epoch_10"
    control_agent = Agent.load(path)

    experiment(method_name="Dagger",
               alg=Dagger,
               n_epochs=100,
               n_steps=1000,
               n_steps_per_fit=1,
               n_episodes_test=5,
               control_agent=control_agent,
               run_id=0,
               visual_policy=path_visual_policy)


