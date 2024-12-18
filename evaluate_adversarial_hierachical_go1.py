import numpy as np
from src.utils.state_preprocessor import StateVariableConcatenation
import matplotlib.pyplot as plt
from mushroom_rl.core import Logger
from src.core.core_quadruped_adversary import Core
from src.envs.go1_env import Go1
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.core import Agent
from src.networks.nets import ActorNetwork, CriticNetwork, PolicyNetwork, ValueNetwork




def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate
def experiment(method_name=None, task_name="obstacles_navigation"):
    np.random.seed()

    logger = Logger(results_dir=None)


    # MDP

    horizon = 1000
    gamma = 0.99
    cfg_rewards=dict()

    cfg = dict(kp=40,
               kd=1,
               antagonist=True,
               max_level=False,
               curriculum=False,
               joystick=False,
               resample=False,
               domain_randomization=False,
               rewards=cfg_rewards)

    mdp = Go1(horizon, gamma,
              timestep=0.005,
              cfg=cfg,
              task_name=task_name)




    path_c ="checkouts/locomotion_policy"
    if method_name == "MIN_MAX":
        path ="src/nominal_models/GO1/obstacles_navigation/Adversarial_SAC/0/Protagonist/epoch_last"
        path_adversary = "src/nominal_models/GO1/obstacles_navigation/Adversarial_SAC/0/ANTAGONIST/epoch_last"

    if method_name == "QARL":
        path ="src/nominal_models/GO1/obstacles_navigation/QARL/0/Protagonist/epoch_last"
        path_adversary = "src/nominal_models/GO1/obstacles_navigation/QARL/0/ANTAGONIST/epoch_last"

    control_agent = Agent.load(path_c)
    agent = Agent.load(path)
    adversary = Agent.load(path_adversary)
    if method_name=="QARL":
        preprocessor = StateVariableConcatenation()
        obs_dim = 18 +1
        core = Core([agent, adversary], mdp,preprocessors=[preprocessor],low_level_agent=control_agent, obs_dim=obs_dim)
        core.set_variables_for_training(0.1 * np.ones(5))
    else:
        obs_dim = 18
        core = Core([agent, adversary], mdp,low_level_agent=control_agent, obs_dim=obs_dim)

    dataset = core.evaluate(n_episodes=5, render=True, id_agent=1)
    s, a, r, s1, a, info, last = parse_dataset(dataset)

    E = agent.policy.entropy(s)
    E_a = adversary.policy.entropy(s)


    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))

    # success_rate = calculate_success_rate(info, last)

    logger.epoch_info(0, J=J, R=R, entropy=E, entropy_adversary=E_a)

    #


if __name__ == '__main__':

    methods = ["MIN_MAX", "QARL"]
    experiment(method_name="MIN_MAX", task_name="multi_obstacles_navigation")
