import numpy as np
from src.utils.state_preprocessor import StateVariableConcatenation
import matplotlib.pyplot as plt
from mushroom_rl.core import Logger
from src.core.core_quadruped_adversary import Core
from src.envs.go1_env import Go1
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.core import Agent
from src.networks.nets import ActorNetwork, CriticNetwork, PolicyNetwork, ValueNetwork


def experiment(
    path_agent, path_adversary, method_name, temp=0, task_name="obstacles_navigation"
):
    np.random.seed()

    logger = Logger(results_dir=None)

    # MDP

    horizon = 1000
    gamma = 0.99
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
    mdp = Go1(horizon, gamma, timestep=0.005, cfg=cfg, task_name=task_name)

    path_c = "src/nominal_models/GO_1/epoch_last"
    control_agent = Agent.load(path_c)
    agent = Agent.load(path_agent)
    adversary = Agent.load(path_adversary)
    if method_name == "QARL":
        preprocessor = StateVariableConcatenation()
        obs_dim = 18 + 1
        core = Core(
            [agent, adversary],
            mdp,
            preprocessors=[preprocessor],
            low_level_agent=control_agent,
            obs_dim=obs_dim,
        )
        core.set_variables_for_training(temp * np.ones(100))
    else:
        obs_dim = 18
        core = Core(
            [agent, adversary], mdp, low_level_agent=control_agent, obs_dim=obs_dim
        )

    dataset = core.evaluate(n_episodes=5, render=True, id_agent=1)
    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    return J, R


if __name__ == "__main__":
    # RARL
    # base_path = "/home/aryaman/Desktop/project-hiqarl/adversarial_locomotion/src/nominal_models/GO1/2024-10-23_20-02-01/obstacles_navigation/Adversarial_SAC"
    # R_list = []
    # for i in range(5):
    #     path_agent = base_path + "/{}/Protagonist/epoch_last".format(i)
    #     path_adversary = base_path + "/{}/ANTAGONIST/epoch_last".format(i)
    #     J, R = experiment(path_agent, path_adversary, "RARL")
    #     print("RARL", i, J, R)
    #     R_list.append(R)
    # print("RARL", np.mean(R_list))

    # QARL
    for temp in [0.001]:
        base_path = "src/nominal_models/GO1/2024-10-23_20-02-18/obstacles_navigation/QARL"
        R_list = []

        path_agent = base_path + "/{}/Protagonist/epoch_last".format(0)
        path_adversary = base_path + "/{}/ANTAGONIST/epoch_last".format(0)
        J, R = experiment(path_agent, path_adversary, "QARL", temp, task_name="multi_obstacles_navigation")

        R_list.append(R)
        print("RARL", temp, np.mean(R_list))
