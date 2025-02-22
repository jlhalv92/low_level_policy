from src.envs.mujoco_gym_env import UnitreeGo1
from torch.distributions.normal import Normal
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mushroom_rl.utils.dataset import compute_J

def make_env():
    def thunk():
        env_args = dict(vx_limit=1.2, wz_limit=1.5)
        env = UnitreeGo1(42, render=False, **env_args)

        return env

    return thunk

def evaluate(eval_agent, env, eval_episodes=10, gamma=0.99, render=False, device=torch.device(0)):
    eval_agent.eval()
    dataset = []
    # env.eval=True
    for i in range(eval_episodes):
        state, _ = env.reset()
        done = False
        episode_return = 0
        steps = 0
        while not done:
            if render:
                env.render()
            a = eval_agent.get_action(torch.Tensor(state).to(device)).detach()
            # a  = env.action_space.sample()
            next_state, reward, absorbing, truncated, info = env.step(a.cpu().numpy())

            done =  truncated|absorbing

            dataset.append((state, a, reward, done))
            state = next_state
            steps += 1

    J = np.mean(compute_J(dataset, 0.99))
    R = np.mean(compute_J(dataset))
    env.close()
    return J,R


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer