import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from src.algorithm.deep_actor_critic.imitation.dagger import DaggerSP
from src.algorithm.policies.torch_policy import GaussianTorchPolicy
from src.algorithm.network.actor_critic import ActorCritic
from mushroom_rl.core import Logger
from src.core.core_imitation_ppo import Core

from src.envs.agile_gymnasium_env import GymnasiumGO1
import torch.nn as nn
from mushroom_rl.core import Agent

from mushroom_rl.utils.dataset import compute_J

from tqdm import trange

class HISTORY_WRAPPER():
    def __init__(self, mdp, history_len=630, device="cuda:0"):
        self.history = np.zeros(history_len)
        self.mdp = mdp
        self.device = device
    def reset(self):
        self.mdp.reset()
        self.history[:] = 0.0
        #self.mdp.env.get_AGILE_robot_observation()
        return {"obs":torch.tensor(self.mdp.env.get_AGILE_robot_observation()).float().to(device=self.device)
            , "obs_history": torch.tensor(self.history).float().to(device=self.device),
                'privileged_obs': None}

    def step(self, action):
        _, reward, terminated, truncated, info = self.mdp.step(action)
        next_obs = self.mdp.env.get_AGILE_robot_observation()
        self.history = np.concatenate(
            (self.history[next_obs.shape[0]:], next_obs),
            axis=-1
        )
        return ({"obs":torch.tensor(next_obs).float().to(device=self.device),
                "obs_history":torch.tensor(self.history).float().to(device=self.device), 'privileged_obs': None},
                reward, terminated, truncated, info)

    def stop(self):
        self.mdp.stop()

    def render(self, record=False):
        self.mdp.render(record=record)





class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.fc1 = nn.Linear(n_input, 2*n_features)
        self.ln1 = nn.LayerNorm(2*n_features)

        self.fc2 = nn.Linear(2*n_features, n_features)
        self.ln2 = nn.LayerNorm(n_features)

        self.fc3 = nn.Linear(n_features, n_features//2)
        self.ln3 = nn.LayerNorm(n_features//2)

        self.fc4 = nn.Linear(n_features//2, n_output)

    def forward(self, x):
        x = torch.squeeze(x, 1).float()

        x = F.silu(self.ln1(self.fc1(x)))
        x = F.silu(self.ln2(self.fc2(x)))
        x = F.silu(self.ln3(self.fc3(x)))

        return self.fc4(x)
class ValueNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, 2*n_features)
        self.ln1 = nn.LayerNorm(2*n_features)

        self._h2 = nn.Linear(2*n_features, n_features)
        self.ln2 = nn.LayerNorm(n_features)

        self._h3 = nn.Linear(n_features, n_features//2)
        self.ln3 = nn.LayerNorm(n_features//2)

        self._h4 = nn.Linear(n_features//2, n_output)

        nn.init.orthogonal_(self._h1.weight, np.sqrt(2))
        nn.init.constant_(self._h1.bias, 0.0)
        nn.init.orthogonal_(self._h2.weight, np.sqrt(2))
        nn.init.constant_(self._h2.bias, 0.0)
        nn.init.orthogonal_(self._h3.weight, np.sqrt(2))
        nn.init.constant_(self._h3.bias, 0.0)
        nn.init.orthogonal_(self._h4.weight, 0.01)
        nn.init.constant_(self._h4.bias, 0.0)

    def forward(self, state, **kwargs):
        features1 = F.silu(self.ln1(self._h1(torch.squeeze(state, 1).float())))
        features2 = F.silu(self.ln2(self._h2(features1)))
        features3 = F.silu(self.ln3(self._h3(features2)))
        V = self._h4(features3)

        return V


def calculate_success_rate(info, last):
    info_array = np.array([d["success"] for d in info])
    success_cnt = np.count_nonzero(info_array)
    last_cnt = np.count_nonzero(last)
    success_rate = success_cnt / last_cnt
    return success_rate


def experiment(render=False,
               joystick=False, vel=[1.0,1.0,1.0]):

    np.random.seed()

    logger = Logger("PPO", results_dir=None)
    logger.strong_line()
    task_name = "nominal"
    # agent_task = f"{task_name}_{name_exp}"
    # logger.info('Experiment Algorithm: ' + alg.__name__ + " task " + agent_task )

    # MDP
    batch_size = 1024 #
    gamma = 0.99
    n_features = 256

    env_args = dict(vx_limit=vel[0],vy_limit=vel[1], wz_limit=vel[2] )
    actor_critic = ActorCritic(
        num_obs=42,
        num_privileged_obs=12,
        num_obs_history=42 * \
                        15,
        num_actions=12)

    weights = torch.load("/home/jose-luis/Project/rapid-locomotion-rl/runs/rapid-locomotion/2026-06-11/train/092827.210612/checkpoints/ac_weights_last.pt")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to("cuda:0")
    policy = actor_critic.act_inference


    mdp_0 = GymnasiumGO1(gamma=gamma,  n_envs=1, joystick=joystick, domain_randomization=False,  **env_args)
    mdp_0.env.eval =True
    mdp_0.env.track_velocities = True
    mdp_0.env.resampling_command_steps = 200

    mdp = HISTORY_WRAPPER(mdp_0, 630)


    #
    action_vector = np.load("/home/jose-luis/Project/rapid-locomotion-rl/action_vector.npy")
    command_vector = np.zeros((10,1000,3))
    vel_vector = np.zeros((10,1000,3))

    for episode in range(10):
        obs = mdp.reset()


        for i in range(1000):


            #obs= state["obs"]


            #action = agent.draw_noisy_action(student_obs)
            action  = policy(obs).cpu().detach().numpy()
            command_vector[episode, i, :] = mdp.mdp.env.get_commands()
            vel_vector[episode, i, :] = mdp.mdp.env.get_local_velocity()


            obs, reward, terminated, truncated, info = mdp.step(action)
            if render:
                mdp.render(record=False)
            if terminated or truncated:
                break



    mdp.stop()

    from matplotlib import pyplot as plt
    num_eval_steps = 10*1000
    fig, axs = plt.subplots(3, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * 0.02, num_eval_steps), vel_vector[:,:,0].flatten(), color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * 0.02, num_eval_steps), command_vector[:,:,0].flatten(), color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * 0.02, num_eval_steps), vel_vector[:,:,1].flatten(), color='black', linestyle="-", label="Measured")
    axs[1].plot(np.linspace(0, num_eval_steps *0.02, num_eval_steps), command_vector[:,:,1].flatten(), color='black', linestyle="--", label="Desired")
    axs[1].legend()
    axs[1].set_title("Forward Linear Velocity")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")

    axs[2].plot(np.linspace(0, num_eval_steps * 0.02, num_eval_steps), vel_vector[:,:,2].flatten(), color='black', linestyle="-", label="Measured")
    axs[2].plot(np.linspace(0, num_eval_steps * 0.02, num_eval_steps), command_vector[:,:,2].flatten(), color='black', linestyle="--", label="Desired")
    axs[2].legend()
    axs[2].set_title("Forward Linear Velocity")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Velocity (m/s)")

    # axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    # axs[1].set_title("Joint Positions")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.show()




if __name__ == '__main__':

    experiment(render=True, joystick=True, vel=[3.,0.5,5.])


