from src.envs.gymnasium_env import GymnasiumGO1
import gymnasium as gym
from gymnasium import spaces as gym_spaces
from src.envs.mujoco_gym_env import UnitreeGo1
from mushroom_rl.utils.dataset import compute_J

# env = UnitreeGo1(42, render=False)
import numpy as np
n_envs = 5
env = gym.vector.AsyncVectorEnv([lambda: (UnitreeGo1(42, render=False))] * n_envs)
# print(env.single_observation_space.shape)
# env = UnitreeGo1(42, render=False)
env.eval=True
# env = VectorizeEnvHandler(env)
obs, infos  = env.reset()
# autoreset = np.zeros(env.num_envs)
class Buffer:
    def __init__(self, n_envs):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminated = []
        self.info = []
        self.truncated = []
    def append(self, state, action, reward, next_state, terminated, truncated, info):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminated.append(terminated)
        self.info =info
        self.truncated.append(truncated)
    def return_dataset(self, single_observation_space, action_space):
        dataset = []
        states_array = np.array(self.states).reshape((-1,) + single_observation_space.shape)
        actions_array = np.array(self.actions).reshape((-1,) + action_space.shape)
        rewards_array = np.array(self.rewards).reshape(-1,)
        next_states_array = np.array(self.next_states).reshape((-1,) + single_observation_space.shape)
        terminated_array = np.array(self.terminated).reshape(-1)
        truncated_array = np.array(self.truncated).reshape(-1)
        for i in range(len(states_array)):
            dataset.append([states_array[i],
                            actions_array[i],
                            rewards_array[i],
                            next_states_array[i],
                            terminated_array[i],
                            truncated_array[i]])
        return dataset






dataset = []
buffer = Buffer(n_envs)

state, _ = env.reset()
done = np.zeros(n_envs)
episode_return = 0
steps = 0
for i in range(5000):
    actions = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(actions)
    done = terminated | truncated
    next_state = next_state
    actual_next_state = next_state.copy()
    for i, single_done in enumerate(done):
        if single_done:
            actual_next_state[i] = info["final_observation"][i]

    buffer.append(state, actions, reward, actual_next_state, terminated, done, info)
    state = next_state
dataset = buffer.return_dataset(env.single_observation_space, env.single_action_space)

J = np.mean(compute_J(dataset, 0.99))
R = np.mean(compute_J(dataset))
print(J, R)