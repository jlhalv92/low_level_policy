import warnings

import gymnasium as gym
from gymnasium import spaces as gym_spaces
from src.envs.go1_low_level_locomotion_env import UnitreeGo1
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box, Discrete
import numpy as np
from mushroom_rl.utils.viewer import ImageViewer
# gym.logger.set_level(40)


def make_env(seed, env_id, env_args, domain_randomization, episode_length_in_seconds):
    def thunk():
        env =UnitreeGo1(seed,
                        env_id=env_id,
                   render=False,
                   domain_randomization=domain_randomization,
                   episode_length_in_seconds=episode_length_in_seconds,
                   **env_args)
        return env
    return thunk



class GymnasiumGO1(Environment):
    """
    Interface for Gymnasium environments. It makes it possible to use every
    Gymnasium environment just providing the id, except for the Atari games that
    are managed in a separate class.

    """

    def __init__(self,
                 horizon=1000,
                 gamma=0.99,
                 n_envs=10,
                 joystick=False,
                 domain_randomization=False
                 ,headless=False,
                 wrappers=None,
                 wrappers_args=None,
                 **env_args):
        """
        Constructor.

        Args:
             name (str): gym id of the environment;
             horizon (int): the horizon. If None, use the one from Gym;
             gamma (float, 0.99): the discount factor;
             headless (bool, False): If True, the rendering is forced to be headless.
             wrappers (list, None): list of wrappers to apply over the environment. It
                is possible to pass arguments to the wrappers by providing
                a tuple with two elements: the gym wrapper class and a
                dictionary containing the parameters needed by the wrapper
                constructor;
            wrappers_args (list, None): list of dictionaries of arguments for each wrapper;
            ** env_args: other gym environment parameters.

        """

        # MDP creation
        self._not_pybullet = True
        self._first = True
        self._headless = headless
        self.horizon = horizon
        episode_length_in_seconds = horizon*0.02
        if n_envs > 1:
            vector_environment_class = gym.vector.AsyncVectorEnv
            print("Parallel envs", n_envs)

            env = vector_environment_class(
                [make_env(42,
                          env_id,
                          env_args,
                          domain_randomization,
                          episode_length_in_seconds) for env_id in range(n_envs)],
                autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
            )


        else:
            env = UnitreeGo1(42,

                             render=False,
                             joystick=joystick,
                             domain_randomization=domain_randomization,
                             episode_length_in_seconds=episode_length_in_seconds,
                             **env_args)

        self.env  = env  # always rgb_array render mode

        if wrappers is not None:
            if wrappers_args is None:
                wrappers_args = [dict()] * len(wrappers)
            for wrapper, args in zip(wrappers, wrappers_args):
                if isinstance(wrapper, tuple):
                    self.env = wrapper[0](self.env, **args, **wrapper[1])
                else:
                    self.env = wrapper(self.env, **args, **env_args)

        # horizon = self._set_horizon(self.env, horizon)

        # MDP properties
        assert not isinstance(self.env.observation_space,
                              gym_spaces.MultiDiscrete)
        assert not isinstance(self.env.action_space, gym_spaces.MultiDiscrete)

        dt = self.env.unwrapped.dt if hasattr(self.env.unwrapped, "dt") else 0.1
        action_space = self._convert_gym_space(self.env.action_space)
        observation_space = self._convert_gym_space(self.env.observation_space)

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        if isinstance(action_space, Discrete):
            self._convert_action = lambda a: a[0]
        else:
            self._convert_action = lambda a: a

        super().__init__(mdp_info)

    def reset(self, state=None):
        # if state is None:
        state, info = self.env.reset()

        return np.atleast_1d(state)

    def step(self, action):
        action = self._convert_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)  # truncated flag is ignored

        return np.atleast_1d(obs), reward, terminated, truncated, info

    def render(self, record=False):
        self.env.render()
        return None

    def stop(self):
        self.env.close()


    @staticmethod
    def _set_horizon(env, horizon):

        while not hasattr(env, '_max_episode_steps') and env.env != env.unwrapped:
            env = env.env

        if horizon is None:
            if not hasattr(env, '_max_episode_steps'):
                raise RuntimeError('This gymnasium environment has no specified time limit!')
            horizon = env._max_episode_steps
            if horizon == np.inf:
                warnings.warn("Horizon can not be infinity.")
                horizon = int(1e4)

        if hasattr(env, '_max_episode_steps'):
            env._max_episode_steps = horizon

        return horizon

    @staticmethod
    def _convert_gym_space(space):
        if isinstance(space, gym_spaces.Discrete):
            return Discrete(space.n)
        elif isinstance(space, gym_spaces.Box):
            return Box(low=space.low, high=space.high, shape=space.shape)
        else:
            raise ValueError