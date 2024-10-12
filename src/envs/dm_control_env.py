import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from src.dm_control import suite
    from dm_control.suite.wrappers import pixels


from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import *
from mushroom_rl.utils.viewer import CV2Viewer


class DMControl(Environment):
    """
    Interface for dm_control suite Mujoco environments. It makes it possible to
    use every dm_control suite Mujoco environment just providing the necessary
    information.

    """
    def __init__(self, domain_name, task_name, horizon=None, gamma=0.99, task_kwargs=None,
                 dt=.025, width_screen=480, height_screen=480, camera_id=0,
                 use_pixels=False, pixels_width=64, pixels_height=64, padding_zeros=None):
        """
        Constructor.

        Args:
             domain_name (str): name of the environment;
             task_name (str): name of the task of the environment;
             horizon (int): the horizon;
             gamma (float): the discount factor;
             task_kwargs (dict, None): parameters of the task;
             dt (float, .01): duration of a control step;
             width_screen (int, 480): width of the screen;
             height_screen (int, 480): height of the screen;
             camera_id (int, 0): position of camera to render the environment;
             use_pixels (bool, False): if True, pixel observations are used
                rather than the state vector;
             pixels_width (int, 64): width of the pixel observation;
             pixels_height (int, 64): height of the pixel observation;

        """
        # MDP creation
        self.padding_zeros = padding_zeros

        self.env = suite.load(domain_name, task_name, task_kwargs=task_kwargs)
        if use_pixels:
            self.env = pixels.Wrapper(self.env, render_kwargs={'width': pixels_width, 'height': pixels_height})

        # get the default horizon
        if horizon is None:
            horizon = self.env._step_limit
        self.horizon = horizon
        # Hack to ignore dm_control time limit.
        self.env._step_limit = np.inf

        if use_pixels:
            self._convert_observation_space = self._convert_observation_space_pixels
            self._convert_observation = self._convert_observation_pixels
        else:
            self._convert_observation_space = self._convert_observation_space_vector
            self._convert_observation = self._convert_observation_vector

        # MDP properties
        action_space = self._convert_action_space(self.env.action_spec())
        observation_space = self._convert_observation_space(self.env.observation_spec())
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)
        self._height_screen = height_screen
        self._width_screen = width_screen
        self._viewer = CV2Viewer("dm_control", dt, self._width_screen, self._height_screen)
        self._camera_id = camera_id
        self.velocity_tracker = []
        self.ep_length_tracker = []
        self.torso_height_tracker = []
        self.velocity_tracker_ep = []
        self.torso_height_tracker_ep = []
        self.distance_tracker = []
        self.distance_cond_tracker = []
        self.env.task._move_speed = 6 # curriculum 4/0
        self.level= 0
        self.step_counter = 0
        self.n_episodes = 0
        self.target_velocity = 4
        self.rewards = []
        self.norm_rewards = [0]
        self.rewards_mean = 0
        self.rewards_std = 1
        self.evaluate=False
        self.absorbing_tracker = []
        self.success = []
        self.count = 0

        super().__init__(mdp_info)

        self._state = None

    def reset(self, state=None):
        self.velocity_tracker_ep = []
        self.torso_height_tracker_ep = []
        self.step_counter = 0


        if state is None:

            if self.padding_zeros:
                state = self._convert_observation(self.env.reset().observation).copy()
                self._state = np.pad(state, (0, self.padding_zeros), mode='constant', constant_values=0)
                # self._state = np.zeros(279)
            else:
                self._state = self._convert_observation(self.env.reset().observation)
                # self._state = np.zeros(24)

        else:
            raise NotImplementedError


        return self._state

    def step(self, action):
        step = self.env.step(action)
        reward = step.reward # curriculum 5 and 0
        self.rewards.append(reward)


        if self.padding_zeros:
            state = self._convert_observation(step.observation).copy()
            self._state = np.pad(state, (0, self.padding_zeros), mode='constant', constant_values=0)

        else:
            self._state = self._convert_observation(step.observation)
            # self._state = np.zeros(24)
        absorbing = step.last()

        self.step_counter += 1
        self.velocity_tracker_ep.append(self.env.task.v_x)
        self.torso_height_tracker_ep.append(self.env.task.h)

        if (self.step_counter >= self.horizon or absorbing) and self.evaluate:
            self.n_episodes += 1
            self.ep_length_tracker.append(self.step_counter)
            self.velocity_tracker.append(np.mean(self.velocity_tracker_ep))
            self.torso_height_tracker.append(np.mean(self.torso_height_tracker_ep))
            self.distance_tracker.append(self.env.physics.torso_pos()[0]-self.env.task.init_pose)
            self.absorbing_tracker.append(absorbing)

            distance = self.env.physics.torso_pos()[0]-self.env.task.init_pose
            current_level = self.level
            max_level = 6
            max_distance = 25
            current_position = max_distance * current_level + distance
            total_length = max_distance * max_level
            success = (current_position / total_length)
            self.success.append(success)

        # if (self.step_counter >= self.horizon or absorbing) and not self.evaluate:
        #     self.distance_cond_tracker.append(self.env.physics.torso_pos()[0] - self.env.task.init_pose)
        #     if len(self.distance_cond_tracker) >= 5:
        #         distance = 50
        #         cond_1 = np.mean(self.distance_cond_tracker[-5:]) >= distance
        #         self.count = np.mean(self.distance_cond_tracker[-5:])
        #         if cond_1  and self.level < 5:
        #             print("Level UP")
        #             self.level += 1
        #             self.distance_cond_tracker = []
        #             self.absorbing_tracker = []
        #             self.env.task.curriculum_level = np.clip(self.level,0,5)



        return self._state, reward, absorbing, {}


    def render(self, record=False):
        img = self.env.physics.render(self._height_screen,
                                      self._width_screen,
                                      self._camera_id)

        self._viewer.display(img)

        if record:
            return img
        else:
            return None

    def stop(self):

        self._viewer.close()

    @staticmethod
    def _convert_observation_space_vector(observation_space):
        observation_shape = 0
        for i in observation_space:
            shape = observation_space[i].shape
            observation_var = 1
            for dim in shape:
                observation_var *= dim
            observation_shape += observation_var

        return Box(low=-np.inf, high=np.inf, shape=(observation_shape,))


    @staticmethod
    def _convert_observation_space_pixels(observation_space):
        img_size = observation_space['pixels'].shape
        return Box(low=0., high=255.,
            shape=(3, img_size[0], img_size[1]))

    @staticmethod
    def _convert_action_space(action_space):
        low = action_space.minimum
        high = action_space.maximum

        return Box(low=np.array(low), high=np.array(high))

    @staticmethod
    def _convert_observation_vector(observation):
        obs = list()
        for i in observation:
            obs.append(np.atleast_1d(observation[i]).flatten())

        return np.concatenate(obs)

    @staticmethod
    def _convert_observation_pixels(observation):
        return observation['pixels'].transpose((2, 0, 1))
