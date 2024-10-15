from audioop import avgpp

from tqdm import tqdm
import numpy as np
from collections import deque

np.random.seed()

class HistoryStore:
    def __init__(self, max_size):
        # Initialize with a maximum size and a deque to store the history
        self.max_size = max_size
        self.history = deque(maxlen=max_size)

    def add_sample(self, sample):
        # Add a new sample to the history
        self.history.append(sample)

    def get_history(self):
        # Return the current history as a list
        return np.array(list(self.history)).flatten()

class Core(object):
    """
    Implements the functions to run a generic algorithm.

    """
    def __init__(self, agent, mdp, callbacks_fit=None, callback_step=None,
                 preprocessors=None, low_level_agent=None, obs_dim=18):
        """
        Constructor.

        Args:
             the agent moving according to a policy;
            mdp (Environment): the environment in which the agent moves;
            callbacks_fit (list): list of callbacks to execute at the end of
                each fit;
            callback_step (Callback): callback to execute after each step;
            preprocessors (list): list of state preprocessors to be
                applied to state variables before feeding them to the
                agent.


        """
        self.joystick=False
        self.history = HistoryStore(5)
        self.use_history = True
        self.agent = agent
        self.nav_obs_dim = obs_dim


        self.action = np.zeros(3)
        self.command = np.zeros(3)
        self.low_level_agent = low_level_agent
        self.mdp = mdp
        self.callbacks_fit = callbacks_fit if callbacks_fit is not None else list()
        self.callback_step = callback_step if callback_step is not None else lambda x: None
        self._preprocessors = preprocessors if preprocessors is not None else list()

        self._state = None


        self._prior_sample_count = 0
        self._prior_success_count = 0
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0
        self._episode_steps = None
        self._n_episodes = None
        self._n_steps_per_fit = None
        self._n_episodes_per_fit = None


    def set_variables_for_training(self, new_variables):
        for preprocessor in self._preprocessors:
            preprocessor.reset_episode_count()
            preprocessor.set_variables(new_variables)



    def learn(self, n_steps=None, n_episodes=None, n_steps_per_fit=None,
              n_episodes_per_fit=None, render=False, quiet=False, get_renders=False, id_agent=None):
        """
        This function moves the agent in the environment and fits the policy
        using the collected samples. The agent can be moved for a given number
        of steps or a given number of episodes and, independently from this
        choice, the policy can be fitted after a given number of steps or a
        given number of episodes. By default, the environment is reset.

        Args:
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            n_steps_per_fit (int, None): number of steps between each fit of the
                policy;
            n_episodes_per_fit (int, None): number of episodes between each fit
                of the policy;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """
        assert (n_episodes_per_fit is not None and n_steps_per_fit is None)\
            or (n_episodes_per_fit is None and n_steps_per_fit is not None)

        self._n_steps_per_fit = n_steps_per_fit
        self._n_episodes_per_fit = n_episodes_per_fit

        if n_steps_per_fit is not None:
            fit_condition =\
                lambda: self._current_steps_counter >= self._n_steps_per_fit
        else:
            fit_condition = lambda: self._current_episodes_counter\
                                     >= self._n_episodes_per_fit

        self._run(n_steps, n_episodes, fit_condition, render, quiet, get_renders, learning=True) # Add bool to signify this is a learning run

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None,
                 render=False, quiet=False, get_renders=False, initial_conditions=None):
        """
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from
        a set of initial states for the whole episode. By default, the
        environment is reset.

        Args:
            initial_states (np.ndarray, None): the starting states of each
                episode;
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """
        fit_condition = lambda: False

        return self._run(n_steps, n_episodes, fit_condition, render, quiet, get_renders,
                         initial_states, initial_conditions)

    def _run(self, n_steps, n_episodes, fit_condition, render, quiet, get_renders=False,
             initial_states=None, initial_conditions=None,  learning=False):
        assert n_episodes is not None and n_steps is None and initial_states is None\
            or n_episodes is None and n_steps is not None and initial_states is None\
            or n_episodes is None and n_steps is None and initial_states is not None

        self._n_episodes = len(
            initial_states) if initial_states is not None else n_episodes

        self._n_episodes = len(
            initial_conditions) if initial_conditions is not None else n_episodes

        if n_steps is not None:
            move_condition =\
                lambda: self._total_steps_counter < n_steps

            steps_progress_bar = tqdm(total=n_steps,
                                      dynamic_ncols=True, disable=quiet,
                                      leave=False)
            episodes_progress_bar = tqdm(disable=True)
        else:
            move_condition =\
                lambda: self._total_episodes_counter < self._n_episodes

            steps_progress_bar = tqdm(disable=True)
            episodes_progress_bar = tqdm(total=self._n_episodes,
                                         dynamic_ncols=True, disable=quiet,
                                         leave=False)

        return self._run_impl(move_condition, fit_condition, steps_progress_bar,
                              episodes_progress_bar, render, get_renders, initial_states,initial_conditions, learning)

    def _run_impl(self, move_condition, fit_condition, steps_progress_bar,
                  episodes_progress_bar, render, get_renders, initial_states, initial_conditions, learning):
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0
        dataset = list()
        dataset_antagonist = list()
        self._prior_sample_count = 0
        self._prior_success_count = 0
        last = True
        self.t_i = 0

        while move_condition():
            if last:
                self.t_i = 0
                self.reset(initial_states, initial_conditions)


            sample = self._step(render, get_renders, learning)

            self.callback_step([sample])

            self._total_steps_counter += 1
            self._current_steps_counter += 1
            steps_progress_bar.update(1)

            if sample[-1]:
                self._total_episodes_counter += 1
                self._current_episodes_counter += 1
                episodes_progress_bar.update(1)
            dataset.append(sample)


            if fit_condition():

                self.agent.fit(dataset)


                self._current_episodes_counter = 0
                self._current_steps_counter = 0

                for c in self.callbacks_fit:
                    c(dataset)

                dataset = list()

            last = sample[-1]

        self.agent.stop()

        self.mdp.stop()

        steps_progress_bar.close()
        episodes_progress_bar.close()

        return dataset

    def wrap_to_pi(self,angle):
        """Wraps an angle to the range [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    def position_controller(self, k_x=1.0, k_y=1.0, k_yaw=1.0):
        # Calculate position errors
        x,y = self.mdp._data.qpos[:2]

        x_target = self.mdp._data.geom("target").xpos[0]
        y_target = self.mdp._data.geom("target").xpos[1]

        x_obstacle = self.mdp._data.geom("obstacle_1").xpos[0]
        y_obstacle = self.mdp._data.geom("obstacle_1").xpos[1]

        e_x_o = x -x_obstacle
        e_y_o = y -y_obstacle
        distance_obs = np.sqrt(e_x_o**2 + e_y_o**2)
        yaw = self.mdp.orientation_euler[2]
        e_x = x_target - x
        e_y = y_target - y

        # Calculate the angle to the target
        theta_target = np.arctan2(e_y, e_x)

        # Calculate yaw error
        e_yaw = self.wrap_to_pi(0 - yaw)

        v_x_obs = (2/(1 +distance_obs**2))*e_x_o if distance_obs < 2 else 0
        v_y_obs =  (2/(1+distance_obs**2))*e_y_o if distance_obs < 2 else 0

        theta_obs_target = np.arctan2(v_x_obs, v_y_obs)
        e_obs_yaw = (2/(1 +distance_obs**2))*self.wrap_to_pi(yaw-theta_obs_target) if distance_obs < 2 else 0

        # Calculate linear and angular velocities
        v_x =  k_x * e_x*0 + v_x_obs
        v_y = k_y * e_y*0 + v_y_obs
        w_z = k_yaw * e_yaw + e_obs_yaw

        # Clamp velocities to the range [-1, 1]
        v_x = np.clip(v_x, -1, 1)
        v_y = np.clip(v_y, -1, 1)
        w_z = np.clip(w_z, -1, 1)

        return np.array([v_x, v_y, w_z])

    def _step(self, render, get_renders=False, learning=False):
        """
        Single step.

        Args:
            render (bool): whether to render or not.
            get_renders (bool): whether to return the render images
            learning (bool): tells us whether this is a learning step or an eval

        Returns:
            A tuple containing the previous state, the action sampled by the
            agent, the reward obtained, the reached state, the absorbing flag
            of the reached state and the last step flag.

        """
        obs_dim = self.nav_obs_dim


        if not self.use_history:
            navigation_state = self._state[-obs_dim:]

        else:
            navigation_state =  self.history.get_history()

        if learning:
            # navigation_state = self._state[-obs_dim:]
            self.action = self.agent.draw_noisy_action(navigation_state)

            robot_state = self._state[:-obs_dim]
            if not self.joystick:
                robot_state[54:57] =  np.array([2,2,0.25]) * self.action
            control_action = self.low_level_agent.draw_action(robot_state)
            next_state, reward, absorbing, info = self.mdp.step(control_action)

        else: # Default
            self.action = self.agent.draw_action(navigation_state)
            robot_state = self._state[:-obs_dim]
            if not self.joystick:
                robot_state[54:57] = np.array([2,2,0.25])* self.action

            control_action = self.low_level_agent.draw_action(robot_state)
            next_state, reward, absorbing, info = self.mdp.step(control_action)

        self._episode_steps += 1

        if render:
            self.mdp.render()

        last = not(
            self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        history_state = self.history.get_history()

        next_state = self._preprocess(next_state.copy())
        next_navigation_state = next_state[-obs_dim:]
        self.history.add_sample(next_navigation_state)

        history_next_state = self.history.get_history()

        self._state = next_state

        if not self.use_history:

            if get_renders:
                img = self.mdp.get_render()
                return img, state[-obs_dim:],  self.action, reward, next_state[-obs_dim:], absorbing, info, last

            return state[-obs_dim:],  self.action, reward, next_state[-obs_dim:], absorbing, info, last
        else:
            if get_renders:
                img = self.mdp.get_render()
                return img, history_state, self.action, reward, history_next_state, absorbing, info, last

            return history_state, self.action, reward, history_next_state, absorbing, info, last


    def reset(self, initial_states=None, initial_conditions=None):
        """
        Reset the state of the agent.

        """
        for preprocessor in self._preprocessors:
            preprocessor.update_episode_count(self._total_episodes_counter)

        if initial_states is None\
            or self._total_episodes_counter == self._n_episodes:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]

        # """
        if initial_conditions is None\
            or self._total_episodes_counter == self._n_episodes:
            initial_condition = None
        else:
            initial_condition = initial_conditions[self._total_episodes_counter]

        self._state = self._preprocess(self.mdp.reset(initial_state, initial_condition).copy())

        if self.mdp.task==2 and self.use_history:
            self.history = HistoryStore(5)
            obs= self._state[-self.nav_obs_dim:]
            for _ in range(5):
                self.history.add_sample(obs)

        self.agent.episode_start()
        self.agent.next_action = None
        self._episode_steps = 0


    def _preprocess(self, state):
        """
        Method to apply state preprocessors.

        Args:
            state (np.ndarray): the state to be preprocessed.

        Returns:
             The preprocessed state.

        """
        for p in self._preprocessors:
            state = p(state)

        return state
