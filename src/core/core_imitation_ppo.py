from audioop import avgpp

from tqdm import tqdm
import numpy as np
from collections import deque
from src.utils.record.record import VideoRecorder
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
                 preprocessors=None, low_level_agent=None, expert_agent=None,
                 obs_dim=28,  record_dictionary=None, joystick=False):
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
        self.joystick=joystick
        self.history = HistoryStore(5)
        self.use_history = False
        self.agent = agent
        self.nav_obs_dim = obs_dim
        self.beta = 0  # Initial probability of using expert actions
        self.beta_decay = 0.9  # Rate at which beta decays per iteration
        self.beta_min = 0.  # Lower bound for beta
        self.action = np.zeros(3)
        self.command = np.zeros(3)
        self.low_level_agent = low_level_agent
        self.expert_agent = expert_agent
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

        if record_dictionary is None:
            record_dictionary = dict()
        self._record = self._build_recorder_class(**record_dictionary)

    def set_variables_for_training(self, new_variables):
        for preprocessor in self._preprocessors:
            preprocessor.reset_episode_count()
            preprocessor.set_variables(new_variables)

    def _build_recorder_class(self, recorder_class=None, fps=None, **kwargs):
        """
        Method to create a video recorder class.

        Args:
            recorder_class (class): the class used to record the video. By default, we use the ``VideoRecorder`` class
                from mushroom. The class must implement the ``__call__`` and ``stop`` methods.

        Returns:
             The recorder object.

        """

        if not recorder_class:
            recorder_class = VideoRecorder

        if not fps:
            fps = int(1 / self.mdp.info.dt)

        return recorder_class(fps=fps, **kwargs)

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
                 render=False, quiet=False, get_renders=False, initial_conditions=None, record=False):
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
        self.record = record
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
        self._prior_sample_count = 0
        self._prior_success_count = 0
        last = True


        while move_condition():
            if last:
                self.reset()


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
                self.beta = max(self.beta_min, self.beta * self.beta_decay)


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

        student_obs= self.mdp.env.get_robot_observation()
        noisy_states = student_obs
        if self.expert_agent:
            expert_action = self.expert_agent.draw_action(self._state)
        else:
            expert_action = np.zeros(12)
        if learning:
            action = self.agent.draw_noisy_action(noisy_states)
            mixed_action = self.beta * expert_action + (1 - self.beta) * action
            next_state, reward, terminated, truncated, info = self.mdp.step(mixed_action)

        else:
            action = self.agent.draw_action(noisy_states)
            next_state, reward,terminated, truncated, info = self.mdp.step(action)

        absorbing = terminated | truncated



        self._episode_steps += 1

        if render:
            frame = self.mdp.render(record=self.record)
            if self.record:
                self._record(frame)

        last = not(
            self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state


        next_state = self._preprocess(next_state.copy())

        self._state = next_state

        return noisy_states,  expert_action, reward, np.zeros_like(noisy_states), absorbing, info, last


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

        self._state = self._preprocess(self.mdp.reset(initial_state).copy())
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
