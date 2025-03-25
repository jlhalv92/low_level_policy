from tqdm import tqdm
import numpy as np
np.random.seed()

from src.utils.record.record import VideoRecorder


class Batch:
    def __init__(self, states, actions, rewards, next_states, terminations, last):
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards
        self.terminations = terminations
        self.last = last




class Buffer:
    def __init__(self):
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
                            {},
                            truncated_array[i]])
        return dataset

class Core(object):
    """
    Implements the functions to run a generic algorithm.

    """
    def __init__(self, agent, mdp, callbacks_fit=None, callback_step=None,
                 preprocessors=None, vectorize=False):
        """
        Constructor.

        Args:
            agent (Agent): the agent moving according to a policy;
            mdp (Environment): the environment in which the agent moves;
            callbacks_fit (list): list of callbacks to execute at the end of
                each fit;
            callback_step (Callback): callback to execute after each step;
            preprocessors (list): list of state preprocessors to be
                applied to state variables before feeding them to the
                agent.
            prior_pretrain_only (bool): tells us whether to only pretrain a policy with samples from a prior
            use_data_prior (bool): tells us whether to use a prior from mdp for biasing data collection

        """
        self.agent = agent
        self.mdp = mdp
        self.vectorize = vectorize
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
        self.record=False



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
              n_episodes_per_fit=None, render=False, quiet=False, get_renders=False):
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
        fit_condition = lambda: False
        self.record =record
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
        self.reset(initial_states, initial_conditions)

        infos_l = []
        datasets = [[] for _ in range(self.mdp.env.num_envs)]
        while move_condition():

            self._total_steps_counter += 1
            self._current_steps_counter += 1
            steps_progress_bar.update(1)

            actions = self.agent.draw_noisy_action(self._state)

            next_state, reward, terminated, truncated, info = self.mdp.step(actions)
            done = terminated | truncated
            actual_next_state = next_state.copy()

            for i, single_done in enumerate(done):
                infos_l.append(info)
                if single_done:
                    actual_next_state[i] = info["final_observation"][i]

            for env in range(self.mdp.env.num_envs):
                datasets[env].append([self._state[env]
                                   ,actions[env],
                                    reward[env],
                                actual_next_state[env],
                                terminated[env],
                                {"env_id": "env_{}".format(env)},
                                    done[env]])
            self._state = actual_next_state


            if fit_condition():
                dataset = []
                for trj in datasets:
                    dataset.extend(trj)

                self.agent.fit(dataset)
                self._current_episodes_counter = 0
                self._current_steps_counter = 0

                for c in self.callbacks_fit:
                    c(dataset)

                datasets = [[] for _ in range(self.mdp.env.num_envs)]
                dataset = list()


        self.agent.stop()

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

        if learning:
            action = self.agent.draw_noisy_action(self._state)
            next_state, reward, absorbing, truncated, info = self.mdp.step(action)
        else: # Default

            action = self.agent.draw_action(self._state)

            next_state, reward, absorbing, truncated, info = self.mdp.step(action)

        self._episode_steps += 1

        if render:
            frame = self.mdp.render(record=self.record)

            # if self.record:
            #     self._record(frame)
        if self.vectorize:
            absorbing = absorbing[0]
        last = not(
            self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state
        # if get_renders:
        #     img = self.mdp.get_render()
        #     return img, state, action, reward, next_state, absorbing, info, last
        if self.vectorize:
            return state[0], action[0], reward[0], next_state[0], absorbing, info, last
        return state, action, reward, next_state, absorbing, info, last

    def reset(self, initial_states=None, initial_conditions=None):
        """
        Reset the state of the agent.

        """
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
