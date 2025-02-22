import numpy as np

import torch
import torch.nn as nn
from mushroom_rl.core import Agent
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.dataset import parse_dataset, compute_J

import torch.optim as optim
from mushroom_rl.utils.torch import update_optimizer_parameters
from src.mushroom.deep_actor_critic.deep_actor_critic import DeepAC
from src.algorithm.policies.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from src.algorithm.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter

from copy import deepcopy
from itertools import chain


class StochasticPolicy(Policy):
    """
    Class used to implement the policy used by the Soft Actor-Critic
    algorithm. The policy is a Gaussian policy squashed by a tanh.
    This class implements the compute_action_and_log_prob and the
    compute_action_and_log_prob_t methods, that are fundamental for
    the internals calculations of the SAC algorithm.

    """

    def __init__(self, mu_approximator, sigma_approximator, min_a, max_a,
                 log_std_min, log_std_max):
        """
        Constructor.

        Args:
            mu_approximator (Regressor): a regressor computing mean in given a
                state;
            sigma_approximator (Regressor): a regressor computing the variance
                in given a state;
            min_a (np.ndarray): a vector specifying the minimum action value
                for each component;
            max_a (np.ndarray): a vector specifying the maximum action value
                for each component.
            log_std_min ([float, Parameter]): min value for the policy log std;
            log_std_max ([float, Parameter]): max value for the policy log std.

        """
        self._mu_approximator = mu_approximator
        self._sigma_approximator = sigma_approximator
        self.use_tanh = True

        self._delta_a = to_float_tensor(.5 * (max_a - min_a), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (max_a + min_a), self.use_cuda)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._eps_log_prob = 1e-6

        use_cuda = self._mu_approximator.model.use_cuda

        if use_cuda:
            self._delta_a = self._delta_a.cuda()
            self._central_a = self._central_a.cuda()

        self._add_save_attr(
            _mu_approximator='mushroom',
            _sigma_approximator='mushroom',
            _delta_a='torch',
            _central_a='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            _eps_log_prob='primitive'
        )

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        a = self.compute_action_and_log_prob_t(
            state, compute_log_prob=False).detach().cpu().numpy()
        return a

    def draw_noisy_action(self, state):
        # Dummy call to draw_action
        return self.draw_action(state)

    def compute_action_and_log_prob(self, state):
        """
        Function that samples actions using the reparametrization trick and
        the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled.

        Returns:
            The actions sampled and the log probability as numpy arrays.

        """
        a, log_prob = self.compute_action_and_log_prob_t(state)
        return a.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_and_log_prob_t(self, state, compute_log_prob=True):
        """
        Function that samples actions using the reparametrization trick and,
        optionally, the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled;
            compute_log_prob (bool, True): whether to compute the log
            probability or not.

        Returns:
            The actions sampled and, optionally, the log probability as torch
            tensors.

        """
        dist = self.distribution(state)

        a_raw = dist.rsample()
        a = torch.tanh(a_raw)

        a_true = a * self._delta_a + self._central_a

        if compute_log_prob:
            log_prob = dist.log_prob(a_raw).sum(dim=1)
            log_prob -= torch.log(1. - a.pow(2) + self._eps_log_prob).sum(dim=1)
            return a_true, log_prob
        else:
            return a_true

    def distribution(self, state):
        """
        Compute the policy distribution in the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is
                computed.

        Returns:
            The torch distribution for the provided states.

        """
        mu = self._mu_approximator.predict(state, output_tensor=True)
        log_sigma = self._sigma_approximator.predict(state, output_tensor=True)
        # Bound the log_s
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())
        return torch.distributions.Normal(mu, log_sigma.exp())

    def entropy(self, state=None):
        """
        Compute the entropy of the policy.

        Args:
            state (np.ndarray): the set of states to consider.

        Returns:
            The value of the entropy of the policy.

        """

        return torch.mean(self.distribution(state).entropy()).detach().cpu().numpy().item()

    def reset(self):
        pass

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by
                the policy.

        """
        mu_weights = weights[:self._mu_approximator.weights_size]
        sigma_weights = weights[self._mu_approximator.weights_size:]

        self._mu_approximator.set_weights(mu_weights)
        self._sigma_approximator.set_weights(sigma_weights)

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        mu_weights = self._mu_approximator.get_weights()
        sigma_weights = self._sigma_approximator.get_weights()

        return np.concatenate([mu_weights, sigma_weights])

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._mu_approximator.model.use_cuda

    def parameters(self):
        """
        Returns the trainable policy parameters, as expected by torch
        optimizers.

        Returns:
            List of parameters to be optimized.

        """
        return chain(self._mu_approximator.model.network.parameters(),
                     self._sigma_approximator.model.network.parameters())




class Dagger(DeepAC):


    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params,
                 actor_optimizer, batch_size,
                 offline_transitions, max_replay_size,
                 log_std_min=-20, log_std_max=2,range=None, iterations=1):

        self._batch_size = to_parameter(batch_size)
        self._replay_memory = ReplayMemory(offline_transitions, max_replay_size)
        self._iterations = to_parameter(iterations)
        self._loss_list = []
        actor_mu_approximator = Regressor(TorchApproximator,
                                          **actor_mu_params)
        actor_sigma_approximator = Regressor(TorchApproximator,
                                             **actor_sigma_params)
        if range is not None:
            low = range[0]
            high = range[1]
        else:
            low = mdp_info.action_space.low
            high = mdp_info.action_space.high

        policy = StochasticPolicy(actor_mu_approximator,
                           actor_sigma_approximator,
                           low,
                           high,
                           log_std_min,
                           log_std_max)

        policy_parameters = chain(actor_mu_approximator.model.network.parameters(),
                                  actor_sigma_approximator.model.network.parameters())

        self._add_save_attr(
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _iterations='mushroom',
            _loss_list='primitive',
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset,  **info):
        self._replay_memory.add(dataset)

        if self._replay_memory.initialized:

            for _ in range(self._iterations()):
                state, action, _, _, _, _ =  self._replay_memory.get(self._batch_size())
                loss = self._loss(state, action)
                self._optimize_actor_parameters(loss)
                self._loss_list.append(loss.clone().detach().cpu().numpy())


    def _loss(self, state, action_expert):
        predicted_action = self.policy.compute_action_and_log_prob_t(state, compute_log_prob=False)
        action_expert_t = torch.from_numpy(action_expert).to(0)

        loss = nn.MSELoss()

        return loss(predicted_action, action_expert_t)

    def _post_load(self):

        self._update_optimizer_parameters(self.policy.parameters())



class DaggerSP(Agent):


    def __init__(self, mdp_info, policy,actor_optimizer, batch_size, iterations):

        self._batch_size = to_parameter(batch_size)
        self._iterations = to_parameter(iterations)
        self._loss_list = []
        self._optimizer = actor_optimizer['class'](policy.parameters(), **actor_optimizer['params'])

        self._add_save_attr(
            _optimizer='torch',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _iterations='mushroom',
            _loss_list='primitive',
        )

        super().__init__(mdp_info, policy, None)

    def fit(self, dataset,  **info):
        x, u, r, xn, absorbing,_, last = parse_dataset(dataset)
        obs = x.astype(np.float32)
        actions = u.astype(np.float32)


        for _ in range(self._iterations()):
            for obs_i, act_i, in minibatch_generator(self._batch_size(), obs, actions):
                self._update_policy(obs_i, act_i)



    def _update_policy(self, state, action_expert):
        self._optimizer.zero_grad()
        state_t = torch.from_numpy(state).to(0).float()
        prob_ratio = self.policy.draw_action_t_attach(state_t)

        # predicted_action = self.policy.draw_action_t_attach(state)

        action_expert_t = torch.from_numpy(action_expert).float().to(0)
        criterion = nn.MSELoss()
        loss = criterion(prob_ratio, action_expert_t)
        self._loss_list.append(loss.clone().detach().cpu().numpy())
        # loss= torch.tensor(predicted_action) - action_expert_t
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5)

        self._optimizer.step()

    def _update_optimizer_parameters(self, parameters):
        self._parameters = list(parameters)
        if self._optimizer is not None:
            update_optimizer_parameters(self._optimizer, self._parameters)

    def _post_load(self):

        self._update_optimizer_parameters(self.policy.parameters())


