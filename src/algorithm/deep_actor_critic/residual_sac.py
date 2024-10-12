import numpy as np

import torch
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter

from copy import deepcopy
from itertools import chain


class SACPolicy(Policy):
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
        return self.compute_action_and_log_prob_t(
            state, compute_log_prob=False).detach().cpu().numpy()

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


class ResidualSAC(DeepAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019.

    """

    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params,
                 actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau,
                 lr_alpha, log_std_min=-20, log_std_max=2, target_entropy=None,
                 critic_fit_params=None):
        """
        Constructor.

        Args:
            actor_mu_params (dict): parameters of the actor mean approximator
                to build;
            actor_sigma_params (dict): parameters of the actor sigm
                approximator to build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size ((int, Parameter)): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            warmup_transitions ([int, Parameter]): number of samples to accumulate in the
                replay memory to start the policy fitting;
            tau ([float, Parameter]): value of coefficient for soft updates;
            lr_alpha ([float, Parameter]): Learning rate for the entropy coefficient;
            log_std_min ([float, Parameter]): Min value for the policy log std;
            log_std_max ([float, Parameter]): Max value for the policy log std;
            target_entropy (float, None): target entropy for the policy, if
                None a default value is computed ;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params
        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)
        self._Q = []
        self._rho = [0]
        self._old_Q = [0]
        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape).astype(np.float32)
        else:
            self._target_entropy = target_entropy

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)

        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        actor_mu_approximator = Regressor(TorchApproximator,
                                          **actor_mu_params)
        actor_sigma_approximator = Regressor(TorchApproximator,
                                             **actor_sigma_params)
        self._actor_last_loss = None  # Store actor loss for logging

        policy = SACPolicy(actor_mu_approximator,
                           actor_sigma_approximator,
                           mdp_info.action_space.low,
                           mdp_info.action_space.high,
                           log_std_min,
                           log_std_max)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)

        self._log_alpha = torch.tensor(0., dtype=torch.float32)

        if policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        policy_parameters = chain(actor_mu_approximator.model.network.parameters(),
                                  actor_sigma_approximator.model.network.parameters())

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _log_alpha='torch',
            _alpha_optim='torch'
        )

        self._boosting = False

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def setup_residual(self, prior_agents, prior_state_dims=None):
        """
            prior_agents ([mushroom object list]): The agent object from agents trained on prior tasks;
            use_kl_on_pi (bool): Whether to use a kl between the prior task policy and the new policy as a loss on the policy
            kl_on_pi_alpha (float): Alpha parameter to weight the KL divergence loss on the policy
        """
        self._boosting = True
        self._Q = []
        self._rho = []
        self._old_Q = []
        self._prior_critic_approximators = list()
        self._prior_policies = list()
        self._prior_state_dims = prior_state_dims

        for prior_agent in prior_agents:
            self._prior_critic_approximators.append(
                prior_agent._target_critic_approximator)  # The target_critic_approximator object from agents trained on prior tasks
            self._prior_policies.append(
                prior_agent.policy)  # The policy object from an agent trained on a prior task

        self._use_kl_on_pi = False  # Whether to use a kl between the prior task policy and the new policy as a loss for the new policy
        self._kl_on_pi_alpha = 0.03  # Alpha parameter to weight the KL divergence loss on the policy
        self._kl_with_prior = [] # KL divergence with previous policy (numpy)
        self._last_kl_penalty = []
        self._kl_with_prior_t = torch.tensor(0.0)  # KL divergence with previous policy (torch)

    def record_kpis(self, q=None, rho=None, old_q=None):
        if q is not None:
            self._Q.append(q.copy().mean())
        if rho is not None:
            self._rho.append(rho.copy().mean())
        if old_q is not None:
            self._old_Q.append(old_q.copy().mean())

    def reset_kpis(self):
        self._Q = []
        self._rho = []
        self._old_Q = []

    def fit(self, dataset):

        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size())

            if self._boosting:
                if self._use_kl_on_pi:
                    # Calculate KL divergence between current policy and previous policy
                    # Note that policies are not residuals so we only need the KL between the immediate previous task and current task
                    prior_state = state[:, 0:self._prior_state_dims]

                    prior_cont_dist = self._prior_policies[-1].distribution(
                        prior_state)  # use prior_state for the immediate previous task
                    curr_cont_dist = self.policy.distribution(state)
                    # Convert to MultivariateNormal distributions (for KL calculation)
                    prior_multiv_cont_dist = torch.distributions.MultivariateNormal(prior_cont_dist.mean,
                                                                                    torch.diag_embed(
                                                                                        prior_cont_dist.variance))
                    curr_multiv_cont_dist = torch.distributions.MultivariateNormal(curr_cont_dist.mean,
                                                                                   torch.diag_embed(
                                                                                       curr_cont_dist.variance))

                    # Use Forward KL instead of reverse KL because prior policy distribution could be peaky
                    self._kl_with_prior_t = torch.distributions.kl.kl_divergence(prior_multiv_cont_dist,
                                                                                 curr_multiv_cont_dist)

                    self._kl_with_prior.append(self._kl_with_prior_t.detach().cpu().numpy().mean())

            if self._replay_memory.size > self._warmup_transitions():
                action_new, log_prob = self.policy.compute_action_and_log_prob_t(state)
                loss = self._loss(state, action_new, log_prob)
                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob.detach())
                self._actor_last_loss = loss.detach().cpu().numpy()
                # Store actor loss for logging

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self.record_kpis(q=q)
            # rho = q  # residual q
            q_old = np.zeros_like(q)

            if self._boosting:
                for idx, prior_critic in enumerate(self._prior_critic_approximators):
                    # # Fitting a 'residual q' i.e 'rho'. So we subtract the prior_rho values
                    # Use prior rho values. Also use appropriate state-spaces as per the prior task
                    prior_state = state[:, 0:self._prior_state_dims]
                    rho_prior = prior_critic.predict(prior_state, action, prediction='min')
                    q_old += rho_prior  # subtract the prior_rho value

            rho = q - q_old
            self.record_kpis(old_q=q_old)
            self.record_kpis(rho=rho)


            self._critic_approximator.fit(state, action, rho,
                                          **self._critic_fit_params)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

    def _loss(self, state, action_new, log_prob):
        q_0 = self._critic_approximator(state, action_new,
                                        output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_new,
                                        output_tensor=True, idx=1)

        q = torch.min(q_0, q_1)

        if self._boosting:
            for idx, prior_critic in enumerate(self._prior_critic_approximators):
                # # For policy loss, use q value as a combination of prior task and current task residual_q (rho) values
                # Use prior rho values. Also use appropriate state-spaces as per the prior task
                prior_state = state[:, 0:self._prior_state_dims]
                # NOTE: We will use the q_prior gradient here for policy fitting but not in the residual_q fitting
                rho_prior = prior_critic.predict(prior_state, action_new, output_tensor=True, prediction='min').values
                q += rho_prior

            if self._use_kl_on_pi:
                # Add a KL penalty for deviating from previous policy (with gradients)
                kl_penalty = torch.tensor(self._kl_on_pi_alpha, device=q.device) * torch.clip(self._kl_with_prior_t,
                                                                                              0.0,
                                                                                      5000.0)
                self._last_kl_penalty.append(kl_penalty.detach().cpu().numpy().mean())
                q -= kl_penalty  # TWEAK: Clip the KL because it can explode

        return (self._alpha * log_prob - q).mean()

    def _update_alpha(self, log_prob):
        alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        a, log_prob_next = self.policy.compute_action_and_log_prob(next_state)

        q = self._target_critic_approximator.predict(next_state, a, prediction='min')

        if self._boosting:
            for idx, prior_critic in enumerate(self._prior_critic_approximators):
                # # 'Next_Q' value should be a combination of prior task and current task residual_q (rho) values
                # Use prior rho values. Also use appropriate state-spaces as per the prior task
                prior_next_state = next_state[:, 0:self._prior_state_dims]
                q_old = prior_critic.predict(prior_next_state, a, prediction='min')
                q += q_old

        q-= self._alpha_np * log_prob_next
        q *= 1 - absorbing

        return q

    def _post_load(self):

        self._update_optimizer_parameters(self.policy.parameters())

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()
