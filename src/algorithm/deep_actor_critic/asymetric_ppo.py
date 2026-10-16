import numpy as np
import torch.nn as nn

import torch
import torch.nn.functional as F

from src.mushroom.agent.agent import Agent
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor, update_optimizer_parameters
#from mushroom_rl.utils.minibatches import minibatch_generator
# from mushroom_rl.utils.dataset import parse_dataset, compute_J
from src.mushroom.dataset.dataset import parse_vectorize_dataset
#from mushroom_rl.utils.value_functions import compute_gae
from mushroom_rl.utils.parameters import to_parameter

from collections import defaultdict

import numpy as np


def minibatch_generator(mini_batch_size, *dataset):
    """
    Generator that creates a minibatch from the full dataset.

    Args:
        mini_batch_size (int): the maximum size of each minibatch;
        dataset: the dataset to be splitted.
        # actor_obs, actions, adv, old_log_p
    Returns:
        The current minibatch.

    """

    steps = dataset[0].size()[0]
    n_envs = dataset[0].size()[1]
    batch_size = n_envs * steps
    size = len(dataset[0])
    num_minibatches = int(np.ceil(size/ mini_batch_size))

    states = dataset[0].flatten(0, 1)
    actions = dataset[1].flatten(0, 1)
    old_log_probs = dataset[2].flatten(0, 1)
    advantages = dataset[3].flatten(0, 1)

    # A fresh shuffle for every epoch.
    indices = torch.randperm(batch_size,device=dataset[0].device)

    for batch_idx in torch.tensor_split(indices, num_minibatches):
        yield (
            states[batch_idx],
            actions[batch_idx],
            old_log_probs[batch_idx],
            advantages[batch_idx],
        )


@torch.no_grad()
def compute_gae(Value, Next_value, r, absorbing, last, gamma, lam):
    """
    Returns:
        value_target, advantage: tensors of shape (N_steps, N_envs, 1).

    absorbing: disables value bootstrapping.
    last: stops advantage propagation across episode boundaries.
    """
    absorbing = absorbing.bool()
    last = last.bool()

    delta = (
        r
        + gamma * Next_value.masked_fill(absorbing, 0.0)
        - Value
    )

    advantage = torch.empty_like(Value)
    next_advantage = torch.zeros_like(Value[0])

    for t in reversed(range(Value.shape[0])):
        continuation = next_advantage.masked_fill(
            last[t] | absorbing[t], 0.0
        )
        advantage[t] = delta[t] + gamma * lam * continuation
        next_advantage = advantage[t]

    return advantage + Value, advantage

def compute_cumulative_rewards(trajectories):
    cumulative_rewards = defaultdict(list)  # Stores rewards per environment
    env_rewards = defaultdict(float)  # Tracks ongoing episode rewards per env

    for step in trajectories:
        s, a, reward,_,_, env_id_dict, done = step  # Unpack step
        env_id = env_id_dict["env_id"]
        env_rewards[env_id] += reward  # Accumulate reward

        if done:  # If episode ends
            cumulative_rewards[env_id].append(env_rewards[env_id])
            env_rewards[env_id] = 0  # Reset for next episode

    # Store any ongoing episode rewards
    for env_id, reward in env_rewards.items():
        if reward > 0:
            cumulative_rewards[env_id].append(reward)

    acc_r = []
    for key in cumulative_rewards.keys():
        for ep_r in cumulative_rewards[key]:
            acc_r.append(ep_r)
    return acc_r


class PPO(Agent):
    """
    Proximal Policy Optimization algorithm.
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al.. 2017.

    """
    def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
                 n_epochs_policy, batch_size, eps_ppo, lam, ent_coeff=0.0,
                 critic_fit_params=None):
        """
        Constructor.

        Args:
            policy (TorchPolicy): torch policy to be learned by the algorithm
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            n_epochs_policy ([int, Parameter]): number of policy updates for every dataset;
            batch_size ([int, Parameter]): size of minibatches for every optimization step
            eps_ppo ([float, Parameter]): value for probability ratio clipping;
            lam ([float, Parameter], 1.): lambda coefficient used by generalized
                advantage estimation;
            ent_coeff ([float, Parameter], 1.): coefficient for the entropy regularization term;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._critic_fit_params = dict(n_epochs=10) if critic_fit_params is None else critic_fit_params
        self.actor_idxs = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                     10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 53, 54, 55, 24, 25, 26, 27, 28, 29, 30,
       31, 32, 33, 34, 35, 44, 45, 46, 47, 48, 49, 50, 51, 52])
        self._n_epochs_policy = to_parameter(n_epochs_policy)
        self._batch_size = to_parameter(batch_size)
        self._eps_ppo = to_parameter(eps_ppo)

        self._optimizer = actor_optimizer['class'](policy.parameters(), **actor_optimizer['params'])
        self.learning_rate = actor_optimizer['params']['lr']
        self._lambda = to_parameter(lam)
        self._ent_coeff = to_parameter(ent_coeff)

        self._V = Regressor(TorchApproximator, **critic_params)

        self._iter = 1

        self._add_save_attr(
            _critic_fit_params='pickle', 
            _n_epochs_policy='mushroom',
            _batch_size='mushroom',
            _eps_ppo='mushroom',
            _ent_coeff='mushroom',
            _optimizer='torch',
            _lambda='mushroom',
            _V='mushroom',
            _iter='primitive',
            learning_rate='primitive',
            actor_idxs='numpy',

        )

        super().__init__(mdp_info, policy, None)

    def fit(self, dataset, **info):

        states, actions, old_log_p, values, next_values, rewards, next_states, terminated, truncated = dataset
        #state, action, old_log_p, r, values, next_state, absorbing, last = parse_vectorize_dataset(dataset)
        #x = state.astype(np.float32)
        #u = action.astype(np.float32)
        #r = r.astype(np.float32)
        #xn = next_state.astype(np.float32)

        #obs = to_float_tensor(x, self.policy.use_cuda)
        #act = to_float_tensor(u, self.policy.use_cuda)
        # Compute GAE


        #v_target, np_adv = compute_gae(self._V, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda())
        #np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        #adv = to_float_tensor(np_adv, self.policy.use_cuda)
        v_target, adv= compute_gae(values, next_values, rewards, terminated, truncated, self.mdp_info.gamma, self._lambda())
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        actor_obs = states[:,:, self.actor_idxs].cuda()
        actions_gpu = actions.cuda()
        old_log_p_gpu = old_log_p.cuda()
        adv_gpu = adv.cuda()
        #old_pol_dist = self.policy.distribution_t(actor_obs)
        #old_log_p = old_pol_dist.log_prob(act)[:, None].detach()
        v_target_b = v_target.flatten(0,1)
        states_b = states.flatten(0,1)
        old_policy_distribution = self.policy.distribution(actor_obs.flatten(0,1))

        self._V.fit(states_b, v_target_b, **self._critic_fit_params)

        #
        self._update_policy(actor_obs, actions_gpu, adv_gpu, old_log_p_gpu)



        #self._log_info(dataset, x, v_target.flatten(0,1), old_policy_distribution)

        new_pol_dist = self.policy.distribution(actor_obs.flatten(0,1))
        kl_mean = torch.mean(torch.distributions.kl.kl_divergence(
            new_pol_dist, old_policy_distribution))
        kl_target = 0.01 #0.01

        if kl_mean > kl_target * 2:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            print("KL too big, setting lr to {}".format(self.learning_rate))
        elif kl_mean < kl_target / 2 and kl_mean > 0.00:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            print("KL too small, setting lr to {}".format(self.learning_rate))

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        self._iter += 1



    def _update_policy(self, obs, act, adv, old_log_p):
        for epoch in range(self._n_epochs_policy()):
            for obs_i, act_i, adv_i, old_log_p_i in minibatch_generator(
                    self._batch_size(), obs, act, adv, old_log_p):
                self._optimizer.zero_grad()
                prob_ratio = torch.exp(
                    self.policy.log_prob_t(obs_i, act_i) - old_log_p_i
                )
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(),
                                            1 + self._eps_ppo.get_value())
                loss = -torch.mean(torch.min(prob_ratio * adv_i,
                                             clipped_ratio * adv_i))
                loss -= self._ent_coeff()*self.policy.entropy_t(obs_i)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy._mu.model.network.parameters(), 1.)
                self._optimizer.step()

    def _log_info(self, dataset, x, v_target, old_pol_dist):
        if self._logger:
            logging_verr = []
            torch_v_targets = torch.tensor(v_target, dtype=torch.float)
            for idx in range(len(self._V)):
                v_pred = torch.tensor(self._V(x, idx=idx), dtype=torch.float)
                v_err = F.mse_loss(v_pred, torch_v_targets)
                logging_verr.append(v_err.item())

            logging_ent = self.policy.entropy(x[:, self.actor_idxs])
            new_pol_dist = self.policy.distribution(x[:, self.actor_idxs])
            logging_kl = torch.mean(torch.distributions.kl.kl_divergence(
                new_pol_dist, old_pol_dist))



            cumulative_rewards = np.mean(compute_cumulative_rewards(dataset))

            avg_rwd = cumulative_rewards
            #avg_rwd = np.mean(compute_J(dataset))
            msg = "Iteration {}:\n\t\t\t\trewards {} vf_loss {}\n\t\t\t\tentropy {}  kl {}".format(
                self._iter, avg_rwd, logging_verr, logging_ent, logging_kl)

            self._logger.info(msg)
            self._logger.weak_line()

    def _post_load(self):
        if self._optimizer is not None:
            update_optimizer_parameters(self._optimizer, list(self.policy.parameters()))