import os
import pickle
import time

import numpy as np
import torch
from src.utils.distributions.torch_distributions import GammaTorchDistribution, BetaTorchDistribution
from mushroom_rl.utils.torch import to_float_tensor
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from mushroom_rl.core import Logger

G_TOL = 1e-4  # Tolerance for termination by the norm of the Lagrangian gradient
X_TOL = 1e-6  # Tolerance for termination by the change of the independent variable


class Doraemon:
    """
    Klink et al.: A Probabilistic Interpretation of Self-Paced Learning with Applications to Reinforcement Learning
    Implementation based on: https://github.com/psclklnk/spdl
    """

    def __init__(
        self,
        initia_alpha: float,
        initia_beta: float,
        target_alpha: float,
        target_beta: float,
        max_kl,
        performance_lower_bound,
        alpha_lower_bound,
        target_kl_threshold,
        temperature_bounds,
    ):



        self.temp_dist = BetaTorchDistribution(np.array([initia_alpha]),np.array([initia_beta]),
                                               use_cuda=False, dtype=torch.float64)

        self.target_dist = BetaTorchDistribution(np.array([target_alpha]),np.array([target_beta]),
                                                 use_cuda=False, dtype=torch.float64 )



        self.new_alpha = initia_alpha
        self.new_beta = initia_beta
        self.max_kl = max_kl
        self.perf_lb = performance_lower_bound
        self.above_perf_lb = False
        self.alpha_lb = alpha_lower_bound
        self.target_kl_threshold = target_kl_threshold
        self.temp_bounds = temperature_bounds

        self.iteration = 0

        self._logger = Logger("self-paced")
        # self._logger = None


            # Data collection
        self.alfa_temperature_data = [self.temp_dist.parameters()[0].data.detach().cpu().numpy()]
        self.beta_temperature_data = [ self.temp_dist.parameters()[1].data.detach().cpu().numpy()]

    def compute_temp_kl(self, old_temp_dist):
        return torch.distributions.kl.kl_divergence(
            self.temp_dist.distribution_t, old_temp_dist.distribution_t
        )

    def compute_target_temp_kl(self):
        kl_div = torch.distributions.kl.kl_divergence(
            self.temp_dist.distribution_t, self.target_dist.distribution_t
        ).detach()
        kl_div = kl_div.cpu().numpy()
        return kl_div

    def compute_expected_performance(
        self, dist, temp_t, old_temp_log_prob_t, temp_value_t
    ):
        temp_ratio_t = torch.exp(dist.log_pdf_t(temp_t) - old_temp_log_prob_t)

        return torch.mean(temp_ratio_t * temp_value_t)

    def update_temperature_distribution(self, temperatures, values, training_iteration):
        # self.iteration and training_iteration are different
        self.iteration += 1

        old_temp_dist = BetaTorchDistribution.from_weights(
                self.temp_dist.get_weights(), dtype=torch.float64
            )
        # temps_t = to_float_tensor(temperatures, use_cuda=False)
        temps_t = torch.tensor(temperatures, dtype=torch.float64)
        old_temp_log_prob_t = old_temp_dist.log_pdf_t(temps_t).detach()

        # Values of initial states after the policy update
        temp_value_t = to_float_tensor(values, use_cuda=False)

        # Define the KL constraint
        def kl_constraint_func(x0):

            dist = BetaTorchDistribution.from_weights(
                [x0[0], x0[1]], dtype=torch.float64
            )
            kl_div = torch.distributions.kl.kl_divergence(
                dist.distribution_t, old_temp_dist.distribution_t
            )


            return kl_div.detach().cpu().numpy()



        def kl_constraint_grad_func(x0):

            dist = BetaTorchDistribution.from_weights(
                [x0[0], x0[1]], dtype=torch.float64
            )
            kl_div = torch.distributions.kl.kl_divergence(
                dist.distribution_t, old_temp_dist.distribution_t
            )
            parameters_grad_t = torch.autograd.grad(kl_div, dist.parameters())
            parameters_grad_numpy = np.array([g.detach().cpu().numpy() for g in parameters_grad_t]).astype(np.float64)
            return parameters_grad_numpy


        kl_constraint = NonlinearConstraint(
            kl_constraint_func,
            -np.inf,
            self.max_kl,
            jac=kl_constraint_grad_func,
            keep_feasible=True,
        )

        # Define the performance constraint
        def perf_constraint_func(x0):

            dist = BetaTorchDistribution.from_weights(
                [x0[0], x0[1]], dtype=torch.float64
            )
            perf = self.compute_expected_performance(
                dist, temps_t, old_temp_log_prob_t, temp_value_t
            )
            return perf.detach().cpu().numpy()

        def perf_constraint_grad_func(x0):

            dist = BetaTorchDistribution.from_weights(
                [x0[0], x0[1]], dtype=torch.float64
            )
            perf = self.compute_expected_performance(
                dist, temps_t, old_temp_log_prob_t, temp_value_t
            )

            parameters_grad_t = torch.autograd.grad(perf, dist.parameters())

            parameters_grad_numpy = np.array([g.detach().cpu().numpy() for g in parameters_grad_t]).astype(np.float64)

            return parameters_grad_numpy

        perf_constraint = NonlinearConstraint(
            fun=perf_constraint_func,
            lb=self.perf_lb,
            ub=1,
            jac=perf_constraint_grad_func,
            keep_feasible=True,
        )

        if self.compute_target_temp_kl() > self.target_kl_threshold:
            lb = np.array([0.1, 0.1])
            ub = np.array([5., 5.])
            bounds = Bounds(lb, ub, keep_feasible=True)

            # If the bounds are active, clip the standard deviation to be in bounds (because we may re-introduce
            # bounds after they have previously been removed)
            alpha_0 = np.clip(self.temp_dist.get_weights()[:1].copy(), lb[0], ub[0])
            beta_0 = np.clip(self.temp_dist.get_weights()[-1:].copy(), lb[1], ub[1])

        else:
            alpha_0 = self.temp_dist.get_weights()[:1].copy()
            beta_0 = self.temp_dist.get_weights()[-1:].copy()


            bounds = None

        x0 = np.array([alpha_0[0], beta_0[0]])

        try:
            if kl_constraint_func(x0) >= self.max_kl:
                print("Warning! KL-Bound of x0 violates constraint already")

            if perf_constraint_func(x0) >= self.perf_lb:
                print("Optimizing KL divergence")
                self.above_perf_lb = True
                constraints = [kl_constraint, perf_constraint]

                # Define the objective plus Jacobian
                def objective(x0):
                    dist = BetaTorchDistribution.from_weights(
                        [x0[0], x0[1]], dtype=torch.float64
                    )
                    kl_div = torch.distributions.kl.kl_divergence(
                        dist.distribution_t, self.target_dist.distribution_t
                    )
                    parameters_grad = torch.autograd.grad(kl_div, dist.parameters())
                    parameters_grad_numpy = np.array([g.detach().cpu().numpy() for g in parameters_grad]).astype(np.float64)

                    return kl_div.detach().cpu().numpy(), parameters_grad_numpy



                res = minimize(
                    objective,
                    x0,
                    method="trust-constr",
                    jac=True,
                    constraints=constraints,
                    options={"gtol": G_TOL, "xtol": X_TOL},
                )

            # Only optimize the temperature distribution if the performance threshold has not yet been exceeded even
            # once
            elif not self.above_perf_lb:
                print("Distribution frozen")
                res = None

            else:
                print("Error in the optimization")
                res = None

        except Exception as e:

            os.makedirs("optimization_errors", exist_ok=True)
            with open(
                os.path.join("optimization_errors", "error_" + str(time.time())), "wb"
            ) as f:
                pickle.dump((self.temp_dist.get_weights(), temperatures, values), f)
            print(
                "Exception occurred during optimization! Storing state and keeping old values!"
            )
            # raise e
            res = None

        if res is not None and res.success:

            self.temp_dist.set_weights(np.array([res.x[0], res.x[1]]))

        elif res is not None:
            # If it was not a success, but the objective value was improved and the bounds are still valid, we still
            # use the result
            print(f"Optimization failed!: optimizer messge {res.message}")
            alpha_0 = self.temp_dist.get_weights()[:1].copy()
            beta_0 = self.temp_dist.get_weights()[-1:].copy()
            x0_0 = np.array([alpha_0[0], beta_0[0]])
            old_f,_gradients = objective(x0_0)
            cons_ok = True
            for constraint in constraints:
                cons_ok = (
                    cons_ok and constraint.lb <= constraint.fun(res.x) <= constraint.ub
                )
            rate_ok = bounds is None or (
                np.all(bounds.lb <= res.x) and np.all(res.x <= bounds.ub)
            )

            if cons_ok and rate_ok and res.fun < old_f:

                self.temp_dist.set_weights([res.x[0], res.x[1]])
            else:
                print(
                    "Warning! Context optimization unsuccessful - will keep old values. Message: %s"
                    % res.message
                )


        self.alfa_temperature_data.extend(
            [self.alfa_temperature_data[-1]]
            * (training_iteration - len(self.alfa_temperature_data))
        )

        self.alfa_temperature_data.append(
            self.temp_dist.parameters()[0].data.detach().cpu().numpy()
        )

        self.beta_temperature_data.extend(
            [self.beta_temperature_data[-1]]
            * (training_iteration - len(self.beta_temperature_data))
        )

        self.beta_temperature_data.append(
            self.temp_dist.parameters()[1].data.detach().cpu().numpy()
        )


        # Logging
        new_alpha = self.temp_dist.get_weights()[0]
        new_beta = self.temp_dist.get_weights()[1]

        x0 = np.array([new_alpha, new_beta])


        perf = perf_constraint_func(x0)
        self.new_alpha = new_alpha
        self.new_beta = new_beta


        self._log_info(
            perf,
            new_alpha,
            new_beta,
            self.compute_temp_kl(old_temp_dist),
            self.compute_target_temp_kl(),
        )
        self.above_perf_lb = False

    def sample(self, shape):
        sample = self.temp_dist.sample(shape)

        numpy_samples = sample.detach().cpu().numpy()
        return numpy_samples

    def set_logger(self, logger):
        self._logger = logger

    def _log_info(self, performance, alpha, beta, temp_kl, target_temp_kl):
        if self._logger:
            msg = "It.: {} E[J]: {} alpha: {} beta: {} KL(old||new): {} KL(new||target): {}".format(
                self.iteration,
                round(float(performance), 6),
                round(float(alpha), 6),
                round(float(beta), 6),
                round(float(temp_kl.detach().cpu().numpy()), 6),
                round(float(target_temp_kl), 6),
            )


            self._logger.info(msg)