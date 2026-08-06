from typing import Any
from collections.abc import Callable
from functools import partial
import numpy as np
from loguru import logger

import diffrax
import jax
import jax.numpy as jnp
from flax.core import frozen_dict
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils

from cellflow import utils
from cellflow.solvers._otfm import OTFlowMatching
from cellflow._types import ArrayLike
from cellflow.networks._velocity_field import ConditionalVelocityField

__all__ = ["GaussianConicalSolver", "GaussianConicalFlow"]


class GaussianConicalFlow(dynamics.BaseFlow):
    """Conical Gaussian flow starting from origin.

    The path interpolates from 0 to x1 with decreasing noise:
    - μ_t = t * x1
    - σ_t = σ * (1 - t)
    - x_t = μ_t + σ_t * noise

    Args:
      sigma: Noise used for computing time-dependent noise schedule.
    """

    def compute_mu_t(
        self, t: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
    ) -> jnp.ndarray:
        """Mean of probability path: starts at 0, ends at x1."""
        del x0  # unused in conical flow
        return t * x1

    def compute_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
        """Noise decreases linearly from sigma to 0."""
        return self.sigma * (1 - t)

    def compute_ut(
        self, t: jnp.ndarray, x: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
    ) -> jnp.ndarray:
        """Conditional vector field for conical flow."""
        del x0  # unused in conical flow
        return (x1 - x) / (1 - t)


class GaussianConicalSolver(OTFlowMatching):
    """(OT) flow matching :cite:`lipman:22` extended to the conditional setting.

    With an extension to OT-CFM :cite:`tong:23,pooladian:23`, and its
    unbalanced version :cite:`eyring:24`.

    Parameters
    ----------
        vf
            Vector field parameterized by a neural network.
        probability_path
            Probability path between the source and the target distributions.
        time_sampler
            Time sampler with a ``(rng, n_samples) -> time`` signature, see e.g.
            :func:`ott.solvers.utils.uniform_sampler`.
        kwargs
            Keyword arguments for :meth:`cellflow.networks.ConditionalVelocityField.create_train_state`.
    """

    def __init__(
        self,
        vf: ConditionalVelocityField,
        probability_path: GaussianConicalFlow,
        time_sampler: Callable[
            [jax.Array, int], jnp.ndarray
        ] = solver_utils.uniform_sampler,
        n_time_per_sample: int = 256,
        **kwargs: Any,
    ):
        self._is_trained: bool = False
        self.vf = vf
        self.condition_encoder_mode = self.vf.condition_mode
        self.condition_encoder_regularization = self.vf.regularization
        self.probability_path = probability_path
        self.time_sampler = time_sampler
        self.ema = kwargs.pop("ema", 1.0)

        self.vf_state = self.vf.create_train_state(
            input_dim=self.vf.output_dims[-1], **kwargs
        )
        self.vf_state_inference = self.vf.create_train_state(
            input_dim=self.vf.output_dims[-1], **kwargs
        )
        self.vf_step_fn = self._get_vf_step_fn()
        self.n_time_per_sample = n_time_per_sample
        self.match_fn = None

    def _predict_jit(
        self,
        x: ArrayLike,
        condition: dict[str, ArrayLike],
        rng: jax.Array | None = None,
        **kwargs: Any,
    ) -> ArrayLike:
        """See :meth:`OTFlowMatching.predict`."""
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault(
            "stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5)
        )
        kwargs = frozen_dict.freeze(kwargs)

        rng_encoder = utils.default_prng_key(rng)
        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = self.condition_encoder_mode == "deterministic"

        encoder_noise = (
            jnp.zeros(noise_dim)
            if use_mean
            else jax.random.normal(rng_encoder, noise_dim)
        )

        def vf(
            t: jnp.ndarray,
            x: jnp.ndarray,
            args: tuple[dict[str, jnp.ndarray], jnp.ndarray],
        ) -> jnp.ndarray:
            params = self.vf_state_inference.params
            condition, encoder_noise = args
            return self.vf_state_inference.apply_fn(
                {"params": params}, t, x, condition, encoder_noise, train=False
            )[0]

        def solve_ode(
            x0: jnp.ndarray,
            condition: dict[str, jnp.ndarray],
            encoder_noise: jnp.ndarray,
        ) -> jnp.ndarray:
            ode_term = diffrax.ODETerm(vf)
            result = diffrax.diffeqsolve(
                ode_term,
                t0=0.0,
                t1=1.0,
                y0=x0,
                args=(condition, encoder_noise),
                max_steps=4096 * 4096,
                **kwargs,
            )
            return result.ys[0]

        x_pred = jax.vmap(solve_ode, in_axes=[0, None, None])(
            x, condition, encoder_noise
        )
        return x_pred

    def predict(
        self,
        x: ArrayLike | dict[str, ArrayLike],
        condition: dict[str, ArrayLike] | dict[str, dict[str, ArrayLike]],
        rng: jax.Array | None = None,
        batched: bool = False,
        **kwargs: Any,
    ) -> ArrayLike | dict[str, ArrayLike]:
        """Predict the translated source ``x`` under condition ``condition``.

        This function solves the ODE learnt with
        the :class:`~cellflow.networks.ConditionalVelocityField`.

        Parameters
        ----------
        x
            A dictionary with keys indicating the name of the condition and values containing
            the input data as arrays. If ``batched=False`` provide an array of shape [batch_size, ...].
        condition
            A dictionary with keys indicating the name of the condition and values containing
            the condition of input data as arrays. If ``batched=False`` provide an array of shape
            [batch_size, ...].
        rng
            Random number generator to sample from the latent distribution,
            only used if ``condition_mode='stochastic'``. If :obj:`None`, the
            mean embedding is used.
        batched
            Whether to use batched prediction. This is only supported if the input has
            the same number of cells for each condition. For example, this works when using
            :class:`~cellflow.data.ValidationSampler` to sample the validation data.
        kwargs
            Keyword arguments for :func:`diffrax.diffeqsolve`.

        Returns
        -------
        The push-forward distribution of ``x`` under condition ``condition``.
        """
        if batched and not x:
            return {}

        if batched:
            logger.info("Using batched prediction with conical flow.")
            rng = utils.default_prng_key(rng)
            rng_noise, rng_encoder = jax.random.split(rng)
            keys = sorted(x.keys())
            condition_keys = sorted(set().union(*(condition[k].keys() for k in keys)))
            sigma = self.probability_path.sigma
            # assert that the number of cells is the same for each condition
            n_cells = x[keys[0]].shape[0]
            for k in keys:
                assert (
                    x[k].shape[0] == n_cells
                ), "The number of cells must be the same for each condition"
            # Stack inputs and conditions
            key_rngs = jax.random.split(rng_noise, len(keys))
            src_inputs = jnp.stack(
                [sigma * jax.random.normal(k_rng, x[k].shape) for k_rng, k in zip(key_rngs, keys)],
                axis=0,
            )
            batched_conditions = {}
            for cond_key in condition_keys:
                batched_conditions[cond_key] = jnp.stack(
                    [condition[k][cond_key] for k in keys]
                )

            # Setup ODE solver kwargs
            kwargs.setdefault("dt0", None)
            kwargs.setdefault("solver", diffrax.Tsit5())
            kwargs.setdefault(
                "stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5)
            )
            kwargs = frozen_dict.freeze(kwargs)

            noise_dim = (1, self.vf.condition_embedding_dim)
            use_mean = rng is None or self.condition_encoder_mode == "deterministic"

            encoder_noise = (
                jnp.zeros(noise_dim)
                if use_mean
                else jax.random.normal(rng_encoder, noise_dim)
            )

            # Get params once outside the jitted function
            # Ensure params are on the default device (GPU if available)
            params = jax.device_put(self.vf_state_inference.params)
            apply_fn = self.vf_state_inference.apply_fn

            # Ensure inputs are on GPU
            src_inputs = jax.device_put(src_inputs)
            batched_conditions = jax.device_put(batched_conditions)
            encoder_noise = jax.device_put(encoder_noise)

            def vf(t, x, args):
                cond, enc_noise = args
                return apply_fn({"params": params}, t, x, cond, enc_noise, train=False)[
                    0
                ]

            def solve_single_sample(x0, cond, enc_noise):
                ode_term = diffrax.ODETerm(vf)
                result = diffrax.diffeqsolve(
                    ode_term,
                    t0=0.0,
                    t1=1.0,
                    y0=x0,
                    args=(cond, enc_noise),
                    max_steps=4096 * 4096,
                    **kwargs,
                )
                return result.ys[0]

            def solve_condition(x_batch, cond):
                # vmap over cells within a condition
                return jax.vmap(solve_single_sample, in_axes=(0, None, None))(
                    x_batch, cond, encoder_noise
                )

            # vmap over conditions, then jit the whole thing
            batched_solve = jax.vmap(
                solve_condition, in_axes=(0, dict.fromkeys(condition_keys, 0))
            )

            pred_targets = batched_solve(src_inputs, batched_conditions)
            return {k: pred_targets[i] for i, k in enumerate(keys)}
        elif isinstance(x, dict):
            rng_noise, rng_encoder = jax.random.split(utils.default_prng_key(rng))
            key_rngs = jax.random.split(rng_noise, len(x))
            src_inputs = {
                k: self.probability_path.sigma
                * jax.random.normal(k_rng, v.shape)
                for k_rng, (k, v) in zip(key_rngs, x.items())
            }
            key_encoders = jax.random.split(rng_encoder, len(x))
            output = {k: self._predict_jit(src_inputs[k], condition[k], k_rng, **kwargs) for (k_rng, k) in zip(key_encoders, x.keys())}
            return output
        else:
            src_input = self.probability_path.sigma * jax.random.normal(
                utils.default_prng_key(rng), x.shape
            )
            x_pred = self._predict_jit(src_input, condition, rng, **kwargs)
            return np.array(x_pred)
