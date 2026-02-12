import jax
import jax.numpy as jnp
from ott.neural.methods.flows import dynamics
from cellflow.solvers._otfm import OTFlowMatching

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


from collections.abc import Callable
from functools import partial
import os
from typing import Any
from loguru import logger
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict
from flax.training import train_state
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils

from cellflow import utils
from cellflow._types import ArrayLike
from cellflow.networks._velocity_field import ConditionalVelocityField
from cellflow.solvers.utils import ema_update


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

    def _get_vf_step_fn(self) -> Callable:  # type: ignore[type-arg]
        @jax.jit
        def vf_step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            time: jnp.ndarray,
            source: jnp.ndarray,
            target: jnp.ndarray,
            conditions: dict[str, jnp.ndarray],
            encoder_noise: jnp.ndarray,
        ):
            def loss_fn(
                params: jnp.ndarray,
                t: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                conditions: dict[str, jnp.ndarray],
                encoder_noise: jnp.ndarray,
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_flow, rng_encoder, rng_dropout = jax.random.split(rng, 3)
                x_t = self.probability_path.compute_xt(rng_flow, t, source, target)
                v_t, mean_cond, logvar_cond = vf_state.apply_fn(
                    {"params": params},
                    t,
                    x_t,
                    conditions,
                    encoder_noise=encoder_noise,
                    rngs={"dropout": rng_dropout, "condition_encoder": rng_encoder},
                )
                u_t = self.probability_path.compute_ut(t, x_t, source, target)

                flow_matching_loss = jnp.mean((v_t - u_t) ** 2)
                condition_mean_regularization = 0.5 * jnp.mean(mean_cond**2)
                condition_var_regularization = -0.5 * jnp.mean(
                    1 + logvar_cond - jnp.exp(logvar_cond)
                )
                if self.condition_encoder_mode == "stochastic":
                    encoder_loss = (
                        condition_mean_regularization + condition_var_regularization
                    )
                elif (self.condition_encoder_mode == "deterministic") and (
                    self.condition_encoder_regularization > 0
                ):
                    encoder_loss = condition_mean_regularization
                else:
                    encoder_loss = 0.0
                return flow_matching_loss + encoder_loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(
                vf_state.params, time, source, target, conditions, encoder_noise, rng
            )
            return vf_state.apply_gradients(grads=grads), loss

        return vf_step_fn

    def step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, ArrayLike],
    ) -> float:
        """Single step function of the solver, but each x_1 is trained with many time points.

        Parameters
        ----------
        rng
            Random number generator.
        batch
            Data batch with keys ``src_cell_data``, ``tgt_cell_data``, and
            optionally ``condition``.

        Returns
        -------
        Loss value.
        """
        _, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        condition = batch.get("condition")
        rng_resample, rng_time, rng_step_fn, rng_encoder_noise = jax.random.split(
            rng, 4
        )

        n_samples = tgt.shape[0]
        encoder_noise = jax.random.normal(
            rng_encoder_noise,
            (n_samples * self.n_time_per_sample, self.vf.condition_embedding_dim),
        )

        # Sample single src point and n_time_per_sample time points for each sample in the batch
        src = jax.random.normal(
            rng_resample, (n_samples, tgt.shape[1])
        )  # (n_samples, d)
        time = self.time_sampler(rng_time, n_samples * self.n_time_per_sample).reshape(
            n_samples,
            self.n_time_per_sample,
        )

        # Reshape
        src = jnp.tile(src, (1, self.n_time_per_sample)).reshape(
            n_samples, self.n_time_per_sample, -1
        )  # (n_samples, n_time_per_sample, d)
        tgt = jnp.tile(tgt, (1, self.n_time_per_sample)).reshape(
            n_samples, self.n_time_per_sample, -1
        )  # (n_samples, n_time_per_sample, d)
        condition = jax.tree.map(
            lambda c: (
                jnp.tile(c, (1, self.n_time_per_sample)).reshape(
                    n_samples, self.n_time_per_sample, -1
                )
                if c is not None
                else None
            ),
            condition,
        )

        # flatten everything
        src = src.reshape(-1, src.shape[-1])
        tgt = tgt.reshape(-1, tgt.shape[-1])
        time = time.reshape(-1, 1)
        condition = jax.tree.map(
            lambda c: c.reshape(-1, c.shape[-1]) if c is not None else None, condition
        )

        self.vf_state, loss = self.vf_step_fn(
            rng_step_fn,
            self.vf_state,
            time,
            src,
            tgt,
            condition,
            encoder_noise,
        )

        if self.ema == 1.0:
            self.vf_state_inference = self.vf_state
        else:
            self.vf_state_inference = self.vf_state_inference.replace(
                params=ema_update(
                    self.vf_state_inference.params, self.vf_state.params, self.ema
                )
            )
        return loss

    def get_condition_embedding(
        self, condition: dict[str, ArrayLike], return_as_numpy=True
    ) -> ArrayLike:
        """Get learnt embeddings of the conditions.

        Parameters
        ----------
        condition
            Conditions to encode
        return_as_numpy
            Whether to return the embeddings as numpy arrays.

        Returns
        -------
        Mean and log-variance of encoded conditions.
        """
        cond_mean, cond_logvar = self.vf.apply(
            {"params": self.vf_state_inference.params},
            condition,
            method="get_condition_embedding",
        )
        if return_as_numpy:
            return np.asarray(cond_mean), np.asarray(cond_logvar)
        return cond_mean, cond_logvar

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
        # kwargs.setdefault("dt0", 0.01)  # Fixed step size: 100 steps total
        # kwargs.setdefault("solver", diffrax.Euler())
        # kwargs.setdefault("stepsize_controller", diffrax.ConstantStepSize())
        kwargs = frozen_dict.freeze(kwargs)

        noise_dim = (1, self.vf.condition_embedding_dim)
        use_mean = rng is None or self.condition_encoder_mode == "deterministic"
        rng = utils.default_prng_key(rng)
        encoder_noise = (
            jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)
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
            x: jnp.ndarray,
            condition: dict[str, jnp.ndarray],
            encoder_noise: jnp.ndarray,
        ) -> jnp.ndarray:
            ode_term = diffrax.ODETerm(vf)
            result = diffrax.diffeqsolve(
                ode_term,
                t0=0.0,
                t1=1.0,
                y0=x,
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
            keys = sorted(x.keys())
            condition_keys = sorted(set().union(*(condition[k].keys() for k in keys)))

            # assert that the number of cells is the same for each condition
            n_cells = x[keys[0]].shape[0]
            for k in keys:
                assert (
                    x[k].shape[0] == n_cells
                ), "The number of cells must be the same for each condition"

            # Stack inputs and conditions
            src_inputs = jnp.stack([x[k] for k in keys], axis=0)
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
            rng = utils.default_prng_key(rng)
            encoder_noise = (
                jnp.zeros(noise_dim) if use_mean else jax.random.normal(rng, noise_dim)
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

            def solve_single_cell(x_cell, cond, enc_noise):
                ode_term = diffrax.ODETerm(vf)
                result = diffrax.diffeqsolve(
                    ode_term,
                    t0=0.0,
                    t1=1.0,
                    y0=x_cell,
                    args=(cond, enc_noise),
                    **kwargs,
                )
                return result.ys[0]

            def solve_condition(x_batch, cond):
                # vmap over cells within a condition
                return jax.vmap(solve_single_cell, in_axes=(0, None, None))(
                    x_batch, cond, encoder_noise
                )

            # vmap over conditions, then jit the whole thing
            batched_solve = jax.vmap(
                solve_condition, in_axes=(0, dict.fromkeys(condition_keys, 0))
            )

            pred_targets = batched_solve(src_inputs, batched_conditions)
            return {k: pred_targets[i] for i, k in enumerate(keys)}
        elif isinstance(x, dict):
            return jax.tree.map(
                partial(self._predict_jit, rng=rng, **kwargs),
                x,
                condition,  # type: ignore[attr-defined]
            )
        else:
            x_pred = self._predict_jit(x, condition, rng, **kwargs)
            return np.array(x_pred)

    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained = value
