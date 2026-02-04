import jax.numpy as jnp
from ott.neural.methods.flows import dynamics


class GaussianConicalFlow(dynamics.BaseFlow):
    """
    Args:
      sigma: Noise used for computing time-dependent noise schedule.
    """

    def compute_mu_t(  # noqa: D102
        self, t: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
    ) -> jnp.ndarray:
        return t * x1

    def compute_sigma_t(self, t: jnp.ndarray) -> jnp.ndarray:
        return self.sigma * (1 - t) ** 2

    def compute_ut(  # noqa: D102
        self, t: jnp.ndarray, x, x0: jnp.ndarray, x1: jnp.ndarray
    ) -> jnp.ndarray:
        return (x1 - x) / (1 - t)
