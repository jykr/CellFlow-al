from collections.abc import Sequence
import functools
import gc
import jax
import numpy as np
from jax import numpy as jnp
from jax.typing import ArrayLike
from ott.geometry import costs, pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import rbf_kernel

__all__ = [
    "compute_metrics",
    "compute_metrics_fast",
    "compute_mean_metrics",
    "compute_scalar_mmd",
    "compute_r_squared",
    "compute_sinkhorn_div",
    "compute_e_distance",
    "compute_e_distance_fast",
    "maximum_mean_discrepancy",
]


def compute_r_squared(x: ArrayLike, y: ArrayLike) -> float:
    """Compute the R squared score between means of the true (x) and predicted (y) distributions.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].

    Returns
    -------
        A scalar denoting the R squared score.
    """
    return r2_score(np.mean(x, axis=0), np.mean(y, axis=0))


def compute_sinkhorn_div(
    x: ArrayLike, y: ArrayLike, epsilon: float | None = None
) -> float:
    """Compute the Sinkhorn divergence between x and y as in :cite:`feydy:19`.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        epsilon
            The regularization parameter.

    Returns
    -------
        A scalar denoting the sinkhorn divergence value.
    """
    # Ensure inputs are JAX arrays on the correct device
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    res = sinkhorn_divergence(
        pointcloud.PointCloud,
        x=x,
        y=y,
        cost_fn=costs.SqEuclidean(),
        relative_epsilon="std",
        epsilon=epsilon,
    )[0]
    return res


def compute_e_distance(
    x: ArrayLike, y: ArrayLike, memory_efficient: bool = True
) -> float:
    """Compute the energy distance between x and y as in :cite:`Peidli2024`.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        memory_efficient
            If True, uses memory-efficient pairwise distance calculation.
            If False, uses direct broadcasting (faster for small d, but more memory).

    Returns
    -------
        A scalar denoting the energy distance value.
    """
    sigma_X = pairwise_squeuclidean(x, x, memory_efficient=memory_efficient).mean()
    sigma_Y = pairwise_squeuclidean(y, y, memory_efficient=memory_efficient).mean()
    delta = pairwise_squeuclidean(x, y, memory_efficient=memory_efficient).mean()
    return 2 * delta - sigma_X - sigma_Y


def pairwise_squeuclidean(
    x: ArrayLike, y: ArrayLike, memory_efficient: bool = True
) -> ArrayLike:
    """Compute pairwise squared euclidean distances.

    Parameters
    ----------
        x
            An array of shape [n, d].
        y
            An array of shape [m, d].
        memory_efficient
            If True, uses ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y> which is O(n*m) memory.
            If False, uses direct broadcasting which is O(n*m*d) memory but may be faster for small d.

    Returns
    -------
        An array of shape [n, m] containing pairwise squared distances.
    """
    if memory_efficient:
        x_sqnorm = jnp.sum(x**2, axis=-1, keepdims=True)  # (n, 1)
        y_sqnorm = jnp.sum(y**2, axis=-1, keepdims=True)  # (m, 1)
        # ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        dist = x_sqnorm + y_sqnorm.T - 2 * jnp.dot(x, y.T)
        return jnp.maximum(
            dist, 0.0
        )  # Clip to avoid small negatives from numerical errors
    else:
        return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)


def compute_e_distance_fast(x: ArrayLike, y: ArrayLike) -> float:
    """Compute the energy distance between x and y as in :cite:`Peidli2024`.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].

    Returns
    -------
        A scalar denoting the energy distance value.
    """
    return compute_e_distance(x, y)


def compute_metrics(x: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """Compute a set of metrics between two distributions x and y.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].

    Returns
    -------
        A dictionary containing the following computed metrics:

        - the r squared score.
        - the sinkhorn divergence with ``epsilon = 1.0``.
        - the sinkhorn divergence with ``epsilon = 10.0``.
        - the sinkhorn divergence with ``epsilon = 100.0``.
        - the energy distance value.
        - the mean maximum discrepancy loss
    """
    metrics = {}
    metrics["r_squared"] = compute_r_squared(x, y)
    metrics["sinkhorn_div_1"] = compute_sinkhorn_div(x, y, epsilon=1.0)
    metrics["sinkhorn_div_10"] = compute_sinkhorn_div(x, y, epsilon=10.0)
    metrics["sinkhorn_div_100"] = compute_sinkhorn_div(x, y, epsilon=100.0)
    metrics["e_distance"] = compute_e_distance_fast(x, y)
    metrics["mmd"] = compute_scalar_mmd(x, y)
    return metrics


def compute_mean_metrics(
    metrics: dict[str, dict[str, float]], prefix: str = ""
) -> dict[str, list[float]]:
    """Compute the mean value of different metrics.

    Parameters
    ----------
        metrics
            A dictionary where the keys indicate the name of the pertubations and the values are
            dictionaries containing computed metrics.
        prefix
            A string definining the prefix of all metrics in the output dictionary.

    Returns
    -------
        A dictionary where the keys indicate the metrics and the values contain the average metric
        values over all pertubations.
    """
    metric_names = list(list(metrics.values())[0].keys())
    metric_dict: dict[str, list[float]] = {
        prefix + met_name: [] for met_name in metric_names
    }
    for met in metric_names:
        stat = 0.0
        for vals in metrics.values():
            stat += vals[met]
        metric_dict[prefix + met] = stat / len(metrics)
    return metric_dict


def rbf_kernel_fast(x: ArrayLike, y: ArrayLike, gamma: float) -> ArrayLike:
    xx = (x**2).sum(1)
    yy = (y**2).sum(1)
    xy = x @ y.T
    sq_distances = xx[:, None] + yy - 2 * xy
    return jnp.exp(-gamma * sq_distances)


def maximum_mean_discrepancy(
    x: ArrayLike, y: ArrayLike, gamma: float = 1.0, exact: bool = False
) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two distributions x and y.

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        gamma
            Parameter for the rbf kernel.
        exact
            Use exact or fast rbf kernel.

    Returns
    -------
        A scalar denoting the squared maximum mean discrepancy loss.
    """
    kernel = rbf_kernel if exact else rbf_kernel_fast
    xx = kernel(x, x, gamma)
    xy = kernel(x, y, gamma)
    yy = kernel(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_scalar_mmd(
    x: ArrayLike, y: ArrayLike, gammas: Sequence[float] | None = None
) -> float:
    """Compute the Mean Maximum Discrepancy (MMD) across different length scales

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].
        gammas
            A sequence of values for the paramater gamma of the rbf kernel.

    Returns
    -------
        A scalar denoting the average MMD over all gammas.
    """
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    mmds = [maximum_mean_discrepancy(x, y, gamma=gamma) for gamma in gammas]  # type: ignore[union-attr]
    return np.nanmean(np.array(mmds))


def compute_metrics_fast(x: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """Compute metrics which are fast to compute

    Parameters
    ----------
        x
            An array of shape [num_samples, num_features].
        y
            An array of shape [num_samples, num_features].

    Returns
    -------
        A dictionary containing the following computed metrics:

        - the r squared score.
        - the energy distance value.
        - the mean maximum discrepancy loss
    """
    metrics = {}
    metrics["r_squared"] = compute_r_squared(x, y)
    metrics["e_distance"] = compute_e_distance_fast(x, y)
    metrics["mmd"] = compute_scalar_mmd(x, y)
    return metrics
