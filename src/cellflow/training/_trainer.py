from collections.abc import Sequence
import copy
import gc
from typing import Any, Literal
from loguru import logger
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

from cellflow.data._dataloader import OOCTrainSampler, TrainSampler, ValidationSampler
from cellflow.solvers import _genot, _otfm, _conical
from cellflow.training._callbacks import (
    BaseCallback,
    CallbackRunner,
    ComputationCallback,
)


def _find_hyperparams_state(opt_state, parent=None):
    """Walk a (possibly nested) optax opt_state and return the first state that
    exposes a mutable ``hyperparams`` dict (i.e. was produced by
    :func:`optax.inject_hyperparams`). Returns ``(state, parent_path)`` where
    ``parent_path`` is the list of attribute chain to reach it from the root,
    or ``(None, None)`` if not found.
    """
    if hasattr(opt_state, "hyperparams") and isinstance(
        getattr(opt_state, "hyperparams"), dict
    ):
        return opt_state, parent or []
    # optax.MultiSteps wraps an inner state in .inner_opt_state
    if hasattr(opt_state, "inner_opt_state"):
        found, path = _find_hyperparams_state(
            opt_state.inner_opt_state, (parent or []) + ["inner_opt_state"]
        )
        if found is not None:
            return found, path
    # optax.chain produces a tuple of sub-states
    if isinstance(opt_state, tuple):
        for idx, sub in enumerate(opt_state):
            found, path = _find_hyperparams_state(sub, (parent or []) + [idx])
            if found is not None:
                return found, path
    return None, None


def _replace_hyperparams_state(opt_state, path, new_hp_state):
    """Return a copy of ``opt_state`` with the state at ``path`` replaced by
    ``new_hp_state``. Mirrors :func:`_find_hyperparams_state`'s path format.
    """
    if not path:
        return new_hp_state
    head, rest = path[0], path[1:]
    if isinstance(head, int):
        return tuple(
            _replace_hyperparams_state(sub, rest, new_hp_state) if i == head else sub
            for i, sub in enumerate(opt_state)
        )
    # attribute replacement (e.g. inner_opt_state on MultiSteps)
    inner = getattr(opt_state, head)
    new_inner = _replace_hyperparams_state(inner, rest, new_hp_state)
    return opt_state._replace(**{head: new_inner})


class CellFlowTrainer:
    """Trainer for the OTFM/GENOT solver with a conditional velocity field.

    Parameters
    ----------
        dataloader
            Data sampler.
        solver
            :class:`~cellflow.solvers._otfm.OTFlowMatching` or
            :class:`~cellflow.solvers._genot.GENOT` solver with a conditional velocity field.
        predict_kwargs
            Keyword arguments for the prediction functions
            :func:`cellflow.solvers._otfm.OTFlowMatching.predict` or
            :func:`cellflow.solvers._genot.GENOT.predict` used during validation.
        seed
            Random seed for subsampling validation data.

    Returns
    -------
        :obj:`None`
    """

    def __init__(
        self,
        solver: _otfm.OTFlowMatching | _genot.GENOT,
        predict_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
        model=None,
    ):
        if not isinstance(
            solver,
            (_otfm.OTFlowMatching | _genot.GENOT | _conical.GaussianConicalSolver),
        ):
            raise NotImplementedError(
                f"Solver must be an instance of OTFlowMatching or GENOT, got {type(solver)}"
            )

        self.solver = solver
        self.predict_kwargs = predict_kwargs or {}
        self.rng_subsampling = np.random.default_rng(seed)
        self.training_logs: dict[str, Any] = {}
        self.model = model

    def _validation_step(
        self,
        val_data: dict[str, ValidationSampler],
        mode: Literal["on_log_iteration", "on_train_end"] = "on_log_iteration",
    ) -> tuple[
        dict[str, dict[str, ArrayLike]],
        dict[str, dict[str, ArrayLike]],
    ]:
        """Compute predictions for validation data."""
        # TODO: Sample fixed number of conditions to validate on

        valid_source_data: dict[str, dict[str, ArrayLike]] = {}
        valid_pred_data: dict[str, dict[str, ArrayLike]] = {}
        valid_true_data: dict[str, dict[str, ArrayLike]] = {}
        for val_key, vdl in val_data.items():
            batch = vdl.sample(mode=mode, rng=self.rng_subsampling)
            batch = jax.device_put(batch, jax.devices()[0])
            src = batch["source"]
            condition = batch.get("condition", None)
            true_tgt = batch["target"]
            valid_source_data[val_key] = src
            valid_pred_data[val_key] = self.solver.predict(
                src, condition=condition, batched=True, **self.predict_kwargs
            )
            valid_true_data[val_key] = true_tgt
        return valid_source_data, valid_true_data, valid_pred_data

    def _update_logs(self, logs: dict[str, Any]) -> None:
        """Update training logs."""
        for k, v in logs.items():
            if k not in self.training_logs:
                self.training_logs[k] = []
            self.training_logs[k].append(v)

    def train(
        self,
        dataloader: TrainSampler | OOCTrainSampler,
        num_iterations: int,
        valid_freq: int,
        valid_loaders: dict[str, ValidationSampler] | None = None,
        monitor_metrics: Sequence[str] = [],
        callbacks: Sequence[BaseCallback] = [],
        profile=False,
        early_stopping=False,
        early_stopping_patience: int = 5,
        min_epochs: int = 100,
        epoch_length: int | None = None,
        early_stopping_metric: str = "valid_sinkhorn_div_mean",
        lr_patience: int | None = None,
        lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        resume_state: dict[str, Any] | None = None,
        initial_epoch_offset: int = 0,
    ) -> _otfm.OTFlowMatching | _genot.GENOT:
        """Trains the model.

        Parameters
        ----------
            dataloader
                Dataloader used.
            num_iterations
                Number of iterations to train the model.
            valid_freq
                Frequency of validation.
            valid_loaders
                Valid loaders.
            callbacks
                Callback functions.
            monitor_metrics
                Metrics to monitor.
            early_stopping_metric
                Metric to use for early stopping.

        Returns
        -------
            The trained model.
        """
        if resume_state is not None:
            self.training_logs = resume_state.get("training_logs", {"loss": []})
        else:
            self.training_logs = {"loss": []}
        rng_jax = jax.random.PRNGKey(0)
        rng_np = np.random.default_rng(0)

        # Initiate callbacks
        valid_loaders = valid_loaders or {}
        crun = CallbackRunner(
            callbacks=callbacks,
        )
        if isinstance(dataloader, OOCTrainSampler):
            train_conditions = (
                dataloader.inner._data.data_manager.perturbation_covariates
            )
        else:
            train_conditions = dataloader.data_manager.perturbation_covariates
        crun.on_train_begin(train_conditions=train_conditions, model=self.model)

        if epoch_length is None:
            epoch_length = valid_freq
        iter_offset = int(initial_epoch_offset) * int(epoch_length)
        remaining_iterations = max(num_iterations - iter_offset, 0)
        pbar = tqdm(
            range(remaining_iterations),
            initial=iter_offset,
            total=num_iterations,
        )
        sampler = dataloader
        if isinstance(dataloader, OOCTrainSampler):
            dataloader.set_sampler(num_iterations=remaining_iterations)
        if resume_state is not None:
            valid_loss = resume_state.get("valid_loss", 1e6)
            waited = resume_state.get("waited", 0)
            best_params = resume_state.get("best_params", None)
            waited_lr = resume_state.get("waited_lr", 0)
            lr_reductions = resume_state.get("lr_reductions", 0)
            logger.info(
                f"Resumed early-stopping state: valid_loss={valid_loss:.6f}, "
                f"waited={waited}, waited_lr={waited_lr}, lr_reductions={lr_reductions}, "
                f"epoch_offset={initial_epoch_offset}"
            )
        else:
            valid_loss = 1e6
            waited = 0
            best_params = None
            waited_lr = 0
            lr_reductions = 0
        adaptive_lr = lr_patience is not None and lr_patience > 0
        profile_started = False
        on_cpu = jax.devices()[0].platform == "cpu"
        for it in pbar:
            rng_jax, rng_step_fn = jax.random.split(rng_jax, 2)
            batch = sampler.sample(rng_np)
            loss = self.solver.step_fn(rng_step_fn, batch)
            self.training_logs["loss"].append(float(loss))

            if on_cpu and it > 0 and it % (valid_freq * 10) == 0:
                jax.clear_caches()
                gc.collect()

            if ((it - 1) % valid_freq == 0) and (it > 1):
                # Get predictions from validation data
                if profile and (it - 1) // valid_freq != 1:
                    try:
                        jax.profiler.start_trace("/scratch/jr7348/tmp_jax-trace")
                        profile_started = True
                        logger.info(
                            "JAX profiler started - will trace first 100 iterations"
                        )
                    except RuntimeError as e:
                        logger.warning(f"Could not start JAX profiler: {e}")
                validate = False
                if any(
                    [
                        isinstance(callback, ComputationCallback)
                        for callback in callbacks
                    ]
                ):
                    validate = True
                    valid_source_data, valid_true_data, valid_pred_data = (
                        self._validation_step(valid_loaders, mode="on_log_iteration")
                    )
                else:
                    valid_source_data = valid_true_data = valid_pred_data = None

                # Run callbacks
                current_epoch = int(it // epoch_length) + int(initial_epoch_offset)
                metrics = crun.on_log_iteration(valid_source_data, valid_true_data, valid_pred_data, self.solver, state=self.solver.vf_state_inference, epoch_id=current_epoch)  # type: ignore[arg-type]

                if validate:
                    self._update_logs(metrics)
                    if (early_stopping or adaptive_lr) and current_epoch > min_epochs:
                        current_metric = metrics[early_stopping_metric]
                        improved = current_metric < valid_loss
                        if improved:
                            valid_loss = current_metric
                            waited = 0
                            waited_lr = 0
                            if early_stopping:
                                best_params = jtu.tree_map(
                                    lambda x: x.copy(),
                                    self.solver.vf_state_inference.params,
                                )
                            logger.info(
                                f"New best {early_stopping_metric}={valid_loss:.6f} at epoch {current_epoch}"
                            )
                        else:
                            waited += 1
                            waited_lr += 1

                        if adaptive_lr and waited_lr >= lr_patience:
                            hp_state, hp_parent = _find_hyperparams_state(
                                self.solver.vf_state.opt_state
                            )
                            if hp_state is not None and "learning_rate" in hp_state.hyperparams:
                                current_lr = float(hp_state.hyperparams["learning_rate"])
                                new_lr = max(current_lr * lr_factor, min_lr)
                                if new_lr < current_lr:
                                    new_hyperparams = dict(hp_state.hyperparams)
                                    new_hyperparams["learning_rate"] = jnp.asarray(
                                        new_lr, dtype=jnp.asarray(current_lr).dtype
                                    )
                                    new_hp_state = hp_state._replace(hyperparams=new_hyperparams)
                                    new_opt_state = _replace_hyperparams_state(
                                        self.solver.vf_state.opt_state, hp_parent, new_hp_state
                                    )
                                    self.solver.vf_state = self.solver.vf_state.replace(
                                        opt_state=new_opt_state
                                    )
                                    lr_reductions += 1
                                    logger.info(
                                        f"Reduced LR from {current_lr:.2e} to {new_lr:.2e} "
                                        f"at epoch {current_epoch} (reduction #{lr_reductions})"
                                    )
                                    # Give the model a chance to improve at the new LR
                                    waited = 0
                                else:
                                    logger.info(
                                        f"LR plateau at epoch {current_epoch} but already at min_lr={min_lr:.2e}"
                                    )
                            else:
                                logger.warning(
                                    "lr_patience set but optimizer does not expose "
                                    "'learning_rate' via optax.inject_hyperparams; "
                                    f"opt_state type: {type(self.solver.vf_state.opt_state).__name__}; "
                                    "skipping LR reduction."
                                )
                            waited_lr = 0

                        if early_stopping and waited >= early_stopping_patience:
                            print(f"Early stopping at iteration {it}")
                            if best_params is not None:
                                self.solver.vf_state_inference = (
                                    self.solver.vf_state_inference.replace(
                                        params=best_params
                                    )
                                )
                                logger.info(
                                    f"Restored best params ({early_stopping_metric}={valid_loss:.6f})"
                                )
                            jax.clear_caches()
                            gc.collect()
                            break

                    # Update progress bar
                    mean_loss = np.mean(self.training_logs["loss"][-valid_freq:])
                    postfix_dict = {
                        metric: round(self.training_logs[metric][-1], 3)
                        for metric in monitor_metrics
                    }
                    postfix_dict["loss"] = round(mean_loss, 3)
                    pbar.set_postfix(postfix_dict)

                # Snapshot early-stopping state so a crash-resume can restore
                # valid_loss/waited/waited_lr/lr_reductions/best_params/training_logs.
                self._es_state = {
                    "valid_loss": valid_loss,
                    "waited": waited,
                    "waited_lr": waited_lr,
                    "lr_reductions": lr_reductions,
                    "best_params": best_params,
                    "training_logs": self.training_logs,
                    "last_epoch": current_epoch,
                }
                crun.on_checkpoint(
                    state=self.solver.vf_state_inference, epoch_id=current_epoch
                )
            # Stop profiler after 100 iterations (outside validation block)
            if profile_started:
                jax.profiler.stop_trace()
                profile_started = False
                logger.info(
                    "JAX profiler stopped - trace saved to /scratch/jr7348/tmp_jax-trace"
                )

        # Stop profiler if still running (e.g., early stopping before 100 iterations)
        if profile_started:
            jax.profiler.stop_trace()
            logger.info(
                "JAX profiler stopped - trace saved to /scratch/jr7348/tmp_jax-trace"
            )

        if num_iterations > 0:
            valid_source_data, valid_true_data, valid_pred_data = self._validation_step(
                valid_loaders, mode="on_train_end"
            )
            metrics = crun.on_train_end(
                valid_source_data,
                valid_true_data,
                valid_pred_data,
                self.solver,
                model=self.model,
            )
            self._update_logs(metrics)
            self._update_logs(
                {
                    "source_data": valid_source_data,
                    "pred_data": valid_pred_data,
                    "true_data": valid_true_data,
                }
            )

        self.solver.is_trained = True
        return self.solver
