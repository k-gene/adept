import argparse
import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import dill as pickle
import numpy as np
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path("/g/g15/kur1/ws/ML_backscatter/adept")
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/sbsbs-1d/sbsbs_cbet_nbeams.yaml"
DEFAULT_DATA_PATH = REPO_ROOT / "adept/Fake_ML_Backscatter_Data.npz"
DEFAULT_OUTDIR = "/g/g15/kur1/ws/ML_backscatter/mlbs_predmodel/runs/sbsbs_train"
DEFAULT_VENV_ACTIVATE = "/g/g15/kur1/ws/ML_backscatter/mlbs_predmodel/myenv_mlbs_local/bin/activate"
DEFAULT_BASE_TEMPDIR = "/tmp/kur1"
DEFAULT_USE_MLFLOW = True
# A safer default for this JAX/XLA workload is to reduce concurrent compilations.
DEFAULT_WORKERS = 32
DEFAULT_CORES_PER_WORKER = 1
DEFAULT_TIMEPOINTS_PER_TASK = 4
DEFAULT_XLA_CPU_MULTI_THREAD_EIGEN = False


@dataclass(frozen=True)
class NpzDataset:
    input_powers: np.ndarray  # (shots, beams, tpts)
    sbs_powers: np.ndarray  # (shots, beams, tpts)
    design_inputs: np.ndarray  # (shots, n_design_params)

    @property
    def n_shots(self) -> int:
        return int(self.input_powers.shape[0])

    @property
    def n_beams(self) -> int:
        return int(self.input_powers.shape[1])

    @property
    def tpts(self) -> int:
        return int(self.input_powers.shape[2])


def compute_reflectivity(input_powers: np.ndarray, sbs_powers: np.ndarray) -> np.ndarray:
    """
    Compute per-beam reflectivity with safe divide-by-zero behavior.

    Both inputs are expected to have identical shapes, typically (beams,) or (beams, tpts).
    """
    input_powers = np.asarray(input_powers)
    sbs_powers = np.asarray(sbs_powers)
    reflectivity = np.zeros_like(input_powers, dtype=float)
    nonzero = input_powers != 0
    reflectivity[nonzero] = sbs_powers[nonzero] / input_powers[nonzero]
    return reflectivity


def _build_run_dir_name(*, mlflow_run_id: str | None) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    parts = [timestamp]
    if slurm_job_id:
        parts.append(f"job{slurm_job_id}")
    if mlflow_run_id:
        run_id_short = mlflow_run_id[:8] if len(mlflow_run_id) >= 8 else mlflow_run_id
        parts.append(f"mlflow_{run_id_short}")

    return "_".join(parts)


def _build_design_inputs(data: dict, n_design_params: int) -> np.ndarray:
    design_params = np.asarray(data["design_params"])
    pulse_durations = np.asarray(data["pulse_durations"])
    if pulse_durations.ndim == 1:
        pulse_durations = pulse_durations.reshape(-1, 1)

    if n_design_params == design_params.shape[1] + pulse_durations.shape[1]:
        return np.concatenate([design_params, pulse_durations], axis=1)
    if n_design_params == design_params.shape[1]:
        return design_params

    raise ValueError(
        "Config/dataset mismatch for design inputs: "
        f"cfg.nn.n_design_params={n_design_params}, "
        f"design_params.shape={design_params.shape}, pulse_durations.shape={pulse_durations.shape}. "
        "Either set cfg.nn.n_design_params to match design_params, or to match design_params+pulse_durations."
    )


def load_npz_dataset(npz_path: str, *, n_design_params: int) -> NpzDataset:
    data = np.load(npz_path)
    input_powers = np.asarray(data["input_powers"])
    sbs_powers = np.asarray(data["sbs_powers"])
    design_inputs = _build_design_inputs(data, n_design_params=n_design_params)
    return NpzDataset(input_powers=input_powers, sbs_powers=sbs_powers, design_inputs=design_inputs)


def _xla_flags_for_worker(*, cores_per_worker: int, xla_cpu_multi_thread_eigen: bool) -> str:
    thread_limit = str(max(1, int(cores_per_worker)))
    eigen_flag = "true" if xla_cpu_multi_thread_eigen else "false"
    return f"--xla_cpu_multi_thread_eigen={eigen_flag} intra_op_parallelism_threads={thread_limit}"


def _configure_worker_env(
    *, base_tempdir: str | None, cores_per_worker: int, xla_cpu_multi_thread_eigen: bool = DEFAULT_XLA_CPU_MULTI_THREAD_EIGEN
) -> None:
    thread_limit = str(max(1, int(cores_per_worker)))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["OMP_NUM_THREADS"] = thread_limit
    os.environ["OPENBLAS_NUM_THREADS"] = thread_limit
    os.environ["MKL_NUM_THREADS"] = thread_limit
    os.environ["NUMEXPR_NUM_THREADS"] = thread_limit
    os.environ["TF_NUM_INTRAOP_THREADS"] = thread_limit
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = _xla_flags_for_worker(
        cores_per_worker=cores_per_worker, xla_cpu_multi_thread_eigen=xla_cpu_multi_thread_eigen
    )
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    if base_tempdir:
        os.environ["BASE_TEMPDIR"] = base_tempdir


def _process_pool_initializer(
    base_tempdir: str | None, cores_per_worker: int, xla_cpu_multi_thread_eigen: bool = DEFAULT_XLA_CPU_MULTI_THREAD_EIGEN
) -> None:
    _configure_worker_env(
        base_tempdir=base_tempdir,
        cores_per_worker=cores_per_worker,
        xla_cpu_multi_thread_eigen=xla_cpu_multi_thread_eigen,
    )


def _thread_env_snapshot() -> dict[str, str | None]:
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "TF_NUM_INTRAOP_THREADS": os.environ.get("TF_NUM_INTRAOP_THREADS"),
        "TF_NUM_INTEROP_THREADS": os.environ.get("TF_NUM_INTEROP_THREADS"),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
        "BASE_TEMPDIR": os.environ.get("BASE_TEMPDIR"),
    }


def _import_training_stack():
    # Delay JAX-heavy imports until after the local process pool has been
    # created. The pool uses spawn, so workers stay clear of fork-after-JAX.
    import equinox as eqx
    import jax
    import optax

    from adept import ergoExo, utils as adept_utils
    from adept._sbsbs1d.base import Train_SBSBS_CBET

    return eqx, jax, optax, ergoExo, adept_utils, Train_SBSBS_CBET


def _run_one_val_and_grad(_run_cfg: dict, diff_modules: dict, static_modules: dict) -> tuple[float, dict]:
    import os
    from adept import ergoExo
    from adept._sbsbs1d.base import Train_SBSBS_CBET

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    exo = ergoExo()
    exo.setup(_run_cfg, adept_module=Train_SBSBS_CBET, log=False)
    val, grad, _ = exo.val_and_grad(
        diff_modules,
        args={
            "static_modules": static_modules,
            "reflectivity": _run_cfg["reflectivity"],
            "beam_mask": _run_cfg["beam_mask"],
        },
        export=False,
        log=False,
    )
    return float(val), grad


def _run_timepoint_batch_val_and_grad(
    run_cfgs: list[dict], diff_modules: dict, static_modules: dict
) -> tuple[float, dict, dict]:
    import os
    import time
    import numpy as np
    from adept import ergoExo, utils as adept_utils
    from adept._sbsbs1d.base import Train_SBSBS_CBET

    start_time = time.perf_counter()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    vals = []
    grads = []
    for run_cfg in run_cfgs:
        exo = ergoExo()
        exo.setup(run_cfg, adept_module=Train_SBSBS_CBET, log=False)
        val, grad, _ = exo.val_and_grad(
            diff_modules,
            args={
                "static_modules": static_modules,
                "reflectivity": run_cfg["reflectivity"],
                "beam_mask": run_cfg["beam_mask"],
            },
            export=False,
            log=False,
        )
        vals.append(val)
        grads.append(grad)

    avg_val = float(np.mean(vals))
    avg_grad = adept_utils.all_reduce_gradients(grads, num=len(grads))
    timing = {
        "elapsed_s": float(time.perf_counter() - start_time),
        "n_timepoints": len(run_cfgs),
    }
    return avg_val, avg_grad, timing


def _run_timepoint_batch_val_and_grad_serialized(
    run_cfgs: list[dict], diff_modules_bytes: bytes, static_modules_bytes: bytes
) -> bytes:
    diff_modules = pickle.loads(diff_modules_bytes)
    static_modules = pickle.loads(static_modules_bytes)
    return pickle.dumps(_run_timepoint_batch_val_and_grad(run_cfgs, diff_modules, static_modules))


def train_model(
    cfg_path: str | Path = DEFAULT_CONFIG_PATH,
    data_path: str | Path = DEFAULT_DATA_PATH,
    outdir: str | Path = DEFAULT_OUTDIR,
    *,
    seed: int = 0,
    workers: int = DEFAULT_WORKERS,
    cores_per_worker: int = DEFAULT_CORES_PER_WORKER,
    venv_activate: str | None = DEFAULT_VENV_ACTIVATE,
    base_tempdir: str | None = DEFAULT_BASE_TEMPDIR,
    use_mlflow: bool = DEFAULT_USE_MLFLOW,
    log_every: int = 10,
    save_every: int = 50,
    max_shots: int | None = None,
    max_timepoints: int | None = None,
    timepoints_per_task: int = DEFAULT_TIMEPOINTS_PER_TASK,
    xla_cpu_multi_thread_eigen: bool = DEFAULT_XLA_CPU_MULTI_THREAD_EIGEN,
) -> float:
    cfg_path = str(cfg_path)
    data_path = str(data_path)
    outdir_root = Path(outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    _configure_worker_env(
        base_tempdir=base_tempdir,
        cores_per_worker=cores_per_worker,
        xla_cpu_multi_thread_eigen=xla_cpu_multi_thread_eigen,
    )

    with open(cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)

    cfg = deepcopy(cfg)
    cfg.setdefault("nn", {})["seed"] = int(seed)

    dataset = load_npz_dataset(data_path, n_design_params=int(cfg["nn"]["n_design_params"]))
    all_ts = np.linspace(0.0, 1.0, dataset.tpts, dtype=np.float64)

    if "tpts" in cfg.get("nn", {}) and int(cfg["nn"]["tpts"]) != dataset.tpts:
        raise ValueError(f"cfg.nn.tpts={cfg['nn']['tpts']} does not match dataset tpts={dataset.tpts}")

    if max_shots is None:
        max_shots = dataset.n_shots
    if max_timepoints is None:
        max_timepoints = dataset.tpts

    rng = np.random.default_rng(seed)

    process_pool = None
    do_parallel = workers > 1
    if do_parallel:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        # This trainer uses a local spawned process pool for per-timepoint work;
        # Parsl is not part of the active execution path here.
        process_pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_process_pool_initializer,
            initargs=(base_tempdir, cores_per_worker, xla_cpu_multi_thread_eigen),
        )

    eqx, jax, optax, ergoExo, adept_utils, Train_SBSBS_CBET = _import_training_stack()

    # Seed a config for initializing ADEPT state/solver quantities.
    init_shot = 0
    init_time_idx = 0
    init_reflectivity = compute_reflectivity(
        dataset.input_powers[init_shot, :, init_time_idx], dataset.sbs_powers[init_shot, :, init_time_idx]
    )
    init_beam_mask = (dataset.input_powers[init_shot, :, init_time_idx] != 0).astype(np.float32)

    init_cfg = {
        **cfg,
        "nn_inputs": {
            "laser_powers": dataset.input_powers[init_shot],
            "design": dataset.design_inputs[init_shot],
            "t": np.array([all_ts[init_time_idx]], dtype=np.float64),
        },
        "intensities": dataset.input_powers[init_shot, :, init_time_idx],
        "reflectivity": init_reflectivity,
        "beam_mask": init_beam_mask,
    }

    exo = ergoExo()
    all_modules = exo.setup(init_cfg, adept_module=Train_SBSBS_CBET, log=use_mlflow)
    diff_modules, static_modules = {}, {}
    diff_modules["hohlnet"], static_modules["hohlnet"] = eqx.partition(
        all_modules["hohlnet"], all_modules["hohlnet"].get_partition_spec()
    )
    static_modules_bytes = pickle.dumps(static_modules)

    run_dir_name = _build_run_dir_name(mlflow_run_id=exo.mlflow_run_id if use_mlflow else None)
    outdir_path = outdir_root / run_dir_name
    outdir_path.mkdir(parents=True, exist_ok=False)
    (outdir_path / "weights").mkdir(parents=True, exist_ok=False)

    thread_env = _thread_env_snapshot()
    run_metadata = {
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "mlflow_run_id": exo.mlflow_run_id if use_mlflow else None,
        "config_path": cfg_path,
        "data_path": data_path,
        "workers": workers,
        "cores_per_worker": cores_per_worker,
        "timepoints_per_task": timepoints_per_task,
        "xla_cpu_multi_thread_eigen": xla_cpu_multi_thread_eigen,
        "thread_env": thread_env,
    }
    with open(outdir_path / "run_metadata.yaml", "w") as f:
        yaml.safe_dump(run_metadata, f, sort_keys=False)

    lr_sched = optax.cosine_decay_schedule(
        init_value=float(cfg["opt"]["learning_rate"]), decay_steps=int(cfg["opt"]["decay_steps"])
    )
    opt = optax.adam(learning_rate=lr_sched)
    opt_state = opt.init(eqx.filter(diff_modules, eqx.is_array))

    if use_mlflow:
        import mlflow

        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        mlflow.start_run(run_id=exo.mlflow_run_id, log_system_metrics=True)

    logger.info(
        "Training settings: workers=%s cores_per_worker=%s timepoints_per_task=%s max_shots=%s max_timepoints=%s mlflow=%s",
        workers,
        cores_per_worker,
        timepoints_per_task,
        max_shots,
        max_timepoints,
        use_mlflow,
    )
    logger.info(
        "Thread controls: OMP=%s OPENBLAS=%s MKL=%s NUMEXPR=%s TF_INTRA=%s TF_INTER=%s JAX_PLATFORM=%s XLA_FLAGS=%s",
        thread_env["OMP_NUM_THREADS"],
        thread_env["OPENBLAS_NUM_THREADS"],
        thread_env["MKL_NUM_THREADS"],
        thread_env["NUMEXPR_NUM_THREADS"],
        thread_env["TF_NUM_INTRAOP_THREADS"],
        thread_env["TF_NUM_INTEROP_THREADS"],
        thread_env["JAX_PLATFORM_NAME"],
        thread_env["XLA_FLAGS"],
    )
    logger.info("Resolved run output directory: %s", outdir_path)

    num_epochs = int(cfg["opt"]["num_epochs"])
    global_step = 0
    last_epoch_loss = float("nan")

    try:
        for epoch in range(num_epochs):
            shot_indices = np.arange(min(max_shots, dataset.n_shots))
            rng.shuffle(shot_indices)

            epoch_losses = []
            epoch_gradnorms = []
            for shot_idx in shot_indices:
                shot_start = time.perf_counter()
                shot_cfg_base = {
                    **cfg,
                    "nn_inputs": {
                        "laser_powers": dataset.input_powers[shot_idx],
                        "design": dataset.design_inputs[shot_idx],
                        "t": None,  # filled per-timepoint
                    },
                }

                run_cfgs = []
                for t_idx in range(min(max_timepoints, dataset.tpts)):
                    intensities = dataset.input_powers[shot_idx, :, t_idx]
                    reflectivity = compute_reflectivity(intensities, dataset.sbs_powers[shot_idx, :, t_idx])
                    beam_mask = (intensities != 0).astype(np.float32)
                    run_cfgs.append({
                        **shot_cfg_base,
                        "nn_inputs": {**shot_cfg_base["nn_inputs"], "t": np.array([all_ts[t_idx]], dtype=np.float64)},
                        "intensities": intensities,
                        "reflectivity": reflectivity,
                        "beam_mask": beam_mask,
                    })

                futures_or_results = []
                diff_modules_bytes = pickle.dumps(diff_modules) if do_parallel else None
                for start_idx in range(0, len(run_cfgs), timepoints_per_task):
                    run_cfg_batch = run_cfgs[start_idx : start_idx + timepoints_per_task]
                    if do_parallel:
                        futures_or_results.append(
                            process_pool.submit(
                                _run_timepoint_batch_val_and_grad_serialized,
                                run_cfg_batch,
                                diff_modules_bytes,
                                static_modules_bytes,
                            )
                        )
                    else:
                        futures_or_results.append(_run_timepoint_batch_val_and_grad(run_cfg_batch, diff_modules, static_modules))

                if do_parallel:
                    vgs = [pickle.loads(f.result()) for f in futures_or_results]
                else:
                    vgs = futures_or_results

                # Weight task losses by timepoint count for consistency with gradient averaging
                task_losses = np.array([v for v, _, _ in vgs], dtype=float)
                task_weights = np.array([meta["n_timepoints"] for _, _, meta in vgs], dtype=float)
                total_timepoints = int(task_weights.sum())
                batch_loss = float(np.sum(task_losses * task_weights) / total_timepoints)
                avg_grad = adept_utils.all_reduce_gradients([g for _, g, _ in vgs], num=total_timepoints)
                flat_grad, _ = jax.flatten_util.ravel_pytree(avg_grad)
                grad_norm = float(np.linalg.norm(np.asarray(flat_grad)))
                task_elapsed = np.array([meta["elapsed_s"] for _, _, meta in vgs], dtype=float)
                timepoints_per_result = np.array([meta["n_timepoints"] for _, _, meta in vgs], dtype=int)
                shot_wall_s = float(time.perf_counter() - shot_start)

                updates, opt_state = opt.update(avg_grad, opt_state, diff_modules)
                diff_modules = eqx.apply_updates(diff_modules, updates)

                epoch_losses.append(batch_loss)
                epoch_gradnorms.append(grad_norm)

                if use_mlflow and (global_step % log_every == 0):
                    import mlflow

                    mlflow.log_metrics(
                        {
                            "batch_loss": batch_loss,
                            "batch_grad_norm": grad_norm,
                            "batch_wall_s": shot_wall_s,
                            "task_elapsed_mean_s": float(np.mean(task_elapsed)),
                            "task_elapsed_max_s": float(np.max(task_elapsed)),
                        },
                        step=global_step,
                    )

                logger.info(
                    "epoch=%s shot=%s tasks=%s timepoints=%s task_tp_mean=%.2f loss=%.6g grad_norm=%.6g shot_wall_s=%.2f task_mean_s=%.2f task_max_s=%.2f",
                    epoch,
                    shot_idx,
                    len(vgs),
                    len(run_cfgs),
                    float(np.mean(timepoints_per_result)),
                    batch_loss,
                    grad_norm,
                    shot_wall_s,
                    float(np.mean(task_elapsed)),
                    float(np.max(task_elapsed)),
                )

                if global_step % save_every == 0:
                    hohlnet = eqx.combine(diff_modules["hohlnet"], static_modules["hohlnet"])
                    weights_path = outdir_path / "weights" / f"weights-step{global_step:06d}.eqx"
                    hohlnet.save(str(weights_path))
                    with open(outdir_path / "opt_state.pkl", "wb") as f:
                        pickle.dump(opt_state, f)

                    if use_mlflow:
                        import mlflow

                        mlflow.log_artifact(str(weights_path))
                        mlflow.log_artifact(str(outdir_path / "opt_state.pkl"))

                global_step += 1

            last_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            if use_mlflow:
                import mlflow

                mlflow.log_metrics(
                    {"epoch_loss": last_epoch_loss, "epoch_grad_norm": float(np.mean(epoch_gradnorms))}, step=epoch
                )

            logger.info("epoch=%s loss=%s", epoch, last_epoch_loss)

    finally:
        if process_pool is not None:
            process_pool.shutdown(wait=True, cancel_futures=True)
        if use_mlflow:
            import mlflow

            mlflow.end_run()

    return last_epoch_loss


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train SBSBS+CBET HohlNet using an NPZ dataset.")
    parser.set_defaults(mlflow=DEFAULT_USE_MLFLOW)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for deterministic shuffling")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Spawned worker processes for timepoint batches (default 32 to reduce concurrent JAX/XLA compile pressure)",
    )
    parser.add_argument(
        "--cores-per-worker",
        type=int,
        default=DEFAULT_CORES_PER_WORKER,
        help="Thread limit per worker process",
    )
    parser.add_argument("--mlflow", dest="mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument("--no-mlflow", dest="mlflow", action="store_false", help="Disable MLflow logging")
    parser.add_argument("--log-every", type=int, default=10, help="Log batch metrics every N steps (when --mlflow)")
    parser.add_argument("--save-every", type=int, default=50, help="Save weights every N steps")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Root directory for per-run output folders")
    parser.add_argument(
        "--base-tempdir",
        type=str,
        default=DEFAULT_BASE_TEMPDIR,
        help="BASE_TEMPDIR for ADEPT temp files (defaults to /tmp/kur1)",
    )

    parser.add_argument("--max-shots", type=int, default=None, help="Debug: limit shots per epoch")
    parser.add_argument("--max-timepoints", type=int, default=None, help="Debug: limit timepoints per shot")
    parser.add_argument(
        "--timepoints-per-task",
        type=int,
        default=DEFAULT_TIMEPOINTS_PER_TASK,
        help="Number of timepoints evaluated serially inside each worker task (default 4 for safer compile pressure)",
    )
    parser.add_argument(
        "--xla-cpu-multi-thread-eigen",
        dest="xla_cpu_multi_thread_eigen",
        action="store_true",
        help="Enable XLA CPU Eigen multithreading inside each worker",
    )
    parser.add_argument(
        "--no-xla-cpu-multi-thread-eigen",
        dest="xla_cpu_multi_thread_eigen",
        action="store_false",
        help="Disable XLA CPU Eigen multithreading inside each worker",
    )
    parser.set_defaults(xla_cpu_multi_thread_eigen=DEFAULT_XLA_CPU_MULTI_THREAD_EIGEN)

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("Using config: %s", DEFAULT_CONFIG_PATH)
    logger.info("Using dataset: %s", DEFAULT_DATA_PATH)
    logger.info("Writing outputs to: %s", args.outdir)
    train_model(
        outdir=args.outdir,
        seed=args.seed,
        workers=args.workers,
        cores_per_worker=args.cores_per_worker,
        base_tempdir=args.base_tempdir,
        use_mlflow=args.mlflow,
        log_every=args.log_every,
        save_every=args.save_every,
        max_shots=args.max_shots,
        max_timepoints=args.max_timepoints,
        timepoints_per_task=args.timepoints_per_task,
        xla_cpu_multi_thread_eigen=args.xla_cpu_multi_thread_eigen,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
