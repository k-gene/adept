import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import dill as pickle
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from adept import ergoExo
from adept._sbsbs1d.base import Train_SBSBS_CBET
from adept.sbsbs_train import (
    DEFAULT_BASE_TEMPDIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTDIR,
    DEFAULT_XLA_CPU_MULTI_THREAD_EIGEN,
    _configure_worker_env,
    _thread_env_snapshot,
    compute_reflectivity,
    load_npz_dataset,
)
from adept.sbsbs_training_utils import masked_mse

logger = logging.getLogger(__name__)

DEFAULT_DEBUG_OUTDIR = Path(DEFAULT_OUTDIR) / "debug_eval"


def _spawn_case_seed(seed_rng, *, shot_idx, time_idx):
    return int(seed_rng.integers(0, np.iinfo(np.int32).max, endpoint=True))


def _resolve_default_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE") or os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(1, cpu_count * 3 // 4)


def _to_numpy(value):
    return np.asarray(jax.device_get(value), dtype=np.float64)


def _array_summary(value):
    arr = _to_numpy(value)
    finite_mask = np.isfinite(arr)
    summary = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite_count": int(finite_mask.sum()),
        "nonfinite_count": int(arr.size - finite_mask.sum()),
        "nan_count": int(np.isnan(arr).sum()),
        "posinf_count": int(np.isposinf(arr).sum()),
        "neginf_count": int(np.isneginf(arr).sum()),
    }
    if finite_mask.any():
        finite_vals = arr[finite_mask]
        summary["min"] = float(finite_vals.min())
        summary["max"] = float(finite_vals.max())
        summary["mean"] = float(finite_vals.mean())
    else:
        summary["min"] = None
        summary["max"] = None
        summary["mean"] = None
    return summary


def _summarize_mapping(mapping):
    return {key: _array_summary(val) for key, val in mapping.items()}


def _dump_mapping_npz(path: Path, mapping):
    np.savez(path, **{key: _to_numpy(val) for key, val in mapping.items()})


def _build_single_run_cfg(cfg, dataset, all_ts, shot_idx, time_idx):
    shot_cfg_base = {
        **deepcopy(cfg),
        "nn_inputs": {
            "laser_powers": dataset.input_powers[shot_idx],
            "design": dataset.design_inputs[shot_idx],
            "t": None,
        },
    }
    intensities = dataset.input_powers[shot_idx, :, time_idx]
    reflectivity = compute_reflectivity(intensities, dataset.sbs_powers[shot_idx, :, time_idx])
    beam_mask = (intensities != 0).astype(np.float32)
    return {
        **shot_cfg_base,
        "nn_inputs": {**shot_cfg_base["nn_inputs"], "t": np.array([all_ts[time_idx]], dtype=np.float64)},
        "intensities": intensities,
        "reflectivity": reflectivity,
        "beam_mask": beam_mask,
    }


def _build_init_cfg(cfg, dataset, all_ts):
    init_shot = 0
    init_time_idx = 0
    init_reflectivity = compute_reflectivity(
        dataset.input_powers[init_shot, :, init_time_idx], dataset.sbs_powers[init_shot, :, init_time_idx]
    )
    init_beam_mask = (dataset.input_powers[init_shot, :, init_time_idx] != 0).astype(np.float32)
    return {
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


def _prepare_modules(run_cfg, cfg, dataset, all_ts, weights_path):
    init_cfg = _build_init_cfg(cfg, dataset, all_ts)
    exo = ergoExo()
    all_modules = exo.setup(init_cfg, adept_module=Train_SBSBS_CBET, log=False)
    if weights_path is not None:
        all_modules["hohlnet"] = eqx.tree_deserialise_leaves(str(weights_path), all_modules["hohlnet"])
    diff_modules, static_modules = {}, {}
    diff_modules["hohlnet"], static_modules["hohlnet"] = eqx.partition(
        all_modules["hohlnet"], all_modules["hohlnet"].get_partition_spec()
    )

    exo = ergoExo()
    exo.setup(run_cfg, adept_module=Train_SBSBS_CBET, log=False)
    return exo, {"hohlnet": diff_modules["hohlnet"], "static_modules": static_modules}


def _collect_nn_outputs(modules, run_cfg, exo_module):
    hohlnet = eqx.combine(modules["hohlnet"], modules["static_modules"]["hohlnet"])
    nn_inputs = run_cfg["nn_inputs"]
    laser_powers = nn_inputs["laser_powers"]
    design = nn_inputs["design"]
    eval_time = nn_inputs["t"]

    leh_outputs = hohlnet.leh_eval(laser_powers, design, eval_time)

    beam_profiles = {}
    thermal_noise = {}
    z_grids = {}
    for beam_idx in range(run_cfg["laser"]["num_beams"]):
        beam_key = f"beam_{beam_idx}"
        z_grid = np.asarray(exo_module.cfg["beam"][str(beam_idx)]["spec"]["z"], dtype=np.float64)
        z_norm = z_grid / float(exo_module.cfg["beam"][str(beam_idx)]["spec"]["zmax"])
        z_grids[beam_key] = z_grid
        beam_profiles[beam_key] = {
            key: jax.vmap(
                lambda z_val, idx=beam_idx: hohlnet.hohl_eval(
                    laser_powers,
                    design,
                    jnp.array([idx]),
                    jnp.array([z_val]),
                    eval_time,
                )[key]
            )(jnp.asarray(z_norm))
            for key in hohlnet.hohl_outputs
        }
        thermal_noise[beam_key] = hohlnet.sbs_source_eval(
            laser_powers,
            design,
            jnp.array([beam_idx]),
            eval_time,
        )
    return leh_outputs, beam_profiles, thermal_noise, z_grids


def _summarize_solution(solution):
    ys = solution.ys
    if isinstance(ys, dict):
        return {key: _array_summary(val) for key, val in ys.items()}
    return {"ys": _array_summary(ys)}


def _first_nonfinite_stage(leh_summary, beam_summaries, cbet_summary, sbs_summaries, loss_summary):
    if any(item["nonfinite_count"] > 0 for item in leh_summary.values()):
        return "leh_eval"
    for beam_key, profile_summary in beam_summaries.items():
        if any(item["nonfinite_count"] > 0 for item in profile_summary["hohl_eval"].values()):
            return f"{beam_key}.hohl_eval"
        if any(item["nonfinite_count"] > 0 for item in profile_summary["sbs_source_eval"].values()):
            return f"{beam_key}.sbs_source_eval"
    if any(item["nonfinite_count"] > 0 for item in cbet_summary.values()):
        return "perform_cbet"
    for beam_key, summary in sbs_summaries.items():
        if any(item["nonfinite_count"] > 0 for item in summary.values()):
            return f"{beam_key}.perform_sbs"
    if loss_summary["prediction_summary"]["nonfinite_count"] > 0 or not np.isfinite(loss_summary["loss"]):
        return "loss_construction"
    return "all_finite"


def _gradient_tree_nonfinite_report(grad_tree):
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(grad_tree)
    report = []
    for path, leaf in leaves_with_path:
        summary = _array_summary(leaf)
        if summary["nonfinite_count"] > 0:
            report.append(
                {
                    "path": jax.tree_util.keystr(path),
                    "summary": summary,
                }
            )
    return report


def _run_one_case_from_payload(payload):
    return _run_one_case(**payload)


def _process_pool_initializer(base_tempdir, cores_per_worker, xla_cpu_multi_thread_eigen):
    _configure_worker_env(
        base_tempdir=base_tempdir,
        cores_per_worker=cores_per_worker,
        xla_cpu_multi_thread_eigen=xla_cpu_multi_thread_eigen,
    )



def _run_one_case(
    cfg,
    dataset,
    all_ts,
    *,
    shot_idx,
    time_idx,
    weights_path,
    outdir,
):
    run_cfg = _build_single_run_cfg(cfg, dataset, all_ts, shot_idx, time_idx)
    exo, module_bundle = _prepare_modules(run_cfg, cfg, dataset, all_ts, weights_path)
    exo_module = exo.adept_module

    case_name = f"shot{shot_idx:04d}_time{time_idx:04d}"
    case_dir = outdir / case_name
    if case_dir.exists():
        raise RuntimeError(
            f"Case directory {case_dir} already exists. "
            "This indicates duplicate (shot_idx, time_idx) in the request or a previous incomplete run. "
            "Remove the directory manually if rerun is intended."
        )
    case_dir.mkdir(parents=True, exist_ok=False)

    leh_outputs, beam_profiles, thermal_noise_outputs, z_grids = _collect_nn_outputs(module_bundle, run_cfg, exo_module)
    run_args = {
        "static_modules": module_bundle["static_modules"],
        "reflectivity": run_cfg["reflectivity"],
        "beam_mask": run_cfg["beam_mask"],
    }
    trainable_modules = {"hohlnet": module_bundle["hohlnet"]}
    out_dict = exo_module(trainable_modules, run_args)[1]
    val, grad, _ = exo.val_and_grad(
        trainable_modules,
        args=run_args,
        export=False,
        log=False,
    )

    cbet_summary = _summarize_solution(out_dict["cbet result"])
    sbs_summaries = {
        f"beam_{beam_idx}": _summarize_solution(solution)
        for beam_idx, solution in enumerate(out_dict["sbs results"])
    }
    nn_summaries = {
        "leh_eval": _summarize_mapping(leh_outputs),
        "per_beam": {
            beam_key: {
                "hohl_eval": _summarize_mapping(beam_profiles[beam_key]),
                "sbs_source_eval": _summarize_mapping(thermal_noise_outputs[beam_key]),
            }
            for beam_key in beam_profiles
        },
    }

    reflectivity_pred = np.asarray(
        jax.device_get(jnp.stack([solution.ys["Jr"][0] for solution in out_dict["sbs results"]])),
        dtype=np.float64,
    )
    loss_summary = {
        "loss": float(val),
        "target_summary": _array_summary(run_cfg["reflectivity"]),
        "prediction_summary": _array_summary(reflectivity_pred),
        "beam_mask_summary": _array_summary(run_cfg["beam_mask"]),
        "masked_mse": float(masked_mse(reflectivity_pred, run_cfg["reflectivity"], run_cfg["beam_mask"])),
    }
    grad_flat, _ = jax.flatten_util.ravel_pytree(grad)
    grad_summary = _array_summary(grad_flat)
    grad_nonfinite_paths = _gradient_tree_nonfinite_report(grad)
    first_nonfinite_stage = _first_nonfinite_stage(
        nn_summaries["leh_eval"],
        nn_summaries["per_beam"],
        cbet_summary,
        sbs_summaries,
        loss_summary,
    )
    if first_nonfinite_stage == "all_finite" and grad_summary["nonfinite_count"] > 0:
        first_nonfinite_stage = "gradient_backprop"

    metadata = {
        "shot_idx": int(shot_idx),
        "time_idx": int(time_idx),
        "time_value": float(all_ts[time_idx]),
        "nn_seed": int(run_cfg["nn"]["seed"]),
        "intensities": run_cfg["intensities"].astype(np.float64).tolist(),
        "reflectivity_target": run_cfg["reflectivity"].astype(np.float64).tolist(),
        "beam_mask": run_cfg["beam_mask"].astype(np.float64).tolist(),
        "first_nonfinite_stage": first_nonfinite_stage,
    }

    _dump_mapping_npz(case_dir / "leh_eval_outputs.npz", leh_outputs)
    np.savez(
        case_dir / "reflectivity_compare.npz",
        target=run_cfg["reflectivity"],
        prediction=reflectivity_pred,
        mask=run_cfg["beam_mask"],
    )
    np.savez(case_dir / "grad_summary.npz", grad_flat=_to_numpy(grad_flat))
    for beam_key, z_grid in z_grids.items():
        np.savez(
            case_dir / f"{beam_key}_hohl_eval_outputs.npz",
            z_um=np.asarray(z_grid, dtype=np.float64),
            **{key: _to_numpy(val) for key, val in beam_profiles[beam_key].items()},
        )
        _dump_mapping_npz(case_dir / f"{beam_key}_sbs_source_outputs.npz", thermal_noise_outputs[beam_key])

    report = {
        "metadata": metadata,
        "nn_summaries": nn_summaries,
        "cbet_summary": cbet_summary,
        "sbs_summaries": sbs_summaries,
        "loss_summary": loss_summary,
        "grad_summary": grad_summary,
        "grad_nonfinite_paths": grad_nonfinite_paths,
    }
    with open(case_dir / "report.yaml", "w") as f:
        yaml.safe_dump(report, f, sort_keys=False)
    with open(case_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(case_dir / "run_cfg_snapshot.pkl", "wb") as f:
        pickle.dump(run_cfg, f)

    logger.info("Wrote SBSBS debug evaluation artifacts to %s", case_dir)
    logger.info("Case %s first nonfinite stage: %s", case_name, first_nonfinite_stage)
    return {
        "case_name": case_name,
        "case_dir": case_dir,
        "nn_seed": int(run_cfg["nn"]["seed"]),
        "first_nonfinite_stage": first_nonfinite_stage,
        "loss": float(val),
        "loss_is_finite": bool(np.isfinite(float(val))),
        "prediction_nonfinite_count": int(loss_summary["prediction_summary"]["nonfinite_count"]),
        "grad_nonfinite_count": int(grad_summary["nonfinite_count"]),
        "grad_nonfinite_paths": grad_nonfinite_paths,
    }


def run_debug_eval(
    cfg_path=DEFAULT_CONFIG_PATH,
    data_path=DEFAULT_DATA_PATH,
    outdir=DEFAULT_DEBUG_OUTDIR,
    *,
    shot_idx=0,
    time_idx=0,
    shot_indices=None,
    time_indices=None,
    max_cases=None,
    weights_path=None,
    seed=0,
    base_tempdir=DEFAULT_BASE_TEMPDIR,
    workers=None,
    cores_per_worker=1,
    xla_cpu_multi_thread_eigen=DEFAULT_XLA_CPU_MULTI_THREAD_EIGEN,
):
    cfg_path = Path(cfg_path).resolve()
    data_path = Path(data_path).resolve()
    outdir = Path(outdir).resolve()
    weights_path = None if weights_path is None else Path(weights_path).resolve()

    if workers is None:
        workers = _resolve_default_workers()
    workers = max(1, int(workers))
    cores_per_worker = max(1, int(cores_per_worker))

    _configure_worker_env(
        base_tempdir=base_tempdir,
        cores_per_worker=cores_per_worker,
        xla_cpu_multi_thread_eigen=xla_cpu_multi_thread_eigen,
    )

    with open(cfg_path, "r") as fi:
        cfg = yaml.safe_load(fi)
    cfg = deepcopy(cfg)
    global_seed = int(seed)
    cfg.setdefault("nn", {})["seed"] = global_seed
    seed_rng = np.random.default_rng(global_seed)

    dataset = load_npz_dataset(str(data_path), n_design_params=int(cfg["nn"]["n_design_params"]))
    all_ts = np.linspace(0.0, 1.0, dataset.tpts, dtype=np.float64)

    def _normalize_indices(indices, upper_bound, label):
        if indices is None:
            return None
        normalized = sorted({int(idx) for idx in indices})
        for idx in normalized:
            if not (0 <= idx < upper_bound):
                raise ValueError(f"{label}={idx} out of bounds for upper_bound={upper_bound}")
        return normalized

    shot_indices = _normalize_indices(shot_indices, dataset.n_shots, "shot_idx")
    time_indices = _normalize_indices(time_indices, dataset.tpts, "time_idx")

    if shot_indices is None:
        if shot_idx is None:
            shot_indices = [0]
        else:
            if not (0 <= shot_idx < dataset.n_shots):
                raise ValueError(f"shot_idx={shot_idx} out of bounds for dataset.n_shots={dataset.n_shots}")
            shot_indices = [int(shot_idx)]
    if time_indices is None:
        if time_idx is None:
            time_indices = [0]
        else:
            if not (0 <= time_idx < dataset.tpts):
                raise ValueError(f"time_idx={time_idx} out of bounds for dataset.tpts={dataset.tpts}")
            time_indices = [int(time_idx)]

    case_indices = [(s_idx, t_idx) for s_idx in shot_indices for t_idx in time_indices]
    if max_cases is not None:
        case_indices = case_indices[: max(1, int(max_cases))]

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = outdir / f"debug_eval_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    effective_parallelism = min(len(case_indices), workers)
    logger.info(
        "Running SBSBS debug eval for %d case(s) with workers=%d (effective parallel cases=%d) and cores_per_worker=%d",
        len(case_indices),
        workers,
        effective_parallelism,
        cores_per_worker,
    )

    case_payloads = []
    for this_shot_idx, this_time_idx in case_indices:
        case_cfg = deepcopy(cfg)
        case_cfg["nn"]["seed"] = _spawn_case_seed(seed_rng, shot_idx=this_shot_idx, time_idx=this_time_idx)
        case_payloads.append(
            {
                "cfg": case_cfg,
                "dataset": dataset,
                "all_ts": all_ts,
                "shot_idx": this_shot_idx,
                "time_idx": this_time_idx,
                "weights_path": weights_path,
                "outdir": run_dir,
            }
        )

    if workers > 1 and len(case_payloads) > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_process_pool_initializer,
            initargs=(base_tempdir, cores_per_worker, xla_cpu_multi_thread_eigen),
        ) as process_pool:
            futures = [process_pool.submit(_run_one_case_from_payload, payload) for payload in case_payloads]
            case_summaries = [future.result() for future in futures]
    else:
        case_summaries = []
        for case_number, payload in enumerate(case_payloads, start=1):
            logger.info(
                "Running case %d/%d: shot_idx=%d time_idx=%d",
                case_number,
                len(case_payloads),
                payload["shot_idx"],
                payload["time_idx"],
            )
            case_summaries.append(_run_one_case_from_payload(payload))

    manifest = {
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "config_path": str(cfg_path),
        "data_path": str(data_path),
        "weights_path": None if weights_path is None else str(weights_path),
        "seed": int(seed),
        "thread_env": _thread_env_snapshot(),
        "requested_workers": workers,
        "cores_per_worker": cores_per_worker,
        "case_count": len(case_summaries),
        "cases": [
            {
                **{k: v for k, v in case.items() if k != "case_dir"},
                "case_dir": str(case["case_dir"]),
            }
            for case in case_summaries
        ],
    }
    with open(run_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote SBSBS debug evaluation artifacts to %s", run_dir)
    return run_dir


def _parse_index_list(raw_value):
    if raw_value is None:
        return None
    values = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values



def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate one or more SBSBS shot/timepoint cases and dump NaN-debug artifacts."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Absolute path to YAML config")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Absolute path to NPZ dataset")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_DEBUG_OUTDIR), help="Root output directory")
    parser.add_argument("--shot-idx", type=int, default=0, help="Single dataset shot index")
    parser.add_argument("--time-idx", type=int, default=0, help="Single dataset time index")
    parser.add_argument(
        "--shot-indices",
        type=str,
        default=None,
        help="Comma-separated shot indices for a debug sweep; overrides --shot-idx when provided",
    )
    parser.add_argument(
        "--time-indices",
        type=str,
        default=None,
        help="Comma-separated time indices for a debug sweep; overrides --time-idx when provided",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on the number of shot/timepoint combinations to evaluate",
    )
    parser.add_argument("--weights", type=str, default=None, help="Optional absolute path to .eqx weights")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic HohlNet initialization")
    parser.add_argument(
        "--base-tempdir",
        type=str,
        default=str(DEFAULT_BASE_TEMPDIR),
        help="BASE_TEMPDIR override (defaults to /tmp/kur1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Node-level worker budget for the sweep manifest; defaults to ~3/4 of visible CPUs or Slurm allocation",
    )
    parser.add_argument("--cores-per-worker", type=int, default=1, help="Thread limit per debug worker (1 recommended)")
    parser.add_argument(
        "--xla-cpu-multi-thread-eigen",
        dest="xla_cpu_multi_thread_eigen",
        action="store_true",
        help="Enable XLA CPU Eigen multithreading",
    )
    parser.add_argument(
        "--no-xla-cpu-multi-thread-eigen",
        dest="xla_cpu_multi_thread_eigen",
        action="store_false",
        help="Disable XLA CPU Eigen multithreading",
    )
    parser.set_defaults(xla_cpu_multi_thread_eigen=DEFAULT_XLA_CPU_MULTI_THREAD_EIGEN)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_dir = run_debug_eval(
        cfg_path=args.config,
        data_path=args.data,
        outdir=args.outdir,
        shot_idx=args.shot_idx,
        time_idx=args.time_idx,
        shot_indices=_parse_index_list(args.shot_indices),
        time_indices=_parse_index_list(args.time_indices),
        max_cases=args.max_cases,
        weights_path=args.weights,
        seed=args.seed,
        base_tempdir=args.base_tempdir,
        workers=args.workers,
        cores_per_worker=args.cores_per_worker,
        xla_cpu_multi_thread_eigen=args.xla_cpu_multi_thread_eigen,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
