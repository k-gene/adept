from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

from adept import sbsbs_train


def _worker_thread_env_snapshot():
    return sbsbs_train._thread_env_snapshot()


def test_compute_reflectivity_handles_zero_input_power():
    input_powers = np.array([0.0, 2.0, 4.0])
    sbs_powers = np.array([1.0, 1.0, 2.0])
    r = sbsbs_train.compute_reflectivity(input_powers, sbs_powers)
    assert np.allclose(r, [0.0, 0.5, 0.5])


def test_xla_flags_for_worker_respects_toggle():
    assert sbsbs_train._xla_flags_for_worker(cores_per_worker=4, xla_cpu_multi_thread_eigen=False) == (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
    )
    assert sbsbs_train._xla_flags_for_worker(cores_per_worker=4, xla_cpu_multi_thread_eigen=True) == (
        "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
    )


def test_configure_worker_env_sets_expected_thread_controls(monkeypatch):
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "XLA_FLAGS",
        "JAX_PLATFORM_NAME",
        "BASE_TEMPDIR",
    ):
        monkeypatch.delenv(key, raising=False)

    sbsbs_train._configure_worker_env(
        base_tempdir="/tmp/kur1/sbsbs-test", cores_per_worker=4, xla_cpu_multi_thread_eigen=False
    )
    snapshot = sbsbs_train._thread_env_snapshot()

    assert snapshot["OMP_NUM_THREADS"] == "4"
    assert snapshot["OPENBLAS_NUM_THREADS"] == "4"
    assert snapshot["MKL_NUM_THREADS"] == "4"
    assert snapshot["NUMEXPR_NUM_THREADS"] == "4"
    assert snapshot["TF_NUM_INTRAOP_THREADS"] == "4"
    assert snapshot["TF_NUM_INTEROP_THREADS"] == "1"
    assert snapshot["JAX_PLATFORM_NAME"] == "cpu"
    assert snapshot["BASE_TEMPDIR"] == "/tmp/kur1/sbsbs-test"
    assert snapshot["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"


def test_process_pool_initializer_propagates_thread_controls():
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
        initializer=sbsbs_train._process_pool_initializer,
        initargs=("/tmp/kur1/sbsbs-spawn", 3, True),
    ) as executor:
        snapshot = executor.submit(_worker_thread_env_snapshot).result(timeout=30)

    assert snapshot["OMP_NUM_THREADS"] == "3"
    assert snapshot["OPENBLAS_NUM_THREADS"] == "3"
    assert snapshot["MKL_NUM_THREADS"] == "3"
    assert snapshot["NUMEXPR_NUM_THREADS"] == "3"
    assert snapshot["TF_NUM_INTRAOP_THREADS"] == "3"
    assert snapshot["TF_NUM_INTEROP_THREADS"] == "1"
    assert snapshot["JAX_PLATFORM_NAME"] == "cpu"
    assert snapshot["BASE_TEMPDIR"] == "/tmp/kur1/sbsbs-spawn"
    assert snapshot["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=3"


def test_default_base_tempdir_uses_tmp_kur1():
    assert sbsbs_train.DEFAULT_BASE_TEMPDIR == "/tmp/kur1"


def test_main_passes_outdir_to_train_model(monkeypatch):
    captured = {}

    def fake_train_model(**kwargs):
        captured.update(kwargs)
        return 0.0

    monkeypatch.setattr(sbsbs_train, "train_model", fake_train_model)

    exit_code = sbsbs_train.main(["--outdir", "/tmp/kur1/sbsbs-cli-out", "--no-mlflow", "--workers", "1"])

    assert exit_code == 0
    assert captured["outdir"] == "/tmp/kur1/sbsbs-cli-out"
    assert captured["workers"] == 1
    assert captured["use_mlflow"] is False
    assert captured["base_tempdir"] == "/tmp/kur1"
