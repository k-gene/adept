import yaml

from adept import ergoExo


def test_reuse_config_dict():
    with open("tests/test_base/configs/example.yaml") as file:
        cfg = yaml.safe_load(file)

    exo = ergoExo()
    exo.setup(cfg)

    exo = ergoExo()
    exo.setup(cfg)


def test_setup_without_mlflow_logging():
    with open("tests/test_base/configs/example.yaml") as file:
        cfg = yaml.safe_load(file)

    exo = ergoExo()
    modules = exo.setup(cfg, log=False)
    assert isinstance(modules, dict)
    assert exo.mlflow_run_id is None
