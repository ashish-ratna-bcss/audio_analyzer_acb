import pathlib, yaml


def test_compose_has_seven_services():
    root = pathlib.Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "docker-compose.yml").read_text())
    services = set(data["services"])
    assert services == {
        "nginx", "api", "redis", "postgres",
        "worker-cpu", "worker-gpu", "flower",
    }


def test_worker_gpu_has_gpu_reservation():
    root = pathlib.Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "docker-compose.yml").read_text())
    devs = data["services"]["worker-gpu"]["deploy"]["resources"]["reservations"]["devices"]
    assert any(d.get("driver") == "nvidia" for d in devs)


def test_api_has_no_gpu_reservation():
    root = pathlib.Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "docker-compose.yml").read_text())
    assert "deploy" not in data["services"]["api"]
