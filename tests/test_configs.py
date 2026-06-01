from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_configs_readable():
    for name in ["dataset.yaml", "experiments.yaml", "models.yaml"]:
        data = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert isinstance(data, dict)


def test_configs_do_not_contain_absolute_paths():
    for path in (ROOT / "configs").glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "/mnt/" not in text
        assert "C:\\" not in text
        assert "/home/" not in text
