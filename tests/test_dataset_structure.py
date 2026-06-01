from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_directories_exist():
    for relative in [
        "data/raw",
        "data/interim",
        "data/processed",
        "configs",
        "reports/tables",
        "reports/figures",
        "reports/metrics",
        "reports/paper",
        "models_trained",
        "models_exported",
    ]:
        assert (ROOT / relative).exists()


def test_dataset_class_mapping_is_stable_when_processed_exists():
    train_dir = ROOT / "data" / "processed" / "train"
    if not train_dir.exists() or not (train_dir / "ok").exists() or not (train_dir / "defective").exists():
        return
    from edge_casting.training.evaluate import CastingImageFolder

    dataset = CastingImageFolder(str(train_dir))
    assert dataset.class_to_idx == {"ok": 0, "defective": 1}
