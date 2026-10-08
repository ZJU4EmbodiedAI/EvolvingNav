"""The published code keeps its runtime helpers inside its own package."""

import importlib.util
from pathlib import Path


def test_habitat_helpers_are_packaged() -> None:
    assert importlib.util.find_spec("evolvingnav.habitat_utils") is not None


def test_perception_does_not_import_the_original_readyagent_stack() -> None:
    source = (Path(__file__).resolve().parents[1] / "evolvingnav/perception.py").read_text()
    assert "readyagent.artifacts" not in source
    assert "readyagent.perception" not in source


def test_navigation_cli_accepts_explicit_asset_paths() -> None:
    from evolvingnav import run

    assert hasattr(run, "arguments")
    args = run.arguments([
        "--dataset", "/data/belief", "--tasks", "/data/tasks",
        "--hssd-root", "/data/hssd", "--navmesh-root", "/data/navmeshes",
        "--checkpoint", "/models/p4d.pt", "--output", "/tmp/p4d-run",
    ])
    assert args.navmesh_root == Path("/data/navmeshes")
    assert args.inspection == "grounded-sam"


def test_visual_verifier_accepts_explicit_asset_paths() -> None:
    from evolvingnav import verify_visual

    assert hasattr(verify_visual, "arguments")
    args = verify_visual.arguments([
        "/tmp/p4d-run", "--tasks", "/data/tasks",
        "--hssd-root", "/data/hssd", "--navmesh-root", "/data/navmeshes",
    ])
    assert args.navmesh_root == Path("/data/navmeshes")


def test_unit_tests_do_not_require_the_local_benchmark_dataset() -> None:
    source = (Path(__file__).resolve().parent / "test_agent_code.py").read_text()
    assert "data/task_datasets" not in source


def test_record_packer_requires_an_explicit_dataset_root(monkeypatch) -> None:
    import sys

    import pytest
    from pack_p4d_hssd_records import arguments

    monkeypatch.setattr(sys, "argv", ["pack_p4d_hssd_records.py"])
    with pytest.raises(SystemExit) as error:
        arguments()
    assert error.value.code == 2
