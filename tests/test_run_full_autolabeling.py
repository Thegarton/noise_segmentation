import importlib.util
from pathlib import Path


def test_actor_heuristic_is_disabled_by_default_when_openpcdet_predictions_are_used():
    module = _load_run_full_autolabeling()

    assert module.should_use_actor_heuristic(openpcdet_predictions="predictions.jsonl", use_actor_heuristic=False) is False
    assert module.should_use_actor_heuristic(openpcdet_predictions="predictions.jsonl", use_actor_heuristic=True) is True
    assert module.should_use_actor_heuristic(openpcdet_predictions=None, use_actor_heuristic=False) is True


def _load_run_full_autolabeling():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_full_autolabeling.py"
    spec = importlib.util.spec_from_file_location("run_full_autolabeling", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
