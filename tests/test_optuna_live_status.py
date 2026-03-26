"""sway.optuna_live_status JSON payload."""

import optuna

from sway.optuna_live_status import build_study_status_payload, write_live_sweep_status


def test_build_study_status_payload(tmp_path) -> None:
    study = optuna.create_study(direction="maximize")

    def obj(t: optuna.Trial) -> float:
        t.set_user_attr("u", 1)
        return t.suggest_float("x", 0, 1)

    study.optimize(obj, n_trials=2)
    p = build_study_status_payload(study, extra={"k": "v"})
    assert p["schema"] == "sway_optuna_sweep_status_v1"
    assert p["n_trials_total"] == 2
    assert p["n_complete"] == 2
    assert p["best"] is not None
    assert p["best"]["number"] in (0, 1)
    assert p["meta"]["k"] == "v"
    assert len(p["trials"]) == 2


def test_write_live_sweep_status_atomic(tmp_path) -> None:
    study = optuna.create_study()
    study.optimize(lambda t: 0.0, n_trials=1)
    path = tmp_path / "s.json"
    write_live_sweep_status(study, path, extra={})
    assert path.is_file()
    assert not (tmp_path / "s.json.tmp").exists()
