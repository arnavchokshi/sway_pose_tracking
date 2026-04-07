from __future__ import annotations

import types
import sys
from pathlib import Path

from sway.detr_detector import DETRDetector


def _bare_detector(model_name: str = "co_dino_swinl") -> DETRDetector:
    det = DETRDetector.__new__(DETRDetector)
    det.model_name = model_name
    det.device = "cpu"
    det.conf_threshold = 0.3
    det.detect_size = 800
    det._model = None
    det._is_ultralytics = False
    return det


def test_resolve_codetr_config_from_sway_codetr_repo(monkeypatch, tmp_path: Path) -> None:
    cfg = (
        tmp_path
        / "projects"
        / "CO-DETR"
        / "configs"
        / "codino"
        / "co_dino_5scale_swin_l_16xb1_16e_o365tococo.py"
    )
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("# test cfg\n", encoding="utf-8")

    monkeypatch.setenv("SWAY_CODETR_REPO", str(tmp_path))
    monkeypatch.delenv("SWAY_MMDET_CODINO_CONFIG", raising=False)
    monkeypatch.delenv("SWAY_MMDET_CODETR_CONFIG", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    det = _bare_detector("co_dino_swinl")
    got = det._resolve_codetr_config_path()
    assert got == cfg.resolve()


def test_resolve_codetr_config_from_pythonpath(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "mmdetection_like"
    cfg = repo / "projects" / "configs" / "co_dino" / "co_dino_5scale_swin_large_1x_coco.py"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("# test cfg\n", encoding="utf-8")

    monkeypatch.delenv("SWAY_CODETR_REPO", raising=False)
    monkeypatch.delenv("SWAY_MMDET_CODINO_CONFIG", raising=False)
    monkeypatch.delenv("SWAY_MMDET_CODETR_CONFIG", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(repo))

    det = _bare_detector("co_dino_swinl")
    got = det._resolve_codetr_config_path()
    assert got == cfg.resolve()


def test_codetr_reuses_codino_weight_when_codetr_missing(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "co_dino_cfg.py"
    cfg.write_text("# cfg\n", encoding="utf-8")

    captured: dict[str, str] = {}

    def fake_init_detector(config_path: str, checkpoint_path: str, device: str = "cpu"):
        captured["config"] = config_path
        captured["checkpoint"] = checkpoint_path
        captured["device"] = device
        return {"ok": True}

    fake_mmdet = types.ModuleType("mmdet")
    fake_apis = types.ModuleType("mmdet.apis")
    fake_apis.init_detector = fake_init_detector  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mmdet", fake_mmdet)
    monkeypatch.setitem(sys.modules, "mmdet.apis", fake_apis)

    det = _bare_detector("co_detr_swinl")
    monkeypatch.setattr(det, "_resolve_codetr_config_path", lambda: cfg.resolve())
    det._load_codetr()

    assert det._model == {"ok": True}
    assert captured["checkpoint"].endswith("co_dino_swinl_coco.pth")
    assert captured["config"] == str(cfg.resolve())
