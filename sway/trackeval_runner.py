"""
Run JonathonLuiten/TrackEval on a single sequence with custom GT/pred folders.
"""

from __future__ import annotations

import configparser
import tempfile

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from sway.mot_format import (
    data_json_to_mot_lines,
    load_mot_lines_from_file,
    mot_lines_to_seq_info,
    write_mot_file,
)


def _write_seqinfo(ini_path: Path, seq_length: int, width: int, height: int) -> None:
    ini_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = configparser.ConfigParser()
    cfg["Sequence"] = {
        "name": ini_path.parent.name,
        "imDir": "img1",
        "frameRate": "30",
        "seqLength": str(seq_length),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": ".jpg",
    }
    with open(ini_path, "w") as f:
        cfg.write(f)


def run_trackeval_single_sequence(
    gt_mot_lines: List[str],
    pred_mot_lines: List[str],
    sequence_name: str = "swayseq",
    im_width: int = 1920,
    im_height: int = 1080,
) -> Dict[str, Any]:
    """
    Evaluate one sequence. Returns flat dict of key metrics (IDF1, IDSW, HOTA, etc.).
    """
    try:
        from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
        from trackeval.eval import Evaluator
        from trackeval.metrics.clear import CLEAR
        from trackeval.metrics.hota import HOTA
        from trackeval.metrics.identity import Identity
    except ImportError as e:
        raise RuntimeError(
            "trackeval is not installed. pip install trackeval"
        ) from e

    max_gt_f, _ = mot_lines_to_seq_info(gt_mot_lines)
    max_pr_f, _ = mot_lines_to_seq_info(pred_mot_lines)
    seq_len = max(max_gt_f, max_pr_f, 1)

    tmp = Path(tempfile.mkdtemp(prefix="sway_trackeval_"))
    try:
        gt_seq = tmp / "gt_root" / sequence_name
        (gt_seq / "gt").mkdir(parents=True)
        write_mot_file(gt_mot_lines, gt_seq / "gt" / "gt.txt")
        _write_seqinfo(gt_seq / "seqinfo.ini", seq_len, im_width, im_height)

        tr_name = "sway_tracker"
        tr_root = tmp / "tr_root" / tr_name / "data"
        tr_root.mkdir(parents=True)
        write_mot_file(pred_mot_lines, tr_root / f"{sequence_name}.txt")

        dataset_config = {
            "GT_FOLDER": str(tmp / "gt_root"),
            "TRACKERS_FOLDER": str(tmp / "tr_root"),
            "TRACKERS_TO_EVAL": [tr_name],
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": {sequence_name: seq_len},
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
            "TRACKER_SUB_FOLDER": "data",
            "PRINT_CONFIG": False,
        }
        dataset = MotChallenge2DBox(dataset_config)

        eval_config = {
            "USE_PARALLEL": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        }
        metrics = [HOTA(), CLEAR(), Identity()]
        evaluator = Evaluator(eval_config)
        res, _msg = evaluator.evaluate([dataset], metrics)

        flat: Dict[str, Any] = {}
        ds = res.get("MotChallenge2DBox", {})
        for trk, seq_data in ds.items():
            if sequence_name not in seq_data:
                continue
            ped = seq_data[sequence_name].get("pedestrian", {})
            for metric_name, fields in ped.items():
                if not isinstance(fields, dict):
                    continue
                for k, v in fields.items():
                    if isinstance(v, float):
                        flat[f"{metric_name}_{k}"] = v
                    elif isinstance(v, (int, np.integer)):
                        flat[f"{metric_name}_{k}"] = int(v)
        flat["sequence"] = sequence_name
        return flat
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def trackeval_from_ground_truth_yaml(
    gt_yaml: Dict[str, Any],
    data_json: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    If gt_yaml contains trackeval.gt_mot_file (path to MOT gt.txt), run eval vs data_json.
    Optional: trackeval.sequence_name, im_width, im_height
    """
    te = gt_yaml.get("trackeval") or {}
    mot_path = te.get("gt_mot_file")
    if not mot_path:
        return None
    p = Path(mot_path)
    if not p.is_file():
        raise FileNotFoundError(f"trackeval.gt_mot_file not found: {mot_path}")

    gt_lines = load_mot_lines_from_file(p)
    pred_lines = data_json_to_mot_lines(data_json)
    seq = te.get("sequence_name", "swayseq")
    w = int(te.get("im_width", data_json.get("metadata", {}).get("frame_width", 1920)))
    h = int(te.get("im_height", data_json.get("metadata", {}).get("frame_height", 1080)))
    return run_trackeval_single_sequence(gt_lines, pred_lines, seq, w, h)


