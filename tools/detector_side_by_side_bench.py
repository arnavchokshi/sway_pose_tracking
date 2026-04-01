#!/usr/bin/env python3
"""
Run person detection only (no tracking/pose) for multiple SWAY detectors on one video,
log timing/detection stats, write per-model panel MP4s, then ffmpeg hstack into one comparison video.

Usage (from sway_pose_mvp):
  CUDA_VISIBLE_DEVICES=0 python -m tools.detector_side_by_side_bench \\
    --video /path/to/BigTest.mov --out-dir output/det_bench_run

Env: uses SWAY_DETECT_SIZE, SWAY_YOLO_CONF, SWAY_DETR_CONF unless overridden.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import _repo_path  # noqa: F401, E402


def _draw_panel(
    frame_bgr: np.ndarray,
    dets: list,
    title: str,
    subtitle: str,
    color: Tuple[int, int, int],
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    band = 48
    cv2.rectangle(out, (0, 0), (w, band), (30, 30, 30), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, title, (8, 22), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, subtitle, (8, 42), font, 0.45, (200, 220, 255), 1, cv2.LINE_AA)
    for d in dets:
        box = getattr(d, "bbox", None)
        if box is None:
            continue
        x1, y1, x2, y2 = map(int, box)
        cf = float(getattr(d, "confidence", 0.0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{cf:.2f}",
            (x1, max(band + 14, y1 - 4)),
            font,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    a = sorted(xs)
    k = int(round((len(a) - 1) * q))
    return float(a[max(0, min(k, len(a) - 1))])


def _panel_writer_start(
    out_path: Path,
    fw: int,
    fh: int,
    fps: float,
    ffmpeg_bin: str,
    crf: int,
    preset: str,
) -> subprocess.Popen:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{fw}x{fh}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=fw * fh * 3 * 2,
    )


def _panel_writer_write(proc: subprocess.Popen, panel: np.ndarray) -> None:
    proc.stdin.write(panel.astype(np.uint8, copy=False).tobytes())


def _panel_writer_close(proc: subprocess.Popen) -> None:
    if proc.stdin:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg panel encode failed (exit {rc})")


def run_one_model(
    video: Path,
    out_panel_mp4: Path,
    primary: str,
    device: str,
    warmup: int,
    colors: Tuple[int, int, int],
    *,
    ffmpeg_bin: str | None,
    panel_crf: int,
    panel_preset: str,
) -> Dict[str, Any]:
    import torch
    from sway.detector_factory import create_detector

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = create_detector(primary=primary, device=device)
    backend_note = primary
    if primary in ("co_detr", "co_dino") and type(detector).__name__ == "_YOLODetectorAdapter":
        backend_note = f"{primary} (fallback YOLO)"

    use_ffmpeg = ffmpeg_bin is not None
    if use_ffmpeg:
        enc = _panel_writer_start(
            out_panel_mp4, fw, fh, fps, ffmpeg_bin, panel_crf, panel_preset
        )
    else:
        enc = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_panel_mp4), fourcc, fps, (fw, fh))

    lat_ms: List[float] = []
    det_counts: List[int] = []
    total_infer = 0.0
    frame_idx = 0

    def sync():
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < warmup:
                try:
                    _ = detector.detect(frame, frame_idx=frame_idx)
                except TypeError:
                    _ = detector.detect(frame)
                sync()
                panel = _draw_panel(
                    frame,
                    [],
                    primary,
                    "warmup",
                    colors,
                )
                if use_ffmpeg:
                    _panel_writer_write(enc, panel)
                else:
                    writer.write(panel)
                frame_idx += 1
                continue

            sync()
            t0 = time.perf_counter()
            try:
                dets = detector.detect(frame, frame_idx=frame_idx)
            except TypeError:
                dets = detector.detect(frame)
            if isinstance(dets, tuple) and len(dets) == 2:
                dets = dets[0]
            sync()
            dt = time.perf_counter() - t0
            total_infer += dt
            ms = dt * 1000.0
            lat_ms.append(ms)
            det_counts.append(len(dets))

            panel = _draw_panel(
                frame,
                dets,
                primary,
                f"{ms:.1f} ms  |  n={len(dets)}",
                colors,
            )
            if use_ffmpeg:
                _panel_writer_write(enc, panel)
            else:
                writer.write(panel)
            frame_idx += 1
    finally:
        cap.release()
        if use_ffmpeg:
            _panel_writer_close(enc)
        else:
            writer.release()

    n = len(lat_ms)
    return {
        "primary": primary,
        "backend_note": backend_note,
        "frames_timed": n,
        "video_frames_reported": nframes,
        "warmup_frames": warmup,
        "total_inference_s": round(total_infer, 4),
        "mean_ms_per_frame": round(sum(lat_ms) / n, 3) if n else 0.0,
        "p50_ms": round(_percentile(lat_ms, 0.50), 3),
        "p95_ms": round(_percentile(lat_ms, 0.95), 3),
        "effective_fps": round(n / total_infer, 3) if total_infer > 0 else 0.0,
        "total_detections": int(sum(det_counts)),
        "mean_dets_per_frame": round(sum(det_counts) / n, 4) if n else 0.0,
        "panel_video": str(out_panel_mp4),
    }


def _ffmpeg_hstack_one_pass(
    panel_paths: List[Path],
    out_mp4: Path,
    ffmpeg_bin: str,
    panel_width: int,
    crf: int,
    preset: str,
) -> None:
    """Decode panel MP4s, Lanczos-scale each column, hstack, single high-quality x264 encode."""
    n = len(panel_paths)
    if n == 0:
        raise SystemExit("No panels to stack")
    ins: List[str] = []
    for p in panel_paths:
        ins.extend(["-i", str(p)])
    scale_parts = [
        f"[{i}:v]scale={panel_width}:-2:flags=lanczos+accurate_rnd+full_chroma_int[v{i}]"
        for i in range(n)
    ]
    stack_in = "".join(f"[v{i}]" for i in range(n))
    fc = ";".join(scale_parts) + f";{stack_in}hstack=inputs={n}[outv]"
    cmd = [
        ffmpeg_bin,
        "-y",
        *ins,
        "-filter_complex",
        fc,
        "-map",
        "[outv]",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def _hstack_panels_cv2(panel_paths: List[Path], out_mp4: Path, target_w: int) -> None:
    """Scale each panel to target_w width and horizontal-stack into one MP4 (no ffmpeg)."""
    caps = [cv2.VideoCapture(str(p)) for p in panel_paths]
    if not all(c.isOpened() for c in caps):
        raise SystemExit("OpenCV could not open one or more panel videos")
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    outs = []
    for c in caps:
        fw = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nh = max(1, int(fh * (target_w / max(1, fw))))
        outs.append((nh, target_w))
    h_max = max(t[0] for t in outs)
    total_w = target_w * len(panel_paths)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (total_w, h_max))
    while True:
        tiles = []
        ok = True
        for c, (nh, nw) in zip(caps, outs):
            ret, fr = c.read()
            if not ret:
                ok = False
                break
            scaled = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA)
            if nh < h_max:
                pad = np.zeros((h_max, nw, 3), dtype=np.uint8)
                pad[(h_max - nh) // 2 : (h_max - nh) // 2 + nh, :] = scaled
                tiles.append(pad)
            else:
                tiles.append(scaled)
        if not ok:
            break
        writer.write(np.hstack(tiles))
    for c in caps:
        c.release()
    writer.release()


def _sorted_panel_paths(out_dir: Path) -> List[Path]:
    return sorted(out_dir.glob("panel_*.mp4"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detection-only benchmark + side-by-side MP4 (high quality when ffmpeg is used)."
    )
    ap.add_argument("--video", type=Path, default=None, help="Source video (not required with --merge-only)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
        "--models",
        nargs="*",
        default=[
            "yolo26l_dancetrack",
            "rt_detr_l",
            "rt_detr_x",
            "co_detr",
            "co_dino",
        ],
    )
    ap.add_argument(
        "--panel-width",
        type=int,
        default=720,
        help="Target width per column after Lanczos scale (default 720; was 360–400 in older runs)",
    )
    ap.add_argument(
        "--panel-crf",
        type=int,
        default=16,
        help="libx264 CRF for each column MP4 (lower = better; 14–18 typical)",
    )
    ap.add_argument(
        "--stitch-crf",
        type=int,
        default=14,
        help="libx264 CRF for final side-by-side (single re-encode after scale+hstack)",
    )
    ap.add_argument(
        "--encoder-preset",
        default="medium",
        help="libx264 preset (slower presets = better compression at same CRF)",
    )
    ap.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip detection; restitch from existing panel_*.mp4 in --out-dir (needs ffmpeg)",
    )
    ap.add_argument(
        "--opencv-panels",
        action="store_true",
        help="Write column videos with OpenCV mp4v (low quality); default is ffmpeg libx264 if available",
    )
    ap.add_argument(
        "--first-n-panels",
        type=int,
        default=None,
        help="Stitch only first N panels (e.g. 3 for YOLO + RT-DETR-L/X) after --merge-only or full run",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    final_mp4 = out_dir / "detectors_side_by_side.mp4"
    ffmpeg_panels = ffmpeg is not None and not args.opencv_panels

    panel_paths: List[Path] = []

    if args.merge_only:
        panel_paths = _sorted_panel_paths(out_dir)
        if not panel_paths:
            raise SystemExit(f"No panel_*.mp4 files in {out_dir}")
    else:
        if args.video is None:
            raise SystemExit("--video is required unless --merge-only")
        video = args.video.expanduser().resolve()
        if not video.is_file():
            raise SystemExit(f"Missing video: {video}")

        colors = [
            (0, 200, 0),
            (0, 165, 255),
            (255, 128, 0),
            (200, 0, 200),
            (255, 255, 0),
        ]
        results: Dict[str, Any] = {"video": str(video), "device": args.device, "models": {}}

        for i, primary in enumerate(args.models):
            panel_path = out_dir / f"panel_{i:02d}_{primary}.mp4"
            c = colors[i % len(colors)]
            print(f"\n=== [{i+1}/{len(args.models)}] {primary} ===", flush=True)
            stat = run_one_model(
                video,
                panel_path,
                primary,
                args.device,
                args.warmup,
                c,
                ffmpeg_bin=ffmpeg if ffmpeg_panels else None,
                panel_crf=args.panel_crf,
                panel_preset=args.encoder_preset,
            )
            results["models"][primary] = stat
            panel_paths.append(panel_path)
            print(json.dumps(stat, indent=2), flush=True)

        metrics_path = out_dir / "detector_bench_metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {metrics_path}", flush=True)

    all_panels = list(panel_paths)
    if args.first_n_panels is not None:
        panel_paths = all_panels[: max(0, args.first_n_panels)]

    if ffmpeg:
        if len(panel_paths) == 1:
            subprocess.run(["cp", str(panel_paths[0]), str(final_mp4)], check=True)
        else:
            _ffmpeg_hstack_one_pass(
                panel_paths,
                final_mp4,
                ffmpeg,
                args.panel_width,
                args.stitch_crf,
                args.encoder_preset,
            )
    else:
        if args.merge_only:
            raise SystemExit("--merge-only requires ffmpeg for Lanczos scale + hstack")
        _hstack_panels_cv2(panel_paths, final_mp4, args.panel_width)

    print(f"Side-by-side: {final_mp4}", flush=True)

    yolo_stack = out_dir / "detectors_yolo_rtdetr_l_x.mp4"
    if ffmpeg and len(all_panels) >= 3:
        _ffmpeg_hstack_one_pass(
            all_panels[:3],
            yolo_stack,
            ffmpeg,
            args.panel_width,
            args.stitch_crf,
            args.encoder_preset,
        )
        print(f"3-up (first 3 panels → YOLO + RT-DETR L/X if run order matches): {yolo_stack}", flush=True)


if __name__ == "__main__":
    main()
