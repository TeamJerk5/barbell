"""End-to-end barbell tracking pipeline.

Usage:
    python run_pipeline.py <input.mp4> [--out <dir>] [--clip <name>]

Stages:
    1. inference_motion_v2.py    YOLO + motion-darkening per-frame L/R plate detection
    2. post_process_seg_v1.py    Anchor-graded fill + plate-size/bar-length priors
    3. select_seeds.py           Auto-pick cotracker seeds from postproc CSV
    4. cotracker_seed_track.py   CoTracker3 offline seed-driven trajectories

Outputs (under <out>/<clip>/):
    detections_selected_v9_v4_g0.15_l0.08.csv
    detections_postproc_seg_v1_g0.15_l0.08.csv
    cotracker_tracks_auto.csv
    cotracker_video_auto.mp4
And <out>/auto_seeds.json, <out>/cotracker_summary_auto.txt.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "5data_best.pt"


def derive_clip_name(video_path: Path) -> str:
    m = re.search(r"(KNSF_day\d+_\d+)", video_path.stem)
    return m.group(1) if m else video_path.stem


def link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def run(cmd: list[str], *, env_extra: dict | None = None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    print(f"\n$ {' '.join(map(str, cmd))}")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"[FAIL] stage exited {r.returncode}: {cmd[1] if len(cmd)>1 else cmd[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="input mp4 path")
    ap.add_argument("--out", default="out", help="output root (default: ./out)")
    ap.add_argument("--clip", default=None, help="clip name (default: derived from filename)")
    args = ap.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        raise SystemExit(f"input video not found: {video}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"model not found: {MODEL_PATH}")

    clip = args.clip or derive_clip_name(video)
    out_root = Path(args.out).resolve()
    clip_dir = out_root / clip
    clip_dir.mkdir(parents=True, exist_ok=True)

    # cotracker expects <video-dir>/<clip>.mp4 — link with canonical name
    video_link = out_root / f"{clip}.mp4"
    link_or_copy(video, video_link)

    print(f"[INFO] clip      : {clip}")
    print(f"[INFO] video     : {video}")
    print(f"[INFO] out root  : {out_root}")
    print(f"[INFO] model     : {MODEL_PATH}")

    # 1) inference_motion_v2.py
    run(
        [sys.executable, str(HERE / "inference_motion_v2.py")],
        env_extra={
            "INFERENCE_RULE_V9_MODEL_PATH": str(MODEL_PATH),
            "INFERENCE_RULE_V9_VIDEO_PATH": str(video_link),
            "INFERENCE_MOTION_V2_OUTPUT_DIR": str(clip_dir),
            "INFERENCE_MOTION_V2_FAST": "1",
        },
    )

    # 2) post_process_seg_v1.py — operates on <root>/<clip>/detections_selected_*.csv
    run([sys.executable, str(HERE / "post_process_seg_v1.py"), str(out_root)])

    # 3) select_seeds.py
    seeds_json = out_root / f"auto_seeds_{clip}.json"
    run([
        sys.executable, str(HERE / "select_seeds.py"),
        "--root", str(out_root),
        "--out", str(seeds_json),
    ])

    # 4) cotracker_seed_track.py
    run(
        [
            sys.executable, str(HERE / "cotracker_seed_track.py"),
            "--video-dir", str(out_root),
            "--run-out",   str(out_root),
            "--seeds-json", str(seeds_json),
            "--only", clip,
            "--out-suffix", "_auto",
        ],
        env_extra={"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
    )

    print(f"\n[DONE] artifacts under {clip_dir}")


if __name__ == "__main__":
    main()
