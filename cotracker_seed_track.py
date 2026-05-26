#!/usr/bin/env python3
"""
cotracker_seed_track.py — post_process_seg_v1 결과의 시드로 CoTracker3 offline 트래킹.

두 가지 입력 모드:
  (A) 내장 SEEDS 딕셔너리 (단일 시드)
  (B) --seeds-json <path>   (select_seeds.py 출력, 다중 시드 지원)

다중 시드 동작:
  per side (L/R) 별로 N_seeds 개의 trajectory 를 CoTracker3 에 의뢰해 (2·N_seeds queries),
  각 frame t 에서 side 별로 |t - seed_t| 가 최소인 trajectory 의 (x, y, visibility) 를 채택.

출력 (batch_run_out/<clip>/):
  cotracker_tracks{suffix}.csv  : frame_idx, side(L/R), x, y, visible
  cotracker_video{suffix}.mp4   : 시각화 영상

전체 요약:
  batch_run_out/cotracker_summary{suffix}.txt

환경변수:
  COTRACKER_MAX_SIDE  : 모델 입력 최대 변 길이 (기본 768)
  COTRACKER_DEVICE    : auto | cpu | mps | cuda

사용:
  python cotracker_seed_track.py                                 # SEEDS dict 사용
  python cotracker_seed_track.py --only KNSF_day4_005
  python cotracker_seed_track.py --seeds-json auto_seeds.json --out-suffix _auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch


DEFAULT_ROOT = Path("/Users/byeol/Desktop/by/4-1/빅데이터 캡스톤/0519_5data")
VIDEO_DIR = DEFAULT_ROOT / "mp4"
RUN_OUT_DIR = DEFAULT_ROOT / "batch_run_out"
CSV_NAME = "detections_postproc_seg_v1_g0.15_l0.08.csv"

SEEDS_BUILTIN: Dict[str, int] = {
    "KNSF_day4_001": 143,
    "KNSF_day4_002": 0,
    "KNSF_day4_003": 105,
    "KNSF_day4_004": 97,
    "KNSF_day4_005": 158,
}

MAX_SIDE = int(os.environ.get("COTRACKER_MAX_SIDE", "768"))

L_COLOR = (60, 220, 80)
R_COLOR = (40, 180, 240)
BAR_COLOR = (200, 200, 200)
SEED_MARKER_COLOR = (0, 255, 255)
TRAIL_LEN = 60

FOURCC_PREF = ("avc1", "mp4v")


@dataclass
class SeedSpec:
    t: int
    Lx: float
    Ly: float
    Rx: float
    Ry: float
    L_src: str = ""
    R_src: str = ""


def pick_device() -> torch.device:
    pref = os.environ.get("COTRACKER_DEVICE", "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def open_writer(path: Path, fps: float, W: int, H: int):
    for codec in FOURCC_PREF:
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (W, H))
        if w.isOpened():
            return w, codec
        w.release()
    return None, None


def read_video_resized(path: Path, max_side: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(max_side / max(W0, H0), 1.0)
    W = int(round(W0 * scale))
    H = int(round(H0 * scale))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if scale < 1.0:
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    arr = np.stack(frames, axis=0)
    return arr, (W0, H0), (W, H), fps, scale


def seed_from_csv(csv_path: Path, t: int) -> SeedSpec:
    df = pd.read_csv(csv_path)
    df["frame_idx"] = df["frame_idx"].astype(int)
    sub = df[df.frame_idx == t]
    if len(sub) == 0:
        raise RuntimeError(f"no rows at frame_idx={t} in {csv_path}")
    L = sub[sub.side == "L"]
    R = sub[sub.side == "R"]
    if len(L) == 0 or len(R) == 0:
        raise RuntimeError(f"missing L or R at frame_idx={t} in {csv_path}")
    Lr, Rr = L.iloc[0], R.iloc[0]
    if pd.isna(Lr.center_x) or pd.isna(Lr.center_y) or pd.isna(Rr.center_x) or pd.isna(Rr.center_y):
        raise RuntimeError(f"NaN seed coords at frame_idx={t} in {csv_path}")
    return SeedSpec(
        t=int(t),
        Lx=float(Lr.center_x), Ly=float(Lr.center_y),
        Rx=float(Rr.center_x), Ry=float(Rr.center_y),
        L_src=str(Lr.source), R_src=str(Rr.source),
    )


def merge_multi_seed(tracks: np.ndarray, vis: np.ndarray,
                     seed_ts: List[int]) -> tuple:
    """
    tracks: (T, 2*N_seeds, 2) — seed i 의 L 은 2i, R 은 2i+1
    vis   : (T, 2*N_seeds) bool
    seed_ts: 각 seed 의 시드 frame_idx
    Returns: out_tracks (T, 2, 2), out_vis (T, 2)
    각 frame t / side 별로 |t - seed_t| 가 최소인 seed 의 trajectory 채택.
    """
    T = tracks.shape[0]
    N = len(seed_ts)
    out_tracks = np.zeros((T, 2, 2), dtype=tracks.dtype)
    out_vis = np.zeros((T, 2), dtype=bool)
    seed_arr = np.array(seed_ts, dtype=int)
    for t in range(T):
        best = int(np.argmin(np.abs(seed_arr - t)))
        for side in range(2):
            qi = 2 * best + side
            out_tracks[t, side] = tracks[t, qi]
            out_vis[t, side] = vis[t, qi]
    return out_tracks, out_vis


def run_clip(name: str, seeds: List[SeedSpec], video_dir: Path, run_out_dir: Path,
             model, device: torch.device, out_suffix: str) -> dict:
    if not seeds:
        raise RuntimeError(f"no seeds provided for {name}")

    video_path = video_dir / f"{name}.mp4"
    out_dir = run_out_dir / name
    out_csv = out_dir / f"cotracker_{name}.csv"

    for i, s in enumerate(seeds):
        print(f"  seed#{i} t={s.t}  L=({s.Lx:.1f},{s.Ly:.1f}) [{s.L_src}]  "
              f"R=({s.Rx:.1f},{s.Ry:.1f}) [{s.R_src}]")

    t0 = time.time()
    arr, (W0, H0), (W, H), fps, scale = read_video_resized(video_path, MAX_SIDE)
    T = arr.shape[0]
    print(f"  loaded {T} frames {W0}x{H0} -> {W}x{H} (scale={scale:.3f})  in {time.time()-t0:.1f}s")

    # Clamp seeds to valid range
    for s in seeds:
        if s.t < 0 or s.t >= T:
            raise RuntimeError(f"seed t={s.t} out of range [0,{T-1}] for {name}")

    vid_t = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    # queries: 2*N_seeds, each [t, x_scaled, y_scaled]
    q_list = []
    for s in seeds:
        q_list.append([float(s.t), s.Lx * scale, s.Ly * scale])
        q_list.append([float(s.t), s.Rx * scale, s.Ry * scale])
    queries = torch.tensor(q_list, device=device, dtype=torch.float32).unsqueeze(0)

    backward = any(s.t > 0 for s in seeds)

    t0 = time.time()
    with torch.no_grad():
        pred_tracks, pred_visibility = model(
            vid_t,
            queries=queries,
            backward_tracking=backward,
        )
    print(f"  cotracker forward done in {time.time()-t0:.1f}s  (N_queries={len(q_list)}, backward={backward})")

    tracks_all = pred_tracks[0].detach().cpu().numpy()       # (T, 2N, 2)
    vis_all = pred_visibility[0].detach().cpu().numpy()
    if vis_all.dtype != bool:
        vis_all = vis_all > 0.5
    tracks_all = tracks_all / scale  # back to original resolution

    if len(seeds) == 1:
        tracks = tracks_all  # (T, 2, 2)
        vis = vis_all        # (T, 2)
    else:
        seed_ts = [s.t for s in seeds]
        tracks, vis = merge_multi_seed(tracks_all, vis_all, seed_ts)

    # ---- tracks CSV ----
    rows = []
    for t in range(T):
        for i, side in enumerate(("L", "R")):
            rows.append({
                "frame_idx": t,
                "side": side,
                "x": float(tracks[t, i, 0]),
                "y": float(tracks[t, i, 1]),
                "visible": int(bool(vis[t, i])),
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    return {
        "name": name,
        "n_seeds": len(seeds),
        "seed_ts": [s.t for s in seeds],
        "seeds": [{
            "t": s.t, "Lx": s.Lx, "Ly": s.Ly, "Rx": s.Rx, "Ry": s.Ry,
            "L_src": s.L_src, "R_src": s.R_src,
        } for s in seeds],
        "frames": T,
        "L_vis_count": int(vis[:, 0].sum()),
        "R_vis_count": int(vis[:, 1].sum()),
        "L_vis_mean": float(vis[:, 0].mean()),
        "R_vis_mean": float(vis[:, 1].mean()),
        "scale": scale,
        "out_csv": str(out_csv),
    }


def write_summary(summaries: list, log_path: Path, device: torch.device, suffix: str, mode: str):
    lines = [
        "CoTracker3 offline — seed-tracked weightlifting plate centers",
        f"device       : {device}",
        f"MAX_SIDE     : {MAX_SIDE}",
        f"seed mode    : {mode}",
        f"out suffix   : '{suffix}'",
        f"clips        : {len(summaries)}",
        "=" * 72,
        "",
    ]
    for r in summaries:
        lines += [
            f"[{r['name']}]",
            f"  N seeds        : {r['n_seeds']}    seed_ts={r['seed_ts']}",
        ]
        for i, s in enumerate(r["seeds"]):
            lines.append(
                f"  seed#{i}        : t={s['t']}  L=({s['Lx']:.2f},{s['Ly']:.2f}) [{s['L_src']}]  "
                f"R=({s['Rx']:.2f},{s['Ry']:.2f}) [{s['R_src']}]"
            )
        lines += [
            f"  frames tracked : {r['frames']}",
            f"  L visibility   : {r['L_vis_count']}/{r['frames']} ({r['L_vis_mean']*100:.2f}%)",
            f"  R visibility   : {r['R_vis_count']}/{r['frames']} ({r['R_vis_mean']*100:.2f}%)",
            f"  scale (resize) : {r['scale']:.4f}",
            f"  out CSV        : {r['out_csv']}",
            "",
        ]
    log_path.write_text("\n".join(lines), encoding="utf-8")


def load_seeds_json(path: Path, run_out_dir: Path) -> Dict[str, List[SeedSpec]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[SeedSpec]] = {}
    for clip, payload in raw.items():
        specs: List[SeedSpec] = []
        for s in payload.get("seeds", []):
            specs.append(SeedSpec(
                t=int(s["t"]),
                Lx=float(s["Lx"]), Ly=float(s["Ly"]),
                Rx=float(s["Rx"]), Ry=float(s["Ry"]),
                L_src="json", R_src="json",
            ))
        out[clip] = specs
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", default=str(VIDEO_DIR))
    p.add_argument("--run-out", default=str(RUN_OUT_DIR))
    p.add_argument("--only", default=None,
                   help="comma-separated clip names (e.g. KNSF_day4_005)")
    p.add_argument("--seeds-json", default=None,
                   help="select_seeds.py 출력 JSON. 지정 시 SEEDS_BUILTIN 무시.")
    p.add_argument("--out-suffix", default="",
                   help="출력 파일 접미사 (예: _auto → cotracker_tracks_auto.csv)")
    args = p.parse_args()

    device = pick_device()
    print(f"[INFO] device: {device}  MAX_SIDE: {MAX_SIDE}")
    print(f"[INFO] video_dir: {args.video_dir}")
    print(f"[INFO] run_out  : {args.run_out}")
    print(f"[INFO] suffix   : '{args.out_suffix}'")

    run_out_dir = Path(args.run_out)
    video_dir = Path(args.video_dir)

    # build per-clip seeds
    seeds_per_clip: Dict[str, List[SeedSpec]] = {}
    mode = "builtin_single"
    if args.seeds_json:
        seeds_per_clip = load_seeds_json(Path(args.seeds_json), run_out_dir)
        mode = f"json:{args.seeds_json}"
        print(f"[INFO] seeds source: {args.seeds_json} ({len(seeds_per_clip)} clips)")
    else:
        for name, t in SEEDS_BUILTIN.items():
            try:
                spec = seed_from_csv(run_out_dir / name / CSV_NAME, t)
                seeds_per_clip[name] = [spec]
            except Exception as e:
                print(f"[WARN] {name}: cannot load seed t={t}: {e}", file=sys.stderr)

    if args.only:
        only = set(args.only.split(","))
        seeds_per_clip = {k: v for k, v in seeds_per_clip.items() if k in only}

    if not seeds_per_clip:
        print("[ERROR] no seeds to process", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] loading cotracker3_offline ...")
    model = torch.hub.load("facebookresearch/co-tracker",
                           "cotracker3_offline").to(device).eval()

    summaries = []
    for name, specs in seeds_per_clip.items():
        if not specs:
            print(f"\n[{name}] SKIP — empty seeds")
            continue
        print(f"\n[{name}] N_seeds={len(specs)}")
        try:
            r = run_clip(name, specs, video_dir, run_out_dir, model, device, args.out_suffix)
            summaries.append(r)
            print(f"  L vis {r['L_vis_count']}/{r['frames']} ({r['L_vis_mean']*100:.1f}%) "
                  f"R vis {r['R_vis_count']}/{r['frames']} ({r['R_vis_mean']*100:.1f}%)")
            print(f"  -> {r['out_csv']}")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}", file=sys.stderr)
            traceback.print_exc()

    log_path = run_out_dir / f"cotracker_summary{args.out_suffix}.txt"
    write_summary(summaries, log_path, device, args.out_suffix, mode)
    print(f"\n[INFO] summary -> {log_path}")


if __name__ == "__main__":
    main()
