#!/usr/bin/env python3
"""
select_seeds.py — post_process_seg_v1 CSV 만 보고 cotracker3 시드를 자동 선정.

전략 문서 (seed_selection_strategy.md) 의 알고리즘을 그대로 구현:
  S1 both-detected, S2 conf floor, S3 bar-length consistency, S4 positional consistency
  -> eligible frame -> contiguous run -> run scoring -> multi-seed 규칙
  -> fallback (tau / L_min 단계적 완화, kalman_pred 허용, 최후엔 conf max)

사용:
  python select_seeds.py [--root /path/to/batch_run_out]
                         [--input-glob 'detections_postproc_seg_v1_*.csv']
                         [--out seeds.json]
                         [--verbose]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 기본 파라미터 (전략 문서 §6 와 일치)
# ============================================================
DEFAULTS = {
    "tau_conf":                  0.50,
    "k_bar_mad":                 3.0,
    "d_max_factor":              0.50,
    "L_min":                     5,
    "alpha_edge":                0.30,
    "delta_t_min":               60,
    "secondary_min_score_ratio": 0.30,
    "n_seeds_max":               3,
    "fallback_tau_seq":          [0.50, 0.30, 0.10],
    "fallback_Lmin_seq":         [5, 3, 1],
    # 영상 자체 sanity (운영 단계 flag 용)
    "barL_min_frame_ratio":      0.05,   # bar_median / frame_width >= 5%
    "barL_max_frame_ratio":      0.60,
}

DEFAULT_ROOT = Path("/Users/byeol/Desktop/by/4-1/빅데이터 캡스톤/0519_5data/batch_run_out")
DEFAULT_GLOB = "detections_postproc_*.csv"


# ============================================================
# 데이터 컨테이너
# ============================================================
@dataclass
class SeedPoint:
    t: int
    Lx: float
    Ly: float
    Rx: float
    Ry: float
    run_start: int
    run_end: int
    run_length: int
    L_conf: float
    R_conf: float
    bar_length: float

@dataclass
class ClipResult:
    clip: str
    seeds: List[SeedPoint] = field(default_factory=list)
    fallback_used: str = "none"            # none | tau_relaxed | Lmin_relaxed | kalman_allowed | conf_max_only
    n_frames: int = 0
    n_both_detected: int = 0
    n_eligible: int = 0
    bar_length_median: Optional[float] = None
    bar_length_mad: Optional[float] = None
    bar_length_cov: Optional[float] = None
    barL_to_frame_ratio: Optional[float] = None  # 영상 폭 미지 → None (운영 단계에서 채움)
    video_flag: Optional[str] = None       # low_confidence_video | etc.
    runs_total: int = 0
    runs_eligible: int = 0
    selected_runs: List[Tuple[int, int, float]] = field(default_factory=list)  # (start, end, score)

    def to_json(self):
        d = asdict(self)
        d["seeds"] = [asdict(s) for s in self.seeds]
        return d


# ============================================================
# 핵심 알고리즘
# ============================================================
def _both_detected_mask(L: pd.DataFrame, R: pd.DataFrame, common: pd.Index) -> np.ndarray:
    ls = L.loc[common, "source"].astype("object").to_numpy()
    rs = R.loc[common, "source"].astype("object").to_numpy()
    return (ls == "detected") & (rs == "detected")


def _compute_priors(L: pd.DataFrame, R: pd.DataFrame, common: pd.Index,
                    mask: np.ndarray) -> dict:
    """양쪽 detected (또는 fallback mask) 에서 prior 계산."""
    if mask.sum() == 0:
        return None
    lcx = L.loc[common, "center_x"].to_numpy()
    lcy = L.loc[common, "center_y"].to_numpy()
    rcx = R.loc[common, "center_x"].to_numpy()
    rcy = R.loc[common, "center_y"].to_numpy()
    barL = np.hypot(rcx - lcx, rcy - lcy)
    barL_m = barL[mask]
    bar_med = float(np.median(barL_m))
    bar_mad = float(1.4826 * np.median(np.abs(barL_m - bar_med)))
    return {
        "bar_median": bar_med,
        "bar_mad":    bar_mad,
        "Lcx_med":    float(np.median(lcx[mask])),
        "Lcy_med":    float(np.median(lcy[mask])),
        "Rcx_med":    float(np.median(rcx[mask])),
        "Rcy_med":    float(np.median(rcy[mask])),
        "barL":       barL,           # 전체 frames
        "lcx": lcx, "lcy": lcy, "rcx": rcx, "rcy": rcy,
    }


def _eligibility_mask(L: pd.DataFrame, R: pd.DataFrame, common: pd.Index,
                      both_det: np.ndarray, priors: dict,
                      tau_conf: float, k_bar_mad: float, d_max_factor: float,
                      require_both_detected: bool = True) -> np.ndarray:
    """S1 ∧ S2 ∧ S3 ∧ S4 (require_both_detected=False 면 S1 무시 — fallback 용)."""
    n = len(common)
    Lc = L.loc[common, "confidence"].to_numpy().astype(float)
    Rc = R.loc[common, "confidence"].to_numpy().astype(float)
    Lc = np.nan_to_num(Lc, nan=0.0)
    Rc = np.nan_to_num(Rc, nan=0.0)

    # S1
    s1 = both_det if require_both_detected else np.ones(n, dtype=bool)
    # S2
    s2 = (np.minimum(Lc, Rc) >= tau_conf)
    # S3
    if priors["bar_mad"] > 1e-6:
        s3 = np.abs(priors["barL"] - priors["bar_median"]) <= k_bar_mad * priors["bar_mad"]
    else:
        s3 = np.ones(n, dtype=bool)  # MAD=0 이면 분산 없음 → 모두 통과
    # S4
    d_max = d_max_factor * priors["bar_median"]
    dL = np.hypot(priors["lcx"] - priors["Lcx_med"], priors["lcy"] - priors["Lcy_med"])
    dR = np.hypot(priors["rcx"] - priors["Rcx_med"], priors["rcy"] - priors["Rcy_med"])
    s4 = (dL <= d_max) & (dR <= d_max)

    return s1 & s2 & s3 & s4


def _runs_from_frames(eligible_frames: List[int]) -> List[Tuple[int, int]]:
    """연속된 frame_idx 를 (start, end) run 으로 묶음."""
    if not eligible_frames:
        return []
    runs = []
    s = p = eligible_frames[0]
    for f in eligible_frames[1:]:
        if f == p + 1:
            p = f
        else:
            runs.append((s, p))
            s = p = f
    runs.append((s, p))
    return runs


def _score_run(run: Tuple[int, int], common: pd.Index, Lc: np.ndarray, Rc: np.ndarray,
               barL: np.ndarray, bar_mad_global: float) -> float:
    s, e = run
    # common 의 위치 인덱스
    pos = np.where((common >= s) & (common <= e))[0]
    length = len(pos)
    if length == 0:
        return 0.0
    mean_conf = float(np.mean(Lc[pos] * Rc[pos]))
    if length >= 2 and bar_mad_global > 1e-6:
        bar_std = float(np.std(barL[pos]))
        decay = float(np.exp(-bar_std / bar_mad_global))
    else:
        decay = 1.0
    return length * mean_conf * decay


def _seed_in_run(run: Tuple[int, int], common: pd.Index, Lc: np.ndarray, Rc: np.ndarray,
                 alpha_edge: float) -> int:
    s, e = run
    pos = np.where((common >= s) & (common <= e))[0]
    frames_in_run = common[pos].to_numpy()
    mid = (s + e) / 2.0
    half = max((e - s) / 2.0, 1.0)
    center_w = 1.0 - alpha_edge * (np.abs(frames_in_run - mid) / half)
    score = (Lc[pos] * Rc[pos]) * center_w
    return int(frames_in_run[int(np.argmax(score))])


def _pick_multi_seeds(scored_runs: List[Tuple[Tuple[int, int], float, int]],
                      delta_t_min: int, ratio: float, n_max: int) -> List[Tuple[Tuple[int, int], float, int]]:
    """scored_runs: [((s,e), score, seed_t), ...] (score desc).
    top1 채택 → 이후엔 시간 분리 + score 비율 조건."""
    if not scored_runs:
        return []
    picked = [scored_runs[0]]
    top_score = scored_runs[0][1]
    for run, score, seed_t in scored_runs[1:]:
        if len(picked) >= n_max:
            break
        if score < ratio * top_score:
            continue
        if all(abs(seed_t - p[2]) >= delta_t_min for p in picked):
            picked.append((run, score, seed_t))
    return picked


def _build_seed_point(seed_t: int, run: Tuple[int, int],
                      L: pd.DataFrame, R: pd.DataFrame, barL_full: np.ndarray,
                      common: pd.Index) -> SeedPoint:
    pos_t = int(np.where(common == seed_t)[0][0])
    return SeedPoint(
        t=int(seed_t),
        Lx=float(L.loc[seed_t, "center_x"]),
        Ly=float(L.loc[seed_t, "center_y"]),
        Rx=float(R.loc[seed_t, "center_x"]),
        Ry=float(R.loc[seed_t, "center_y"]),
        run_start=int(run[0]),
        run_end=int(run[1]),
        run_length=int(run[1] - run[0] + 1),
        L_conf=float(np.nan_to_num(L.loc[seed_t, "confidence"], nan=0.0)),
        R_conf=float(np.nan_to_num(R.loc[seed_t, "confidence"], nan=0.0)),
        bar_length=float(barL_full[pos_t]),
    )


def select_seeds(csv_path: Path, params: dict = None) -> ClipResult:
    p = dict(DEFAULTS)
    if params:
        p.update(params)

    df = pd.read_csv(csv_path)
    df["frame_idx"] = df["frame_idx"].astype(int)
    Lall = df[df.side == "L"].set_index("frame_idx").sort_index()
    Rall = df[df.side == "R"].set_index("frame_idx").sort_index()
    common = Lall.index.intersection(Rall.index)

    # valid coords 만 (NaN 좌표는 어쨌든 시드 불가)
    valid_mask = (
        Lall.loc[common, "center_x"].notna() & Lall.loc[common, "center_y"].notna() &
        Rall.loc[common, "center_x"].notna() & Rall.loc[common, "center_y"].notna()
    )
    common = common[valid_mask.values]
    L = Lall.loc[common]
    R = Rall.loc[common]

    res = ClipResult(clip=csv_path.parent.name or csv_path.stem,
                     n_frames=int(len(common)))

    if len(common) == 0:
        res.video_flag = "no_valid_coords"
        return res

    both_det = _both_detected_mask(L, R, common)
    res.n_both_detected = int(both_det.sum())

    # ----- Prior 계산 -----
    priors_mask = both_det if both_det.sum() >= 3 else np.ones(len(common), dtype=bool)
    priors = _compute_priors(L, R, common, priors_mask)
    res.bar_length_median = priors["bar_median"]
    res.bar_length_mad = priors["bar_mad"]
    if priors["bar_median"] and priors["bar_median"] > 0:
        res.bar_length_cov = priors["bar_mad"] / priors["bar_median"]

    Lc = L["confidence"].to_numpy().astype(float)
    Rc = R["confidence"].to_numpy().astype(float)
    Lc = np.nan_to_num(Lc, nan=0.0)
    Rc = np.nan_to_num(Rc, nan=0.0)

    # ----- Eligibility + run 분할 (fallback 단계) -----
    chosen_seeds: List[SeedPoint] = []
    chosen_runs: List[Tuple[Tuple[int, int], float]] = []
    fallback_used = "none"

    for tau in p["fallback_tau_seq"]:
        for Lmin in p["fallback_Lmin_seq"]:
            elig = _eligibility_mask(
                L, R, common, both_det, priors,
                tau_conf=tau, k_bar_mad=p["k_bar_mad"],
                d_max_factor=p["d_max_factor"],
                require_both_detected=True,
            )
            n_elig = int(elig.sum())
            eligible_frames = common[elig].tolist()
            runs = _runs_from_frames(eligible_frames)
            runs = [r for r in runs if (r[1] - r[0] + 1) >= Lmin]
            if runs:
                if tau < p["fallback_tau_seq"][0]:
                    fallback_used = "tau_relaxed"
                elif Lmin < p["fallback_Lmin_seq"][0]:
                    fallback_used = "Lmin_relaxed"
                res.n_eligible = n_elig
                # run scoring + seed pick
                scored = []
                for run in runs:
                    sc = _score_run(run, common, Lc, Rc, priors["barL"], priors["bar_mad"])
                    seed_t = _seed_in_run(run, common, Lc, Rc, p["alpha_edge"])
                    scored.append((run, sc, seed_t))
                scored.sort(key=lambda x: -x[1])
                picked = _pick_multi_seeds(scored, p["delta_t_min"],
                                           p["secondary_min_score_ratio"],
                                           p["n_seeds_max"])
                for run, sc, seed_t in picked:
                    chosen_seeds.append(_build_seed_point(seed_t, run, L, R, priors["barL"], common))
                    chosen_runs.append((run, sc))
                res.runs_total = len(runs)
                res.runs_eligible = len(runs)
                res.selected_runs = [(r[0], r[1], sc) for (r, sc) in chosen_runs]
                res.seeds = chosen_seeds
                res.fallback_used = fallback_used
                break
        if chosen_seeds:
            break

    # ----- Fallback: S1 제거 (kalman_pred 허용) -----
    if not chosen_seeds:
        elig = _eligibility_mask(
            L, R, common, both_det, priors,
            tau_conf=0.0, k_bar_mad=p["k_bar_mad"],
            d_max_factor=p["d_max_factor"],
            require_both_detected=False,
        )
        eligible_frames = common[elig].tolist()
        if eligible_frames:
            # 단일 frame: conf_proxy 최대 (proxy = Lc*Rc, kalman_pred 면 0.5*0.5=0.25)
            scores_elig = (Lc * Rc)[elig]
            best_pos = int(np.argmax(scores_elig))
            seed_t = int(common[elig][best_pos])
            run = (seed_t, seed_t)
            chosen_seeds.append(_build_seed_point(seed_t, run, L, R, priors["barL"], common))
            chosen_runs.append((run, float(scores_elig[best_pos])))
            fallback_used = "kalman_allowed"

    # ----- 최후 fallback: 전체 frame 중 conf 최대 -----
    if not chosen_seeds:
        scores_all = Lc * Rc
        best_pos = int(np.argmax(scores_all))
        seed_t = int(common[best_pos])
        run = (seed_t, seed_t)
        chosen_seeds.append(_build_seed_point(seed_t, run, L, R, priors["barL"], common))
        chosen_runs.append((run, float(scores_all[best_pos])))
        fallback_used = "conf_max_only"

    res.seeds = chosen_seeds
    res.selected_runs = [(r[0], r[1], sc) for (r, sc) in chosen_runs]
    res.fallback_used = fallback_used

    # ----- video sanity flag -----
    # frame width 를 모르므로 ratio 는 None. bar_length 가 비현실적이면 flag.
    if priors["bar_median"] < 20 or priors["bar_median"] > 4000:
        res.video_flag = "bar_length_abnormal"

    return res


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help="batch_run_out 디렉토리 (각 영상 서브폴더에 CSV)")
    ap.add_argument("--input-glob", default=DEFAULT_GLOB)
    ap.add_argument("--out", default=None,
                    help="seeds JSON 출력 경로 (기본: <root>/auto_seeds.json)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    csvs = sorted(root.glob(f"*/{args.input_glob}"))
    if not csvs:
        csvs = sorted(root.glob(args.input_glob))
    if not csvs:
        print(f"[ERROR] no CSVs under {root} matching {args.input_glob}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else (root / "auto_seeds.json")
    results = {}
    for csv in csvs:
        clip_name = csv.parent.name
        try:
            res = select_seeds(csv)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[ERROR] {clip_name}: {e}", file=sys.stderr)
            continue
        results[clip_name] = res.to_json()

        # 콘솔 요약
        sd_str = ", ".join(
            f"t={s['t']} L=({s['Lx']:.1f},{s['Ly']:.1f}) R=({s['Rx']:.1f},{s['Ry']:.1f}) "
            f"[run {s['run_start']}..{s['run_end']} len={s['run_length']}]"
            for s in res.to_json()["seeds"]
        )
        print(f"[{clip_name}]")
        print(f"  frames={res.n_frames}  both_det={res.n_both_detected}  eligible={res.n_eligible}  "
              f"runs_used={len(res.selected_runs)}  fallback={res.fallback_used}  flag={res.video_flag}")
        bm = res.bar_length_median
        bcov = res.bar_length_cov
        print(f"  bar_median={bm:.1f}px  CoV={bcov:.4f}" if bm and bcov else f"  bar_median={bm}  CoV={bcov}")
        for i, s in enumerate(res.to_json()["seeds"]):
            print(f"  seed#{i}: t={s['t']}  L=({s['Lx']:.2f},{s['Ly']:.2f})  R=({s['Rx']:.2f},{s['Ry']:.2f})  "
                  f"L_conf={s['L_conf']:.3f} R_conf={s['R_conf']:.3f} bar={s['bar_length']:.1f}  "
                  f"[run {s['run_start']}..{s['run_end']} len={s['run_length']}]")
        if args.verbose:
            for (rs, re_, sc) in res.selected_runs:
                print(f"    chosen run [{rs}..{re_}] score={sc:.3f}")
        print()

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] seeds JSON -> {out_path}")


if __name__ == "__main__":
    main()
