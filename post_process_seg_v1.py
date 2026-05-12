#!/usr/bin/env python3
"""
post_process_seg_v1.py — inference_seg_v1.py 결과 selected CSV 후처리 (개선판).

post_process_v4_init.py 의 개선:
  [Tier 1]
    T1-1 Anchor 등급화 (Strong: 양쪽 detected / Weak: 양쪽 좌표만 valid)
    T1-2 봉 길이 prior (영상 전체 strong-anchor median 으로 강체 제약)
    T1-3 Anchor outlier filtering (bar length / 위치 시계열 / Y-align 3-pass)
    T1-4 is_interpolated 다단계 (0=detected, 1=kalman_pred, 2~5=pp_*)
  [Tier 2]
    T2-1 Plate size prior (W/H median 클램프)
    T2-2 봉 벡터 (magnitude, angle) 분해 처리 — 각도만 lerp, 크기는 prior
    T2-3 Side-swap (L.cx > R.cx) 자동 보정 + 로그
    T2-4 Final trajectory smoothing (옵션 플래그)
  [Tier 3]
    T3-1 Confidence proxy 부여 (NaN 대신 신뢰도 점수)
    T3-2 Frame area median 추정 (단일 행 의존 제거)
    T3-3 외삽 step magnitude 클램프
    T3-4 Anchor 0개 영상 fallback (warning 로그 + 원본 그대로)
    T3-5 확장 QC stats (bar_length_cov, outliers_demoted, side_swaps_fixed, …)

신뢰도 코드 (is_interpolated):
  0 = detected           실측 (YOLO)
  1 = kalman_pred        Kalman 예측 (inference 단계)
  2 = pp_bar_vector      한쪽 fill, 봉 강체 prior 사용
  3 = pp_lerp            양쪽 fill, 선형 보간
  4 = pp_extrap          경계 갭 등속 외삽
  5 = pp_freeze          외삽 한계, anchor freeze

Confidence proxy:
  detected      : 원본 유지 (보통 0.6~0.95)
  kalman_pred   : 0.5
  pp_bar_vector : 0.4
  pp_lerp       : 0.35
  pp_extrap     : 0.2
  pp_freeze     : 0.05

사용법:
  python post_process_seg_v1.py [root_dir]
  환경변수:
    POST_PROCESS_SEG_V1_SMOOTH = "off" | "filled_only" | "all"   (기본: off)
    POST_PROCESS_SEG_V1_INPUT_GLOB = "detections_selected_v9_v4_*.csv"
"""

from __future__ import annotations

import bisect
import glob
import os
import sys
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# =========================
# 경로/입출력 설정
# =========================
DEFAULT_ROOT = os.environ.get(
    "POST_PROCESS_SEG_V1_ROOT",
    "/Users/byeol/Desktop/inference/v4_init_얇은원판",
)
INPUT_GLOB = os.environ.get(
    "POST_PROCESS_SEG_V1_INPUT_GLOB",
    "detections_selected_v9_v4_*.csv",
)
OUTPUT_PREFIX = "detections_postproc_seg_v1_"

# =========================
# 후처리 동작 파라미터
# =========================

# 경계 외삽 최대 프레임
EXTRAP_MAX_FRAMES = 10

# T1-1: strong anchor 가 ±이 프레임 안에 있으면 우선 사용, 아니면 weak fallback
STRONG_GAP_MAX = 60

# T1-3: outlier 검출 임계 (k * MAD, MAD 는 std 환산 1.4826 곱)
BAR_LENGTH_MAD_K = 3.0       # 봉 길이 outlier
POS_RESIDUAL_MAD_K = 4.0     # 위치 시계열 residual
POS_ROLLING_WINDOW = 5       # 시계열 rolling median 윈도
Y_MISALIGN_RATIO_MAX = 0.30  # |L.cy - R.cy| / bar_length 상한 (이보다 큰 anchor 강등)

# T2-1: plate size 클램프 임계
SIZE_MAD_K = 3.0

# T3-3: 외삽 한 프레임당 이동량 상한 (화면폭/높이 비율)
EXTRAP_MAX_STEP_RATIO = 0.10

# T2-4: smoothing
SMOOTH_MODE = os.environ.get("POST_PROCESS_SEG_V1_SMOOTH", "off").lower()  # off | filled_only | all
SMOOTH_WINDOW = 5   # 홀수 권장
SMOOTH_POLYORDER = 2

# =========================
# Source / 신뢰도 코드
# =========================
SRC_DETECTED = "detected"
SRC_KALMAN = "kalman_pred"
SRC_ONE_BAR = "pp_bar_vector"
SRC_BOTH_LERP = "pp_lerp"
SRC_BOTH_EXTRAP = "pp_extrap"
SRC_BOTH_FREEZE = "pp_freeze"

INTERP_CODE = {
    SRC_DETECTED: 0,
    SRC_KALMAN: 1,
    SRC_ONE_BAR: 2,
    SRC_BOTH_LERP: 3,
    SRC_BOTH_EXTRAP: 4,
    SRC_BOTH_FREEZE: 5,
}

# T3-1: confidence proxy (fill 단계별)
CONF_PROXY = {
    SRC_KALMAN: 0.50,
    SRC_ONE_BAR: 0.40,
    SRC_BOTH_LERP: 0.35,
    SRC_BOTH_EXTRAP: 0.20,
    SRC_BOTH_FREEZE: 0.05,
}

# =========================
# 컬럼 정의
# =========================
COORD_COLS = ("center_x", "center_y", "width", "height")
SCORE_COLS_TO_NAN = ("assoc_score", "pred_dist_score", "size_score", "motion_score")


# =========================
# 기본 유틸
# =========================
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def safe_median(xs):
    arr = np.asarray([x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))],
                     dtype=float)
    if len(arr) == 0:
        return None
    return float(np.median(arr))


def safe_mad(xs, med: float) -> float:
    """std-환산 MAD (1.4826 * median(|x - med|))"""
    arr = np.asarray([x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))],
                     dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(arr - med)))


def has_valid_coords(side_df: pd.DataFrame, t: int) -> bool:
    if t not in side_df.index:
        return False
    row = side_df.loc[t]
    for c in COORD_COLS:
        v = row[c]
        if pd.isna(v):
            return False
    return True


def reconstruct_box(cx: float, cy: float, w: float, h: float):
    half_w = w / 2.0
    half_h = h / 2.0
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h, w * h


def find_anchor_neighbors(anchor_arr: np.ndarray, t: int) -> Tuple[Optional[int], Optional[int]]:
    """sorted 1D int array 에서 t 앞/뒤 anchor 인덱스 반환."""
    if len(anchor_arr) == 0:
        return None, None
    idx = bisect.bisect_left(anchor_arr, t)
    before = int(anchor_arr[idx - 1]) if idx > 0 else None
    if idx < len(anchor_arr) and int(anchor_arr[idx]) == t:
        after = int(anchor_arr[idx + 1]) if idx + 1 < len(anchor_arr) else None
    else:
        after = int(anchor_arr[idx]) if idx < len(anchor_arr) else None
    return before, after


def find_anchor_neighbors_tiered(
    strong_arr: np.ndarray,
    weak_arr: np.ndarray,
    t: int,
    strong_gap_max: int,
) -> Tuple[Optional[int], Optional[int]]:
    """
    T1-1: strong 을 우선 사용, ±strong_gap_max 밖이면 weak fallback.
    before / after 를 독립적으로 결정.
    """
    s_b, s_a = find_anchor_neighbors(strong_arr, t)
    use_strong_before = s_b is not None and (t - s_b) <= strong_gap_max
    use_strong_after  = s_a is not None and (s_a - t) <= strong_gap_max

    if use_strong_before and use_strong_after:
        return s_b, s_a

    w_b, w_a = find_anchor_neighbors(weak_arr, t)
    before = s_b if use_strong_before else w_b
    after  = s_a if use_strong_after  else w_a
    return before, after


# =========================
# T3-2: Frame area robust 추정
# =========================
def derive_frame_area_robust(side_dfs: List[pd.DataFrame]) -> Optional[float]:
    ratios = []
    for df_part in side_dfs:
        valid = df_part[
            df_part["area"].notna()
            & df_part["area_ratio"].notna()
            & (df_part["area_ratio"] > 0)
        ]
        if len(valid) > 0:
            ratios.extend((valid["area"].astype(float) / valid["area_ratio"].astype(float)).tolist())
    if not ratios:
        return None
    return float(np.median(ratios))


# =========================
# T1-1: Anchor 등급화
# =========================
def classify_anchors(L: pd.DataFrame, R: pd.DataFrame, frames: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    weak_arr : 양쪽 좌표만 valid (detected, kalman_pred 무관)
    strong_arr: 양쪽이 모두 source == 'detected' AND 좌표 valid
    """
    weak = []
    strong = []
    for t in frames:
        lv = has_valid_coords(L, t)
        rv = has_valid_coords(R, t)
        if not (lv and rv):
            continue
        weak.append(t)
        l_src = str(L.loc[t, "source"]) if "source" in L.columns else ""
        r_src = str(R.loc[t, "source"]) if "source" in R.columns else ""
        if l_src == SRC_DETECTED and r_src == SRC_DETECTED:
            strong.append(t)
    return np.array(strong, dtype=int), np.array(weak, dtype=int)


# =========================
# T1-3: Outlier filtering (3-pass)
# =========================
def filter_anchor_outliers(
    L: pd.DataFrame,
    R: pd.DataFrame,
    strong_arr: np.ndarray,
) -> Tuple[np.ndarray, Set[int], dict]:
    """
    Returns:
        cleaned_strong_arr  : 살아남은 strong anchor
        demoted_anchors_set : 강등된 anchor set
        diag                : 진단 dict
    """
    diag = {
        "pass1_bar_len_removed": 0,
        "pass2_pos_residual_removed": 0,
        "pass3_y_misalign_removed": 0,
        "bar_length_median": None,
        "bar_length_mad": None,
    }
    if len(strong_arr) < 4:
        # 통계 의존 필터링 불가 — 그대로 반환
        return strong_arr.copy(), set(), diag

    keep_mask = np.ones(len(strong_arr), dtype=bool)

    # ---------- Pass 1: 봉 길이 outlier ----------
    bar_lens = []
    for t in strong_arr:
        dx = float(R.loc[t, "center_x"]) - float(L.loc[t, "center_x"])
        dy = float(R.loc[t, "center_y"]) - float(L.loc[t, "center_y"])
        bar_lens.append(np.hypot(dx, dy))
    bar_lens = np.array(bar_lens)
    med_len = float(np.median(bar_lens))
    mad_len = float(1.4826 * np.median(np.abs(bar_lens - med_len)))
    diag["bar_length_median"] = med_len
    diag["bar_length_mad"] = mad_len

    if mad_len > 1e-6:
        threshold = BAR_LENGTH_MAD_K * mad_len
        bad = np.abs(bar_lens - med_len) > threshold
        diag["pass1_bar_len_removed"] = int(bad.sum())
        keep_mask &= ~bad

    # ---------- Pass 2: 위치 시계열 residual ----------
    # L.cx, L.cy, R.cx, R.cy 각각 rolling median 잔차 검사
    def _rolling_residuals(values: np.ndarray, window: int) -> np.ndarray:
        n = len(values)
        out = np.zeros(n)
        for i in range(n):
            lo = max(0, i - window // 2)
            hi = min(n, i + window // 2 + 1)
            local = values[lo:hi]
            out[i] = values[i] - np.median(local)
        return out

    cols = [
        ("L_cx", np.array([float(L.loc[t, "center_x"]) for t in strong_arr])),
        ("L_cy", np.array([float(L.loc[t, "center_y"]) for t in strong_arr])),
        ("R_cx", np.array([float(R.loc[t, "center_x"]) for t in strong_arr])),
        ("R_cy", np.array([float(R.loc[t, "center_y"]) for t in strong_arr])),
    ]
    pass2_bad = np.zeros(len(strong_arr), dtype=bool)
    for _, vals in cols:
        resid = _rolling_residuals(vals, POS_ROLLING_WINDOW)
        med_r = float(np.median(resid))
        mad_r = float(1.4826 * np.median(np.abs(resid - med_r)))
        if mad_r > 1e-6:
            threshold = POS_RESIDUAL_MAD_K * mad_r
            pass2_bad |= np.abs(resid - med_r) > threshold
    diag["pass2_pos_residual_removed"] = int((pass2_bad & keep_mask).sum())
    keep_mask &= ~pass2_bad

    # ---------- Pass 3: Y-misalignment ----------
    y_misalign = np.zeros(len(strong_arr), dtype=bool)
    for i, t in enumerate(strong_arr):
        dx = float(R.loc[t, "center_x"]) - float(L.loc[t, "center_x"])
        dy = float(R.loc[t, "center_y"]) - float(L.loc[t, "center_y"])
        length = max(np.hypot(dx, dy), 1.0)
        if abs(dy) / length > Y_MISALIGN_RATIO_MAX:
            y_misalign[i] = True
    diag["pass3_y_misalign_removed"] = int((y_misalign & keep_mask).sum())
    keep_mask &= ~y_misalign

    cleaned = strong_arr[keep_mask]
    demoted = set(strong_arr[~keep_mask].tolist())
    return cleaned, demoted, diag


# =========================
# T1-2, T2-1: prior 계산
# =========================
def compute_priors(
    L: pd.DataFrame,
    R: pd.DataFrame,
    strong_arr: np.ndarray,
    weak_arr: np.ndarray,
) -> dict:
    """
    strong 이 충분히 있으면 strong 기준, 아니면 weak 기준으로 fallback.
    """
    use_arr = strong_arr if len(strong_arr) >= 3 else weak_arr
    if len(use_arr) == 0:
        return {
            "bar_length_median": None,
            "bar_length_mad": 0.0,
            "L_W_median": None,
            "L_H_median": None,
            "R_W_median": None,
            "R_H_median": None,
            "L_W_mad": 0.0,
            "L_H_mad": 0.0,
            "R_W_mad": 0.0,
            "R_H_mad": 0.0,
            "prior_source": "none",
        }

    lens = []
    L_w, L_h, R_w, R_h = [], [], [], []
    for t in use_arr:
        dx = float(R.loc[t, "center_x"]) - float(L.loc[t, "center_x"])
        dy = float(R.loc[t, "center_y"]) - float(L.loc[t, "center_y"])
        lens.append(np.hypot(dx, dy))
        L_w.append(float(L.loc[t, "width"]))
        L_h.append(float(L.loc[t, "height"]))
        R_w.append(float(R.loc[t, "width"]))
        R_h.append(float(R.loc[t, "height"]))

    bar_med = float(np.median(lens))
    bar_mad = float(1.4826 * np.median(np.abs(np.array(lens) - bar_med)))
    return {
        "bar_length_median": bar_med,
        "bar_length_mad": bar_mad,
        "L_W_median": float(np.median(L_w)),
        "L_H_median": float(np.median(L_h)),
        "R_W_median": float(np.median(R_w)),
        "R_H_median": float(np.median(R_h)),
        "L_W_mad": float(1.4826 * np.median(np.abs(np.array(L_w) - np.median(L_w)))),
        "L_H_mad": float(1.4826 * np.median(np.abs(np.array(L_h) - np.median(L_h)))),
        "R_W_mad": float(1.4826 * np.median(np.abs(np.array(R_w) - np.median(R_w)))),
        "R_H_mad": float(1.4826 * np.median(np.abs(np.array(R_h) - np.median(R_h)))),
        "prior_source": "strong" if len(strong_arr) >= 3 else "weak",
    }


def clamp_size(value: float, median: Optional[float], mad: float, k: float = SIZE_MAD_K) -> float:
    """T2-1: size 가 median ± k*mad 밖이면 median 으로 클램프."""
    if median is None:
        return value
    if mad <= 1e-6:
        return value
    if abs(value - median) > k * mad:
        return median
    return value


# =========================
# T1-2, T2-2: bar vector (magnitude/angle 분해)
# =========================
def _to_polar(dx: float, dy: float) -> Tuple[float, float]:
    return float(np.hypot(dx, dy)), float(np.arctan2(dy, dx))


def _from_polar(mag: float, ang: float) -> Tuple[float, float]:
    return mag * float(np.cos(ang)), mag * float(np.sin(ang))


def _shortest_arc_lerp(ang_a: float, ang_b: float, alpha: float) -> float:
    diff = ang_b - ang_a
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return float(ang_a + diff * alpha)


def interpolated_bar_vector_v2(
    L: pd.DataFrame,
    R: pd.DataFrame,
    before: Optional[int],
    after: Optional[int],
    t: int,
    bar_length_prior: Optional[float],
) -> Optional[Tuple[float, float]]:
    """
    T2-2: 봉 벡터를 (magnitude, angle) 분해해서 처리.
    magnitude 는 bar_length_prior 가 있으면 그것으로 강제 (T1-2 강체 제약).
    angle 은 shortest-arc lerp.
    """
    def v_at(s):
        return (
            float(R.loc[s, "center_x"]) - float(L.loc[s, "center_x"]),
            float(R.loc[s, "center_y"]) - float(L.loc[s, "center_y"]),
        )

    if before is not None and after is not None and after != before:
        alpha = (t - before) / (after - before)
        mag_b, ang_b = _to_polar(*v_at(before))
        mag_a, ang_a = _to_polar(*v_at(after))
        ang_t = _shortest_arc_lerp(ang_b, ang_a, alpha)
        mag_t = bar_length_prior if bar_length_prior is not None else lerp(mag_b, mag_a, alpha)
        return _from_polar(mag_t, ang_t)

    if before is not None:
        mag_b, ang_b = _to_polar(*v_at(before))
        mag_t = bar_length_prior if bar_length_prior is not None else mag_b
        return _from_polar(mag_t, ang_b)

    if after is not None:
        mag_a, ang_a = _to_polar(*v_at(after))
        mag_t = bar_length_prior if bar_length_prior is not None else mag_a
        return _from_polar(mag_t, ang_a)

    return None


# =========================
# Size 보간 (+ T2-1 prior 클램프)
# =========================
def interpolated_size_with_prior(
    side_df: pd.DataFrame,
    before: Optional[int],
    after: Optional[int],
    t: int,
    W_median: Optional[float],
    H_median: Optional[float],
    W_mad: float,
    H_mad: float,
    use_prior_freeze: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """
    use_prior_freeze=True 면 W/H 를 무조건 median (extrap/freeze 용).
    아니면 보간 후 median ± k*MAD 밖이면 median 으로 클램프.
    """
    if use_prior_freeze and W_median is not None and H_median is not None:
        return float(W_median), float(H_median)

    if before is not None and after is not None and after != before:
        alpha = (t - before) / (after - before)
        w = lerp(float(side_df.loc[before, "width"]),  float(side_df.loc[after, "width"]),  alpha)
        h = lerp(float(side_df.loc[before, "height"]), float(side_df.loc[after, "height"]), alpha)
    elif before is not None:
        w = float(side_df.loc[before, "width"])
        h = float(side_df.loc[before, "height"])
    elif after is not None:
        w = float(side_df.loc[after, "width"])
        h = float(side_df.loc[after, "height"])
    else:
        return None, None

    w = clamp_size(w, W_median, W_mad)
    h = clamp_size(h, H_median, H_mad)
    return w, h


# =========================
# 위치 보간 (양쪽 결측, T3-3 외삽 step 클램프)
# =========================
def interpolated_position(
    side_df: pd.DataFrame,
    anchor_arr: np.ndarray,
    before: Optional[int],
    after: Optional[int],
    t: int,
    frame_W: Optional[float],
    frame_H: Optional[float],
) -> Tuple[Optional[float], Optional[float], str]:
    """양쪽 결측 시 side 의 center_x, center_y 보간/외삽 (T3-3 step 클램프 포함)."""

    if before is not None and after is not None and after != before:
        alpha = (t - before) / (after - before)
        cx = lerp(float(side_df.loc[before, "center_x"]), float(side_df.loc[after, "center_x"]), alpha)
        cy = lerp(float(side_df.loc[before, "center_y"]), float(side_df.loc[after, "center_y"]), alpha)
        return cx, cy, SRC_BOTH_LERP

    # 경계: before 만 있음 (= t 가 마지막 anchor 이후)
    if before is not None:
        a1 = before
        # before 직전 anchor 를 찾아 등속 외삽
        idx = int(np.searchsorted(anchor_arr, a1))
        a0 = int(anchor_arr[idx - 1]) if idx >= 1 else None
        gap = t - a1

        if a0 is not None and a0 != a1 and gap <= EXTRAP_MAX_FRAMES:
            dt = max(a1 - a0, 1)
            vx_raw = (float(side_df.loc[a1, "center_x"]) - float(side_df.loc[a0, "center_x"])) / dt
            vy_raw = (float(side_df.loc[a1, "center_y"]) - float(side_df.loc[a0, "center_y"])) / dt
            # T3-3: per-frame step magnitude 클램프
            if frame_W is not None:
                vx_raw = float(np.clip(vx_raw, -EXTRAP_MAX_STEP_RATIO * frame_W,
                                       +EXTRAP_MAX_STEP_RATIO * frame_W))
            if frame_H is not None:
                vy_raw = float(np.clip(vy_raw, -EXTRAP_MAX_STEP_RATIO * frame_H,
                                       +EXTRAP_MAX_STEP_RATIO * frame_H))
            cx = float(side_df.loc[a1, "center_x"]) + vx_raw * gap
            cy = float(side_df.loc[a1, "center_y"]) + vy_raw * gap
            return cx, cy, SRC_BOTH_EXTRAP

        # 외삽 불가 → freeze
        return (float(side_df.loc[a1, "center_x"]),
                float(side_df.loc[a1, "center_y"]),
                SRC_BOTH_FREEZE)

    # 경계: after 만 있음 (= t 가 첫 anchor 이전)
    if after is not None:
        a1 = after
        idx = int(np.searchsorted(anchor_arr, a1))
        a0 = int(anchor_arr[idx + 1]) if idx + 1 < len(anchor_arr) else None
        gap = a1 - t

        if a0 is not None and a0 != a1 and gap <= EXTRAP_MAX_FRAMES:
            dt = max(a0 - a1, 1)
            vx_raw = (float(side_df.loc[a0, "center_x"]) - float(side_df.loc[a1, "center_x"])) / dt
            vy_raw = (float(side_df.loc[a0, "center_y"]) - float(side_df.loc[a1, "center_y"])) / dt
            if frame_W is not None:
                vx_raw = float(np.clip(vx_raw, -EXTRAP_MAX_STEP_RATIO * frame_W,
                                       +EXTRAP_MAX_STEP_RATIO * frame_W))
            if frame_H is not None:
                vy_raw = float(np.clip(vy_raw, -EXTRAP_MAX_STEP_RATIO * frame_H,
                                       +EXTRAP_MAX_STEP_RATIO * frame_H))
            cx = float(side_df.loc[a1, "center_x"]) - vx_raw * gap
            cy = float(side_df.loc[a1, "center_y"]) - vy_raw * gap
            return cx, cy, SRC_BOTH_EXTRAP

        return (float(side_df.loc[a1, "center_x"]),
                float(side_df.loc[a1, "center_y"]),
                SRC_BOTH_FREEZE)

    return None, None, ""


# =========================
# Fill row 작성 (T1-4 is_interpolated 다단계, T3-1 confidence proxy)
# =========================
def write_filled_row(
    side_df: pd.DataFrame,
    t: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    src: str,
    frame_area_total: Optional[float],
    class_id_default: int,
    class_name_default,
):
    x1, y1, x2, y2, area = reconstruct_box(cx, cy, w, h)
    side_df.at[t, "center_x"] = cx
    side_df.at[t, "center_y"] = cy
    side_df.at[t, "width"] = w
    side_df.at[t, "height"] = h
    side_df.at[t, "x1"] = x1
    side_df.at[t, "y1"] = y1
    side_df.at[t, "x2"] = x2
    side_df.at[t, "y2"] = y2
    side_df.at[t, "area"] = area
    if frame_area_total is not None and frame_area_total > 0:
        side_df.at[t, "area_ratio"] = area / frame_area_total
    if h > 1e-6:
        side_df.at[t, "aspect_ratio"] = w / h
    side_df.at[t, "source"] = src
    # T1-4 다단계 신뢰도
    side_df.at[t, "is_interpolated"] = INTERP_CODE.get(src, 1)
    # T3-1 confidence proxy
    if src in CONF_PROXY:
        side_df.at[t, "confidence"] = CONF_PROXY[src]
    # inference 단계 점수들은 fill 행에 의미 없음 — NaN
    for c in SCORE_COLS_TO_NAN:
        if c in side_df.columns:
            side_df.at[t, c] = np.nan
    # class 정보 누락 보강
    if pd.isna(side_df.at[t, "class_id"]) or side_df.at[t, "class_id"] == -1:
        side_df.at[t, "class_id"] = class_id_default
    if pd.isna(side_df.at[t, "class_name"]) or side_df.at[t, "class_name"] == "":
        side_df.at[t, "class_name"] = class_name_default


def update_existing_rows_metadata(L: pd.DataFrame, R: pd.DataFrame):
    """
    T1-4: 이미 존재하는 detected / kalman_pred 행의 is_interpolated 를 다단계 코드로 갱신.
    T3-1: kalman_pred 행에 confidence proxy 부여 (NaN 일 경우).
    """
    for df in (L, R):
        if "source" not in df.columns:
            continue
        for t in df.index:
            src = df.at[t, "source"]
            if pd.isna(src):
                continue
            src = str(src)
            if src in INTERP_CODE:
                df.at[t, "is_interpolated"] = INTERP_CODE[src]
            if src == SRC_KALMAN:
                # 원본 inference 는 confidence=NaN → proxy 부여
                c = df.at[t, "confidence"]
                if pd.isna(c):
                    df.at[t, "confidence"] = CONF_PROXY[SRC_KALMAN]


# =========================
# T2-3: Side-swap (L.cx > R.cx) 자동 보정
# =========================
def correct_side_swaps(L: pd.DataFrame, R: pd.DataFrame, frames: List[int]) -> List[int]:
    """
    매 프레임 L.cx <= R.cx 검증. 위반 시 두 행 전체를 교환.
    반환: 교환된 frame_idx 리스트.
    """
    swapped = []
    swappable_cols = [c for c in L.columns if c not in ("frame_idx", "side", "track_id")]
    for t in frames:
        if t not in L.index or t not in R.index:
            continue
        lcx = L.at[t, "center_x"]
        rcx = R.at[t, "center_x"]
        if pd.isna(lcx) or pd.isna(rcx):
            continue
        if float(lcx) > float(rcx):
            for c in swappable_cols:
                if c not in R.columns:
                    continue
                lv = L.at[t, c]
                rv = R.at[t, c]
                L.at[t, c] = rv
                R.at[t, c] = lv
            swapped.append(t)
    return swapped


# =========================
# T2-4: Final trajectory smoothing
# =========================
def _moving_average_centered(arr: np.ndarray, window: int) -> np.ndarray:
    """홀수 window 의 중앙 평균. NaN 이 섞이면 그 구간은 NaN 유지."""
    n = len(arr)
    if window <= 1 or n < window:
        return arr.copy()
    half = window // 2
    out = arr.copy().astype(float)
    for i in range(half, n - half):
        chunk = arr[i - half:i + half + 1]
        if np.any(np.isnan(chunk)):
            out[i] = arr[i]
        else:
            out[i] = float(np.mean(chunk))
    return out


def _savgol_or_ma(arr: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """scipy 가 있으면 Savitzky-Golay, 없으면 moving average."""
    try:
        from scipy.signal import savgol_filter
        if len(arr) < window or window <= polyorder:
            return arr.copy()
        # NaN 은 그대로 두고 나머지에만 적용
        if np.any(np.isnan(arr)):
            return _moving_average_centered(arr, window)
        return np.array(savgol_filter(arr, window, polyorder))
    except ImportError:
        return _moving_average_centered(arr, window)


def smooth_trajectory(
    L: pd.DataFrame,
    R: pd.DataFrame,
    frames: List[int],
    mode: str,
    frame_area_total: Optional[float],
):
    """
    mode = "off"           : 그대로
    mode = "filled_only"   : pp_* 행만 부드럽게 (이미 보간이라 영향 적음)
    mode = "all"           : detected/kalman_pred/pp_* 모두 부드럽게

    center_x/y, width/height 모두 smoothing 후 box 재구성.
    """
    if mode == "off":
        return

    for df in (L, R):
        for col in ("center_x", "center_y", "width", "height"):
            if col not in df.columns:
                continue
            vals = df.sort_index()[col].to_numpy(dtype=float)
            if mode == "filled_only":
                # detected (0) 와 kalman_pred (1) 행 보호
                is_interp = df.sort_index()["is_interpolated"].to_numpy()
                smoothed = _savgol_or_ma(vals, SMOOTH_WINDOW, SMOOTH_POLYORDER)
                # filled (>=2) 만 교체
                mask = (is_interp >= 2)
                vals_out = vals.copy()
                vals_out[mask] = smoothed[mask]
            else:  # "all"
                vals_out = _savgol_or_ma(vals, SMOOTH_WINDOW, SMOOTH_POLYORDER)

            # 다시 df 에 기록 (sort_index 순서로 작성됐으므로 동일 정렬 사용)
            sorted_idx = df.sort_index().index
            for i, t in enumerate(sorted_idx):
                df.at[t, col] = float(vals_out[i])

        # box 재구성
        for t in df.index:
            cx, cy = df.at[t, "center_x"], df.at[t, "center_y"]
            w, h = df.at[t, "width"], df.at[t, "height"]
            if pd.isna(cx) or pd.isna(cy) or pd.isna(w) or pd.isna(h):
                continue
            x1, y1, x2, y2, area = reconstruct_box(float(cx), float(cy), float(w), float(h))
            df.at[t, "x1"], df.at[t, "y1"] = x1, y1
            df.at[t, "x2"], df.at[t, "y2"] = x2, y2
            df.at[t, "area"] = area
            if frame_area_total is not None and frame_area_total > 0:
                df.at[t, "area_ratio"] = area / frame_area_total
            if float(h) > 1e-6:
                df.at[t, "aspect_ratio"] = float(w) / float(h)


# =========================
# 메인 CSV 후처리
# =========================
def post_process_csv(csv_path: str) -> Tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    df["frame_idx"] = df["frame_idx"].astype(int)

    # split L/R, frame_idx 인덱스
    L = df[df["side"] == "L"].copy().set_index("frame_idx", drop=False).sort_index()
    R = df[df["side"] == "R"].copy().set_index("frame_idx", drop=False).sort_index()

    frames = sorted(set(L.index) | set(R.index))

    stats = {
        "input_rows": len(df),
        "frames": len(frames),
        "weak_anchors": 0,
        "strong_anchors_raw": 0,
        "strong_anchors_clean": 0,
        "outliers_demoted": 0,
        "pass1_bar_len_removed": 0,
        "pass2_pos_residual_removed": 0,
        "pass3_y_misalign_removed": 0,
        "fill_one_bar": 0,
        "fill_two_lerp": 0,
        "fill_two_extrap": 0,
        "fill_two_freeze": 0,
        "side_swaps_fixed": 0,
        "still_nan": 0,
        "bar_length_median": None,
        "bar_length_cov": None,
        "L_W_median": None, "L_H_median": None,
        "R_W_median": None, "R_H_median": None,
        "prior_source": None,
        "smooth_mode": SMOOTH_MODE,
    }

    # T1-1: anchor 분류
    strong_arr_raw, weak_arr = classify_anchors(L, R, frames)
    stats["strong_anchors_raw"] = int(len(strong_arr_raw))
    stats["weak_anchors"] = int(len(weak_arr))

    # T3-4: anchor 0 fallback
    if len(weak_arr) == 0:
        print(f"  [WARN] no anchors at all — leaving NaN as-is")
        out = pd.concat([L.reset_index(drop=True), R.reset_index(drop=True)], ignore_index=True)
        out = out.sort_values(["frame_idx", "side"]).reset_index(drop=True)
        stats["still_nan"] = int(((out["center_x"].isna()) | (out["center_y"].isna())).sum())
        return out, stats

    # T1-3: outlier filtering on strong anchors
    strong_arr, demoted_set, outlier_diag = filter_anchor_outliers(L, R, strong_arr_raw)
    stats["strong_anchors_clean"] = int(len(strong_arr))
    stats["outliers_demoted"] = int(len(demoted_set))
    stats["pass1_bar_len_removed"] = outlier_diag["pass1_bar_len_removed"]
    stats["pass2_pos_residual_removed"] = outlier_diag["pass2_pos_residual_removed"]
    stats["pass3_y_misalign_removed"] = outlier_diag["pass3_y_misalign_removed"]

    # T1-2 / T2-1: prior 계산 (cleaned strong 우선)
    priors = compute_priors(L, R, strong_arr, weak_arr)
    stats["bar_length_median"] = priors["bar_length_median"]
    if priors["bar_length_median"] and priors["bar_length_median"] > 0:
        stats["bar_length_cov"] = priors["bar_length_mad"] / priors["bar_length_median"]
    stats["L_W_median"] = priors["L_W_median"]
    stats["L_H_median"] = priors["L_H_median"]
    stats["R_W_median"] = priors["R_W_median"]
    stats["R_H_median"] = priors["R_H_median"]
    stats["prior_source"] = priors["prior_source"]

    # T3-2: frame area
    frame_area_total = derive_frame_area_robust([L, R])
    frame_W = None
    frame_H = None
    # frame_W/H 추정: width 컬럼이 box width 라서 직접 못 얻음.
    # area_total = W * H 이고, aspect 정보가 없음 → 16:9 가정은 위험.
    # 대신 EXTRAP step 클램프용으로 sqrt(area_total) 정도의 대표 길이만 쓰는 게 안전.
    if frame_area_total is not None and frame_area_total > 0:
        # 비율 미지 → 정사각형 가정에 가까운 보수적 추정 (양쪽 클램프에 동일하게 작동)
        frame_W = frame_H = float(np.sqrt(frame_area_total))

    # anchor 0개 (이론상 일어나기 어려움) 또는 사용 가능한 anchor 가 없는 경우 처리
    if len(weak_arr) == 0:
        out = pd.concat([L.reset_index(drop=True), R.reset_index(drop=True)], ignore_index=True)
        out = out.sort_values(["frame_idx", "side"]).reset_index(drop=True)
        stats["still_nan"] = int(((out["center_x"].isna()) | (out["center_y"].isna())).sum())
        return out, stats

    # class default — strong (or weak) anchor 첫 행
    base_arr = strong_arr if len(strong_arr) > 0 else weak_arr
    anchor_t0 = int(base_arr[0])
    L_class_id = int(L.loc[anchor_t0, "class_id"]) if not pd.isna(L.loc[anchor_t0, "class_id"]) else 0
    L_class_name = L.loc[anchor_t0, "class_name"]
    R_class_id = int(R.loc[anchor_t0, "class_id"]) if not pd.isna(R.loc[anchor_t0, "class_id"]) else 0
    R_class_name = R.loc[anchor_t0, "class_name"]

    # T1-4: 기존 행 metadata 갱신 (detected/kalman_pred 의 is_interpolated 코드 통일)
    update_existing_rows_metadata(L, R)

    # ===== 메인 fill 루프 =====
    bar_prior = priors["bar_length_median"]

    for t in frames:
        Lv = has_valid_coords(L, t)
        Rv = has_valid_coords(R, t)

        if Lv and Rv:
            continue  # anchor — 채움 불필요

        # before/after anchor 결정 (strong 우선, weak fallback)
        before, after = find_anchor_neighbors_tiered(
            strong_arr, weak_arr, t, STRONG_GAP_MAX
        )

        if not Lv and not Rv:
            # 양쪽 결측: 각각 독립 보간
            Lcx, Lcy, src_pos_l = interpolated_position(
                L, weak_arr, before, after, t, frame_W, frame_H
            )
            Rcx, Rcy, src_pos_r = interpolated_position(
                R, weak_arr, before, after, t, frame_W, frame_H
            )

            # source 는 더 보수적인 쪽을 채택
            # (extrap > freeze 보수성: freeze > extrap > lerp 순서로 보수)
            order = {SRC_BOTH_FREEZE: 3, SRC_BOTH_EXTRAP: 2, SRC_BOTH_LERP: 1, "": 0}
            src_chosen = src_pos_l if order.get(src_pos_l, 0) >= order.get(src_pos_r, 0) else src_pos_r

            # size: T2-1 prior 적용
            use_freeze = (src_chosen == SRC_BOTH_FREEZE)
            Lw, Lh = interpolated_size_with_prior(
                L, before, after, t,
                priors["L_W_median"], priors["L_H_median"],
                priors["L_W_mad"], priors["L_H_mad"],
                use_prior_freeze=use_freeze,
            )
            Rw, Rh = interpolated_size_with_prior(
                R, before, after, t,
                priors["R_W_median"], priors["R_H_median"],
                priors["R_W_mad"], priors["R_H_mad"],
                use_prior_freeze=use_freeze,
            )

            if Lcx is None or Lw is None or Rcx is None or Rw is None:
                continue

            write_filled_row(L, t, Lcx, Lcy, Lw, Lh, src_chosen, frame_area_total,
                             L_class_id, L_class_name)
            write_filled_row(R, t, Rcx, Rcy, Rw, Rh, src_chosen, frame_area_total,
                             R_class_id, R_class_name)

            if src_chosen == SRC_BOTH_LERP:
                stats["fill_two_lerp"] += 1
            elif src_chosen == SRC_BOTH_EXTRAP:
                stats["fill_two_extrap"] += 1
            else:
                stats["fill_two_freeze"] += 1

        elif Lv and not Rv:
            # 한쪽 fill: 봉 벡터 (방향 lerp + 크기 prior)
            v = interpolated_bar_vector_v2(L, R, before, after, t, bar_prior)
            Rw, Rh = interpolated_size_with_prior(
                R, before, after, t,
                priors["R_W_median"], priors["R_H_median"],
                priors["R_W_mad"], priors["R_H_mad"],
            )
            if v is None or Rw is None:
                continue
            Lcx_t = float(L.loc[t, "center_x"])
            Lcy_t = float(L.loc[t, "center_y"])
            Rcx = Lcx_t + v[0]
            Rcy = Lcy_t + v[1]
            write_filled_row(R, t, Rcx, Rcy, Rw, Rh, SRC_ONE_BAR, frame_area_total,
                             R_class_id, R_class_name)
            stats["fill_one_bar"] += 1

        else:  # not Lv and Rv
            v = interpolated_bar_vector_v2(L, R, before, after, t, bar_prior)
            Lw, Lh = interpolated_size_with_prior(
                L, before, after, t,
                priors["L_W_median"], priors["L_H_median"],
                priors["L_W_mad"], priors["L_H_mad"],
            )
            if v is None or Lw is None:
                continue
            Rcx_t = float(R.loc[t, "center_x"])
            Rcy_t = float(R.loc[t, "center_y"])
            Lcx = Rcx_t - v[0]
            Lcy = Rcy_t - v[1]
            write_filled_row(L, t, Lcx, Lcy, Lw, Lh, SRC_ONE_BAR, frame_area_total,
                             L_class_id, L_class_name)
            stats["fill_one_bar"] += 1

    # ===== T2-3 Side-swap 보정 =====
    swapped_frames = correct_side_swaps(L, R, frames)
    stats["side_swaps_fixed"] = len(swapped_frames)

    # ===== T2-4 Final smoothing =====
    smooth_trajectory(L, R, frames, SMOOTH_MODE, frame_area_total)

    # ===== 결과 정리 =====
    out = pd.concat([L.reset_index(drop=True), R.reset_index(drop=True)], ignore_index=True)
    out = out.sort_values(["frame_idx", "side"]).reset_index(drop=True)
    stats["still_nan"] = int(((out["center_x"].isna()) | (out["center_y"].isna())).sum())
    return out, stats


# =========================
# 출력 파일 경로
# =========================
def output_path_for(in_csv: str) -> str:
    base = os.path.basename(in_csv)
    suffix = base.replace("detections_selected_v9_v4_", "").replace(".csv", "")
    out_name = f"{OUTPUT_PREFIX}{suffix}.csv"
    return os.path.join(os.path.dirname(in_csv), out_name)


# =========================
# main
# =========================
def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    csvs = sorted(glob.glob(os.path.join(root, "*", INPUT_GLOB)))
    if not csvs:
        # 평면 디렉토리도 시도
        csvs = sorted(glob.glob(os.path.join(root, INPUT_GLOB)))
    if not csvs:
        print(f"[ERROR] no input CSVs found under {root}/* (or {root}/) matching {INPUT_GLOB}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] post_process_seg_v1: processing {len(csvs)} CSV(s)")
    print(f"[INFO] root           : {root}")
    print(f"[INFO] input pattern  : {INPUT_GLOB}")
    print(f"[INFO] output prefix  : {OUTPUT_PREFIX}")
    print(f"[INFO] smooth mode    : {SMOOTH_MODE}")
    print(f"[INFO] EXTRAP_MAX_FRAMES        : {EXTRAP_MAX_FRAMES}")
    print(f"[INFO] STRONG_GAP_MAX           : {STRONG_GAP_MAX}")
    print(f"[INFO] BAR_LENGTH_MAD_K         : {BAR_LENGTH_MAD_K}")
    print(f"[INFO] POS_RESIDUAL_MAD_K       : {POS_RESIDUAL_MAD_K}")
    print(f"[INFO] Y_MISALIGN_RATIO_MAX     : {Y_MISALIGN_RATIO_MAX}")
    print(f"[INFO] SIZE_MAD_K               : {SIZE_MAD_K}")
    print(f"[INFO] EXTRAP_MAX_STEP_RATIO    : {EXTRAP_MAX_STEP_RATIO}")
    print()

    grand_keys = [
        "input_rows", "frames", "weak_anchors",
        "strong_anchors_raw", "strong_anchors_clean", "outliers_demoted",
        "pass1_bar_len_removed", "pass2_pos_residual_removed", "pass3_y_misalign_removed",
        "fill_one_bar", "fill_two_lerp", "fill_two_extrap", "fill_two_freeze",
        "side_swaps_fixed", "still_nan",
    ]
    grand = {k: 0 for k in grand_keys}

    for csv_path in csvs:
        out_df, stats = post_process_csv(csv_path)
        out_path = output_path_for(csv_path)
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        rel_dir = os.path.basename(os.path.dirname(csv_path)) or os.path.basename(csv_path)
        print(f"[{rel_dir}]")
        print(f"  rows={stats['input_rows']}  frames={stats['frames']}")
        print(f"  anchors: weak={stats['weak_anchors']}  "
              f"strong_raw={stats['strong_anchors_raw']}  "
              f"strong_clean={stats['strong_anchors_clean']}  "
              f"(demoted: total={stats['outliers_demoted']}, "
              f"bar={stats['pass1_bar_len_removed']}, "
              f"pos={stats['pass2_pos_residual_removed']}, "
              f"y={stats['pass3_y_misalign_removed']})")
        bm = stats['bar_length_median']
        bcov = stats['bar_length_cov']
        bar_str = f"median={bm:.1f}px  cov={bcov:.4f}" if bm is not None and bcov is not None \
                  else f"median={bm}  cov={bcov}"
        print(f"  prior source={stats['prior_source']}  bar_length: {bar_str}")
        if stats['L_W_median'] is not None:
            print(f"    L plate ~ ({stats['L_W_median']:.1f} x {stats['L_H_median']:.1f})  "
                  f"R plate ~ ({stats['R_W_median']:.1f} x {stats['R_H_median']:.1f})")
        print(f"  fill: one_bar={stats['fill_one_bar']}  "
              f"two_lerp={stats['fill_two_lerp']}  "
              f"two_extrap={stats['fill_two_extrap']}  "
              f"two_freeze={stats['fill_two_freeze']}")
        print(f"  side_swaps_fixed={stats['side_swaps_fixed']}  still_nan={stats['still_nan']}")
        print(f"  -> {out_path}")
        print()

        for k in grand:
            grand[k] += stats[k]

    print("===== summary =====")
    for k in grand_keys:
        print(f"  {k}: {grand[k]}")


if __name__ == "__main__":
    main()
