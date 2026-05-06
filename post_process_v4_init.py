#!/usr/bin/env python3
"""
inference_rule_v9_v4_init.py 결과 selected CSV 후처리.

목표: 모든 프레임의 양쪽(L, R) 원판 좌표가 채워진 CSV 생성.

채움 규칙:
  - anchor frame = 양쪽 모두 valid 좌표(center_x, center_y, width, height 모두 not-NaN)인 프레임
                   (source 가 detected / kalman_pred / 후처리 fill 무관, 좌표 존재 여부만 본다)
  - 한쪽만 결측: 가장 가까운 앞/뒤 anchor 의 봉 벡터(R - L)를 보간하여 visible 쪽에 더해 채움
                양 anchor 가 모두 있으면 선형 보간, 한쪽만 있으면 그쪽 freeze
  - 양쪽 결측:  앞/뒤 anchor 좌표를 좌·우 각각 선형 보간
                한쪽 anchor 만 있는 경계 갭은 가까운 두 anchor 의 속도로 EXTRAP_MAX_FRAMES 까지 외삽,
                그 이상은 가장 가까운 anchor 좌표로 freeze

폭/높이는 anchor 들의 값을 동일 방식(선형 보간)으로 채우며, x1/y1/x2/y2/area 는 center+size 로 재구성.

채워진 행은 source 컬럼에 다음 값 중 하나가 들어간다:
  - pp_bar_vector   : 한쪽 결측 → 봉 벡터 fill
  - pp_lerp         : 양쪽 결측 → 두 anchor 사이 선형 보간
  - pp_extrap       : 경계 갭 외삽 (≤ EXTRAP_MAX_FRAMES)
  - pp_freeze       : 외삽 한계를 넘는 경계 갭 → anchor 좌표 동결

is_interpolated 는 모든 채움 행에 대해 1 로 설정.

사용법:
  python post_process_v4_init.py [root_dir]
  디폴트 root_dir = /Users/byeol/Desktop/inference/v4_init_얇은원판
"""

from __future__ import annotations

import bisect
import glob
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_ROOT = "/Users/byeol/Desktop/inference/v4_init_얇은원판"
INPUT_GLOB = "detections_selected_v9_v4_*.csv"
OUTPUT_PREFIX = "detections_postproc_v9_v4_"

EXTRAP_MAX_FRAMES = 10  # 경계 외삽 최대 프레임

SRC_ONE_BAR = "pp_bar_vector"
SRC_BOTH_LERP = "pp_lerp"
SRC_BOTH_EXTRAP = "pp_extrap"
SRC_BOTH_FREEZE = "pp_freeze"

COORD_COLS = ("center_x", "center_y", "width", "height")
BOX_COLS = ("x1", "y1", "x2", "y2", "width", "height", "center_x", "center_y", "area")
NULL_SCORE_COLS = ("confidence", "assoc_score", "pred_dist_score", "size_score", "motion_score")


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def has_valid_coords(side_df: pd.DataFrame, t: int) -> bool:
    if t not in side_df.index:
        return False
    row = side_df.loc[t]
    for c in COORD_COLS:
        v = row[c]
        if pd.isna(v):
            return False
    return True


def derive_frame_area(side_dfs: List[pd.DataFrame]) -> Optional[float]:
    """area / area_ratio 가 모두 valid 한 첫 행에서 frame_W * frame_H 추정."""
    for df_part in side_dfs:
        valid = df_part[
            df_part["area"].notna()
            & df_part["area_ratio"].notna()
            & (df_part["area_ratio"] > 0)
        ]
        if len(valid) > 0:
            r = valid.iloc[0]
            return float(r["area"]) / float(r["area_ratio"])
    return None


def reconstruct_box(cx: float, cy: float, w: float, h: float):
    half_w = w / 2.0
    half_h = h / 2.0
    x1 = cx - half_w
    y1 = cy - half_h
    x2 = cx + half_w
    y2 = cy + half_h
    area = w * h
    return x1, y1, x2, y2, area


def write_filled_row(side_df: pd.DataFrame, t: int, cx: float, cy: float,
                     w: float, h: float, src: str, frame_area_total: Optional[float],
                     class_id_default: int, class_name_default):
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
    side_df.at[t, "is_interpolated"] = 1
    # source / 점수 / conf 는 더 이상 의미 없음 — None 으로
    for c in NULL_SCORE_COLS:
        if c in side_df.columns:
            side_df.at[t, c] = np.nan
    # class 정보가 비어 있으면 anchor 의 값을 채움
    if pd.isna(side_df.at[t, "class_id"]) or side_df.at[t, "class_id"] == -1:
        side_df.at[t, "class_id"] = class_id_default
    if pd.isna(side_df.at[t, "class_name"]) or side_df.at[t, "class_name"] == "":
        side_df.at[t, "class_name"] = class_name_default


def find_anchor_neighbors(anchor_arr: np.ndarray, t: int) -> Tuple[Optional[int], Optional[int]]:
    """anchor_arr (정렬된 1D int array) 에서 t 의 앞/뒤 anchor 인덱스 반환."""
    idx = bisect.bisect_left(anchor_arr, t)
    before = int(anchor_arr[idx - 1]) if idx > 0 else None
    if idx < len(anchor_arr) and int(anchor_arr[idx]) == t:
        # t 가 anchor 인 경우 (정상 흐름에서는 호출되지 않음)
        after = int(anchor_arr[idx + 1]) if idx + 1 < len(anchor_arr) else None
    else:
        after = int(anchor_arr[idx]) if idx < len(anchor_arr) else None
    return before, after


def interpolated_bar_vector(L: pd.DataFrame, R: pd.DataFrame,
                            before: Optional[int], after: Optional[int],
                            t: int) -> Optional[Tuple[float, float]]:
    """앞/뒤 anchor 의 봉 벡터 (R-L) 를 t 시점에 맞춰 선형 보간."""
    def v_at(s):
        return (R.loc[s, "center_x"] - L.loc[s, "center_x"],
                R.loc[s, "center_y"] - L.loc[s, "center_y"])

    if before is not None and after is not None and after != before:
        a = (t - before) / (after - before)
        v_b = v_at(before)
        v_a = v_at(after)
        return lerp(v_b[0], v_a[0], a), lerp(v_b[1], v_a[1], a)
    if before is not None:
        return v_at(before)
    if after is not None:
        return v_at(after)
    return None


def interpolated_size(side_df: pd.DataFrame,
                      before: Optional[int], after: Optional[int],
                      t: int) -> Tuple[Optional[float], Optional[float]]:
    if before is not None and after is not None and after != before:
        a = (t - before) / (after - before)
        return (lerp(side_df.loc[before, "width"], side_df.loc[after, "width"], a),
                lerp(side_df.loc[before, "height"], side_df.loc[after, "height"], a))
    if before is not None:
        return float(side_df.loc[before, "width"]), float(side_df.loc[before, "height"])
    if after is not None:
        return float(side_df.loc[after, "width"]), float(side_df.loc[after, "height"])
    return None, None


def interpolated_position(side_df: pd.DataFrame, anchor_arr: np.ndarray,
                          before: Optional[int], after: Optional[int],
                          t: int) -> Tuple[Optional[float], Optional[float], str]:
    """양쪽 결측 시 side 의 center_x, center_y 보간/외삽."""
    if before is not None and after is not None and after != before:
        a = (t - before) / (after - before)
        return (lerp(side_df.loc[before, "center_x"], side_df.loc[after, "center_x"], a),
                lerp(side_df.loc[before, "center_y"], side_df.loc[after, "center_y"], a),
                SRC_BOTH_LERP)

    if before is not None:
        # t > 모든 anchor 또는 before==after
        a1 = before
        idx = int(np.searchsorted(anchor_arr, a1))
        a0 = int(anchor_arr[idx - 1]) if idx >= 1 else None
        gap = t - a1
        if a0 is not None and a0 != a1 and gap <= EXTRAP_MAX_FRAMES:
            dt = max(a1 - a0, 1)
            steps = gap / dt
            cx = float(side_df.loc[a1, "center_x"]) + (
                float(side_df.loc[a1, "center_x"]) - float(side_df.loc[a0, "center_x"])
            ) * steps
            cy = float(side_df.loc[a1, "center_y"]) + (
                float(side_df.loc[a1, "center_y"]) - float(side_df.loc[a0, "center_y"])
            ) * steps
            return cx, cy, SRC_BOTH_EXTRAP
        return (float(side_df.loc[a1, "center_x"]),
                float(side_df.loc[a1, "center_y"]),
                SRC_BOTH_FREEZE)

    if after is not None:
        a1 = after
        idx = int(np.searchsorted(anchor_arr, a1))
        a0 = int(anchor_arr[idx + 1]) if idx + 1 < len(anchor_arr) else None
        gap = a1 - t
        if a0 is not None and a0 != a1 and gap <= EXTRAP_MAX_FRAMES:
            dt = max(a0 - a1, 1)
            steps = -gap / dt  # 뒤쪽 segment 의 속도를 역방향으로 적용
            cx = float(side_df.loc[a1, "center_x"]) + (
                float(side_df.loc[a0, "center_x"]) - float(side_df.loc[a1, "center_x"])
            ) * steps
            cy = float(side_df.loc[a1, "center_y"]) + (
                float(side_df.loc[a0, "center_y"]) - float(side_df.loc[a1, "center_y"])
            ) * steps
            return cx, cy, SRC_BOTH_EXTRAP
        return (float(side_df.loc[a1, "center_x"]),
                float(side_df.loc[a1, "center_y"]),
                SRC_BOTH_FREEZE)

    return None, None, ""


def post_process_csv(csv_path: str) -> Tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    df["frame_idx"] = df["frame_idx"].astype(int)

    # split L/R
    L = df[df["side"] == "L"].copy().set_index("frame_idx", drop=False).sort_index()
    R = df[df["side"] == "R"].copy().set_index("frame_idx", drop=False).sort_index()

    frames = sorted(set(L.index) | set(R.index))

    anchor_arr = np.array(
        [t for t in frames if has_valid_coords(L, t) and has_valid_coords(R, t)],
        dtype=int,
    )

    stats = {
        "input_rows": len(df),
        "frames": len(frames),
        "anchors": int(len(anchor_arr)),
        "fill_one_bar": 0,
        "fill_two_lerp": 0,
        "fill_two_extrap": 0,
        "fill_two_freeze": 0,
        "still_nan": 0,
    }

    if len(anchor_arr) == 0:
        out = pd.concat([L.reset_index(drop=True), R.reset_index(drop=True)], ignore_index=True)
        out = out.sort_values(["frame_idx", "side"]).reset_index(drop=True)
        stats["still_nan"] = int(
            ((out["center_x"].isna()) | (out["center_y"].isna())).sum()
        )
        return out, stats

    frame_area_total = derive_frame_area([L, R])

    # class default — anchor 의 첫 행에서
    anchor_t0 = int(anchor_arr[0])
    L_class_id = int(L.loc[anchor_t0, "class_id"])
    L_class_name = L.loc[anchor_t0, "class_name"]
    R_class_id = int(R.loc[anchor_t0, "class_id"])
    R_class_name = R.loc[anchor_t0, "class_name"]

    for t in frames:
        Lv = has_valid_coords(L, t)
        Rv = has_valid_coords(R, t)

        if Lv and Rv:
            continue

        before, after = find_anchor_neighbors(anchor_arr, t)

        if not Lv and not Rv:
            Lcx, Lcy, src_pos = interpolated_position(L, anchor_arr, before, after, t)
            Rcx, Rcy, _ = interpolated_position(R, anchor_arr, before, after, t)
            Lw, Lh = interpolated_size(L, before, after, t)
            Rw, Rh = interpolated_size(R, before, after, t)

            if Lcx is None or Lw is None or Rcx is None or Rw is None:
                continue

            write_filled_row(L, t, Lcx, Lcy, Lw, Lh, src_pos, frame_area_total,
                             L_class_id, L_class_name)
            write_filled_row(R, t, Rcx, Rcy, Rw, Rh, src_pos, frame_area_total,
                             R_class_id, R_class_name)
            if src_pos == SRC_BOTH_LERP:
                stats["fill_two_lerp"] += 1
            elif src_pos == SRC_BOTH_EXTRAP:
                stats["fill_two_extrap"] += 1
            else:
                stats["fill_two_freeze"] += 1

        elif Lv and not Rv:
            v = interpolated_bar_vector(L, R, before, after, t)
            Rw, Rh = interpolated_size(R, before, after, t)
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
            v = interpolated_bar_vector(L, R, before, after, t)
            Lw, Lh = interpolated_size(L, before, after, t)
            if v is None or Lw is None:
                continue
            Rcx_t = float(R.loc[t, "center_x"])
            Rcy_t = float(R.loc[t, "center_y"])
            Lcx = Rcx_t - v[0]
            Lcy = Rcy_t - v[1]
            write_filled_row(L, t, Lcx, Lcy, Lw, Lh, SRC_ONE_BAR, frame_area_total,
                             L_class_id, L_class_name)
            stats["fill_one_bar"] += 1

    out = pd.concat([L.reset_index(drop=True), R.reset_index(drop=True)], ignore_index=True)
    out = out.sort_values(["frame_idx", "side"]).reset_index(drop=True)

    stats["still_nan"] = int(
        ((out["center_x"].isna()) | (out["center_y"].isna())).sum()
    )
    return out, stats


def output_path_for(in_csv: str) -> str:
    base = os.path.basename(in_csv)
    suffix = base.replace("detections_selected_v9_v4_", "").replace(".csv", "")
    out_name = f"{OUTPUT_PREFIX}{suffix}.csv"
    return os.path.join(os.path.dirname(in_csv), out_name)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    csvs = sorted(glob.glob(os.path.join(root, "*", INPUT_GLOB)))
    if not csvs:
        print(f"[ERROR] no input CSVs found under {root}/*/", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] processing {len(csvs)} CSVs under {root}")
    print()

    grand = {"input_rows": 0, "frames": 0, "anchors": 0,
             "fill_one_bar": 0, "fill_two_lerp": 0, "fill_two_extrap": 0,
             "fill_two_freeze": 0, "still_nan": 0}

    for csv_path in csvs:
        out_df, stats = post_process_csv(csv_path)
        out_path = output_path_for(csv_path)
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        rel_dir = os.path.basename(os.path.dirname(csv_path))
        print(f"[{rel_dir}]")
        print(f"  rows={stats['input_rows']}  frames={stats['frames']}  anchors={stats['anchors']}")
        print(f"  fill: one_bar={stats['fill_one_bar']}  two_lerp={stats['fill_two_lerp']}"
              f"  two_extrap={stats['fill_two_extrap']}  two_freeze={stats['fill_two_freeze']}")
        print(f"  still_nan_rows={stats['still_nan']}")
        print(f"  -> {out_path}")
        print()

        for k in grand:
            grand[k] += stats[k]

    print("===== summary =====")
    for k, v in grand.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
