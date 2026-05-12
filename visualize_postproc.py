#!/usr/bin/env python3
"""
post_process_seg_v1.py 결과 CSV 를 원본 mp4 영상에 다시 오버레이해 시각화 영상 생성.

입력 : batch_run_out/<영상명>/detections_postproc_seg_v1_*.csv
출력 : batch_run_out/<영상명>/annotated_video_postproc_seg_v1_*.mp4

표시 내용:
  - L (좌) / R (우) 각 원판 박스 + 중심점
  - L-R 중심을 잇는 봉 라인
  - 소스별 색상 / 두께 차등 (detected 가장 진한 색 + 두꺼움, pp_freeze 가장 옅음 + 얇음)
  - 상단 헤더: frame_idx, 좌·우 source, bar length, 신뢰도 코드

색상 규칙 (BGR):
  detected      : L = 강한 초록 (60,220,80)   R = 강한 청록 (240,180,40)   thickness=3
  kalman_pred   : L = 노랑초록 (90,220,180)   R = 청록           (220,200,60) thickness=3
  pp_bar_vector : L = 옅은 초록 (140,220,180) R = 옅은 청록      (220,210,140) thickness=2
  pp_lerp       : L = 보라기   (220,140,220)  R = 자홍           (220,80,220)  thickness=2
  pp_extrap     : L = 주황     (60,140,255)   R = 짙은 주황      (40,90,220)   thickness=2
  pp_freeze     : L = 회색     (160,160,160)  R = 회색           (160,160,160) thickness=2 (점선)
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VIDEO_DIR = Path("/Users/byeol/Desktop/by/4-1/빅데이터 캡스톤/배경 제거/얇은원판")
RUN_OUT_DIR = Path("/Users/byeol/Desktop/by/4-1/빅데이터 캡스톤/배경 제거/batch_run_out")
OUTPUT_PREFIX = "annotated_video_postproc_seg_v1_"
INPUT_CSV_GLOB = "detections_postproc_seg_v1_*.csv"

# (L_color, R_color, thickness, dashed)
STYLE_BY_SOURCE = {
    "detected":      ((60, 220, 80),   (240, 180, 40),  3, False),
    "kalman_pred":   ((90, 220, 180),  (220, 200, 60),  3, False),
    "pp_bar_vector": ((140, 220, 180), (220, 210, 140), 2, False),
    "pp_lerp":       ((220, 140, 220), (220, 80, 220),  2, False),
    "pp_extrap":     ((60, 140, 255),  (40, 90, 220),   2, False),
    "pp_freeze":     ((160, 160, 160), (160, 160, 160), 2, True),
}
DEFAULT_STYLE = ((180, 180, 180), (180, 180, 180), 1, True)

# 헤더 박스 배경/글자
HDR_BG = (0, 0, 0)
HDR_FG = (255, 255, 255)


def get_style(source: str, side: str):
    sty = STYLE_BY_SOURCE.get(str(source), DEFAULT_STYLE)
    color = sty[0] if side == "L" else sty[1]
    thickness = sty[2]
    dashed = sty[3]
    return color, thickness, dashed


def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash=8):
    x1, y1 = pt1
    x2, y2 = pt2
    # top / bottom
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness, cv2.LINE_AA)
    # left / right
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness, cv2.LINE_AA)


def draw_side(vis, row):
    if pd.isna(row.get("center_x")) or pd.isna(row.get("center_y")):
        return
    cx, cy = float(row["center_x"]), float(row["center_y"])
    w  = float(row["width"]) if not pd.isna(row.get("width")) else 0.0
    h  = float(row["height"]) if not pd.isna(row.get("height")) else 0.0
    if w <= 0 or h <= 0:
        return
    side = row["side"]
    src = str(row.get("source", ""))

    color, thickness, dashed = get_style(src, side)

    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))

    if dashed:
        draw_dashed_rect(vis, (x1, y1), (x2, y2), color, thickness)
    else:
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # center dot
    cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, color, -1, cv2.LINE_AA)

    # label
    conf = row.get("confidence")
    conf_txt = "" if pd.isna(conf) else f" conf={float(conf):.2f}"
    interp = row.get("is_interpolated")
    interp_txt = "" if pd.isna(interp) else f" i={int(interp)}"
    label = f"{side} {src}{conf_txt}{interp_txt}"

    text_pos = (x1, max(20, y1 - 6))
    # 글자 가독성용 검은 외곽선 → 본문
    cv2.putText(vis, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color, 1, cv2.LINE_AA)


def draw_bar_line(vis, lrow, rrow):
    if lrow is None or rrow is None:
        return
    if (pd.isna(lrow.get("center_x")) or pd.isna(lrow.get("center_y"))
            or pd.isna(rrow.get("center_x")) or pd.isna(rrow.get("center_y"))):
        return
    p1 = (int(round(float(lrow["center_x"]))), int(round(float(lrow["center_y"]))))
    p2 = (int(round(float(rrow["center_x"]))), int(round(float(rrow["center_y"]))))
    # 두 행의 source 중 더 보수적인 색 사용
    src_l, src_r = str(lrow.get("source", "")), str(rrow.get("source", ""))
    fallback_priority = {
        "detected": 0, "kalman_pred": 1, "pp_bar_vector": 2,
        "pp_lerp": 3, "pp_extrap": 4, "pp_freeze": 5,
    }
    src = src_l if fallback_priority.get(src_l, 9) >= fallback_priority.get(src_r, 9) else src_r
    color = STYLE_BY_SOURCE.get(src, DEFAULT_STYLE)[0]
    cv2.line(vis, p1, p2, color, 2, cv2.LINE_AA)


def draw_header(vis, frame_idx, lrow, rrow, total_frames):
    H, W = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (W, 56), HDR_BG, -1)

    src_l = str(lrow.get("source", "?")) if lrow is not None else "?"
    src_r = str(rrow.get("source", "?")) if rrow is not None else "?"

    bar_len = float("nan")
    if (lrow is not None and rrow is not None
            and not pd.isna(lrow.get("center_x")) and not pd.isna(rrow.get("center_x"))):
        dx = float(rrow["center_x"]) - float(lrow["center_x"])
        dy = float(rrow["center_y"]) - float(lrow["center_y"])
        bar_len = float(np.hypot(dx, dy))

    header1 = f"frame={frame_idx}/{total_frames-1}  L={src_l}  R={src_r}"
    if not np.isnan(bar_len):
        header1 += f"  bar={bar_len:.1f}px"
    cv2.putText(vis, header1, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, HDR_FG, 2, cv2.LINE_AA)

    # legend (color sample lines)
    legend = "[detected/kalman/bar_vec/lerp/extrap/freeze]"
    cv2.putText(vis, legend, (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)


def process_one(video_path: Path, csv_path: Path, out_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df["frame_idx"] = df["frame_idx"].astype(int)
    # quick lookup: dict[frame_idx] = {"L": row, "R": row}
    lookup: dict = {}
    for _, r in df.iterrows():
        d = lookup.setdefault(int(r["frame_idx"]), {})
        d[r["side"]] = r

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": video_path.name, "error": "cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        return {"video": video_path.name, "error": "VideoWriter open failed (mp4v unsupported?)"}

    frame_idx = 0
    frames_written = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        rows = lookup.get(frame_idx, {})
        lrow = rows.get("L")
        rrow = rows.get("R")

        vis = frame.copy()
        # 봉 라인 먼저 그리고 그 위에 박스 (선이 박스 뒤로 깔리게)
        draw_bar_line(vis, lrow, rrow)
        if lrow is not None:
            draw_side(vis, lrow)
        if rrow is not None:
            draw_side(vis, rrow)
        draw_header(vis, frame_idx, lrow, rrow, total)

        writer.write(vis)
        frame_idx += 1
        frames_written += 1

    cap.release()
    writer.release()

    return {
        "video": video_path.name,
        "frames_written": frames_written,
        "csv_frames": int(df["frame_idx"].nunique()),
        "video_total": total,
        "fps": fps,
        "size": f"{W}x{H}",
        "out": str(out_path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default=str(RUN_OUT_DIR),
                   help="batch_run_out 디렉토리 (각 영상별 서브폴더에 CSV 가 있어야 함)")
    p.add_argument("--video-dir", default=str(VIDEO_DIR), help="원본 mp4 들이 있는 디렉토리")
    args = p.parse_args()

    root = Path(args.root)
    video_dir = Path(args.video_dir)
    if not root.exists():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        sys.exit(1)

    subs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not subs:
        print(f"[ERROR] no subdirectories under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] visualize {len(subs)} clip(s)")
    print()

    results = []
    for sub in subs:
        name = sub.name
        video_path = video_dir / f"{name}.mp4"
        if not video_path.exists():
            print(f"[skip] no source video for: {name}")
            continue
        csvs = sorted(sub.glob(INPUT_CSV_GLOB))
        if not csvs:
            print(f"[skip] no postproc CSV in: {sub}")
            continue
        csv_path = csvs[0]
        suffix = csv_path.stem.replace("detections_postproc_seg_v1_", "")
        out_path = sub / f"{OUTPUT_PREFIX}{suffix}.mp4"

        print(f"[{name}]")
        r = process_one(video_path, csv_path, out_path)
        results.append(r)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  frames_written={r['frames_written']}  "
                  f"csv_frames={r['csv_frames']}  video_total={r['video_total']}  "
                  f"fps={r['fps']:.2f}  size={r['size']}")
            print(f"  -> {r['out']}")
        print()

    print("===== summary =====")
    ok = [r for r in results if "error" not in r]
    fail = [r for r in results if "error" in r]
    print(f"  success: {len(ok)}")
    print(f"  failed : {len(fail)}")
    total_frames = sum(r.get("frames_written", 0) for r in ok)
    print(f"  total frames written: {total_frames}")


if __name__ == "__main__":
    main()
