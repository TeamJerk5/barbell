import glob
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


# =========================
# 사용자 설정 (플랫폼 자동 감지 디폴트)
# =========================
def _default_path(win_path: str, posix_fallback: str):
    if platform.system() == "Windows":
        return win_path
    return posix_fallback


MODEL_PATH = os.environ.get(
    "INFERENCE_RULE_V9_MODEL_PATH",
    _default_path(
        r"/Users/byeol/Desktop/inference/best_yolo26m_4data_77.pt",
        os.path.expanduser("~/Desktop/inference/best.pt"),
    ),
)
VIDEO_PATH = os.environ.get(
    "INFERENCE_RULE_V9_VIDEO_PATH",
    _default_path(
        r"/Users/byeol/Desktop/inference/KNSF_day3_34.mp4",
        os.path.expanduser("~/Desktop/inference/KNSF_day3_34.mp4"),
    ),
)
OUTPUT_DIR = os.environ.get(
    "INFERENCE_MOTION_V2_OUTPUT_DIR",
    _default_path(
        r"/Users/byeol/Desktop/inference/v2/109_sna_motion_v2",
        os.path.expanduser("~/Desktop/inference/inference_motion_v2_out"),
    ),
)

# =========================
# v1-motion: Temporal Median 기반 배경 darkening
# =========================
# v4 위에 inference 전 단계에서 motion-based 전경/배경 분리를 추가한다.
# 방식:
#   1) 영상 시작 시 N 프레임 균등 샘플 → 채널별 median = "정적 배경" 영상
#   2) 매 프레임에서 |frame - median| → 픽셀별 motion likelihood
#   3) 임계+soft range 로 0..1 weight (이진 마스크 아님 — 정지 봉 보호)
#   4) det_frame = frame * (alpha + (1-alpha)*weight)
#      → 배경 픽셀은 alpha 배 어둡게, 전경 픽셀은 원본 유지
#   5) plate YOLO 에는 det_frame 입력, 시각화/optical flow 는 원본 frame 사용
#
# seg_v1 과의 차이:
#   - seg 는 "선수 영역만 보존" (의미 기반)
#   - motion 은 "정적 픽셀만 darkening" (시간 기반)
#   - motion 의 강점: 모델 의존성 0, 배경 정적 plate 제거 더 강력
#   - motion 의 약점: 카메라 흔들림에 취약 (본 데이터셋은 OK)
USE_MOTION_BG_DARKEN = True
# 영상에서 균등 샘플링할 프레임 개수.
# 6초 영상 기준: 12개면 ~0.5초마다 1개 → 락아웃 자세가 ≤ 절반에 머묾.
MEDIAN_N_SAMPLES = 12
# diff 임계 (이 값 이하면 배경으로 간주)
MOTION_DIFF_THRESHOLD = 18
# soft range — diff 가 THRESHOLD ~ THRESHOLD+SOFT_RANGE 사이면 weight 0→1 부드럽게.
# 정지 봉이 살짝만 움직여도 점진적 보존을 받도록.
MOTION_DIFF_SOFT_RANGE = 50.0
# 배경 픽셀 (motion=0) 곱셈 계수. seg_v1 과 동일하게 0.25.
MOTION_BG_DARKEN_ALPHA = 0.25
# foreground mask 가장자리 dilation (morph) — 봉 가장자리 보존 + 노이즈 흡수
MOTION_FG_DILATE_PX = 5

# 전역 탐색용 conf
# v2 retune: motion darkening 이 배경 FP 를 깎았으므로 진짜 sub-threshold 회수.
GLOBAL_CONF_THRES = 0.15        # v1: 0.25
# 추적 중 local 보조 탐색용 conf
LOCAL_CONF_THRES = 0.08

IOU_THRES = 0.50
IMGSZ = 1280

SAVE_VIDEO = True
SAVE_FIRST_FRAME = False
SKIP_COMPLETED_THRESHOLDS = False
FORCE_TRANSCODE_TO_H264 = False

# =========================
# 최소 bbox sanity check
# =========================
MIN_BOX_SIZE_PX = 4.0

# =========================
# 트래킹 관련 (v4: 60fps 기준 재조정)
# =========================
MAX_MISSED_FRAMES = 5            # v4: 10→5, kalman_pred fill 한도 (~83ms)
SIDE_REACQUIRE_FRAMES = 6        # v4: 12→6, kalman_pred 직후 즉시 그쪽 재시도
MAX_REACQUIRE_FRAMES = 9         # v4: 18→9, SIDE+3 갭 (~150ms 만에 reset 사이클)

# local gating (v4: 3-tier)
LOCAL_GATE_DIST_RATIO = 0.18         # v4: 0.16→0.18 narrow tier 약간 완화
MID_GATE_DIST_RATIO = 0.24           # v4 신규: narrow(0.18)와 wide(0.32) 사이
REACQUIRE_GATE_DIST_RATIO = 0.32     # v4: 0.28→0.32 wide tier 빠른 동작 대비
LOCAL_GATE_Y_RATIO = 0.20            # 그대로
LOCAL_GATE_X_RATIO = 0.25            # 그대로

# motion 관련
STATIC_BG_MOTION_THRESH_RATIO = 0.0025
VERY_LARGE_JUMP_RATIO = 0.23

# Kalman / tracking
KALMAN_PROCESS_NOISE = 0.03
KALMAN_MEASURE_NOISE = 0.20

# 광류 보조
USE_OPTICAL_FLOW_ASSIST = True
OPTICAL_FLOW_MAX_CORNERS = 20
OPTICAL_FLOW_WIN_SIZE = (21, 21)
OPTICAL_FLOW_MAX_LEVEL = 3
# Optical flow를 Kalman measurement처럼 다룰 때의 가중 계수 (0=무시, 1=완전 신뢰)
OPTICAL_FLOW_BLEND_ALPHA = 0.5

# =========================
# 초기화 / 재획득 pair score
# v2 retune: motion darkening 후 conf 가 더 신뢰할 수 있는 신호 → conf 가중 ↑.
# motion 의 경우 seg 만큼 ROI 중앙성을 강제하지 않으므로 W_CENTER_SYM 은
# seg_v2 (0.9→0.6) 보다 약하게 (0.9→0.7) 축소.
# =========================
W_CONF = 1.7              # v1: 1.3
W_HEIGHT_SIM = 1.0
W_Y_ALIGN = 1.8
W_CENTER_SYM = 0.7        # v1: 0.9 — motion 은 seg 보다 보수적 축소
MIN_ACCEPTABLE_PAIR_SCORE = 1.5   # v1: 1.8

# =========================
# 초기 단일 detection mirror-search fallback
# =========================
# global 추론이 한 객체만 잡고 init이 막혔을 때, frame_cx 반대편에
# 낮은 conf 로 ROI 추론을 1회 돌려 파트너를 찾아 init 시도.
INIT_FALLBACK_ENABLED = True
INIT_FALLBACK_ROI_W_RATIO = 3.0       # ROI 폭 = single_det.width * 비율
INIT_FALLBACK_ROI_H_RATIO = 2.5       # ROI 높이 = single_det.height * 비율
INIT_FALLBACK_ROI_MIN_W = 240.0
INIT_FALLBACK_ROI_MIN_H = 200.0
INIT_FALLBACK_HEIGHT_SIM_MIN = 0.45   # v1: 0.55 — 완화
INIT_FALLBACK_Y_DIFF_RATIO_MAX = 0.25 # v1: 0.20 — 완화

# =========================
# 좌/우 개별 association score
# =========================
W_ASSOC_CONF = 1.5        # v1: 1.2 — conf 가중 ↑
W_ASSOC_PRED_DIST = 3.8
W_ASSOC_SIZE = 1.0
W_ASSOC_Y = 1.2
W_ASSOC_MOTION = 1.4
W_ASSOC_EDGE = 0.5
MIN_ASSOC_SCORE = -2.0

# 디버깅 로그
LOG_EVERY_N_FRAMES = 20

# =========================
# Local inference gating (A + E) (v4: 강화)
# =========================
LOCAL_INFERENCE_IMGSZ = 640                # E. local 전용 imgsz
LOCAL_TRIGGER_LOW_CONF = 0.40              # v1: 0.55 — 불필요 local 호출 감소
LOCAL_TRIGGER_FORCE_EVERY = 3              # v4: 5→3, 안전판 50ms/회 (60fps)
LOCAL_TRIGGER_PRED_DIST_RATIO = 0.07       # v4: 0.06→0.07, 게이트 완화에 맞춤

# =========================
# Velocity-aware ROI margin (v4 신규, 🅔)
# =========================
ROI_VELOCITY_LOOKAHEAD = 2.0      # frames forward 만큼 velocity 반영
ROI_MAX_MARGIN_RATIO = 1.5        # margin 폭 상한 (roi_w/h 대비) — 폭주 방지

# =========================
# CSV chunked flush
# =========================
CSV_FLUSH_EVERY_FRAMES = 2000


RAW_COLUMNS = [
    "frame_idx", "time_sec", "det_idx", "class_id", "class_name", "confidence",
    "x1", "y1", "x2", "y2", "width", "height", "center_x", "center_y", "area",
    "area_ratio", "aspect_ratio", "final_conf_thres"
]

SELECTED_COLUMNS = [
    "frame_idx", "time_sec", "side", "track_id", "source",
    "class_id", "class_name", "confidence",
    "x1", "y1", "x2", "y2", "width", "height", "center_x", "center_y", "area",
    "area_ratio", "aspect_ratio", "final_conf_thres",
    "assoc_score", "pred_dist_score", "size_score", "motion_score",
    "is_interpolated", "missed_count"
]

PAIR_SCORE_COLUMNS = [
    "frame_idx",
    "left_det_idx", "right_det_idx",
    "pair_score_total",
    "conf_score", "height_score",
    "y_align_score", "center_sym_score",
    "left_cx", "left_cy", "right_cx", "right_cy",
]


# =========================
# 유틸
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def log(msg: str, log_path: str):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def reopen_capture_at_frame(video_path: str, target_frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    if target_frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return None, None

    return cap, frame


def format_conf_suffix(global_conf: float, local_conf: float):
    return f"g{global_conf:.2f}_l{local_conf:.2f}"


def get_inference_device():
    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_inference_device_if_needed(inference_device):
    if torch.cuda.is_available():
        if inference_device == 0:
            torch.cuda.synchronize()
            return
        if isinstance(inference_device, str) and inference_device.startswith("cuda"):
            torch.cuda.synchronize()
            return

    if inference_device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def build_output_paths(output_dir: str, global_conf: float, local_conf: float):
    suffix = format_conf_suffix(global_conf, local_conf)
    return {
        "raw_csv_path": os.path.join(output_dir, f"detections_raw_v9_v4_{suffix}.csv"),
        "selected_csv_path": os.path.join(output_dir, f"detections_selected_v9_v4_{suffix}.csv"),
        "pairs_csv_path": os.path.join(output_dir, f"pairs_scored_v9_v4_{suffix}.csv"),
        "annotated_video_path": os.path.join(output_dir, f"annotated_video_original_v9_v4_{suffix}.mp4"),
        "annotated_video_rule_path": os.path.join(output_dir, f"annotated_video_tracking_rule_v9_v4_{suffix}.mp4"),
        "first_frame_path": os.path.join(output_dir, f"first_frame_v9_v4_{suffix}.jpg"),
        "readable_video_path": os.path.join(output_dir, "input_readable_h264_v9_v4.mp4"),
        "log_path": os.path.join(output_dir, f"run_log_v9_v4_{suffix}.txt"),
    }


def frame_diag(width: int, height: int):
    return float(np.hypot(width, height))


def similarity_ratio(v1: float, v2: float):
    denom = max(v1, v2, 1e-6)
    return min(v1, v2) / denom


def center_distance_xy(x1: float, y1: float, x2: float, y2: float):
    return float(np.hypot(x1 - x2, y1 - y2))


def center_distance(det_a: dict, det_b: dict):
    return center_distance_xy(det_a["center_x"], det_a["center_y"], det_b["center_x"], det_b["center_y"])


def clip_box(x1, y1, x2, y2, width, height):
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    return x1, y1, x2, y2


def enrich_detection_row(row: dict, frame_width: int, frame_height: int) -> dict:
    area_ratio = row["area"] / max(frame_width * frame_height, 1)
    aspect_ratio = row["width"] / max(row["height"], 1e-6)
    row["area_ratio"] = area_ratio
    row["aspect_ratio"] = aspect_ratio
    return row


def detection_is_candidate(det: dict) -> bool:
    if det["width"] < MIN_BOX_SIZE_PX or det["height"] < MIN_BOX_SIZE_PX:
        return False
    if det["x2"] <= det["x1"] or det["y2"] <= det["y1"]:
        return False
    return True


# =========================
# Kalman helper
# =========================
class SimpleKalman2D:
    """
    state: [cx, cy, vx, vy]
    measurement: [cx, cy]
    """
    def __init__(self, process_noise=0.03, measure_noise=0.20):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measure_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def init(self, cx: float, cy: float, vx: float = 0.0, vy: float = 0.0):
        self.kf.statePost = np.array([[cx], [cy], [vx], [vy]], dtype=np.float32)
        self.initialized = True

    def predict(self) -> Tuple[float, float, float, float]:
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0]), float(pred[2, 0]), float(pred[3, 0])

    def correct(self, cx: float, cy: float) -> Tuple[float, float, float, float]:
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        est = self.kf.correct(measurement)
        return float(est[0, 0]), float(est[1, 0]), float(est[2, 0]), float(est[3, 0])

    def set_position(self, cx: float, cy: float):
        """state vector의 위치 성분만 갱신 (속도/공분산 유지)."""
        if not self.initialized:
            self.init(cx, cy)
            return
        self.kf.statePost[0, 0] = np.float32(cx)
        self.kf.statePost[1, 0] = np.float32(cy)


# =========================
# Track state
# =========================
@dataclass
class SideTrack:
    side: str
    track_id: int
    det: Optional[dict] = None
    prev_det: Optional[dict] = None
    last_measured_det: Optional[dict] = None     # ③ 마지막 실측 (size 보존용)
    missed_frames: int = 0
    initialized: bool = False
    kalman: SimpleKalman2D = field(default_factory=lambda: SimpleKalman2D(
        process_noise=KALMAN_PROCESS_NOISE,
        measure_noise=KALMAN_MEASURE_NOISE,
    ))
    pred_cx: Optional[float] = None
    pred_cy: Optional[float] = None
    pred_vx: float = 0.0
    pred_vy: float = 0.0

    def predict(self):
        if not self.initialized or not self.kalman.initialized:
            self.pred_cx = self.det["center_x"] if self.det is not None else None
            self.pred_cy = self.det["center_y"] if self.det is not None else None
            self.pred_vx = 0.0
            self.pred_vy = 0.0
            return self.pred_cx, self.pred_cy, self.pred_vx, self.pred_vy

        cx, cy, vx, vy = self.kalman.predict()
        self.pred_cx = cx
        self.pred_cy = cy
        self.pred_vx = vx
        self.pred_vy = vy
        return cx, cy, vx, vy

    def update_with_det(self, det: dict):
        prev = self.det.copy() if self.det is not None else None
        self.prev_det = prev
        self.det = det.copy()
        self.last_measured_det = det.copy()

        if not self.kalman.initialized:
            vx = 0.0
            vy = 0.0
            if prev is not None:
                vx = det["center_x"] - prev["center_x"]
                vy = det["center_y"] - prev["center_y"]
            self.kalman.init(det["center_x"], det["center_y"], vx, vy)
        else:
            self.kalman.correct(det["center_x"], det["center_y"])

        self.missed_frames = 0
        self.initialized = True

    def mark_missed(self):
        """
        v3 결함 ① 수정: 매칭 실패 1회를 카운트.
        update_with_prediction 호출 여부와 무관하게 매 매칭 실패에서 호출되어야 한다.
        이것이 분리되지 않으면 missed_frames 가 MAX_MISSED_FRAMES 에서 천장에 막혀
        SIDE_REACQUIRE_FRAMES (>=12) / 양측 reset (>10) 조건이 영원히 발동되지 않는다.
        """
        self.missed_frames += 1

    def update_with_prediction(self, pred_box: dict):
        """
        예측 박스를 self.det 로 채워 다음 프레임의 비교 기준으로 사용.
        v3: missed_frames 증분은 mark_missed 가 담당하므로 여기서는 +1 하지 않는다.
        """
        prev = self.det.copy() if self.det is not None else None
        self.prev_det = prev
        self.det = pred_box.copy()
        # NOTE: ③ last_measured_det 은 갱신하지 않아 size 정보 freeze를 막음
        self.initialized = True

    def apply_optical_flow_correction(self, fx: float, fy: float,
                                      alpha: float = OPTICAL_FLOW_BLEND_ALPHA):
        """
        ② Optical flow 결과를 Kalman state에도 반영.
        pred_cx/pred_cy 블렌드 + Kalman statePost의 위치 성분도 같은 값으로 동기화.
        """
        if self.pred_cx is None or self.pred_cy is None:
            return
        new_cx = (1.0 - alpha) * self.pred_cx + alpha * fx
        new_cy = (1.0 - alpha) * self.pred_cy + alpha * fy
        self.pred_cx = new_cx
        self.pred_cy = new_cy
        if self.kalman.initialized:
            self.kalman.set_position(new_cx, new_cy)

    def reset(self):
        self.det = None
        self.prev_det = None
        self.last_measured_det = None
        self.missed_frames = 0
        self.initialized = False
        self.kalman = SimpleKalman2D(
            process_noise=KALMAN_PROCESS_NOISE,
            measure_noise=KALMAN_MEASURE_NOISE,
        )
        self.pred_cx = None
        self.pred_cy = None
        self.pred_vx = 0.0
        self.pred_vy = 0.0


@dataclass
class TrackState:
    left: SideTrack = field(default_factory=lambda: SideTrack(side="L", track_id=0))
    right: SideTrack = field(default_factory=lambda: SideTrack(side="R", track_id=1))
    initialized: bool = False
    global_missed_frames: int = 0


# =========================
# Optical flow assist
# =========================
def estimate_flow_center(prev_frame_gray, curr_frame_gray, det: dict, width: int, height: int):
    if prev_frame_gray is None or curr_frame_gray is None or det is None:
        return None

    x1 = int(max(0, min(width - 1, det["x1"])))
    y1 = int(max(0, min(height - 1, det["y1"])))
    x2 = int(max(0, min(width - 1, det["x2"])))
    y2 = int(max(0, min(height - 1, det["y2"])))

    if x2 <= x1 or y2 <= y1:
        return None

    roi = prev_frame_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    points = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=OPTICAL_FLOW_MAX_CORNERS,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7
    )
    if points is None or len(points) == 0:
        return None

    points[:, 0, 0] += x1
    points[:, 0, 1] += y1

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_frame_gray, curr_frame_gray, points, None,
        winSize=OPTICAL_FLOW_WIN_SIZE,
        maxLevel=OPTICAL_FLOW_MAX_LEVEL,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )

    if next_pts is None or status is None:
        return None

    good_old = points[status.flatten() == 1]
    good_new = next_pts[status.flatten() == 1]

    if len(good_old) < 3 or len(good_new) < 3:
        return None

    motion = np.median((good_new - good_old).reshape(-1, 2), axis=0)
    dx, dy = float(motion[0]), float(motion[1])

    pred_cx = det["center_x"] + dx
    pred_cy = det["center_y"] + dy
    return pred_cx, pred_cy


# =========================
# v1-motion: Temporal median 기반 배경 darkening 유틸
# =========================
def compute_temporal_median_background(video_path: str, n_samples: int) -> Optional[np.ndarray]:
    """
    영상에서 n_samples 개 프레임을 균등 sampling 해 channel-wise median 영상 반환.
    실패 시 None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    # 균등 간격 + 양 끝 제외 (락아웃이 마지막 절반에 몰릴 가능성 흡수 위해 양 끝 살짝 빼고)
    indices = np.linspace(0, max(total - 1, 0), n_samples).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if ok and f is not None:
            frames.append(f)
    cap.release()
    if len(frames) < max(3, n_samples // 2):
        return None
    arr = np.stack(frames, axis=0).astype(np.float32)
    # channel-wise median (BGR 각각). uint8 로 다시 변환.
    med = np.median(arr, axis=0).astype(np.uint8)
    return med


def motion_based_darkening(frame: np.ndarray, median_bg: np.ndarray,
                           threshold: float, soft_range: float,
                           alpha_bg: float, dilate_px: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    frame 과 median 의 픽셀별 max-channel diff → soft foreground weight (0..1).
    배경(weight=0) 픽셀은 alpha_bg 로 darkening, 전경(weight=1) 은 원본.

    Returns:
        det_frame : 어두워진 frame (plate YOLO 입력용)
        fg_weight : (H, W) float32 weight map (디버그/시각화용)
    """
    if median_bg.shape != frame.shape:
        # 크기 다르면 원본 그대로 반환 (fallback)
        return frame.copy(), np.ones(frame.shape[:2], dtype=np.float32)

    diff = np.abs(frame.astype(np.int16) - median_bg.astype(np.int16))   # (H,W,3)
    diff_max = np.max(diff, axis=-1).astype(np.float32)                  # (H,W)
    # 임계 미만 → weight 0, 임계+soft_range 이상 → weight 1, 사이는 선형
    weight = np.clip((diff_max - threshold) / max(soft_range, 1e-6), 0.0, 1.0)

    # 가장자리 보존 dilation (foreground 영역을 살짝 확장)
    if dilate_px > 0:
        # weight 를 0/1 mask 로 thresholding 한 뒤 dilate, 다시 weight 와 합쳐 max
        mask01 = (weight > 0.5).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask01_d = cv2.dilate(mask01, kernel, iterations=1).astype(np.float32) / 255.0
        weight = np.maximum(weight, mask01_d)

    weight_3 = weight[..., None]   # (H,W,1) broadcasting
    blended_alpha = alpha_bg + (1.0 - alpha_bg) * weight_3   # 배경=alpha_bg, 전경=1.0
    out = frame.astype(np.float32) * blended_alpha
    return np.clip(out, 0, 255).astype(np.uint8), weight


def compute_motion_darkened_frame(frame: np.ndarray, median_bg: Optional[np.ndarray]) \
        -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    매 프레임 1회 호출. median_bg 가 None 이면 원본 그대로 반환 (fallback).
    """
    if not USE_MOTION_BG_DARKEN or median_bg is None:
        return frame, None
    det_frame, weight = motion_based_darkening(
        frame, median_bg,
        threshold=MOTION_DIFF_THRESHOLD,
        soft_range=MOTION_DIFF_SOFT_RANGE,
        alpha_bg=MOTION_BG_DARKEN_ALPHA,
        dilate_px=MOTION_FG_DILATE_PX,
    )
    return det_frame, weight


# =========================
# Detection / prediction utils
# =========================
def predict_with_thresholds(model, frame, conf_thres: float, iou_thres: float, inference_device,
                            imgsz: int = IMGSZ):
    results = model.predict(
        source=frame,
        conf=conf_thres,
        iou=iou_thres,
        imgsz=imgsz,
        verbose=False,
        device=inference_device,
    )
    result = results[0]
    boxes = result.boxes
    return result, boxes


def box_iou(det_a: dict, det_b: dict):
    x1 = max(det_a["x1"], det_b["x1"])
    y1 = max(det_a["y1"], det_b["y1"])
    x2 = min(det_a["x2"], det_b["x2"])
    y2 = min(det_a["y2"], det_b["y2"])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    # 함수 내부에서 area 직접 계산하여 외부 dict 의존도 제거
    area_a = max((det_a["x2"] - det_a["x1"]) * (det_a["y2"] - det_a["y1"]), 0.0)
    area_b = max((det_b["x2"] - det_b["x1"]) * (det_b["y2"] - det_b["y1"]), 0.0)
    union_area = max(area_a + area_b - inter_area, 1e-6)
    return inter_area / union_area


def merge_detections(global_dets: List[dict], local_dets: List[dict], width: int, height: int):
    merged = []

    all_dets = [(det, "global") for det in global_dets] + [(det, "local") for det in local_dets]
    diag = frame_diag(width, height)
    # 0.02*diag 는 작은 box들에서 과머지가 잦아 0.012로 축소
    dup_center_thresh = 0.012 * diag
    dup_iou_thresh = 0.50
    dup_size_sim_thresh = 0.70

    for det, source in all_dets:
        if not detection_is_candidate(det):
            continue

        candidate = det.copy()
        candidate["source"] = det.get("source", source)

        duplicate_idx = None
        for idx, kept in enumerate(merged):
            iou = box_iou(candidate, kept)
            center_close = center_distance(candidate, kept) <= dup_center_thresh
            size_sim = (
                similarity_ratio(candidate["width"], kept["width"]) +
                similarity_ratio(candidate["height"], kept["height"])
            ) / 2.0
            if iou >= dup_iou_thresh or (center_close and size_sim >= dup_size_sim_thresh):
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            merged.append(candidate)
            continue

        kept = merged[duplicate_idx]
        candidate_conf = candidate.get("confidence") or 0.0
        kept_conf = kept.get("confidence") or 0.0
        prefer_candidate = (
            candidate_conf > kept_conf or
            (
                candidate_conf == kept_conf and
                candidate.get("source") == "local" and
                kept.get("source") != "local"
            )
        )
        if prefer_candidate:
            merged[duplicate_idx] = candidate

    for idx, det in enumerate(merged):
        det["det_idx"] = idx

    return merged


def build_pred_box_from_track(side_track: SideTrack, width: int, height: int, source: str = "kalman_pred"):
    """
    예측 박스의 width/height는 last_measured_det 을 우선 사용.
    이렇게 해야 ③ size_score 가 매 프레임 1.0 으로 freeze 되는 문제를 회피.
    """
    if side_track.det is None and side_track.last_measured_det is None:
        return None

    size_src = side_track.last_measured_det if side_track.last_measured_det is not None else side_track.det
    base = side_track.det.copy() if side_track.det is not None else size_src.copy()

    pred_cx = side_track.pred_cx if side_track.pred_cx is not None else base["center_x"]
    pred_cy = side_track.pred_cy if side_track.pred_cy is not None else base["center_y"]

    w = size_src["width"]
    h = size_src["height"]

    x1 = pred_cx - w / 2.0
    y1 = pred_cy - h / 2.0
    x2 = pred_cx + w / 2.0
    y2 = pred_cy + h / 2.0
    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, width, height)

    pred_box = {
        "frame_idx": base.get("frame_idx"),
        "time_sec": base.get("time_sec"),
        "det_idx": -1,
        "class_id": base.get("class_id", -1),
        "class_name": base.get("class_name", ""),
        "confidence": None,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
        "center_x": (x1 + x2) / 2.0,
        "center_y": (y1 + y2) / 2.0,
        "area": (x2 - x1) * (y2 - y1),
        "final_conf_thres": None,
    }
    pred_box = enrich_detection_row(pred_box, width, height)
    pred_box["source"] = source
    return pred_box


# =========================
# Pair init / reacquire
# =========================
def score_pair_for_init(left_det: dict, right_det: dict, frame_width: int, frame_height: int):
    frame_cx = frame_width / 2.0

    conf_score = ((left_det["confidence"] or 0.0) + (right_det["confidence"] or 0.0)) / 2.0
    height_score = similarity_ratio(left_det["height"], right_det["height"])

    center_y_diff_norm = abs(right_det["center_y"] - left_det["center_y"]) / max(frame_height, 1)
    y_align_score = 1.0 - center_y_diff_norm

    left_dist_to_center = abs(left_det["center_x"] - frame_cx)
    right_dist_to_center = abs(right_det["center_x"] - frame_cx)
    symmetry_gap = abs(left_dist_to_center - right_dist_to_center) / max(frame_width, 1)
    center_sym_score = 1.0 - symmetry_gap

    total = (
        W_CONF * conf_score
        + W_HEIGHT_SIM * height_score
        + W_Y_ALIGN * y_align_score
        + W_CENTER_SYM * center_sym_score
    )

    breakdown = {
        "pair_score_total": total,
        "conf_score": conf_score,
        "height_score": height_score,
        "y_align_score": y_align_score,
        "center_sym_score": center_sym_score,
    }
    return total, breakdown


def choose_best_pair_for_init(current_frame_dets: List[dict], frame_width: int, frame_height: int, frame_idx: int):
    candidate_dets = [d for d in current_frame_dets if detection_is_candidate(d)]
    if len(candidate_dets) < 2:
        return [], None, []

    candidate_dets = sorted(candidate_dets, key=lambda d: d["center_x"])

    pair_rows = []
    best_pair = None
    best_score = None

    for left_det, right_det in combinations(candidate_dets, 2):
        if left_det["center_x"] >= right_det["center_x"]:
            continue

        total_score, breakdown = score_pair_for_init(
            left_det=left_det,
            right_det=right_det,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        pair_rows.append({
            "frame_idx": frame_idx,
            "left_det_idx": left_det["det_idx"],
            "right_det_idx": right_det["det_idx"],
            "pair_score_total": total_score,
            "conf_score": breakdown["conf_score"],
            "height_score": breakdown["height_score"],
            "y_align_score": breakdown["y_align_score"],
            "center_sym_score": breakdown["center_sym_score"],
            "left_cx": left_det["center_x"],
            "left_cy": left_det["center_y"],
            "right_cx": right_det["center_x"],
            "right_cy": right_det["center_y"],
        })

        if best_score is None or total_score > best_score:
            best_score = total_score
            best_pair = [left_det, right_det]

    if best_pair is None or best_score < MIN_ACCEPTABLE_PAIR_SCORE:
        return [], None, pair_rows

    return best_pair, best_score, pair_rows


def try_mirror_search_init(
    model,
    frame,
    single_det: dict,
    frame_width: int,
    frame_height: int,
    frame_idx: int,
    fps: float,
    inference_device,
):
    """
    init 미달성 + global이 한 객체만 잡았을 때, frame_cx 반대편에 ROI를 잘라
    LOCAL_CONF_THRES 로 1회 추론하여 파트너를 찾는다.
    찾으면 (left_det, right_det) 페어 반환, 못 찾으면 None.
    """
    frame_cx = frame_width / 2.0
    is_left = single_det["center_x"] < frame_cx

    mirror_cx = 2.0 * frame_cx - single_det["center_x"]
    expected_cy = single_det["center_y"]

    roi_w = max(single_det["width"] * INIT_FALLBACK_ROI_W_RATIO, INIT_FALLBACK_ROI_MIN_W)
    roi_h = max(single_det["height"] * INIT_FALLBACK_ROI_H_RATIO, INIT_FALLBACK_ROI_MIN_H)

    rx1 = int(np.floor(max(0.0, mirror_cx - roi_w / 2.0)))
    ry1 = int(np.floor(max(0.0, expected_cy - roi_h / 2.0)))
    rx2 = int(np.ceil(min(float(frame_width), mirror_cx + roi_w / 2.0)))
    ry2 = int(np.ceil(min(float(frame_height), expected_cy + roi_h / 2.0)))

    if rx2 <= rx1 or ry2 <= ry1:
        return None

    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None

    sync_inference_device_if_needed(inference_device)
    _, boxes = predict_with_thresholds(
        model=model,
        frame=crop,
        conf_thres=LOCAL_CONF_THRES,
        iou_thres=IOU_THRES,
        inference_device=inference_device,
        imgsz=LOCAL_INFERENCE_IMGSZ,
    )
    sync_inference_device_if_needed(inference_device)

    candidates = extract_detections_from_boxes(
        model=model,
        boxes=boxes,
        frame_idx=frame_idx,
        fps=fps,
        width=frame_width,
        height=frame_height,
        used_conf=LOCAL_CONF_THRES,
        x_offset=rx1,
        y_offset=ry1,
        clip_width=rx2 - rx1,
        clip_height=ry2 - ry1,
    )
    candidates = [d for d in candidates if detection_is_candidate(d)]
    if not candidates:
        return None

    best = None
    best_score = -1.0
    for cand in candidates:
        # single_det 과 같은 박스(자기 자신) 또는 같은 쪽 후보 제외
        if is_left and cand["center_x"] <= frame_cx:
            continue
        if not is_left and cand["center_x"] >= frame_cx:
            continue

        h_sim = similarity_ratio(cand["height"], single_det["height"])
        if h_sim < INIT_FALLBACK_HEIGHT_SIM_MIN:
            continue
        y_diff_norm = abs(cand["center_y"] - single_det["center_y"]) / max(frame_height, 1)
        if y_diff_norm > INIT_FALLBACK_Y_DIFF_RATIO_MAX:
            continue

        score = (cand.get("confidence") or 0.0) + h_sim + (1.0 - y_diff_norm)
        if score > best_score:
            best_score = score
            best = cand

    if best is None:
        return None

    if is_left:
        return single_det, best
    return best, single_det


# =========================
# Association
# =========================
def side_expected_sign(side: str):
    return -1.0 if side == "L" else 1.0


def _tier_gate_dist_ratio(tier: str) -> float:
    """v4 신규: 3-tier (narrow/mid/wide) → 거리 게이트 비율."""
    if tier == "narrow":
        return LOCAL_GATE_DIST_RATIO
    if tier == "mid":
        return MID_GATE_DIST_RATIO
    if tier == "wide":
        return REACQUIRE_GATE_DIST_RATIO
    raise ValueError(f"unknown tier: {tier}")


def gating_pass(det: dict, side_track: SideTrack, width: int, height: int,
                tier: str = "narrow"):
    """
    v4: tier 인자로 게이트 임계 차등 적용.
        tier="narrow"  → LOCAL_GATE_DIST_RATIO (정상 매칭)
        tier="mid"     → MID_GATE_DIST_RATIO (1차 실패 후 같은 풀에서 재시도)
        tier="wide"    → REACQUIRE_GATE_DIST_RATIO (재획득)

    ④ 미초기화 분기 보강:
    side_track 이 한 번도 측정된 적 없을 때만 무조건 통과.
    예측 위치가 있으면 거리 게이트는 적용하되, 더 관대한 임계를 사용.
    """
    has_measurement = side_track.last_measured_det is not None or side_track.det is not None
    has_prediction = side_track.pred_cx is not None and side_track.pred_cy is not None

    if not side_track.initialized and not has_measurement and not has_prediction:
        return True

    if not has_prediction:
        return True

    diag = frame_diag(width, height)
    base_ratio = _tier_gate_dist_ratio(tier)
    # 측정이 없는 상태(즉 reset 후 예측만 있음)에서는 게이트를 1.5배 완화
    gate_dist_ratio = base_ratio * (1.5 if not has_measurement else 1.0)
    gate_dist = gate_dist_ratio * diag

    dx = det["center_x"] - side_track.pred_cx
    dy = det["center_y"] - side_track.pred_cy
    dist = np.hypot(dx, dy)

    if dist > gate_dist:
        return False

    if abs(dx) > LOCAL_GATE_X_RATIO * width * (1.5 if not has_measurement else 1.0):
        return False

    if abs(dy) > LOCAL_GATE_Y_RATIO * height * (1.5 if not has_measurement else 1.0):
        return False

    return True


def resolve_two_side_association(
    detections: List[dict],
    track_state: TrackState,
    width: int,
    height: int,
    tier: str = "narrow",
):
    used = set()

    left_idx, left_det, left_info = choose_detection_for_side(
        detections=detections,
        side_track=track_state.left,
        other_track=track_state.right,
        width=width,
        height=height,
        used_det_indices=used,
        tier=tier,
    )
    if left_idx is not None:
        used.add(left_idx)

    right_idx, right_det, right_info = choose_detection_for_side(
        detections=detections,
        side_track=track_state.right,
        other_track=track_state.left,
        width=width,
        height=height,
        used_det_indices=used,
        tier=tier,
    )

    if left_det is not None and right_det is not None:
        if left_det["center_x"] >= right_det["center_x"]:
            if left_info["assoc_score"] >= right_info["assoc_score"]:
                right_idx, right_det, right_info = None, None, None
            else:
                left_idx, left_det, left_info = None, None, None

    return (left_det, left_info), (right_det, right_info)


# =========================
# Selected row / drawing
# =========================
def copy_det_as_selected_row(det: dict, frame_idx: int, fps: float, side: str, source: str, missed_count: int, track_id: int,
                             assoc_info: Optional[dict] = None):
    assoc_info = assoc_info or {}
    return {
        "frame_idx": frame_idx,
        "time_sec": frame_idx / fps if fps > 0 else None,
        "side": side,
        "track_id": track_id,
        "source": source,
        "class_id": det.get("class_id", -1),
        "class_name": det.get("class_name", ""),
        "confidence": det.get("confidence", None),
        "x1": det.get("x1", None),
        "y1": det.get("y1", None),
        "x2": det.get("x2", None),
        "y2": det.get("y2", None),
        "width": det.get("width", None),
        "height": det.get("height", None),
        "center_x": det.get("center_x", None),
        "center_y": det.get("center_y", None),
        "area": det.get("area", None),
        "area_ratio": det.get("area_ratio", None),
        "aspect_ratio": det.get("aspect_ratio", None),
        "final_conf_thres": det.get("final_conf_thres", None),
        "assoc_score": assoc_info.get("assoc_score", None),
        "pred_dist_score": assoc_info.get("pred_dist_score", None),
        "size_score": assoc_info.get("size_score", None),
        "motion_score": assoc_info.get("motion_score", None),
        "is_interpolated": 1 if source == "kalman_pred" else 0,
        "missed_count": missed_count,
    }


def make_placeholder_row(frame_idx: int, fps: float, side: str, track_id: int,
                         missed_count: int, source: str) -> dict:
    """
    detected/kalman_pred 어느 쪽으로도 채울 수 없는 (frame, side)를 위한 NaN 행.
    source: "missing" (트랙은 살아있지만 좌표 없음) 또는 "uninitialized" (페어 init 전)
    is_interpolated 는 항상 0 (Kalman 예측 행만 1로 유지).
    """
    return {
        "frame_idx": frame_idx,
        "time_sec": frame_idx / fps if fps > 0 else None,
        "side": side,
        "track_id": track_id,
        "source": source,
        "class_id": -1,
        "class_name": "",
        "confidence": None,
        "x1": None,
        "y1": None,
        "x2": None,
        "y2": None,
        "width": None,
        "height": None,
        "center_x": None,
        "center_y": None,
        "area": None,
        "area_ratio": None,
        "aspect_ratio": None,
        "final_conf_thres": None,
        "assoc_score": None,
        "pred_dist_score": None,
        "size_score": None,
        "motion_score": None,
        "is_interpolated": 0,
        "missed_count": missed_count,
    }


def draw_rule_boxes(frame, selected_rows: List[dict], frame_idx: int, global_conf: float, local_conf: float, track_state: TrackState):
    vis = frame.copy()

    for row in selected_rows:
        if row["x1"] is None:
            continue

        x1 = int(round(row["x1"]))
        y1 = int(round(row["y1"]))
        x2 = int(round(row["x2"]))
        y2 = int(round(row["y2"]))

        side = row["side"]
        color = (80, 220, 120) if side == "L" else (80, 160, 255)

        src = row["source"]
        conf = row["confidence"]
        conf_text = "None" if conf is None else f"{conf:.2f}"
        label = f"{side} {src} conf={conf_text}"

        thickness = 3 if src == "detected" else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        cv2.circle(vis, (int(row["center_x"]), int(row["center_y"])), 5, color, -1)

        cv2.putText(
            vis,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    header = (
        f"frame={frame_idx} "
        f"Gconf={global_conf:.2f} Lconf={local_conf:.2f} "
        f"Lmiss={track_state.left.missed_frames} Rmiss={track_state.right.missed_frames}"
    )
    cv2.putText(
        vis,
        header,
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return vis


def append_track_row(
    selected_rows: List[dict],
    det: dict,
    frame_idx: int,
    fps: float,
    side: str,
    track_id: int,
    missed_count: int,
    assoc_info: Optional[dict],
):
    selected_rows.append(
        copy_det_as_selected_row(
            det, frame_idx, fps, side=side, source="detected",
            missed_count=missed_count, track_id=track_id,
            assoc_info=assoc_info,
        )
    )


def check_file_exists(path: str, desc: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{desc} file does not exist: {path}")


def auto_find_best_pt():
    search_roots = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        OUTPUT_DIR,
    ]
    candidates = []
    seen = set()
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        pattern = os.path.join(root, "**", "best.pt")
        for candidate in glob.glob(pattern, recursive=True):
            norm = os.path.normpath(candidate)
            if norm in seen:
                continue
            seen.add(norm)
            candidates.append(norm)
    return sorted(candidates)


def ffprobe_codec(video_path: str):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,codec_long_name,width,height,r_frame_rate,avg_frame_rate",
            "-of", "default=noprint_wrappers=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except FileNotFoundError:
        return "ffprobe failed: ffprobe binary not found in PATH"
    except subprocess.CalledProcessError as e:
        return f"ffprobe failed (return code {e.returncode}): {e.stderr.strip() if e.stderr else ''}"
    except Exception as e:
        return f"ffprobe failed: {e}"


def transcode_to_h264(input_path: str, output_path: str, log_path: str):
    log(f"[INFO] H264 transcode start: {input_path} -> {output_path}", log_path)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg binary not found in PATH. Install ffmpeg or set FORCE_TRANSCODE_TO_H264=False.") from e
    log("[INFO] H264 transcode complete", log_path)


def open_video_safely(video_path: str, readable_video_path: str, log_path: str):
    if FORCE_TRANSCODE_TO_H264:
        transcode_to_h264(video_path, readable_video_path, log_path)
        cap = cv2.VideoCapture(readable_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open transcoded video: {readable_video_path}")
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError(f"Failed to read first frame after transcode: {readable_video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap, readable_video_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log("[WARN] Original video open failed. Retrying after H264 transcode.", log_path)
        transcode_to_h264(video_path, readable_video_path, log_path)
        cap = cv2.VideoCapture(readable_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video after transcode: {readable_video_path}")
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError(f"Failed to read first frame after transcode: {readable_video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap, readable_video_path

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        log("[WARN] Original first-frame read failed. Retrying after H264 transcode.", log_path)
        transcode_to_h264(video_path, readable_video_path, log_path)
        cap = cv2.VideoCapture(readable_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video after transcode: {readable_video_path}")
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError(f"Failed to read first frame after transcode: {readable_video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap, readable_video_path

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, video_path


def recover_frame_read(
    cap,
    original_video_path: str,
    actual_video_path: str,
    readable_video_path: str,
    frame_idx: int,
    frame_count: int,
    log_path: str,
):
    expected_more_frames = frame_count <= 0 or frame_idx < frame_count
    if not expected_more_frames:
        return False, None, cap, actual_video_path

    if cap is not None:
        cap.release()

    log(f"[WARN] frame_idx={frame_idx} read failed. Attempting recovery from the same frame.", log_path)

    recovery_candidates = [actual_video_path]
    if readable_video_path not in recovery_candidates:
        recovery_candidates.append(readable_video_path)

    for candidate in recovery_candidates:
        if candidate == readable_video_path and not os.path.exists(readable_video_path):
            transcode_to_h264(original_video_path, readable_video_path, log_path)

        recovered_cap, recovered_frame = reopen_capture_at_frame(candidate, frame_idx)
        if recovered_cap is not None and recovered_frame is not None:
            log(f"[INFO] frame_idx={frame_idx} recovery succeeded. Switching to: {candidate}", log_path)
            return True, recovered_frame, recovered_cap, candidate

    log(f"[ERROR] frame_idx={frame_idx} recovery failed.", log_path)
    return False, None, None, actual_video_path


def is_threshold_already_completed(log_path: str):
    if not os.path.exists(log_path):
        return False
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return "===== save complete =====" in f.read()
    except OSError:
        return False


def association_score(
    det: dict,
    side_track: SideTrack,
    other_track: Optional[SideTrack],
    frame_width: int,
    frame_height: int,
    tier: str = "narrow",
):
    """v4: tier 인자로 게이트 임계 차등 적용 (narrow/mid/wide)."""
    conf_score = float(det["confidence"] or 0.0)

    pred_dist_score = 0.0
    motion_score = 0.0
    size_score = 0.0
    y_score = 0.0
    edge_penalty = 0.0

    gate_dist_ratio = _tier_gate_dist_ratio(tier)

    # ③ size_score 는 가능한 last_measured_det 기준으로 비교 (예측 박스로 freeze되는 문제 회피)
    size_ref = side_track.last_measured_det if side_track.last_measured_det is not None else side_track.det
    if size_ref is not None:
        size_score = (
            similarity_ratio(det["width"], size_ref["width"]) +
            similarity_ratio(det["height"], size_ref["height"]) +
            similarity_ratio(det["area"], size_ref["area"])
        ) / 3.0

    if side_track.pred_cx is not None and side_track.pred_cy is not None:
        diag = frame_diag(frame_width, frame_height)
        dist = center_distance_xy(det["center_x"], det["center_y"], side_track.pred_cx, side_track.pred_cy)
        dist_norm = dist / max(diag, 1e-6)
        pred_dist_score = 1.0 - min(dist_norm / gate_dist_ratio, 1.5)

        if side_track.det is not None:
            curr_dx = det["center_x"] - side_track.det["center_x"]
            curr_dy = det["center_y"] - side_track.det["center_y"]
            prev_vx = side_track.pred_vx
            prev_vy = side_track.pred_vy

            motion_diff = np.hypot(curr_dx - prev_vx, curr_dy - prev_vy) / max(diag, 1e-6)
            motion_score = 1.0 - min(motion_diff / VERY_LARGE_JUMP_RATIO, 1.6)

            curr_motion = np.hypot(curr_dx, curr_dy) / max(diag, 1e-6)
            # v2: -0.4 → -0.2. motion-darkening 이 정지된 배경 plate 를 이미 제거하므로
            # 정지 페널티의 진단력 감소. 진짜 락아웃 홀드에서 진짜 plate 보호.
            if curr_motion < STATIC_BG_MOTION_THRESH_RATIO:
                motion_score -= 0.2

            expected_sign = side_expected_sign(side_track.side)
            if curr_dx * expected_sign < -1.0:
                motion_score -= 0.4

    if side_track.pred_cy is not None:
        y_diff_norm = abs(det["center_y"] - side_track.pred_cy) / max(frame_height, 1)
        y_score = 1.0 - min(y_diff_norm / LOCAL_GATE_Y_RATIO, 1.5)

    if det["x1"] <= 1 or det["x2"] >= frame_width - 1:
        edge_penalty -= 0.25

    if other_track is not None and other_track.pred_cx is not None:
        if side_track.side == "L" and det["center_x"] >= other_track.pred_cx:
            pred_dist_score -= 0.8
        if side_track.side == "R" and det["center_x"] <= other_track.pred_cx:
            pred_dist_score -= 0.8

    total = (
        W_ASSOC_CONF * conf_score
        + W_ASSOC_PRED_DIST * pred_dist_score
        + W_ASSOC_SIZE * size_score
        + W_ASSOC_Y * y_score
        + W_ASSOC_MOTION * motion_score
        + W_ASSOC_EDGE * edge_penalty
    )

    breakdown = {
        "assoc_score": total,
        "pred_dist_score": pred_dist_score,
        "size_score": size_score,
        "motion_score": motion_score,
        "edge_penalty": edge_penalty,
    }
    return total, breakdown


def choose_detection_for_side(
    detections: List[dict],
    side_track: SideTrack,
    other_track: Optional[SideTrack],
    width: int,
    height: int,
    used_det_indices: set,
    tier: str = "narrow",
):
    """v4: tier 인자로 narrow/mid/wide 게이트 차등 적용."""
    candidates = []
    for idx, det in enumerate(detections):
        if idx in used_det_indices:
            continue
        if not detection_is_candidate(det):
            continue
        if not gating_pass(det, side_track, width, height, tier=tier):
            continue

        score, breakdown = association_score(
            det,
            side_track,
            other_track,
            width,
            height,
            tier=tier,
        )
        if score >= MIN_ASSOC_SCORE:
            candidates.append((idx, det, score, breakdown))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: x[2], reverse=True)
    best_idx, best_det, best_score, best_breakdown = candidates[0]
    return best_idx, best_det, {
        "assoc_score": best_score,
        **best_breakdown
    }


LOCAL_ROI_MARGIN_RATIO = 0.75
LOCAL_ROI_MIN_SIZE_PX = 160
LOCAL_ROI_SKIP_AREA_RATIO = 0.90


def extract_detections_from_boxes(
    model,
    boxes,
    frame_idx: int,
    fps: float,
    width: int,
    height: int,
    used_conf: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    clip_width: Optional[int] = None,
    clip_height: Optional[int] = None,
):
    rows = []
    if boxes is None or len(boxes) == 0:
        return rows

    local_width = clip_width if clip_width is not None else width
    local_height = clip_height if clip_height is not None else height

    for det_idx, box in enumerate(boxes):
        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
        conf = float(box.conf[0].item()) if box.conf is not None else None

        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(float, xyxy)
        x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, local_width, local_height)

        x1 += x_offset
        y1 += y_offset
        x2 += x_offset
        y2 += y_offset
        x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, width, height)

        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = w * h

        row = {
            "frame_idx": frame_idx,
            "time_sec": frame_idx / fps if fps > 0 else None,
            "det_idx": det_idx,
            "class_id": cls_id,
            "class_name": model.names.get(cls_id, str(cls_id)),
            "confidence": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": w,
            "height": h,
            "center_x": cx,
            "center_y": cy,
            "area": area,
            "final_conf_thres": used_conf,
        }
        row = enrich_detection_row(row, width, height)
        rows.append(row)
    return rows


def build_local_tracking_roi(track_state: TrackState, frame_width: int, frame_height: int):
    boxes = []
    for side_track in (track_state.left, track_state.right):
        if side_track.det is None:
            continue

        det = side_track.det
        boxes.append((det["x1"], det["y1"], det["x2"], det["y2"]))

        pred_cx = side_track.pred_cx if side_track.pred_cx is not None else det["center_x"]
        pred_cy = side_track.pred_cy if side_track.pred_cy is not None else det["center_y"]
        half_w = max(det["width"] / 2.0, LOCAL_ROI_MIN_SIZE_PX / 4.0)
        half_h = max(det["height"] / 2.0, LOCAL_ROI_MIN_SIZE_PX / 4.0)
        boxes.append((pred_cx - half_w, pred_cy - half_h, pred_cx + half_w, pred_cy + half_h))

    if not boxes:
        return None

    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)

    roi_w = max(x2 - x1, float(LOCAL_ROI_MIN_SIZE_PX))
    roi_h = max(y2 - y1, float(LOCAL_ROI_MIN_SIZE_PX))

    # v4 🅔: Velocity-aware margin — Kalman state 의 |vx|,|vy| 평균을 진행 방향 마진으로 반영
    v_factors = []
    for st in (track_state.left, track_state.right):
        if st.initialized and st.kalman.initialized:
            v_factors.append((abs(st.pred_vx), abs(st.pred_vy)))
    if v_factors:
        avg_vx = sum(v[0] for v in v_factors) / len(v_factors)
        avg_vy = sum(v[1] for v in v_factors) / len(v_factors)
    else:
        avg_vx = avg_vy = 0.0

    vel_margin_x = ROI_VELOCITY_LOOKAHEAD * avg_vx * 2.0
    vel_margin_y = ROI_VELOCITY_LOOKAHEAD * avg_vy * 2.0

    margin_x = max(roi_w * LOCAL_ROI_MARGIN_RATIO, vel_margin_x, 48.0)
    margin_y = max(roi_h * LOCAL_ROI_MARGIN_RATIO, vel_margin_y, 48.0)

    # 폭주 방지: roi_w/h * ROI_MAX_MARGIN_RATIO + 48 으로 상한 둠
    margin_x = min(margin_x, roi_w * ROI_MAX_MARGIN_RATIO + 48.0)
    margin_y = min(margin_y, roi_h * ROI_MAX_MARGIN_RATIO + 48.0)

    x1 -= margin_x
    y1 -= margin_y
    x2 += margin_x
    y2 += margin_y
    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, frame_width, frame_height)

    ix1 = int(np.floor(x1))
    iy1 = int(np.floor(y1))
    ix2 = int(np.ceil(x2))
    iy2 = int(np.ceil(y2))

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    roi_area_ratio = ((ix2 - ix1) * (iy2 - iy1)) / max(frame_width * frame_height, 1)
    if roi_area_ratio >= LOCAL_ROI_SKIP_AREA_RATIO:
        return None

    return ix1, iy1, ix2, iy2


# =========================
# A. Conditional local 트리거
# =========================
def should_run_local_inference(track_state: TrackState,
                               global_dets: List[dict],
                               width: int,
                               height: int,
                               tracked_frame_counter: int) -> Tuple[bool, str]:
    # 1) 강제 주기 (드리프트/scene change 보호)
    if tracked_frame_counter % LOCAL_TRIGGER_FORCE_EVERY == 0:
        return True, "force_periodic"

    # 2) 직전 프레임 한쪽이라도 missed
    if track_state.left.missed_frames > 0 or track_state.right.missed_frames > 0:
        return True, "side_missed"

    diag = frame_diag(width, height)
    near_thresh = LOCAL_TRIGGER_PRED_DIST_RATIO * diag

    for side_track in (track_state.left, track_state.right):
        if side_track.pred_cx is None or side_track.pred_cy is None:
            return True, "no_prediction"

        nearby = [
            d for d in global_dets
            if detection_is_candidate(d) and
               center_distance_xy(d["center_x"], d["center_y"],
                                  side_track.pred_cx, side_track.pred_cy) <= near_thresh
        ]
        if not nearby:
            return True, "no_global_near_pred"

        best_conf = max((d.get("confidence") or 0.0) for d in nearby)
        if best_conf < LOCAL_TRIGGER_LOW_CONF:
            return True, "low_global_conf"

    return False, "skip_local"


# =========================
# v4 🅐 Tier-3: 같은 프레임 ROI local 추론 (즉시 회복 시도)
# =========================
def run_emergency_local_inference(model, frame, track_state, width, height, fps,
                                  frame_idx, inference_device):
    """
    Tier 1(narrow) + Tier 2(mid) 둘 다 실패한 측이 있을 때 같은 프레임에 즉시
    ROI local 추론을 1회 돌려 detection 풀을 확장한다.
    실패한 측의 ROI 만 잘라 LOCAL_INFERENCE_IMGSZ 로 추론.
    추론 비용: 작은 ROI 1회.
    """
    roi = build_local_tracking_roi(track_state, width, height)
    if roi is None:
        return []
    rx1, ry1, rx2, ry2 = roi
    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return []

    sync_inference_device_if_needed(inference_device)
    _, boxes = predict_with_thresholds(
        model=model, frame=crop,
        conf_thres=LOCAL_CONF_THRES, iou_thres=IOU_THRES,
        inference_device=inference_device,
        imgsz=LOCAL_INFERENCE_IMGSZ,
    )
    sync_inference_device_if_needed(inference_device)

    return extract_detections_from_boxes(
        model=model, boxes=boxes, frame_idx=frame_idx, fps=fps,
        width=width, height=height, used_conf=LOCAL_CONF_THRES,
        x_offset=rx1, y_offset=ry1,
        clip_width=rx2 - rx1, clip_height=ry2 - ry1,
    )


def _find_det_idx_in_pool(pool: List[dict], det_to_find: dict, eps: float = 1e-3):
    """merge 후 새 det_idx 찾기 (center_x/y 동등성으로, float-safe)."""
    if det_to_find is None:
        return None
    cx = det_to_find.get("center_x")
    cy = det_to_find.get("center_y")
    if cx is None or cy is None:
        return None
    for i, d in enumerate(pool):
        if abs(d.get("center_x", -1e9) - cx) < eps and abs(d.get("center_y", -1e9) - cy) < eps:
            return i
    return None


# =========================
# CSV chunked writer helper
# =========================
class ChunkedCsvWriter:
    """rows를 누적했다가 임계 이상이면 append 모드로 flush. 메모리 폭증 방지."""
    def __init__(self, path: str, columns: List[str], flush_every: int = CSV_FLUSH_EVERY_FRAMES):
        self.path = path
        self.columns = columns
        self.flush_every = flush_every
        self.buffer: List[dict] = []
        self.header_written = False
        if os.path.exists(path):
            os.remove(path)

    def add(self, rows):
        if isinstance(rows, dict):
            self.buffer.append(rows)
        else:
            self.buffer.extend(rows)
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer, columns=self.columns)
        df.to_csv(
            self.path,
            mode="a" if self.header_written else "w",
            header=not self.header_written,
            index=False,
            encoding="utf-8-sig",
        )
        self.header_written = True
        self.buffer.clear()

    def finalize(self):
        if not self.header_written and not self.buffer:
            # 빈 CSV 라도 헤더는 남김
            pd.DataFrame(columns=self.columns).to_csv(self.path, index=False, encoding="utf-8-sig")
            self.header_written = True
            return
        self.flush()


# =========================
# Main robust inference
# =========================
def run_inference_robust(model):
    overall_start_time = time.perf_counter()
    paths = build_output_paths(OUTPUT_DIR, GLOBAL_CONF_THRES, LOCAL_CONF_THRES)
    log_path = paths["log_path"]
    inference_device = get_inference_device()

    if SKIP_COMPLETED_THRESHOLDS and is_threshold_already_completed(log_path):
        print("[INFO] Skip completed run.")
        return

    if os.path.exists(log_path):
        os.remove(log_path)

    log("===== inference start (motion_v2) =====", log_path)
    log("[INFO] motion_v2: motion_v1 위에 conf/score 임계 재튜닝", log_path)
    log("[INFO]   GLOBAL_CONF_THRES  : 0.25 -> 0.15", log_path)
    log("[INFO]   LOCAL_TRIGGER_LOW  : 0.55 -> 0.40", log_path)
    log("[INFO]   MIN_PAIR_SCORE     : 1.8  -> 1.5", log_path)
    log("[INFO]   W_CONF (pair)      : 1.3  -> 1.7", log_path)
    log("[INFO]   W_CENTER_SYM       : 0.9  -> 0.7  (motion: seg 보다 약하게)", log_path)
    log("[INFO]   W_ASSOC_CONF       : 1.2  -> 1.5", log_path)
    log("[INFO]   static penalty     : -0.4 -> -0.2", log_path)
    log("[INFO]   INIT_FB height_sim : 0.55 -> 0.45", log_path)
    log("[INFO]   INIT_FB y_diff     : 0.20 -> 0.25", log_path)
    log("[INFO] motion_v1: v4 + temporal-median 기반 배경 darkening", log_path)
    log(f"[INFO]   USE_MOTION_BG_DARKEN: {USE_MOTION_BG_DARKEN}", log_path)
    log(f"[INFO]   MEDIAN_N_SAMPLES: {MEDIAN_N_SAMPLES}", log_path)
    log(f"[INFO]   MOTION_DIFF_THRESHOLD: {MOTION_DIFF_THRESHOLD}", log_path)
    log(f"[INFO]   MOTION_DIFF_SOFT_RANGE: {MOTION_DIFF_SOFT_RANGE}", log_path)
    log(f"[INFO]   MOTION_BG_DARKEN_ALPHA: {MOTION_BG_DARKEN_ALPHA}", log_path)
    log(f"[INFO]   MOTION_FG_DILATE_PX: {MOTION_FG_DILATE_PX}", log_path)
    log("[INFO] v4 changes:", log_path)
    log("[INFO]   🅑 MAX_MISSED_FRAMES 10 -> 5 (kalman_pred fill 단축)", log_path)
    log("[INFO]   🅒 SIDE_REACQUIRE 12->6, MAX_REACQUIRE 18->9 (회복 빠르게)", log_path)
    log("[INFO]   🅐 Tiered gating: narrow(0.18) → mid(0.24) → wide(0.32)", log_path)
    log("[INFO]      + same-frame ROI local 추론 (Tier 3)", log_path)
    log("[INFO]   🅓 LOCAL_TRIGGER_FORCE_EVERY 5->3, LOW_CONF 0.45->0.55", log_path)
    log("[INFO]   🅔 Velocity-aware ROI margin (Kalman vx/vy 반영)", log_path)
    log("[INFO]   6-3 init pair detected row 의 score breakdown = 1.0 채움", log_path)
    log("[INFO] v3 carry-over: missed_frames is incremented on every match failure", log_path)
    log(f"[INFO] GLOBAL_CONF_THRES: {GLOBAL_CONF_THRES:.2f}", log_path)
    log(f"[INFO] LOCAL_CONF_THRES: {LOCAL_CONF_THRES:.2f}", log_path)
    log(f"[INFO] IOU_THRES: {IOU_THRES:.2f}", log_path)
    log(f"[INFO] INFERENCE_DEVICE: {inference_device}", log_path)
    log(f"[INFO] MODEL_PATH: {MODEL_PATH}", log_path)
    log(f"[INFO] VIDEO_PATH: {VIDEO_PATH}", log_path)
    log(f"[INFO] LOCAL_INFERENCE_IMGSZ: {LOCAL_INFERENCE_IMGSZ}", log_path)
    log(f"[INFO] LOCAL_TRIGGER_FORCE_EVERY: {LOCAL_TRIGGER_FORCE_EVERY}", log_path)
    log(f"[INFO] LOCAL_TRIGGER_LOW_CONF: {LOCAL_TRIGGER_LOW_CONF}", log_path)
    log(f"[INFO] LOCAL_TRIGGER_PRED_DIST_RATIO: {LOCAL_TRIGGER_PRED_DIST_RATIO}", log_path)
    log(f"[INFO] MIN_ACCEPTABLE_PAIR_SCORE: {MIN_ACCEPTABLE_PAIR_SCORE}", log_path)
    log("[INFO] ffprobe result:", log_path)
    log(ffprobe_codec(VIDEO_PATH), log_path)

    cap = None
    writer = None
    rule_writer = None
    raw_csv = ChunkedCsvWriter(paths["raw_csv_path"], RAW_COLUMNS)
    selected_csv = ChunkedCsvWriter(paths["selected_csv_path"], SELECTED_COLUMNS)
    pairs_csv = ChunkedCsvWriter(paths["pairs_csv_path"], PAIR_SCORE_COLUMNS)

    try:
        video_open_start_time = time.perf_counter()
        cap, actual_video_path = open_video_safely(VIDEO_PATH, paths["readable_video_path"], log_path)
        video_open_time = time.perf_counter() - video_open_start_time
        log(f"[INFO] actual_video_path: {actual_video_path}", log_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        log(f"[INFO] FPS: {fps}", log_path)
        log(f"[INFO] WIDTH x HEIGHT: {width} x {height}", log_path)
        log(f"[INFO] FRAME COUNT: {frame_count}", log_path)

        # === v1-motion: temporal median 영상 precompute (1회) ===
        median_bg = None
        median_time = 0.0
        if USE_MOTION_BG_DARKEN:
            median_start = time.perf_counter()
            median_bg = compute_temporal_median_background(
                actual_video_path, MEDIAN_N_SAMPLES,
            )
            median_time = time.perf_counter() - median_start
            if median_bg is None:
                log("[WARN] temporal median computation failed → fallback to original frames",
                    log_path)
            else:
                log(f"[INFO] temporal median computed: {median_bg.shape} "
                    f"from {MEDIAN_N_SAMPLES} samples ({median_time*1000:.1f}ms)",
                    log_path)

        if SAVE_FIRST_FRAME:
            ret, first_frame = cap.read()
            if ret and first_frame is not None:
                cv2.imwrite(paths["first_frame_path"], first_frame)
                log(f"[INFO] first_frame_saved: {paths['first_frame_path']}", log_path)
            else:
                log("[WARN] first_frame_save_failed", log_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if SAVE_VIDEO:
            # 사용자 요청에 따라 mp4v 코덱 유지
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(paths["annotated_video_path"], fourcc, fps, (width, height))
            rule_writer = cv2.VideoWriter(paths["annotated_video_rule_path"], fourcc, fps, (width, height))
            if not writer.isOpened():
                log("[WARN] annotated_video writer.isOpened() == False (mp4v 미지원 가능). 영상 저장 실패할 수 있음.", log_path)
            if not rule_writer.isOpened():
                log("[WARN] rule_video writer.isOpened() == False (mp4v 미지원 가능). 영상 저장 실패할 수 있음.", log_path)

        track_state = TrackState()

        frame_idx = 0
        processed_frames = 0
        zero_det_frames = 0
        recovered_read_failures = 0
        total_raw_dets = 0
        total_selected_dets = 0    # detected + kalman_pred 행 수만 카운트
        total_missing_rows = 0     # NaN row (source="missing")
        total_uninitialized_rows = 0  # NaN row (source="uninitialized")
        init_frames = 0
        reacquire_frames = 0
        side_reacquire_frames = 0
        one_side_update_frames = 0
        kalman_pred_events = 0   # ⑤ 기존 kalman_pred_frames 의 의미 명확화 (side 단위)
        # v4 🅐 Tier-2/3 카운터
        mid_tier_recoveries = 0
        emergency_local_runs = 0
        emergency_local_recoveries = 0
        # init mirror-search fallback 카운터
        init_fallback_attempts = 0
        init_fallback_frames = 0

        # local trigger 관련
        tracked_frame_counter = 0
        local_inference_runs = 0
        local_trigger_counter = {}

        # v1-motion 통계
        motion_darken_time_total = 0.0
        motion_darken_active_frames = 0
        motion_darken_fallback_frames = 0

        frame_read_time = 0.0
        frame_recover_time = 0.0
        gray_time = 0.0
        global_detect_time = 0.0
        global_extract_time = 0.0
        local_detect_time = 0.0
        local_extract_time = 0.0
        merge_time = 0.0
        init_pair_time = 0.0
        track_predict_time = 0.0
        optical_flow_time = 0.0
        association_time = 0.0
        prediction_fill_time = 0.0
        reacquire_time = 0.0
        draw_time = 0.0
        video_write_time = 0.0
        csv_save_time = 0.0

        prev_frame_gray = None

        while True:
            frame_read_start_time = time.perf_counter()
            ret, frame = cap.read()
            frame_read_time += time.perf_counter() - frame_read_start_time
            if not ret or frame is None:
                frame_recover_start_time = time.perf_counter()
                recovered, frame, cap, actual_video_path = recover_frame_read(
                    cap=cap,
                    original_video_path=VIDEO_PATH,
                    actual_video_path=actual_video_path,
                    readable_video_path=paths["readable_video_path"],
                    frame_idx=frame_idx,
                    frame_count=frame_count,
                    log_path=log_path,
                )
                frame_recover_time += time.perf_counter() - frame_recover_start_time
                if not recovered:
                    break
                recovered_read_failures += 1

            processed_frames += 1

            # grayscale은 optical flow가 사용될 가능성이 있을 때만 계산
            curr_gray = None
            need_gray = USE_OPTICAL_FLOW_ASSIST and track_state.initialized

            # === v1-motion: 매 프레임 motion-based darkening ===
            # det_frame  : plate 모델에 들어갈 입력 (배경 어두움)
            # fg_weight  : 디버깅 시각화용 weight map (사용 안 하면 None)
            motion_darken_start = time.perf_counter()
            det_frame, fg_weight = compute_motion_darkened_frame(frame, median_bg)
            motion_darken_time_total += time.perf_counter() - motion_darken_start
            if median_bg is not None:
                motion_darken_active_frames += 1
            else:
                motion_darken_fallback_frames += 1

            sync_inference_device_if_needed(inference_device)
            global_detect_start_time = time.perf_counter()
            global_result, global_boxes = predict_with_thresholds(
                model=model,
                frame=det_frame,
                conf_thres=GLOBAL_CONF_THRES,
                iou_thres=IOU_THRES,
                inference_device=inference_device,
                imgsz=IMGSZ,
            )
            sync_inference_device_if_needed(inference_device)
            global_detect_time += time.perf_counter() - global_detect_start_time

            global_extract_start_time = time.perf_counter()
            global_dets = extract_detections_from_boxes(
                model=model,
                boxes=global_boxes,
                frame_idx=frame_idx,
                fps=fps,
                width=width,
                height=height,
                used_conf=GLOBAL_CONF_THRES,
            )
            global_extract_time += time.perf_counter() - global_extract_start_time

            local_dets = []
            current_selected_rows: List[dict] = []

            if track_state.initialized:
                if need_gray:
                    gray_start_time = time.perf_counter()
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_time += time.perf_counter() - gray_start_time

                track_predict_start_time = time.perf_counter()
                track_state.left.predict()
                track_state.right.predict()
                track_predict_time += time.perf_counter() - track_predict_start_time

                if USE_OPTICAL_FLOW_ASSIST and prev_frame_gray is not None and curr_gray is not None:
                    if track_state.left.det is not None:
                        optical_flow_start_time = time.perf_counter()
                        flow_est = estimate_flow_center(prev_frame_gray, curr_gray, track_state.left.det, width, height)
                        optical_flow_time += time.perf_counter() - optical_flow_start_time
                        if flow_est is not None:
                            fx, fy = flow_est
                            track_state.left.apply_optical_flow_correction(fx, fy)

                    if track_state.right.det is not None:
                        optical_flow_start_time = time.perf_counter()
                        flow_est = estimate_flow_center(prev_frame_gray, curr_gray, track_state.right.det, width, height)
                        optical_flow_time += time.perf_counter() - optical_flow_start_time
                        if flow_est is not None:
                            fx, fy = flow_est
                            track_state.right.apply_optical_flow_correction(fx, fy)

                # === A. Conditional Local trigger ===
                run_local, local_reason = should_run_local_inference(
                    track_state=track_state,
                    global_dets=global_dets,
                    width=width,
                    height=height,
                    tracked_frame_counter=tracked_frame_counter,
                )
                local_trigger_counter[local_reason] = local_trigger_counter.get(local_reason, 0) + 1

                if run_local:
                    local_roi = build_local_tracking_roi(track_state, width, height)
                    if local_roi is not None:
                        roi_x1, roi_y1, roi_x2, roi_y2 = local_roi
                        # v1-motion: local crop 도 darkened frame 에서
                        local_frame = det_frame[roi_y1:roi_y2, roi_x1:roi_x2]
                        if local_frame.size > 0:
                            sync_inference_device_if_needed(inference_device)
                            local_detect_start_time = time.perf_counter()
                            _, local_boxes = predict_with_thresholds(
                                model=model,
                                frame=local_frame,
                                conf_thres=LOCAL_CONF_THRES,
                                iou_thres=IOU_THRES,
                                inference_device=inference_device,
                                imgsz=LOCAL_INFERENCE_IMGSZ,   # E.
                            )
                            sync_inference_device_if_needed(inference_device)
                            local_detect_time += time.perf_counter() - local_detect_start_time

                            local_extract_start_time = time.perf_counter()
                            local_dets = extract_detections_from_boxes(
                                model=model,
                                boxes=local_boxes,
                                frame_idx=frame_idx,
                                fps=fps,
                                width=width,
                                height=height,
                                used_conf=LOCAL_CONF_THRES,
                                x_offset=roi_x1,
                                y_offset=roi_y1,
                                clip_width=roi_x2 - roi_x1,
                                clip_height=roi_y2 - roi_y1,
                            )
                            local_extract_time += time.perf_counter() - local_extract_start_time
                            local_inference_runs += 1

                tracked_frame_counter += 1

            merge_start_time = time.perf_counter()
            current_frame_dets = merge_detections(global_dets, local_dets, width, height)
            merge_time += time.perf_counter() - merge_start_time

            if len(current_frame_dets) == 0:
                zero_det_frames += 1
            else:
                total_raw_dets += len(current_frame_dets)
                for row in current_frame_dets:
                    raw_csv.add(row.copy())

            if not track_state.initialized:
                init_pair_start_time = time.perf_counter()
                best_pair, best_score, pair_rows = choose_best_pair_for_init(
                    current_frame_dets=current_frame_dets,
                    frame_width=width,
                    frame_height=height,
                    frame_idx=frame_idx,
                )
                init_pair_time += time.perf_counter() - init_pair_start_time
                pairs_csv.add(pair_rows)

                if len(best_pair) == 2:
                    best_pair = sorted(best_pair, key=lambda d: d["center_x"])
                    track_state.left.update_with_det(best_pair[0])
                    track_state.right.update_with_det(best_pair[1])
                    track_state.initialized = True
                    track_state.global_missed_frames = 0
                    init_frames += 1

                    # v4 6-3: init pair detected 행에 score breakdown 채움 (정의상 1.0)
                    init_assoc_info = {
                        "assoc_score": best_score,
                        "pred_dist_score": 1.0,
                        "size_score": 1.0,
                        "motion_score": 1.0,
                    }
                    current_selected_rows = [
                        copy_det_as_selected_row(best_pair[0], frame_idx, fps, side="L", source="detected", missed_count=track_state.left.missed_frames, track_id=0, assoc_info=init_assoc_info),
                        copy_det_as_selected_row(best_pair[1], frame_idx, fps, side="R", source="detected", missed_count=track_state.right.missed_frames, track_id=1, assoc_info=init_assoc_info),
                    ]
                else:
                    # === init mirror-search fallback ===
                    # global이 단일 객체만 잡아 페어가 형성 안 되는 경우, 반대편 ROI 추론으로 파트너 탐색
                    fallback_pair = None
                    if INIT_FALLBACK_ENABLED:
                        candidate_dets = [d for d in current_frame_dets if detection_is_candidate(d)]
                        if len(candidate_dets) == 1:
                            init_fallback_attempts += 1
                            fallback_pair = try_mirror_search_init(
                                model=model,
                                frame=det_frame,  # v1-motion
                                single_det=candidate_dets[0],
                                frame_width=width,
                                frame_height=height,
                                frame_idx=frame_idx,
                                fps=fps,
                                inference_device=inference_device,
                            )

                    if fallback_pair is not None:
                        left_det_fb, right_det_fb = fallback_pair
                        track_state.left.update_with_det(left_det_fb)
                        track_state.right.update_with_det(right_det_fb)
                        track_state.initialized = True
                        track_state.global_missed_frames = 0
                        init_frames += 1
                        init_fallback_frames += 1

                        fb_assoc_info = {
                            "assoc_score": None,
                            "pred_dist_score": 1.0,
                            "size_score": 1.0,
                            "motion_score": 1.0,
                        }
                        current_selected_rows = [
                            copy_det_as_selected_row(left_det_fb, frame_idx, fps, side="L", source="detected", missed_count=0, track_id=0, assoc_info=fb_assoc_info),
                            copy_det_as_selected_row(right_det_fb, frame_idx, fps, side="R", source="detected", missed_count=0, track_id=1, assoc_info=fb_assoc_info),
                        ]
                    else:
                        track_state.global_missed_frames += 1
            else:
                association_start_time = time.perf_counter()
                # === v4 🅐 Tier 1: narrow 게이트로 정상 매칭 ===
                (left_det, left_info), (right_det, right_info) = resolve_two_side_association(
                    detections=current_frame_dets,
                    track_state=track_state,
                    width=width,
                    height=height,
                    tier="narrow",
                )
                used_idx = set()
                if left_det is not None:
                    li = _find_det_idx_in_pool(current_frame_dets, left_det)
                    if li is not None: used_idx.add(li)
                if right_det is not None:
                    ri = _find_det_idx_in_pool(current_frame_dets, right_det)
                    if ri is not None: used_idx.add(ri)

                # === v4 🅐 Tier 2: 실패한 측에 mid 게이트로 같은 풀 재시도 ===
                if left_det is None:
                    l_idx, l_d, l_info = choose_detection_for_side(
                        detections=current_frame_dets,
                        side_track=track_state.left,
                        other_track=track_state.right,
                        width=width, height=height,
                        used_det_indices=used_idx, tier="mid",
                    )
                    if l_d is not None:
                        left_det, left_info = l_d, l_info
                        used_idx.add(l_idx)
                        mid_tier_recoveries += 1
                if right_det is None:
                    r_idx, r_d, r_info = choose_detection_for_side(
                        detections=current_frame_dets,
                        side_track=track_state.right,
                        other_track=track_state.left,
                        width=width, height=height,
                        used_det_indices=used_idx, tier="mid",
                    )
                    if r_d is not None:
                        right_det, right_info = r_d, r_info
                        used_idx.add(r_idx)
                        mid_tier_recoveries += 1

                # === v4 🅐 Tier 3: 그래도 실패한 측이 있으면 즉시 ROI local 추론 1회 + mid 재시도 ===
                if left_det is None or right_det is None:
                    emergency_dets = run_emergency_local_inference(
                        model=model, frame=det_frame, track_state=track_state,  # v1-motion
                        width=width, height=height, fps=fps, frame_idx=frame_idx,
                        inference_device=inference_device,
                    )
                    if emergency_dets:
                        emergency_local_runs += 1
                        extended_pool = merge_detections(current_frame_dets, emergency_dets, width, height)
                        # det_idx 가 재할당되므로 used_idx 재계산
                        used_idx = set()
                        if left_det is not None:
                            li = _find_det_idx_in_pool(extended_pool, left_det)
                            if li is not None: used_idx.add(li)
                        if right_det is not None:
                            ri = _find_det_idx_in_pool(extended_pool, right_det)
                            if ri is not None: used_idx.add(ri)
                        current_frame_dets = extended_pool

                        if left_det is None:
                            l_idx, l_d, l_info = choose_detection_for_side(
                                detections=current_frame_dets,
                                side_track=track_state.left,
                                other_track=track_state.right,
                                width=width, height=height,
                                used_det_indices=used_idx, tier="mid",
                            )
                            if l_d is not None:
                                left_det, left_info = l_d, l_info
                                used_idx.add(l_idx)
                                emergency_local_recoveries += 1
                        if right_det is None:
                            r_idx, r_d, r_info = choose_detection_for_side(
                                detections=current_frame_dets,
                                side_track=track_state.right,
                                other_track=track_state.left,
                                width=width, height=height,
                                used_det_indices=used_idx, tier="mid",
                            )
                            if r_d is not None:
                                right_det, right_info = r_d, r_info
                                used_idx.add(r_idx)
                                emergency_local_recoveries += 1
                association_time += time.perf_counter() - association_start_time

                updated_count = 0

                if left_det is not None:
                    track_state.left.update_with_det(left_det)
                    append_track_row(current_selected_rows, left_det, frame_idx, fps, side="L", track_id=0, missed_count=track_state.left.missed_frames, assoc_info=left_info)
                    updated_count += 1
                else:
                    # v3 결함 ① 수정: 매 매칭 실패마다 missed_frames +1 (kalman_pred 추가 여부와 분리)
                    track_state.left.mark_missed()
                    prediction_fill_start_time = time.perf_counter()
                    pred_box = build_pred_box_from_track(track_state.left, width, height, source="kalman_pred")
                    prediction_fill_time += time.perf_counter() - prediction_fill_start_time
                    if pred_box is not None and track_state.left.missed_frames <= MAX_MISSED_FRAMES:
                        track_state.left.update_with_prediction(pred_box)
                        current_selected_rows.append(copy_det_as_selected_row(pred_box, frame_idx, fps, side="L", source="kalman_pred", missed_count=track_state.left.missed_frames, track_id=0, assoc_info=None))
                        kalman_pred_events += 1

                if right_det is not None:
                    track_state.right.update_with_det(right_det)
                    append_track_row(current_selected_rows, right_det, frame_idx, fps, side="R", track_id=1, missed_count=track_state.right.missed_frames, assoc_info=right_info)
                    updated_count += 1
                else:
                    # v3 결함 ① 수정: 동일 (R 측)
                    track_state.right.mark_missed()
                    prediction_fill_start_time = time.perf_counter()
                    pred_box = build_pred_box_from_track(track_state.right, width, height, source="kalman_pred")
                    prediction_fill_time += time.perf_counter() - prediction_fill_start_time
                    if pred_box is not None and track_state.right.missed_frames <= MAX_MISSED_FRAMES:
                        track_state.right.update_with_prediction(pred_box)
                        current_selected_rows.append(copy_det_as_selected_row(pred_box, frame_idx, fps, side="R", source="kalman_pred", missed_count=track_state.right.missed_frames, track_id=1, assoc_info=None))
                        kalman_pred_events += 1

                if updated_count == 1:
                    one_side_update_frames += 1

                if updated_count >= 1:
                    track_state.global_missed_frames = 0
                else:
                    track_state.global_missed_frames += 1

                # === ⑥ side-level reacquire: 한쪽이 SIDE_REACQUIRE_FRAMES 이상 분실되면 그쪽만 재시도 ===
                for side_track in (track_state.left, track_state.right):
                    if side_track.missed_frames >= SIDE_REACQUIRE_FRAMES:
                        used_indices = set()
                        # 반대쪽이 이번 프레임 매칭에 성공했다면 그 idx 는 사용 중으로 표시
                        other = track_state.right if side_track.side == "L" else track_state.left
                        if other.det is not None and other.missed_frames == 0:
                            for i, d in enumerate(current_frame_dets):
                                if d.get("center_x") == other.det.get("center_x") and \
                                   d.get("center_y") == other.det.get("center_y"):
                                    used_indices.add(i)
                                    break
                        reacq_idx, reacq_det, reacq_info = choose_detection_for_side(
                            detections=current_frame_dets,
                            side_track=side_track,
                            other_track=other,
                            width=width,
                            height=height,
                            used_det_indices=used_indices,
                            tier="wide",
                        )
                        if reacq_det is not None:
                            side_track.update_with_det(reacq_det)
                            append_track_row(current_selected_rows, reacq_det, frame_idx, fps,
                                             side=side_track.side, track_id=side_track.track_id,
                                             missed_count=side_track.missed_frames,
                                             assoc_info=reacq_info)
                            side_reacquire_frames += 1

                if track_state.global_missed_frames > MAX_REACQUIRE_FRAMES:
                    reacquire_start_time = time.perf_counter()
                    (reacq_left_det, reacq_left_info), (reacq_right_det, reacq_right_info) = resolve_two_side_association(
                        detections=current_frame_dets,
                        track_state=track_state,
                        width=width,
                        height=height,
                        tier="wide",
                    )

                    reacquired_any = False
                    if reacq_left_det is not None:
                        track_state.left.update_with_det(reacq_left_det)
                        reacquired_any = True
                    if reacq_right_det is not None:
                        track_state.right.update_with_det(reacq_right_det)
                        reacquired_any = True

                    if reacquired_any:
                        current_selected_rows = []
                        if reacq_left_det is not None:
                            append_track_row(current_selected_rows, reacq_left_det, frame_idx, fps, side="L", track_id=0, missed_count=track_state.left.missed_frames, assoc_info=reacq_left_info)
                        if reacq_right_det is not None:
                            append_track_row(current_selected_rows, reacq_right_det, frame_idx, fps, side="R", track_id=1, missed_count=track_state.right.missed_frames, assoc_info=reacq_right_info)
                        track_state.initialized = True
                        track_state.global_missed_frames = 0
                        reacquire_frames += 1
                    else:
                        best_pair, best_score, pair_rows = choose_best_pair_for_init(current_frame_dets=current_frame_dets, frame_width=width, frame_height=height, frame_idx=frame_idx)
                        pairs_csv.add(pair_rows)
                        if len(best_pair) == 2:
                            best_pair = sorted(best_pair, key=lambda d: d["center_x"])
                            track_state.left.reset()
                            track_state.right.reset()
                            track_state.left.update_with_det(best_pair[0])
                            track_state.right.update_with_det(best_pair[1])
                            track_state.initialized = True
                            track_state.global_missed_frames = 0
                            reacquire_frames += 1
                            # v4 6-3: 재획득 init pair detected 행 score breakdown 채움
                            reinit_assoc_info = {
                                "assoc_score": best_score,
                                "pred_dist_score": 1.0,
                                "size_score": 1.0,
                                "motion_score": 1.0,
                            }
                            current_selected_rows = [
                                copy_det_as_selected_row(best_pair[0], frame_idx, fps, side="L", source="detected", missed_count=track_state.left.missed_frames, track_id=0, assoc_info=reinit_assoc_info),
                                copy_det_as_selected_row(best_pair[1], frame_idx, fps, side="R", source="detected", missed_count=track_state.right.missed_frames, track_id=1, assoc_info=reinit_assoc_info),
                            ]
                        else:
                            if track_state.left.missed_frames > MAX_MISSED_FRAMES and track_state.right.missed_frames > MAX_MISSED_FRAMES:
                                track_state.left.reset()
                                track_state.right.reset()
                                track_state.initialized = False
                    reacquire_time += time.perf_counter() - reacquire_start_time

            # === 모든 프레임 × 좌/우 보장: 빠진 측을 NaN placeholder 행으로 채움 ===
            present_sides = {row["side"] for row in current_selected_rows}
            placeholder_source = "missing" if track_state.initialized else "uninitialized"
            if "L" not in present_sides:
                current_selected_rows.append(
                    make_placeholder_row(
                        frame_idx=frame_idx, fps=fps, side="L", track_id=0,
                        missed_count=track_state.left.missed_frames,
                        source=placeholder_source,
                    )
                )
            if "R" not in present_sides:
                current_selected_rows.append(
                    make_placeholder_row(
                        frame_idx=frame_idx, fps=fps, side="R", track_id=1,
                        missed_count=track_state.right.missed_frames,
                        source=placeholder_source,
                    )
                )

            current_selected_rows = sorted(current_selected_rows, key=lambda r: (r["side"], r["track_id"]))
            selected_csv.add(current_selected_rows)
            total_selected_dets += sum(
                1 for r in current_selected_rows if r["source"] in ("detected", "kalman_pred")
            )
            total_missing_rows += sum(1 for r in current_selected_rows if r["source"] == "missing")
            total_uninitialized_rows += sum(1 for r in current_selected_rows if r["source"] == "uninitialized")

            if SAVE_VIDEO:
                draw_start_time = time.perf_counter()
                # v1-motion: 박스를 원본 frame 위에 그림 (det_frame 아님)
                plotted_original = global_result.plot(img=frame)
                plotted_rule = draw_rule_boxes(frame=frame, selected_rows=current_selected_rows, frame_idx=frame_idx, global_conf=GLOBAL_CONF_THRES, local_conf=LOCAL_CONF_THRES, track_state=track_state)
                draw_time += time.perf_counter() - draw_start_time
                video_write_start_time = time.perf_counter()
                if writer is not None:
                    writer.write(plotted_original)
                if rule_writer is not None:
                    rule_writer.write(plotted_rule)
                video_write_time += time.perf_counter() - video_write_start_time

            if frame_idx < 5 or frame_idx % LOG_EVERY_N_FRAMES == 0:
                lxy = None
                rxy = None
                if track_state.left.det is not None:
                    lxy = (round(track_state.left.det["center_x"], 1), round(track_state.left.det["center_y"], 1))
                if track_state.right.det is not None:
                    rxy = (round(track_state.right.det["center_x"], 1), round(track_state.right.det["center_y"], 1))
                log(f"[INFO] frame_idx={frame_idx} initialized={track_state.initialized} L={lxy} missL={track_state.left.missed_frames} R={rxy} missR={track_state.right.missed_frames} selected={len(current_selected_rows)}", log_path)

            # 다음 프레임 optical flow를 위해 gray 보존 (필요할 때만)
            if curr_gray is not None:
                prev_frame_gray = curr_gray
            elif track_state.initialized and USE_OPTICAL_FLOW_ASSIST:
                gray_start_time = time.perf_counter()
                prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_time += time.perf_counter() - gray_start_time
            else:
                prev_frame_gray = None
            frame_idx += 1

        overall_total_time = time.perf_counter() - overall_start_time
        tracked_frames = max(processed_frames - init_frames, 0)

        csv_save_start_time = time.perf_counter()
        raw_csv.finalize()
        selected_csv.finalize()
        pairs_csv.finalize()
        csv_save_time += time.perf_counter() - csv_save_start_time

        def log_timing(name: str, value_sec: float, denom_frames: Optional[int] = None):
            log(f"[TIME] {name}_sec: {value_sec:.6f}", log_path)
            if denom_frames is not None and denom_frames > 0:
                log(f"[TIME] {name}_ms_per_frame: {value_sec * 1000.0 / denom_frames:.3f}", log_path)

        log("===== save complete =====", log_path)
        log(f"[INFO] processed_frames: {processed_frames}", log_path)
        log(f"[INFO] zero_det_frames: {zero_det_frames}", log_path)
        log(f"[INFO] recovered_read_failures: {recovered_read_failures}", log_path)
        log(f"[INFO] total_raw_dets: {total_raw_dets}", log_path)
        log(f"[INFO] total_selected_dets: {total_selected_dets}", log_path)
        log(f"[INFO] total_missing_rows: {total_missing_rows}", log_path)
        log(f"[INFO] total_uninitialized_rows: {total_uninitialized_rows}", log_path)
        log(f"[INFO] init_frames: {init_frames}", log_path)
        log(f"[INFO] reacquire_frames: {reacquire_frames}", log_path)
        log(f"[INFO] side_reacquire_frames: {side_reacquire_frames}", log_path)
        log(f"[INFO] one_side_update_frames: {one_side_update_frames}", log_path)
        log(f"[INFO] kalman_pred_events: {kalman_pred_events}", log_path)
        log(f"[INFO] mid_tier_recoveries: {mid_tier_recoveries}", log_path)
        log(f"[INFO] emergency_local_runs: {emergency_local_runs}", log_path)
        log(f"[INFO] emergency_local_recoveries: {emergency_local_recoveries}", log_path)
        log(f"[INFO] init_fallback_attempts: {init_fallback_attempts}", log_path)
        log(f"[INFO] init_fallback_frames: {init_fallback_frames}", log_path)
        log(f"[INFO] tracked_frame_counter: {tracked_frame_counter}", log_path)
        log(f"[INFO] local_inference_runs: {local_inference_runs}", log_path)
        if tracked_frame_counter > 0:
            skip_ratio = 1.0 - local_inference_runs / tracked_frame_counter
            log(f"[INFO] local_skip_ratio: {skip_ratio:.3f}", log_path)
        log(f"[INFO] local_trigger_breakdown: {local_trigger_counter}", log_path)
        log(f"[INFO] RAW CSV: {paths['raw_csv_path']}", log_path)
        log(f"[INFO] SELECTED CSV: {paths['selected_csv_path']}", log_path)
        log(f"[INFO] PAIRS CSV: {paths['pairs_csv_path']}", log_path)
        if SAVE_VIDEO:
            log(f"[INFO] original_bbox_video: {paths['annotated_video_path']}", log_path)
            log(f"[INFO] rule_bbox_video: {paths['annotated_video_rule_path']}", log_path)
        if SAVE_FIRST_FRAME:
            log(f"[INFO] first_frame_image: {paths['first_frame_path']}", log_path)
        log(f"[INFO] log_file: {log_path}", log_path)
        log_timing("overall_total", overall_total_time)
        log_timing("video_open", video_open_time)
        log_timing("frame_read", frame_read_time, processed_frames)
        log_timing("frame_recover", frame_recover_time)
        log_timing("grayscale", gray_time, processed_frames)
        log(f"[INFO] motion_darken_active_frames: {motion_darken_active_frames}", log_path)
        log(f"[INFO] motion_darken_fallback_frames: {motion_darken_fallback_frames}", log_path)
        log_timing("median_bg_compute", median_time)
        log_timing("motion_darken", motion_darken_time_total, processed_frames)
        log_timing("global_detect", global_detect_time, processed_frames)
        log_timing("global_extract", global_extract_time, processed_frames)
        log_timing("local_detect", local_detect_time, max(local_inference_runs, 1))
        log_timing("local_extract", local_extract_time, max(local_inference_runs, 1))
        log_timing("merge", merge_time, processed_frames)
        log_timing("init_pair", init_pair_time, init_frames)
        log_timing("track_predict", track_predict_time, tracked_frames)
        log_timing("optical_flow", optical_flow_time, tracked_frames)
        log_timing("association", association_time, tracked_frames)
        log_timing("prediction_fill", prediction_fill_time, tracked_frames)
        log_timing("reacquire", reacquire_time, reacquire_frames)
        log_timing("draw", draw_time, processed_frames if SAVE_VIDEO else None)
        log_timing("video_write", video_write_time, processed_frames if SAVE_VIDEO else None)
        log_timing("csv_save", csv_save_time)

        if processed_frames == 0:
            raise RuntimeError("Failed to process any frames. Check VIDEO_PATH or video decoding.")

    finally:
        # try-finally 로 자원 해제 보장
        try:
            raw_csv.finalize()
            selected_csv.finalize()
            pairs_csv.finalize()
        except Exception:
            pass
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if rule_writer is not None:
            rule_writer.release()


def main():
    ensure_dir(OUTPUT_DIR)
    check_file_exists(VIDEO_PATH, "input video")

    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] MODEL_PATH does not exist: {MODEL_PATH}")
        candidates = auto_find_best_pt()
        if len(candidates) == 0:
            raise FileNotFoundError(f"Could not find best.pt. Current MODEL_PATH: {MODEL_PATH}. Please verify the path manually.")
        print("[INFO] Detected nearby best.pt candidates:")
        for candidate in candidates:
            print(f"  - {candidate}")
        raise FileNotFoundError("MODEL_PATH is likely incorrect. Update MODEL_PATH to one of the actual best.pt paths above.")

    model = YOLO(MODEL_PATH)
    run_inference_robust(model)


if __name__ == "__main__":
    main()
