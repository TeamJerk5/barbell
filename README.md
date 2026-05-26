# Barbell_final — 역도 바벨 원판 자동 트래킹 파이프라인

mp4 영상 하나를 넣으면 4단계 파이프라인을 거쳐 **프레임별 좌/우 원판 중심 좌표 CSV**를 만들어 주는 자동화 도구입니다.

이 README는 사전 요구사항·설치·사용법·주의사항에 대해 다루고 있습니다.

---

## 1. 한눈에 보는 파이프라인

```
input.mp4 (권장: 60 fps, 1920x1080)
   │
   ▼  Stage 1) inference_motion_v2.py
   │      YOLO + temporal-median motion darkening 기반
   │      매 프레임 좌/우 plate 검출 + 트래킹
   │
detections_selected_<clip>.csv
   │
   ▼  Stage 2) post_process_seg_v1.py
   │      Strong/Weak anchor 분류, 봉 길이/플레이트 크기 prior,
   │      다단계 fill (kalman → bar-vector → lerp → extrap → freeze)
   │
detections_postproc_<clip>.csv
   │
   ▼  Stage 3) select_seeds.py
   │      CoTracker 시드 자동 선정
   │      (S1 both-detected ∧ S2 confidence ∧ S3 bar-length 일관성
   │       ∧ S4 위치 일관성, fallback 포함)
   │
auto_seeds_<clip>.json
   │
   ▼  Stage 4) cotracker_seed_track.py
          CoTracker3 offline (forward + backward)
          시드 기반 정밀 트래킹

cotracker_<clip>.csv           ← 최종 결과 (frame_idx, side, x, y, visible)
```

---

## 2. 사전 요구사항

### 2.1 OS / 하드웨어

| 구분 | 권장 | 비고 |
|---|---|---|
| OS | macOS (Apple Silicon) / Linux | Windows는 미검증 |
| Python | **3.9 이상** | 3.10~3.12 권장 |
| RAM | 16 GB+ | CoTracker3가 영상 전체를 메모리로 로드 |
| 가속기 | CUDA GPU > Apple MPS > CPU | 없어도 동작하지만 매우 느림 |

처리 시간 참고 (598 프레임, 1920x1080, 60 fps 약 10초 영상 기준)
- Apple M2 (MPS, 일부 CPU fallback): **약 6분** (inference ~72s + cotracker ~280s)
- NVIDIA CUDA GPU: **2~3분**
- CPU only: **20분 이상**

> 대부분의 시간은 Stage 4 (CoTracker3) 가 차지합니다. macOS에서는 5.1 항목의 MPS fallback 때문에 더 느립니다. 빨리 끝내고 싶으면 5.4의 `COTRACKER_MAX_SIDE` 를 낮추거나 CUDA 환경에서 돌리세요.

### 2.2 시스템 도구

- **`5data_best.pt` 모델 가중치 (131 MB)** — GitHub 100 MB 단일 파일 제한 때문에 저장소에 포함되어 있지 **않습니다**. 아래 3.1 항목을 따라 별도로 다운로드해 `Barbell_final/` 폴더 안에 넣어 주세요.
- **ffmpeg / ffprobe** — 선택. 영상 메타데이터 로깅에만 사용, 없어도 파이프라인은 정상 동작

### 2.3 Python 패키지

`requirements.txt`로 일괄 설치합니다. 주요 항목:
- `torch`, `torchvision`
- `ultralytics >= 8.0`
- `opencv-python >= 4.8`
- `numpy`, `pandas`, `openpyxl`
- `einops`, `imageio`, `imageio-ffmpeg`, `tqdm` (CoTracker3 의존성)

---

## 3. 설치 단계별 가이드

### 3.1 저장소 clone + 모델 다운로드

```bash
# (1) 저장소 clone
git clone <repo-url>
cd Barbell_final

# (2) 5data_best.pt 가중치를 아래 링크에서 받아 Barbell_final/ 폴더 안에 넣어주세요.
```

**모델 다운로드 링크 (Google Drive, 131 MB)**:
https://drive.google.com/file/d/1P1TfvX5zjHwE97-fRkn2zITloppjbkl7/view?usp=sharing

브라우저로 열어 우측 상단의 다운로드 버튼을 눌러 받으신 뒤 `Barbell_final/5data_best.pt` 위치에 두시면 됩니다.

CLI에서 받고 싶다면 `gdown` 사용:
```bash
pip install gdown
cd Barbell_final
gdown 1P1TfvX5zjHwE97-fRkn2zITloppjbkl7 -O 5data_best.pt
```

검증:
```bash
ls -lh 5data_best.pt    # 약 131 MB (정확히는 131,260,284 byte) 이면 OK
```

> ⚠️ `5data_best.pt` 파일이 없거나 크기가 다르면 Stage 1에서 `FileNotFoundError` 또는 `_pickle.UnpicklingError` 가 발생합니다. 이 경우 다시 다운로드받아 위치/크기를 확인하세요.

### 3.2 Python 가상환경 + 패키지 설치

```bash
# venv 생성 & 활성화
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

설치 검증:
```bash
python -c "import torch, cv2, ultralytics, pandas; print('OK')"
```

### 3.3 CoTracker3 가중치 (첫 실행 시 자동 다운로드)

`run_pipeline.py` 첫 실행 시 `torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")`가 호출되어 약 **50 MB** 가중치를 `~/.cache/torch/hub/` 아래에 캐시합니다. **인터넷 연결이 1회 필요**합니다. 이후로는 오프라인에서도 동작합니다.

미리 받아두려면:
```bash
python -c "import torch; torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline')"
```

---

## 4. 실행

### 4.1 가장 기본적인 사용

```bash
python run_pipeline.py /path/to/input.mp4
```

기본값:
- `--out ./out` — 출력 루트
- `--clip` — 파일명에서 `KNSF_dayN_NNN` 패턴이 있으면 자동 추출, 없으면 파일명 stem(확장자 제외) 사용

### 4.2 옵션 지정

```bash
python run_pipeline.py /data/videos/clip.mp4 \
    --out ./result \
    --clip my_lift_01
```

### 4.3 여러 영상 일괄 처리

```bash
for f in /data/videos/*.mp4; do
    python run_pipeline.py "$f" --out ./batch_out
done
```
각 영상은 `./batch_out/<clip>/` 하위에 자체 디렉토리를 가집니다.

### 4.4 출력 결과물

```
<out>/
├── <clip>.mp4                              입력 영상 심볼릭링크 (Stage 4가 canonical 이름 요구)
├── auto_seeds_<clip>.json                  Stage 3 결과 (시드 좌표 + 점수 + run 정보)
├── cotracker_summary_auto.txt              Stage 4 실행 요약
└── <clip>/
    ├── detections_selected_<clip>.csv      Stage 1: 매 프레임 L/R 추적 결과
    ├── detections_postproc_<clip>.csv      Stage 2: 후처리된 CSV
    ├── run_log_v9_v4_g0.15_l0.08.txt       Stage 1 로그 (suffix는 conf threshold)
    └── cotracker_<clip>.csv                ⭐ 최종 결과 (frame_idx, side, x, y, visible)
```

**가장 중요한 산출물은 `cotracker_<clip>.csv`** 입니다. CoTracker3 단계는 CSV만 출력하며 별도의 시각화 mp4는 생성하지 않습니다 (시각화가 필요하면 9번 FAQ 참고).

---

## 5. 주의사항 / 트러블슈팅 (반드시 한 번씩 읽어주세요)

### 5.1 macOS MPS의 `grid_sampler_3d` 미지원

CoTracker3가 내부적으로 사용하는 `aten::grid_sampler_3d` 가 Apple MPS 백엔드에 아직 구현되어 있지 않아 해당 op은 CPU로 fallback 됩니다. `run_pipeline.py`는 자동으로 환경변수 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 설정하므로 **별도 작업 없이 잘 돕니다**. 다만 해당 op 부분은 CPU로 처리되어 ~5~10배 느려집니다 (cotracker 단계 전체에서 60fps 10초 영상 기준 5~6분).

### 5.2 모델 가중치 누락 / 손상 → 모델 로드 실패

`5data_best.pt`는 GitHub 저장소에 포함되어 있지 **않습니다**. 3.1 안내에 따라 별도로 다운로드해야 합니다.

증상 (Stage 1에서 발생):
```
FileNotFoundError: ... 5data_best.pt
```
또는
```
_pickle.UnpicklingError: invalid load key, 'v'.
RuntimeError: PytorchStreamReader failed reading zip archive
```

후자(`_pickle.UnpicklingError`)는 다운로드가 중간에 끊겨 파일이 손상된 경우 흔합니다.

해결:
```bash
ls -lh 5data_best.pt    # 정상이면 131 MB (131,260,284 byte)
```
파일이 없거나 크기가 다르면 3.1의 Drive 링크에서 다시 받으세요.

### 5.3 OpenCV가 AV1/HEVC 영상을 못 여는 경우

OpenCV 빌드에 따라 일부 코덱(AV1, HEVC 등) 디코딩이 안 될 수 있습니다. Stage 1에서 `Failed to process any frames` 같은 오류가 나면 H.264로 트랜스코드 후 재시도하세요:

```bash
ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p -crf 18 input_h264.mp4
python run_pipeline.py input_h264.mp4
```

대부분의 60 fps 1080p mp4는 H.264이므로 문제가 없습니다.

### 5.4 GPU 메모리 부족 (OOM)

CoTracker3는 영상 전체 프레임을 GPU 메모리에 올립니다. 10초 60fps 1080p 영상 기준 약 6 GB VRAM이면 충분하지만, 더 긴 영상이면 OOM이 발생할 수 있습니다. 환경변수로 우회:

```bash
# 1) longest side를 줄여 메모리 절감 (기본 768)
COTRACKER_MAX_SIDE=512 python run_pipeline.py input.mp4

# 2) CoTracker만 CPU로 강제 (느리지만 OOM 회피)
COTRACKER_DEVICE=cpu python run_pipeline.py input.mp4
```

### 5.5 스크립트 안에 박혀 있는 `/Users/byeol/...` 경로

`inference_motion_v2.py`, `post_process_seg_v1.py`, `select_seeds.py`, `cotracker_seed_track.py` 내부에 개발자(byeol)의 macOS 절대경로가 **기본 fallback default**로 남아 있습니다. **`run_pipeline.py`로 실행하는 한 이 경로는 절대 사용되지 않습니다** (env/CLI 인자로 항상 덮어씁니다). 개별 스크립트를 직접 호출하는 경우에만 8번 섹션을 참고해 모든 경로를 명시적으로 넘겨주세요.

### 5.6 입력 영상 권장 사양

| 항목 | 권장 | 동작은 함 (정확도 저하 가능) |
|---|---|---|
| 해상도 | 1920x1080 | 1280x720 이상 |
| 프레임레이트 | 60 fps | 30 fps |
| 컨테이너/코덱 | mp4 / H.264 | mp4 / AV1, HEVC (5.3 참고) |
| 길이 | ≤ 15초 | 더 길어도 OK이나 OOM 주의 (5.4) |

### 5.7 ultralytics 첫 사용 시 홈디렉토리에 폴더 생성

ultralytics 라이브러리가 `~/Library/Application Support/Ultralytics/` (macOS) 또는 `~/.config/Ultralytics/` (Linux)에 설정/폰트 폴더를 만듭니다. 무해합니다. 권한 문제 발생 시:
```bash
YOLO_CONFIG_DIR=./yolo_config python run_pipeline.py input.mp4
```

### 5.8 인터넷 없는 환경

CoTracker3 가중치를 미리 캐시해 두면 (3.4 마지막 명령) 그 후로는 오프라인 동작 가능합니다.

---

## 6. 디렉토리 / 파일 구성

```
Barbell_final/
├── 5data_best.pt              YOLO weights (131 MB, GitHub 미포함 — 3.1 참고)
├── README.md                  이 문서
├── requirements.txt           Python 의존성
├── .gitignore                 *.pt, out/, __pycache__/, .DS_Store 등 제외
│
├── run_pipeline.py            ⭐ Entry point — 4단계 오케스트레이터
│
├── inference_motion_v2.py     Stage 1: YOLO + motion darkening
├── post_process_seg_v1.py     Stage 2: anchor 기반 후처리
├── select_seeds.py            Stage 3: cotracker 시드 선정
└── cotracker_seed_track.py    Stage 4: CoTracker3 offline
```

---

## 7. 파라미터 튜닝 위치

| 항목 | 파일 / 위치 |
|---|---|
| YOLO confidence threshold | `inference_motion_v2.py` 상단 `GLOBAL_CONF_THRES`, `LOCAL_CONF_THRES` |
| motion darkening 강도 | `inference_motion_v2.py` 상단 `MOTION_BG_DARKEN_ALPHA`, `MOTION_DIFF_THRESHOLD` |
| Kalman / tracking 게이트 | `inference_motion_v2.py` 상단 `LOCAL_GATE_*`, `MAX_MISSED_FRAMES` 등 |
| post-process anchor 임계 | `post_process_seg_v1.py` 상단 `STRONG_GAP_MAX`, `BAR_LENGTH_MAD_K`, `POS_RESIDUAL_MAD_K` 등 |
| seed 선정 임계 | `select_seeds.py`의 `DEFAULTS` dict |
| cotracker 입력 리사이즈 | 환경변수 `COTRACKER_MAX_SIDE` (기본 768) |
| cotracker device | 환경변수 `COTRACKER_DEVICE=auto|cpu|mps|cuda` |

---

## 8. 개별 스크립트 직접 실행 (고급)

`run_pipeline.py`를 사용하면 모든 게 자동이지만, 디버깅이나 일부 단계만 다시 실행하고 싶을 때 다음과 같이 개별 호출할 수 있습니다.

```bash
CLIP=KNSF_day5_035
OUT=./out
mkdir -p "$OUT/$CLIP"
ln -sf /abs/path/to/input.mp4 "$OUT/$CLIP.mp4"

# Stage 1: inference (env-var 주입)
INFERENCE_RULE_V9_MODEL_PATH="$(pwd)/5data_best.pt" \
INFERENCE_RULE_V9_VIDEO_PATH="$OUT/$CLIP.mp4" \
INFERENCE_MOTION_V2_OUTPUT_DIR="$OUT/$CLIP" \
INFERENCE_MOTION_V2_FAST=1 \
python inference_motion_v2.py

# Stage 2: post-process (positional arg = output root)
python post_process_seg_v1.py "$OUT"

# Stage 3: seed selection
python select_seeds.py --root "$OUT" --out "$OUT/auto_seeds_$CLIP.json"

# Stage 4: cotracker (CSV만 생성, mp4 없음)
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python cotracker_seed_track.py \
    --video-dir "$OUT" --run-out "$OUT" \
    --seeds-json "$OUT/auto_seeds_$CLIP.json" \
    --only "$CLIP" --out-suffix _auto
```

> 참고: Stage 4의 `--out-suffix`는 현재 `cotracker_summary{suffix}.txt` 요약 파일명에만 영향을 미칩니다. 트랙 CSV 본문은 항상 `cotracker_<clip>.csv` 로 고정 출력됩니다.

---

## 9. 자주 묻는 질문 (FAQ)

**Q1. 시드를 수동으로 지정하고 싶어요.**
`auto_seeds_<clip>.json`을 직접 편집하거나, `cotracker_seed_track.py`의 `SEEDS_BUILTIN` 딕셔너리에 `{clip_name: frame_idx}` 형태로 추가한 뒤 `--seeds-json` 없이 실행하세요.

**Q2. 중간 단계까지만 다시 돌리고 싶어요.**
8번 섹션의 개별 호출을 사용하세요. 예: Stage 1, 2 결과가 이미 있다면 Stage 3, 4만 실행 가능.

**Q3. 어느 단계에서 실패했는지 확인하려면?**
- 콘솔에 stage별 명령이 `$ python ...` 형태로 출력됩니다.
- Stage 1 상세 로그: `<out>/<clip>/run_log_v9_v4_*.txt`
- `run_pipeline.py`는 한 stage라도 non-zero exit이면 즉시 중단하고 어느 stage가 실패했는지 출력합니다.

**Q4. CoTracker3가 너무 느려요.**
- CUDA GPU 사용: 자동 감지됨
- 영상 해상도 낮추기: `COTRACKER_MAX_SIDE=512`
- 영상 자체를 더 짧게 자르기 (10초 이하 권장)

**Q5. 트래킹 결과를 영상으로 시각화하고 싶어요.**
현 버전의 Stage 4는 의도적으로 CSV만 출력합니다. 시각화가 필요하면 `cotracker_<clip>.csv`(`frame_idx, side, x, y, visible`)와 입력 영상을 같이 읽어 OpenCV로 점·라인을 그리는 간단한 스크립트를 별도로 작성해 사용하세요. 컬러 컨벤션 참고:
- 좌측(L): 초록, 우측(R): 파랑
- 양쪽 visible일 때 두 점을 잇는 회색 봉 line
- `visible=0` 인 점은 hollow circle, `?` 라벨로 구분

---

## 10. 라이선스 / 출처

- YOLO 모델 학습: [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- CoTracker3: [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker)
- 본 파이프라인 코드: 팀 내부 사용 / 공유 목적

---

## 문제가 있다면

1. 5번 섹션 (트러블슈팅) 먼저 확인
2. 콘솔 출력과 `<out>/<clip>/run_log_v9_v4_*.txt` 첨부해 문의
3. 가능하면 입력 영상의 첫 1~2초 샘플도 함께 공유
