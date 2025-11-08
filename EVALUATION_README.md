# Fitzpatrick17k 데이터셋 평가 가이드

DermLIP 모델을 Fitzpatrick17k 데이터셋으로 평가하는 스크립트입니다.

## 기능

1. **Fitzpatrick17k 데이터셋 자동 다운로드**
   - GitHub에서 메타데이터 CSV 자동 다운로드
   - 이미지 URL에서 필요한 이미지 자동 다운로드

2. **랜덤 샘플링 및 진단**
   - 지정된 개수만큼 랜덤 샘플링 (기본값: 1000개)
   - DermLIP 모델로 각 이미지 진단
   - Top-1 및 Top-3 예측 결과 생성

3. **결과 분석**
   - Top-1, Top-3 정확도 계산
   - 진단 실패 케이스 분석
   - 가장 많이 실패한 질환 Top 5 추출

4. **CSV 출력**
   - 전체 진단 결과 (`diagnosis_results.csv`)
   - 실패 케이스 (`failed_cases.csv`)
   - Top 5 실패 질환 (`top5_failed_diseases.csv`)
   - 각 질환별 상세 실패 케이스 (`failures_<disease_key>.csv`)
   - 요약 통계 (`summary.csv`)

## 설치

### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

주요 패키지:
- `torch` (PyTorch)
- `open_clip_torch` (DermLIP 모델용)
- `pandas` (데이터 처리)
- `requests` (데이터셋 다운로드)
- `tqdm` (진행 표시)
- `Pillow` (이미지 처리)

### 2. GPU 사용 (권장)

GPU가 있는 경우 훨씬 빠르게 실행됩니다.

```bash
# CUDA가 설치된 경우 PyTorch GPU 버전 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 사용법

### 기본 실행 (1000개 샘플)

```bash
python evaluate_fitzpatrick17k.py
```

### 샘플 수 지정

```bash
# 100개 샘플만 평가
python evaluate_fitzpatrick17k.py --n_samples 100

# 전체 데이터셋 평가
python evaluate_fitzpatrick17k.py --n_samples 17000
```

### CPU로 실행

```bash
python evaluate_fitzpatrick17k.py --device cpu
```

### 출력 디렉토리 지정

```bash
python evaluate_fitzpatrick17k.py --output_dir my_evaluation
```

### 전체 옵션

```bash
python evaluate_fitzpatrick17k.py \
  --n_samples 1000 \
  --model hf-hub:redlessone/DermLIP_ViT-B-16 \
  --output_dir evaluation_results \
  --device cuda
```

## 출력 파일

평가 실행 후 `evaluation_results/` 디렉토리에 다음 파일들이 생성됩니다:

### 1. `diagnosis_results.csv`
전체 진단 결과

| 열 이름 | 설명 |
|--------|------|
| `image_id` | 이미지 고유 ID (md5hash) |
| `image_path` | 로컬 이미지 경로 |
| `gt_label` | Ground truth 레이블 (원본) |
| `gt_disease_key` | 매핑된 질환 키 |
| `pred_disease_key` | 예측된 질환 키 |
| `pred_disease_name` | 예측된 질환 한글명 |
| `pred_confidence` | 예측 신뢰도 (%) |
| `top3_predictions` | Top-3 예측 질환들 |
| `is_correct_top1` | Top-1 정답 여부 |
| `is_correct_top3` | Top-3 정답 여부 |
| `has_gt_mapping` | Ground truth 매핑 가능 여부 |
| `fitzpatrick_scale` | Fitzpatrick skin type |

### 2. `failed_cases.csv`
진단 실패한 케이스만 필터링

### 3. `top5_failed_diseases.csv`
가장 많이 실패한 질환 Top 5

| 열 이름 | 설명 |
|--------|------|
| `rank` | 순위 (1-5) |
| `disease_key` | 질환 키 |
| `disease_name_ko` | 질환 한글명 |
| `disease_name_en` | 질환 영문명 |
| `failure_count` | 실패 횟수 |
| `failure_percentage` | 전체 실패 중 비율 (%) |

### 4. `failures_<disease_key>.csv`
각 Top 5 질환별 상세 실패 케이스

예: `failures_acne.csv`, `failures_eczema.csv`, ...

### 5. `summary.csv`
전체 요약 통계

| 열 이름 | 설명 |
|--------|------|
| `total_samples` | 전체 평가 샘플 수 |
| `mapped_samples` | 매핑 가능한 샘플 수 |
| `top1_accuracy` | Top-1 정확도 (%) |
| `top3_accuracy` | Top-3 정확도 (%) |
| `failed_cases` | 실패 케이스 수 |

## 실행 예시

```bash
$ python evaluate_fitzpatrick17k.py --n_samples 1000

======================================================================
Fitzpatrick17k 평가 시스템
======================================================================
모델: hf-hub:redlessone/DermLIP_ViT-B-16
디바이스: cuda

✓ 모델 로드 완료!
✓ 46개 피부 질환 진단 가능
======================================================================

📥 Fitzpatrick17k 메타데이터 로드 중...
✓ 총 16577개 샘플 발견
✓ 1000개 샘플 랜덤 추출

🔍 DermLIP 모델로 진단 시작...

진단 중: 100%|████████████████████| 1000/1000 [15:30<00:00,  1.08it/s]

✓ 전체 진단 결과 저장: evaluation_results/diagnosis_results.csv

======================================================================
📊 평가 결과 (매핑 가능한 650개 샘플)
======================================================================
Top-1 정확도: 45.23%
Top-3 정확도: 68.15%
======================================================================

✓ 실패 케이스 저장: evaluation_results/failed_cases.csv

======================================================================
🔴 가장 많이 진단 실패한 질환 Top 5
======================================================================
1. 여드름 (Acne Vulgaris)
   실패 횟수: 85회 (23.9% of failures)
2. 습진 (아토피 피부염) (Atopic Dermatitis / Eczema)
   실패 횟수: 62회 (17.4% of failures)
3. 건선 (Psoriasis)
   실패 횟수: 48회 (13.5% of failures)
4. 모반 (점) (Melanocytic Nevus / Mole)
   실패 횟수: 41회 (11.5% of failures)
5. 주사 (안면홍조) (Rosacea)
   실패 횟수: 35회 (9.8% of failures)
======================================================================

✓ Top 5 실패 질환 저장: evaluation_results/top5_failed_diseases.csv
✓ 여드름 실패 케이스 상세: evaluation_results/failures_acne.csv
✓ 습진 (아토피 피부염) 실패 케이스 상세: evaluation_results/failures_eczema.csv
✓ 건선 실패 케이스 상세: evaluation_results/failures_psoriasis.csv
✓ 모반 (점) 실패 케이스 상세: evaluation_results/failures_nevus.csv
✓ 주사 (안면홍조) 실패 케이스 상세: evaluation_results/failures_rosacea.csv

✓ 요약 통계 저장: evaluation_results/summary.csv

✅ 평가 완료! 결과는 evaluation_results/ 디렉토리에 저장되었습니다.
```

## 데이터셋 정보

### Fitzpatrick17k

- **출처**: https://github.com/mattgroh/fitzpatrick17k
- **크기**: 약 16,577개 임상 이미지
- **특징**:
  - 다양한 피부 톤 (Fitzpatrick skin type I-VI)
  - 114개 피부 질환
  - 실제 임상 이미지

### 데이터 저장 위치

```
data/
└── fitzpatrick17k/
    ├── fitzpatrick17k.csv          # 메타데이터
    └── images/                     # 다운로드된 이미지들
        ├── <md5hash1>.jpg
        ├── <md5hash2>.jpg
        └── ...
```

## 주의사항

### 1. 레이블 매핑

Fitzpatrick17k의 114개 질환과 우리 시스템의 46개 질환 간 매핑이 필요합니다.
매핑되지 않은 질환은 평가에서 제외됩니다.

현재 매핑된 질환:
- Acne → acne
- Atopic Dermatitis / Eczema → eczema
- Psoriasis → psoriasis
- Melanoma → melanoma
- Basal Cell Carcinoma → basal_cell_carcinoma
- 기타 약 25개 질환

매핑은 `map_label_to_disease_key()` 함수에서 정의됩니다.

### 2. 실행 시간

- **GPU 사용 시**: 1000개 샘플 약 15-20분
- **CPU 사용 시**: 1000개 샘플 약 2-3시간

### 3. 디스크 공간

이미지 다운로드로 인해 디스크 공간이 필요합니다:
- 1000개 샘플: 약 500MB
- 전체 데이터셋: 약 8GB

## 문제 해결

### 1. 메타데이터 다운로드 실패

수동으로 다운로드:
```bash
mkdir -p data/fitzpatrick17k
cd data/fitzpatrick17k
wget https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv
```

### 2. GPU 메모리 부족

배치 크기를 줄이거나 CPU 사용:
```bash
python evaluate_fitzpatrick17k.py --device cpu
```

### 3. 이미지 다운로드 실패

일부 이미지는 URL이 만료되었을 수 있습니다.
스크립트는 실패한 이미지를 건너뜁니다.

## 확장

### 레이블 매핑 추가

`evaluate_fitzpatrick17k.py`의 `map_label_to_disease_key()` 함수를 수정하여 더 많은 질환 매핑 추가:

```python
def map_label_to_disease_key(self, label):
    mapping = {
        # 여기에 새로운 매핑 추가
        'new disease name': 'disease_key',
    }
    # ...
```

### 다른 모델 사용

```bash
python evaluate_fitzpatrick17k.py \
  --model hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256
```

## 참고 문헌

- Fitzpatrick17k: https://github.com/mattgroh/fitzpatrick17k
- DermLIP: https://huggingface.co/redlessone/DermLIP_ViT-B-16
- Derm1M dataset: https://github.com/JamesQFreeman/Derm1M
