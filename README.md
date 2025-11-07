# DermLIP 피부 질환 진단 정확도 평가

DermLIP (Dermatology Language-Image Pretraining) 모델을 사용하여 피부 질환 이미지에 대한 진단 정확도를 평가하는 프로젝트입니다.

## 프로젝트 소개

이 프로젝트는 최신 Vision-Language 모델인 DermLIP을 활용하여 피부 질환을 진단하고, 그 정확도를 평가합니다. DermLIP은 1백만 개 이상의 피부과 이미지-텍스트 쌍으로 훈련되어 390개 이상의 피부 질환을 인식할 수 있습니다.

### DermLIP 모델 정보

- **훈련 데이터**: Derm1M (1,029,761 이미지-텍스트 쌍)
- **지원 질환**: 390개 피부 질환
- **모델 종류**:
  - DermLIP-ViT-B/16: Vision Transformer 기반 (빠른 속도)
  - DermLIP-PanDerm: PanDerm 아키텍처 기반 (최고 성능)

### 주요 기능

- ✅ Zero-shot 피부 질환 분류
- ✅ 배치 단위 이미지 평가
- ✅ 단일 이미지 진단
- ✅ 정확도, 정밀도, 재현율, F1 스코어 계산
- ✅ Confusion Matrix 시각화
- ✅ 클래스별 성능 분석
- ✅ Jupyter Notebook 및 Python 스크립트 지원

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd SA-project
```

### 2. Python 환경 설정

Python 3.9 이상이 필요합니다.

```bash
# Conda 환경 생성 (권장)
conda create -n dermlip python=3.9
conda activate dermlip

# 또는 venv 사용
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 방법 1: Jupyter Notebook 사용

```bash
jupyter notebook dermlip_diagnosis_evaluation.ipynb
```

노트북을 열고 셀을 순서대로 실행하면 됩니다.

### 방법 2: Python 스크립트 사용

#### CSV 파일에서 데이터 로드

```bash
python evaluate_dermlip.py --data_csv data/labels.csv --output_dir results/
```

CSV 파일 형식:
```csv
image_path,label
data/images/acne/image1.jpg,0
data/images/eczema/image2.jpg,1
...
```

#### 폴더 구조에서 데이터 로드

```bash
python evaluate_dermlip.py --data_dir data/images/ --output_dir results/
```

폴더 구조:
```
data/images/
├── acne/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── eczema/
│   ├── image1.jpg
│   └── ...
└── ...
```

#### 단일 이미지 테스트

```bash
python evaluate_dermlip.py --test_image path/to/image.jpg --output_dir results/
```

#### 고성능 모델 사용 (PanDerm)

```bash
python evaluate_dermlip.py \
    --data_csv data/labels.csv \
    --model hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256 \
    --output_dir results/
```

### 전체 옵션

```bash
python evaluate_dermlip.py --help
```

```
옵션:
  --data_csv PATH       평가 데이터 CSV 파일 경로
  --data_dir PATH       이미지가 클래스별 폴더에 정리된 디렉토리
  --test_image PATH     단일 이미지 테스트 경로
  --model MODEL         사용할 DermLIP 모델 (기본: DermLIP_ViT-B-16)
  --output_dir PATH     결과 저장 디렉토리 (기본: results/)
  --batch_size N        배치 크기 (기본: 32)
  --device DEVICE       사용할 디바이스 (cuda/cpu)
```

## 피부 질환 클래스

현재 지원하는 10개 피부 질환 클래스 (실제 데이터에 맞게 수정 가능):

| 클래스 ID | 질환명 | 설명 |
|---------|--------|------|
| 0 | Acne | 여드름 - 염증성 피부 질환 |
| 1 | Eczema | 습진 - 가려운 붉은 피부 염증 |
| 2 | Psoriasis | 건선 - 붉은 비늘 모양 피부 질환 |
| 3 | Melanoma | 흑색종 - 악성 피부암 |
| 4 | Basal Cell Carcinoma | 기저세포암 - 흔한 피부암 |
| 5 | Seborrheic Keratosis | 지루성 각화증 - 양성 갈색 성장물 |
| 6 | Rosacea | 주사 - 얼굴 홍조 및 혈관 확장 |
| 7 | Vitiligo | 백반증 - 피부 색소 손실 |
| 8 | Herpes | 헤르페스 - 바이러스성 수포 |
| 9 | Warts | 사마귀 - HPV로 인한 작은 돌기 |

**주의**: `evaluate_dermlip.py` 파일에서 `SKIN_DISEASE_CLASSES`와 `CLASS_DESCRIPTIONS`를 실제 데이터에 맞게 수정하세요.

## 출력 결과

평가를 실행하면 다음 파일들이 생성됩니다:

```
results/
├── evaluation_results.json      # 평가 지표 (JSON)
├── confusion_matrix.png         # Confusion Matrix 히트맵
├── class_performance.png        # 클래스별 성능 막대 그래프
└── single_prediction.png        # 단일 이미지 예측 결과 (--test_image 사용시)
```

### 평가 지표

- **Accuracy (정확도)**: 전체 예측 중 올바른 예측의 비율
- **Precision (정밀도)**: 양성으로 예측한 것 중 실제 양성의 비율
- **Recall (재현율)**: 실제 양성 중 올바르게 예측한 비율
- **F1 Score**: Precision과 Recall의 조화 평균

## 데이터 준비 가이드

### 1. 이미지 수집

- 고품질의 피부 질환 이미지를 수집합니다.
- 다양한 각도, 조명, 피부 톤을 포함하는 것이 좋습니다.
- 이미지 형식: JPG, JPEG, PNG, BMP

### 2. 레이블링

각 이미지에 정확한 진단 레이블을 할당합니다.

### 3. 데이터 구성

**옵션 A: CSV 파일**
```csv
image_path,label
/path/to/image1.jpg,0
/path/to/image2.jpg,1
```

**옵션 B: 폴더 구조**
```
data/images/
├── acne/
├── eczema/
├── psoriasis/
└── ...
```

## 성능 최적화 팁

### GPU 사용

CUDA 지원 GPU가 있으면 자동으로 GPU를 사용합니다.

```bash
# GPU 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 배치 크기 조정

메모리에 따라 배치 크기를 조정하세요:

```bash
# GPU 메모리가 충분한 경우
python evaluate_dermlip.py --data_csv data.csv --batch_size 64

# GPU 메모리가 부족한 경우
python evaluate_dermlip.py --data_csv data.csv --batch_size 16
```

### 모델 선택

- **빠른 테스트**: `DermLIP_ViT-B-16` (기본값)
- **최고 성능**: `DermLIP_PanDerm-base-w-PubMed-256`

## 클래스 설명 커스터마이징

더 나은 성능을 위해 클래스 설명을 상세하게 작성할 수 있습니다:

```python
CLASS_DESCRIPTIONS = [
    "a clinical photo of acne vulgaris with comedones, papules, and pustules on oily skin",
    "a dermatological image of atopic eczema showing erythematous, pruritic, and scaly patches",
    # ... 나머지 클래스
]
```

DermLIP은 임상적 용어와 상세한 설명으로 훈련되었으므로, 의학적으로 정확한 설명이 효과적입니다.

## 문제 해결

### CUDA Out of Memory 오류

```bash
# 배치 크기 감소
python evaluate_dermlip.py --data_csv data.csv --batch_size 8

# 또는 CPU 사용
python evaluate_dermlip.py --data_csv data.csv --device cpu
```

### 모델 다운로드 실패

인터넷 연결을 확인하고, Hugging Face Hub에 접근 가능한지 확인하세요.

```bash
# 수동으로 모델 다운로드
python -c "import open_clip; open_clip.create_model_and_transforms('hf-hub:redlessone/DermLIP_ViT-B-16')"
```

### 이미지 로드 오류

- 이미지 경로가 올바른지 확인
- 이미지 파일이 손상되지 않았는지 확인
- 지원 형식: JPG, JPEG, PNG, BMP

## 참고 자료

- **DermLIP GitHub**: https://github.com/SiyuanYan1/Derm1M
- **Hugging Face Models**:
  - https://huggingface.co/redlessone/DermLIP_ViT-B-16
  - https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256
- **논문**: ICCV 2025 (Highlight)

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 사용할 수 있습니다. DermLIP 모델과 Derm1M 데이터셋은 CC BY-NC-4.0 라이선스 하에 제공됩니다.

## 기여

버그 리포트, 기능 제안, Pull Request를 환영합니다!

## 면책 조항

이 도구는 연구 및 교육 목적으로만 사용되어야 합니다. 실제 의료 진단에는 반드시 자격을 갖춘 의료 전문가의 판단이 필요합니다.
