# DermLIP 피부 질환 진단 시스템

DermLIP (Dermatology Language-Image Pretraining) 모델을 사용하여 피부 사진으로부터 질환을 진단하고, 증상, 영향받는 부위, 설명을 제공하는 AI 진단 도구입니다.

## 📋 프로젝트 소개

이 프로젝트는 최신 Vision-Language 모델인 DermLIP을 활용하여 피부 질환을 진단합니다. 사진을 입력하면 가능성 있는 질환들을 제시하고, 각 질환에 대한 상세 정보(증상, 부위, 설명)를 함께 제공합니다.

### DermLIP 모델 정보

- **훈련 데이터**: Derm1M (1,029,761 피부 이미지-텍스트 쌍)
- **커버리지**: 390개 이상의 피부 질환
- **모델 종류**:
  - **DermLIP-ViT-B/16**: Vision Transformer 기반 (빠른 속도)
  - **DermLIP-PanDerm**: PanDerm 아키텍처 기반 (최고 성능)

### 진단 가능한 질환 (46개)

#### 주요 피부암 및 전암성 병변
- 악성 흑색종 (Malignant Melanoma)
- 기저세포암 (Basal Cell Carcinoma)
- 편평세포암 (Squamous Cell Carcinoma)
- 일광 각화증 (Actinic Keratosis)

#### 염증성 피부 질환
- 여드름 (Acne Vulgaris)
- 습진/아토피 피부염 (Atopic Dermatitis)
- 건선 (Psoriasis)
- 주사/안면홍조 (Rosacea)
- 접촉성 피부염 (Contact Dermatitis)
- 편평태선 (Lichen Planus)

#### 감염성 질환
- 헤르페스 (Herpes Simplex)
- 대상포진 (Herpes Zoster)
- 사마귀 (Warts)
- 물사마귀 (Molluscum Contagiosum)
- 피부 진균 감염/무좀 (Tinea)
- 어루러기 (Pityriasis Versicolor)
- 농가진 (Impetigo)
- 봉와직염 (Cellulitis)
- 옴 (Scabies)

#### 알레르기 및 면역 질환
- 두드러기 (Urticaria)
- 혈관부종 (Angioedema)
- 약물 발진 (Drug Eruption)
- 다형홍반 (Erythema Multiforme)
- 홍반성 루푸스 (Lupus Erythematosus)
- 경피증 (Scleroderma)
- 피부근염 (Dermatomyositis)
- 천포창 (Pemphigus)
- 수포성 유천포창 (Bullous Pemphigoid)

#### 색소 질환
- 백반증 (Vitiligo)
- 기미 (Melasma)
- 흑자/점 (Lentigo)
- 모반 (Nevus)

#### 양성 종양 및 성장물
- 지루성 각화증 (Seborrheic Keratosis)
- 체리 혈관종 (Cherry Angioma)
- 연성 섬유종/쥐젖 (Skin Tag)
- 지방종 (Lipoma)
- 화농육아종 (Pyogenic Granuloma)
- 포도주색 반점 (Port-Wine Stain)
- 혈관종 (Hemangioma)

#### 흉터 및 피부 변화
- 켈로이드 (Keloid)
- 비후성 반흔 (Hypertrophic Scar)
- 모공각화증/닭살 (Keratosis Pilaris)

#### 탈모 질환
- 원형 탈모 (Alopecia Areata)
- 남성형/여성형 탈모 (Androgenetic Alopecia)
- 휴지기 탈모 (Telogen Effluvium)

#### 기타
- 장미색 비강진 (Pityriasis Rosea)

## 🚀 주요 기능

- ✅ **AI 기반 진단**: DermLIP 모델을 사용한 46개 피부 질환 진단
- ✅ **상세 정보 제공**: 각 질환의 증상, 영향받는 부위, 설명
- ✅ **중증도 분류**: 각 질환별 중증도 단계 정보 제공
- ✅ **다중 결과**: 상위 3~5개의 가능성 있는 질환 제시
- ✅ **신뢰도 표시**: 각 진단의 신뢰도 퍼센트 제공
- ✅ **시각화**: 이미지와 진단 결과를 그래프로 표시
- ✅ **리포트 생성**: JSON 및 텍스트 형식의 진단 리포트 자동 생성
- ✅ **다양한 사용 방법**: Python 스크립트 및 Jupyter Notebook 지원

## 📦 설치 방법

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

## 💻 사용 방법

### 방법 1: Python 스크립트 사용 (권장)

가장 간단하고 빠른 방법입니다.

```bash
# 기본 사용 (상위 3개 결과)
python dermlip_diagnosis.py --image path/to/skin_image.jpg

# 상위 5개 결과 보기
python dermlip_diagnosis.py --image path/to/skin_image.jpg --top_k 5

# 고성능 모델 사용 (PanDerm)
python dermlip_diagnosis.py --image path/to/skin_image.jpg \
    --model hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256

# 파일 저장 없이 콘솔 출력만
python dermlip_diagnosis.py --image path/to/skin_image.jpg --no_save
```

### 방법 2: Jupyter Notebook 사용

대화형으로 사용하고 결과를 즉시 확인할 수 있습니다.

```bash
jupyter notebook dermlip_diagnosis.ipynb
```

노트북을 열고 셀을 순서대로 실행하세요:
1. 라이브러리 설치 및 임포트
2. 피부 질환 데이터베이스 로드
3. 진단 시스템 클래스 정의
4. 진단 시스템 초기화
5. **이미지 경로 설정 후 진단 실행** ← 여기서 이미지 경로 입력

### 명령줄 옵션

```bash
python dermlip_diagnosis.py --help
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--image PATH` | 진단할 피부 이미지 경로 (필수) | - |
| `--top_k N` | 상위 N개 결과 표시 | 3 |
| `--model MODEL` | 사용할 DermLIP 모델 | ViT-B-16 |
| `--output_dir PATH` | 결과 저장 디렉토리 | diagnoses/ |
| `--device DEVICE` | 사용할 디바이스 (cuda/cpu) | cuda (가능시) |
| `--no_save` | 파일 저장하지 않음 | False |

## 📊 출력 결과

### 콘솔 출력 예시

```
==================================================================
🏥 진단 결과
==================================================================

[1위] 여드름 (Acne Vulgaris)
└─ 신뢰도: 85.3%

📍 주로 영향받는 부위:
   • 얼굴
   • 이마
   • 뺨
   • 턱
   • 등
   • 가슴

🔍 주요 증상:
   • 피부에 작은 붉은 돌기
   • 화농성 병변 (고름이 찬 여드름)
   • 블랙헤드와 화이트헤드 (면포)
   • 피지 과다 분비로 인한 기름진 피부
   • 염증 및 붓기
   • 여드름 자국 또는 흉터

📋 설명:
   모낭과 피지선의 만성 염증성 질환입니다. 피지 과다 분비, 모공 막힘,
   박테리아 감염이 주요 원인이며, 주로 사춘기에 많이 발생하지만
   성인에게도 나타날 수 있습니다.

📊 중증도 분류:
   • 경증 (면포성)
   • 중등도 (구진/농포성)
   • 중증 (결절성)
   • 최중증 (낭종성)

──────────────────────────────────────────────────────────────────

[2위] 주사 (Rosacea)
└─ 신뢰도: 8.7%

...
```

### 생성되는 파일

진단을 실행하면 `diagnoses/` 디렉토리에 다음 파일들이 생성됩니다:

```
diagnoses/
├── diagnosis_20250108_143022.json    # JSON 형식 진단 리포트
├── diagnosis_20250108_143022.txt     # 텍스트 형식 진단 리포트
└── diagnosis_20250108_143022.png     # 시각화 이미지
```

#### JSON 리포트 예시

```json
{
  "timestamp": "20250108_143022",
  "image_path": "skin_image.jpg",
  "model": "hf-hub:redlessone/DermLIP_ViT-B-16",
  "diagnoses": [
    {
      "rank": 1,
      "confidence": 85.3,
      "disease_name_ko": "여드름",
      "disease_name_en": "Acne Vulgaris",
      "affected_areas": ["얼굴", "이마", "뺨", "턱", "등", "가슴"],
      "symptoms": [
        "피부에 작은 붉은 돌기",
        "화농성 병변 (고름이 찬 여드름)",
        ...
      ],
      "description": "모낭과 피지선의 만성 염증성 질환입니다..."
    }
  ]
}
```

## 🎯 사용 예시

### 예시 1: 기본 진단

```bash
python dermlip_diagnosis.py --image examples/acne_sample.jpg
```

### 예시 2: 더 많은 결과 보기

```bash
python dermlip_diagnosis.py --image examples/rash_sample.jpg --top_k 5
```

### 예시 3: 최고 성능 모델 사용

```bash
python dermlip_diagnosis.py \
    --image examples/melanoma_sample.jpg \
    --model hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256 \
    --top_k 3
```

## 🖼️ 이미지 준비 팁

더 정확한 진단을 위해 다음 사항을 권장합니다:

### 촬영 권장사항
- ✅ **선명한 사진**: 초점이 맞고 흔들리지 않은 사진
- ✅ **충분한 조명**: 자연광 또는 밝은 실내 조명
- ✅ **가까운 거리**: 병변 부위가 명확히 보이도록 촬영
- ✅ **정면 촬영**: 병변을 정면에서 촬영
- ✅ **배경 단순화**: 피부 부위만 보이도록 촬영

### 피해야 할 것
- ❌ 흐릿하거나 초점이 맞지 않은 사진
- ❌ 너무 어둡거나 과다 노출된 사진
- ❌ 너무 멀리서 찍은 사진
- ❌ 강한 그림자가 있는 사진

## 🔬 진단 결과 해석

### 신뢰도 이해하기

- **80% 이상**: 매우 높은 신뢰도, 해당 질환일 가능성이 높음
- **50-80%**: 높은 신뢰도, 가능성이 있음
- **30-50%**: 중간 신뢰도, 다른 진단도 고려 필요
- **30% 미만**: 낮은 신뢰도, 참고용

### 주의사항

⚠️ **중요**: 이 도구는 AI 기반 예측이며 **참고용**입니다.

- 이 진단 결과는 의사의 진단을 대체할 수 없습니다
- 정확한 진단과 치료는 **반드시 피부과 전문의와 상담**하세요
- 특히 피부암이 의심되는 경우 즉시 병원을 방문하세요
- 여러 질환이 비슷한 증상을 보일 수 있습니다

## 🛠️ 문제 해결

### GPU 메모리 부족 오류

```bash
# CPU 사용
python dermlip_diagnosis.py --image image.jpg --device cpu
```

### 모델 다운로드 실패

인터넷 연결을 확인하고, Hugging Face Hub 접근이 가능한지 확인하세요.

```bash
# 모델 수동 다운로드 테스트
python -c "import open_clip; open_clip.create_model_and_transforms('hf-hub:redlessone/DermLIP_ViT-B-16')"
```

### 이미지 로드 오류

- 이미지 경로가 올바른지 확인
- 지원 형식: JPG, JPEG, PNG, BMP
- 파일이 손상되지 않았는지 확인

## 📚 참고 자료

### DermLIP 프로젝트
- **GitHub**: https://github.com/SiyuanYan1/Derm1M
- **논문**: ICCV 2025 (Highlight)

### Hugging Face 모델
- **ViT-B/16**: https://huggingface.co/redlessone/DermLIP_ViT-B-16
- **PanDerm**: https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256

## 📄 라이선스

이 프로젝트는 연구 및 교육 목적으로 사용할 수 있습니다. DermLIP 모델과 Derm1M 데이터셋은 CC BY-NC-4.0 라이선스 하에 제공됩니다.

## ⚖️ 면책 조항

**이 도구는 연구 및 교육 목적으로만 사용되어야 합니다.**

- 이 소프트웨어는 "있는 그대로" 제공되며, 어떠한 보증도 하지 않습니다
- 실제 의료 진단에는 반드시 자격을 갖춘 의료 전문가의 판단이 필요합니다
- 진단 결과에 따른 어떠한 의료적 결정도 전문의와 상담 후 내려야 합니다
- 개발자는 이 도구의 사용으로 인한 어떠한 결과에 대해서도 책임지지 않습니다

## 🤝 기여

버그 리포트, 기능 제안, Pull Request를 환영합니다!

---

**Made with DermLIP** | 피부 건강을 위한 AI 진단 도구
