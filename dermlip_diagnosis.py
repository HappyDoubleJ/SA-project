#!/usr/bin/env python3
"""
DermLIP 모델을 사용한 피부 질환 진단 시스템

사용법:
    python dermlip_diagnosis.py --image path/to/skin_image.jpg
    python dermlip_diagnosis.py --image path/to/skin_image.jpg --top_k 5
"""

import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# 피부 질환 데이터베이스 (50개 이상의 주요 질환)
SKIN_DISEASE_DATABASE = {
    'acne': {
        'name_ko': '여드름',
        'name_en': 'Acne Vulgaris',
        'affected_areas': ['얼굴', '이마', '뺨', '턱', '등', '가슴'],
        'symptoms': [
            '피부에 작은 붉은 돌기',
            '화농성 병변 (고름이 찬 여드름)',
            '블랙헤드와 화이트헤드 (면포)',
            '피지 과다 분비로 인한 기름진 피부',
            '염증 및 붓기',
            '여드름 자국 또는 흉터'
        ],
        'description': '모낭과 피지선의 만성 염증성 질환입니다. 피지 과다 분비, 모공 막힘, 박테리아 감염이 주요 원인이며, 주로 사춘기에 많이 발생하지만 성인에게도 나타날 수 있습니다.',
        'severity_levels': ['경증 (면포성)', '중등도 (구진/농포성)', '중증 (결절성)', '최중증 (낭종성)']
    },
    'eczema': {
        'name_ko': '습진 (아토피 피부염)',
        'name_en': 'Atopic Dermatitis / Eczema',
        'affected_areas': ['팔꿈치 안쪽', '무릎 뒤쪽', '손목', '발목', '얼굴', '목', '손'],
        'symptoms': [
            '극심한 가려움증 (특히 밤에 심함)',
            '붉고 염증이 있는 피부',
            '건조하고 비늘처럼 벗겨지는 피부',
            '피부 갈라짐 및 진물',
            '만성적인 긁음으로 인한 피부 두꺼워짐',
            '색소 침착 또는 탈색'
        ],
        'description': '만성 재발성 염증성 피부 질환으로 심한 가려움증이 특징입니다. 유전적 요인, 면역 체계 이상, 피부 장벽 기능 저하 등이 복합적으로 작용합니다.',
        'severity_levels': ['경증 (BSA <10%, 경미한 가려움)', '중등도 (BSA 10-50%, 중간 가려움)', '중증 (BSA >50%, 심한 가려움)', '최중증 (광범위한 병변, 삶의 질 저하)']
    },
    'psoriasis': {
        'name_ko': '건선',
        'name_en': 'Psoriasis',
        'affected_areas': ['팔꿈치', '무릎', '두피', '허리', '손바닥', '발바닥', '손발톱'],
        'symptoms': [
            '은백색 비늘로 덮인 붉은 반점',
            '건조하고 갈라진 피부 (출혈 가능)',
            '가려움증, 작열감 또는 통증',
            '손발톱 두꺼워짐, 변색 또는 함몰',
            '관절 통증 및 뻣뻣함 (건선 관절염)',
            '두피의 비듬 같은 각질'
        ],
        'description': '면역 체계 이상으로 인한 만성 자가면역 질환입니다. 피부 세포가 비정상적으로 빠르게 증식하여 각질이 쌓이고, 스트레스, 감염, 약물 등이 악화 요인이 될 수 있습니다.',
        'severity_levels': ['경증 (BSA <3%)', '중등도 (BSA 3-10%)', '중증 (BSA >10% 또는 PASI >10)', '최중증 (전신형, 관절염 동반)']
    },
    'melanoma': {
        'name_ko': '악성 흑색종',
        'name_en': 'Malignant Melanoma',
        'affected_areas': ['전신 (특히 등, 다리, 팔, 얼굴)', '손발톱 밑', '점막'],
        'symptoms': [
            '비대칭적인 점 또는 병변 (A: Asymmetry)',
            '불규칙하고 들쭉날쭉한 경계 (B: Border)',
            '색상이 고르지 않거나 여러 색 혼재 (C: Color)',
            '직경 6mm 이상 (D: Diameter)',
            '크기, 모양, 색의 변화 (E: Evolution)',
            '가려움증, 출혈, 궤양'
        ],
        'description': '멜라닌 세포에서 발생하는 악성 종양으로 가장 위험한 피부암입니다. 자외선 노출이 주요 위험 요인이며, 조기 발견 시 완치 가능성이 높지만 전이되면 치명적일 수 있습니다.',
        'severity_levels': ['병기 0 (in situ, 제자리암)', '병기 I-II (국소 병변)', '병기 III (림프절 전이)', '병기 IV (원격 전이)']
    },
    'basal_cell_carcinoma': {
        'name_ko': '기저세포암',
        'name_en': 'Basal Cell Carcinoma',
        'affected_areas': ['얼굴', '코', '이마', '뺨', '목', '두피', '어깨'],
        'symptoms': [
            '진주 같은 광택이 있는 작은 돌기',
            '중앙이 함몰되고 가장자리가 융기된 병변',
            '잘 낫지 않는 궤양',
            '쉽게 출혈하는 병변',
            '반짝이는 분홍색, 붉은색 또는 흰색 반점',
            '평평하고 흉터 같은 병변'
        ],
        'description': '가장 흔한 피부암으로 느리게 성장하며 다른 장기로 전이는 드뭅니다. 장기간의 자외선 노출이 주요 원인이며, 조기 발견 및 치료 시 완치율이 매우 높습니다.',
        'severity_levels': ['저위험 (작고 명확한 경계)', '중등도 위험 (중간 크기)', '고위험 (큰 크기, 침습적)', '최고위험 (재발성, 침윤성)']
    },
    'squamous_cell_carcinoma': {
        'name_ko': '편평세포암',
        'name_en': 'Squamous Cell Carcinoma',
        'affected_areas': ['얼굴', '귀', '입술', '손등', '팔', '두피'],
        'symptoms': [
            '딱지나 껍질로 덮인 붉은 결절',
            '평평하고 비늘이 있는 병변',
            '잘 낫지 않는 궤양',
            '사마귀 같은 성장물',
            '출혈하기 쉬운 병변',
            '딱딱하고 융기된 병변'
        ],
        'description': '표피의 편평세포에서 발생하는 피부암으로, 기저세포암 다음으로 흔합니다. 자외선 노출, 만성 상처, HPV 감염이 위험 요인이며, 전이 가능성이 있어 조기 치료가 중요합니다.',
        'severity_levels': ['저위험 (작은 크기, 명확한 경계)', '중등도 위험', '고위험 (큰 크기, 침습적)', '전이성 (림프절이나 원격 전이)']
    },
    'seborrheic_keratosis': {
        'name_ko': '지루성 각화증',
        'name_en': 'Seborrheic Keratosis',
        'affected_areas': ['얼굴', '가슴', '어깨', '등', '복부'],
        'symptoms': [
            '갈색, 검은색 또는 황갈색의 사마귀 같은 성장물',
            '왁스를 바른 듯한 표면',
            '약간 융기되고 "붙여놓은 듯한" 외관',
            '거칠고 비늘 같은 질감',
            '대부분 통증이나 가려움 없음',
            '크기는 수 mm에서 수 cm까지 다양'
        ],
        'description': '양성 피부 종양으로 나이가 들면서 흔히 발생합니다. 악성이 아니며 건강에 해롭지 않지만, 미용적 이유나 자극을 받는 부위에 있을 경우 제거할 수 있습니다.',
        'severity_levels': ['단일 병변', '소수 병변 (2-5개)', '다발성 병변 (6-20개)', '광범위 병변 (20개 이상)']
    },
    'rosacea': {
        'name_ko': '주사 (안면홍조)',
        'name_en': 'Rosacea',
        'affected_areas': ['볼', '코', '이마', '턱', '눈 주변'],
        'symptoms': [
            '얼굴 중앙부의 지속적인 홍조',
            '확장된 혈관 (모세혈관 확장)',
            '여드름 같은 붉은 돌기와 농포',
            '피부 비후 (특히 코 - 딸기코)',
            '눈 자극, 충혈, 건조함',
            '화끈거림 또는 따끔거림'
        ],
        'description': '만성 염증성 피부 질환으로 얼굴에 홍조와 혈관 확장을 유발합니다. 정확한 원인은 불명이나 유전, 환경 요인이 관여하며, 스트레스, 알코올, 매운 음식, 온도 변화 등이 악화 요인입니다.',
        'severity_levels': ['1형 (홍반혈관확장형)', '2형 (구진농포형)', '3형 (비후형)', '4형 (눈 주사)']
    },
    'vitiligo': {
        'name_ko': '백반증',
        'name_en': 'Vitiligo',
        'affected_areas': ['얼굴', '손', '팔', '발', '입 주변', '눈 주변', '생식기', '겨드랑이'],
        'symptoms': [
            '피부의 색소 완전 상실',
            '유백색 또는 흰색 반점',
            '대칭적 또는 비대칭적 분포',
            '반점 부위 모발의 조기 백발화',
            '점막(입, 코, 생식기)의 탈색',
            '색소 상실 부위의 점진적 확대'
        ],
        'description': '멜라닌 세포가 파괴되어 피부 색소가 소실되는 자가면역 질환입니다. 건강에는 해롭지 않지만 자외선 차단이 중요하며, 심리적·미용적 영향이 클 수 있습니다.',
        'severity_levels': ['국소형 (한 부위)', '분절형 (일측성)', '전신형 (대칭적 분포)', '범발형 (체표면 >80%)']
    },
    'herpes': {
        'name_ko': '헤르페스',
        'name_en': 'Herpes Simplex',
        'affected_areas': ['입술', '입 주변', '구강 내', '생식기', '엉덩이', '허벅지'],
        'symptoms': [
            '작고 통증이 있는 물집 (수포)',
            '물집이 터진 후 생기는 궤양',
            '발진 전 가려움증, 따끔거림 (전구 증상)',
            '발열, 두통, 근육통',
            '림프절 부종',
            '재발성 발진 (같은 부위)'
        ],
        'description': '헤르페스 바이러스(HSV-1, HSV-2)에 의한 감염입니다. 1형은 주로 구강, 2형은 생식기에 발생하며, 바이러스는 평생 잠복 상태로 남아 있다가 면역력 저하, 스트레스 시 재발합니다.',
        'severity_levels': ['경증 (소수 병변)', '중등도 (다발성 병변)', '중증 (광범위 병변, 전신 증상)', '면역저하자 중증형']
    },
    'herpes_zoster': {
        'name_ko': '대상포진',
        'name_en': 'Herpes Zoster / Shingles',
        'affected_areas': ['몸통', '얼굴', '목', '팔', '다리 (일측성 분포)'],
        'symptoms': [
            '한쪽 신경 분포 영역을 따라 나타나는 발진',
            '타는 듯한 통증, 따끔거림',
            '군집된 물집 (수포)',
            '발진 전 통증 (전구 증상)',
            '발열, 피로감, 두통',
            '발진 후 지속되는 신경통 (대상포진 후 신경통)'
        ],
        'description': '수두 바이러스가 재활성화되어 발생하는 질환입니다. 나이가 들거나 면역력이 저하되면 발생하며, 심한 통증과 신경통이 특징입니다.',
        'severity_levels': ['경증 (제한적 병변)', '중등도 (한 신경절)', '중증 (다발성, 얼굴 침범)', '복잡형 (눈 침범, 파종성)']
    },
    'warts': {
        'name_ko': '사마귀',
        'name_en': 'Verruca / Warts',
        'affected_areas': ['손', '발', '얼굴', '무릎', '팔꿈치', '손발톱 주변'],
        'symptoms': [
            '작고 거칠고 단단한 피부 성장물',
            '표면에 검은 점 (혈전된 모세혈관)',
            '융기된 돌기 또는 평평한 병변',
            '손발바닥 사마귀의 경우 압통',
            '여러 개가 모여 모자이크 패턴 형성',
            '손발톱 변형 (손발톱 주변 사마귀)'
        ],
        'description': '인유두종 바이러스(HPV)에 의한 양성 피부 감염입니다. 전염성이 있으며 직접 접촉이나 간접 접촉으로 전파됩니다. 대부분 양성이며 자연 소실되기도 하지만 치료로 제거 가능합니다.',
        'severity_levels': ['단일 사마귀', '소수 (2-5개)', '다발성 (6-20개)', '난치성 (20개 이상 또는 치료 저항성)']
    },
    'dermatitis': {
        'name_ko': '접촉성 피부염',
        'name_en': 'Contact Dermatitis',
        'affected_areas': ['손', '얼굴', '목', '팔', '접촉 부위 (국소적)'],
        'symptoms': [
            '붉고 가려운 발진',
            '부종 및 물집',
            '피부 건조, 갈라짐, 벗겨짐',
            '화끈거림 또는 따끔거림',
            '접촉 부위의 명확한 경계',
            '삼출액 및 딱지 형성'
        ],
        'description': '피부가 자극 물질이나 알레르겐에 접촉하여 발생하는 염증 반응입니다. 자극성 접촉 피부염과 알레르기성 접촉 피부염으로 구분되며, 원인 물질 제거가 치료의 핵심입니다.',
        'severity_levels': ['경증 (홍반만)', '중등도 (부종, 수포)', '중증 (미란, 궤양)', '만성 (태선화, 색소 침착)']
    },
    'fungal_infection': {
        'name_ko': '피부 진균 감염 (무좀, 백선)',
        'name_en': 'Tinea / Fungal Infection',
        'affected_areas': ['발', '사타구니', '몸통', '두피', '손톱', '발톱'],
        'symptoms': [
            '원형의 붉은 발진 (가장자리 융기)',
            '중앙은 깨끗하고 가장자리만 활성화',
            '심한 가려움증',
            '피부 벗겨짐, 갈라짐, 각질',
            '손발톱 변색, 두꺼워짐, 부서짐',
            '악취 (특히 발)'
        ],
        'description': '피부사상균에 의한 감염으로 따뜻하고 습한 환경에서 잘 발생합니다. 무좀(발), 완선(사타구니), 체부백선(몸통) 등 발생 부위에 따라 명칭이 다르며, 전염성이 있고 항진균제로 치료합니다.',
        'severity_levels': ['경증 (국소 병변)', '중등도 (광범위 병변)', '중증 (손발톱 침범)', '만성/재발성']
    },
    'impetigo': {
        'name_ko': '농가진',
        'name_en': 'Impetigo',
        'affected_areas': ['얼굴', '입 주변', '코', '손', '팔', '다리'],
        'symptoms': [
            '꿀색 딱지가 있는 붉은 반점',
            '쉽게 터지는 물집',
            '가려움증',
            '병변 주변 림프절 부종',
            '전염성이 매우 높음',
            '긁으면 빠르게 확산'
        ],
        'description': '세균(주로 황색포도상구균)에 의한 피부 감염으로 주로 어린이에게 발생합니다. 전염성이 매우 높으며 항생제 치료가 필요합니다.',
        'severity_levels': ['비수포성 농가진 (딱지)', '수포성 농가진 (큰 물집)', '농피증 (깊은 감염)', '반복성']
    },
    'cellulitis': {
        'name_ko': '봉와직염',
        'name_en': 'Cellulitis',
        'affected_areas': ['다리', '발', '팔', '얼굴'],
        'symptoms': [
            '붉고 부은 피부',
            '통증 및 압통',
            '피부 열감',
            '경계가 불명확한 발적',
            '발열 및 오한',
            '림프절 부종'
        ],
        'description': '피부와 피하조직의 세균 감염으로 주로 연쇄상구균이나 포도상구균이 원인입니다. 즉시 항생제 치료가 필요하며, 치료하지 않으면 패혈증으로 진행될 수 있습니다.',
        'severity_levels': ['경증 (국소 감염)', '중등도 (광범위, 전신 증상)', '중증 (괴사성, 패혈증)', '재발성']
    },
    'urticaria': {
        'name_ko': '두드러기',
        'name_en': 'Urticaria / Hives',
        'affected_areas': ['전신 (어디든 가능)'],
        'symptoms': [
            '가려운 붉은 두드러기',
            '융기된 병변',
            '병변의 빠른 출현과 소실 (24시간 이내)',
            '중앙이 창백한 팽진',
            '때때로 혈관부종 동반',
            '긁으면 악화'
        ],
        'description': '비만세포에서 히스타민이 방출되어 발생하는 알레르기 반응입니다. 음식, 약물, 감염, 스트레스 등 다양한 원인이 있으며, 급성과 만성으로 구분됩니다.',
        'severity_levels': ['경증 (소수 병변)', '중등도 (다발성)', '중증 (광범위, 혈관부종)', '아나필락시스 (생명 위협)']
    },
    'angioedema': {
        'name_ko': '혈관부종',
        'name_en': 'Angioedema',
        'affected_areas': ['얼굴', '입술', '혀', '목', '손', '발', '생식기'],
        'symptoms': [
            '피부 깊은 층의 부종',
            '통증이나 압통',
            '피부색 변화 없음',
            '호흡곤란 (목 부종 시)',
            '복통 (내장 부종 시)',
            '24-72시간 지속'
        ],
        'description': '피부와 점막의 깊은 층에 발생하는 부종으로, 두드러기와 함께 나타날 수 있습니다. 기도 부종 시 생명을 위협할 수 있어 응급 치료가 필요합니다.',
        'severity_levels': ['경증 (말단 부위)', '중등도 (얼굴)', '중증 (후두 부종)', '생명 위협 (기도 폐쇄)']
    },
    'drug_eruption': {
        'name_ko': '약물 발진',
        'name_en': 'Drug Eruption',
        'affected_areas': ['전신'],
        'symptoms': [
            '약물 복용 후 발생하는 발진',
            '다양한 형태 (반점, 구진, 수포 등)',
            '가려움증',
            '발열',
            '점막 침범 가능',
            '약물 중단 후 호전'
        ],
        'description': '약물에 대한 면역 반응으로 발생하는 피부 발진입니다. 경증부터 생명을 위협하는 중증까지 다양하며, 원인 약물 확인과 중단이 중요합니다.',
        'severity_levels': ['경증 (단순 발진)', '중등도 (광범위)', '중증 (SJS/TEN)', '생명 위협 (DRESS)']
    },
    'lichen_planus': {
        'name_ko': '편평태선',
        'name_en': 'Lichen Planus',
        'affected_areas': ['손목', '발목', '구강 점막', '생식기', '두피'],
        'symptoms': [
            '자줏빛 편평한 구진',
            '다각형 모양',
            '표면에 흰색 선 (Wickham striae)',
            '심한 가려움증',
            '구강 내 흰색 망상 패턴',
            '손발톱 변형'
        ],
        'description': '만성 염증성 피부 질환으로 정확한 원인은 불명이나 자가면역 반응으로 추정됩니다. 피부뿐 아니라 점막도 침범할 수 있습니다.',
        'severity_levels': ['국소형', '광범위형', '침식성 (구강/생식기)', '비후성']
    },
    'pityriasis_rosea': {
        'name_ko': '장미색 비강진',
        'name_en': 'Pityriasis Rosea',
        'affected_areas': ['몸통', '목', '팔', '다리'],
        'symptoms': [
            '전구 반점 (herald patch) - 큰 타원형 반점',
            '크리스마스 트리 패턴의 여러 작은 반점',
            '약간의 가려움증',
            '피부 껍질 벗겨짐',
            '6-8주 후 자연 소실',
            '주로 봄과 가을에 발생'
        ],
        'description': '원인 불명의 자가 제한적 피부 질환으로, 바이러스 감염이 원인으로 추정됩니다. 특별한 치료 없이 자연 치유되지만 가려움증 완화 치료를 할 수 있습니다.',
        'severity_levels': ['경증 (경미한 가려움)', '중등도 (다수 병변)', '중증 (심한 가려움)', '비전형 (얼굴 침범)']
    },
    'pityriasis_versicolor': {
        'name_ko': '어루러기',
        'name_en': 'Pityriasis Versicolor / Tinea Versicolor',
        'affected_areas': ['가슴', '등', '어깨', '목', '팔'],
        'symptoms': [
            '색소 변화 (희거나 갈색)',
            '약간 비늘이 있는 반점',
            '햇빛에 노출 후 더 명확해짐',
            '가려움증 (경미하거나 없음)',
            '여러 개의 반점이 합쳐짐',
            '따뜻하고 습한 환경에서 악화'
        ],
        'description': '정상 피부 상재균인 말라세지아가 과증식하여 발생하는 진균 감염입니다. 전염성은 없으며 항진균제로 치료하지만 재발이 흔합니다.',
        'severity_levels': ['경증 (소수 병변)', '중등도 (다수 병변)', '광범위형', '재발성']
    },
    'molluscum_contagiosum': {
        'name_ko': '물사마귀 (전염성 연속종)',
        'name_en': 'Molluscum Contagiosum',
        'affected_areas': ['얼굴', '목', '팔', '손', '몸통', '생식기'],
        'symptoms': [
            '작은 진주빛 돌기',
            '중앙에 배꼽 모양 함몰',
            '단단하고 매끈한 표면',
            '통증 없음',
            '긁거나 짜면 확산',
            '면역저하자에게 광범위 발생'
        ],
        'description': '폭스바이러스에 의한 양성 피부 감염으로 주로 어린이에게 발생합니다. 전염성이 있으나 대부분 6-12개월 내에 자연 소실됩니다.',
        'severity_levels': ['소수 (1-5개)', '중등도 (6-20개)', '다발성 (20-50개)', '광범위 (50개 이상, 면역저하)']
    },
    'scabies': {
        'name_ko': '옴',
        'name_en': 'Scabies',
        'affected_areas': ['손가락 사이', '손목', '겨드랑이', '생식기', '엉덩이'],
        'symptoms': [
            '심한 가려움증 (밤에 악화)',
            'S자 모양의 터널 (피부 속 진드기 이동 경로)',
            '작은 붉은 구진',
            '긁은 자국과 이차 감염',
            '가족 내 집단 발생',
            '청결과 무관하게 발생'
        ],
        'description': '옴 진드기가 피부에 기생하여 발생하는 질환입니다. 전염성이 매우 높으며 밀접 접촉으로 전파됩니다. 가족 전체를 동시에 치료해야 합니다.',
        'severity_levels': ['일반형', '결절성', '딱지형 (crusted scabies)', '이차 감염 동반']
    },
    'melasma': {
        'name_ko': '기미',
        'name_en': 'Melasma',
        'affected_areas': ['얼굴 (특히 볼, 이마, 윗입술, 코)'],
        'symptoms': [
            '대칭적인 갈색 또는 회갈색 반점',
            '경계가 명확하거나 불명확',
            '표면 변화 없음 (평평)',
            '햇빛 노출 시 악화',
            '임신이나 피임약 복용 시 발생',
            '여성에게 더 흔함'
        ],
        'description': '멜라닌 색소 과다 침착으로 인한 색소 질환입니다. 자외선, 호르몬, 유전적 요인이 관여하며, 치료가 어렵고 재발이 흔합니다.',
        'severity_levels': ['경증 (옅은 색소)', '중등도 (진한 색소)', '중증 (광범위)', '난치성']
    },
    'lentigo': {
        'name_ko': '흑자 (점)',
        'name_en': 'Lentigo / Solar Lentigo',
        'affected_areas': ['얼굴', '손등', '팔', '어깨', '등'],
        'symptoms': [
            '평평한 갈색 또는 검은색 반점',
            '명확한 경계',
            '자외선 노출 부위',
            '나이가 들면서 증가',
            '크기 변화 없음',
            '통증이나 가려움 없음'
        ],
        'description': '멜라닌 세포의 국소적 증가로 발생하는 양성 색소 병변입니다. 일광 흑자는 자외선 노출로 인해 발생하며 "노인성 점"이라고도 합니다.',
        'severity_levels': ['단일', '소수 (2-10개)', '다발성 (10개 이상)', '융합형']
    },
    'nevus': {
        'name_ko': '모반 (점)',
        'name_en': 'Melanocytic Nevus / Mole',
        'affected_areas': ['전신'],
        'symptoms': [
            '갈색, 검은색 또는 피부색 병변',
            '평평하거나 융기됨',
            '대칭적',
            '명확한 경계',
            '균일한 색상',
            '대부분 6mm 미만'
        ],
        'description': '멜라닌 세포의 양성 증식으로 발생하는 가장 흔한 피부 병변입니다. 대부분 무해하나 악성 흑색종과 감별이 필요하며, ABCDE 규칙으로 변화를 관찰해야 합니다.',
        'severity_levels': ['정상 모반', '비정형 모반', '선천성 거대 모반', '변화하는 모반 (흑색종 의심)']
    },
    'keratosis_pilaris': {
        'name_ko': '모공각화증 (닭살)',
        'name_en': 'Keratosis Pilaris',
        'affected_areas': ['팔 뒤쪽', '허벅지', '엉덩이', '뺨'],
        'symptoms': [
            '작고 거친 돌기',
            '닭살 같은 외관',
            '건조한 피부',
            '가려움증 (경미하거나 없음)',
            '붉은빛 띔 (염증성)',
            '겨울에 악화'
        ],
        'description': '모낭 주변 각질이 과다 축적되어 발생하는 흔한 양성 질환입니다. 유전적 경향이 있으며 무해하지만 미용적으로 신경 쓰일 수 있습니다.',
        'severity_levels': ['경증 (제한적)', '중등도 (광범위)', '염증성 (붉음)', '아토피 동반']
    },
    'actinic_keratosis': {
        'name_ko': '일광 각화증',
        'name_en': 'Actinic Keratosis / Solar Keratosis',
        'affected_areas': ['얼굴', '두피', '귀', '손등', '팔'],
        'symptoms': [
            '거칠고 비늘이 있는 반점',
            '붉거나 피부색',
            '딱딱한 표면',
            '여러 개 발생',
            '햇빛 노출 부위',
            '편평세포암으로 진행 가능'
        ],
        'description': '만성적인 자외선 노출로 인한 전암성 병변입니다. 방치 시 일부는 편평세포암으로 진행할 수 있어 치료가 필요합니다.',
        'severity_levels': ['Ⅰ등급 (가벼운 각질)', 'Ⅱ등급 (중등도 각질)', 'Ⅲ등급 (두꺼운 각질)', '편평세포암 전환']
    },
    'cherry_angioma': {
        'name_ko': '체리 혈관종',
        'name_en': 'Cherry Angioma',
        'affected_areas': ['몸통', '팔', '다리', '어깨'],
        'symptoms': [
            '밝은 빨간색 작은 돌기',
            '매끈하고 돔 모양',
            '크기 1-5mm',
            '통증 없음',
            '나이가 들면서 증가',
            '압력으로 창백해지지 않음'
        ],
        'description': '혈관의 양성 증식으로 발생하는 매우 흔한 병변입니다. 건강에 무해하며 치료가 필요 없지만, 미용적 이유로 제거할 수 있습니다.',
        'severity_levels': ['단일', '소수 (2-10개)', '다발성 (10-50개)', '매우 다발성 (50개 이상)']
    },
    'skin_tag': {
        'name_ko': '연성 섬유종 (쥐젖)',
        'name_en': 'Skin Tag / Acrochordon',
        'affected_areas': ['목', '겨드랑이', '사타구니', '눈꺼풀', '유방 아래'],
        'symptoms': [
            '피부색 또는 갈색의 작은 돌기',
            '얇은 줄기로 연결됨',
            '부드럽고 주름진 표면',
            '통증 없음',
            '마찰 부위에 발생',
            '비만이나 당뇨와 연관'
        ],
        'description': '피부가 접히는 부위에 발생하는 양성 종양입니다. 건강에 무해하나 미용적 이유나 자극으로 제거할 수 있습니다.',
        'severity_levels': ['단일', '소수 (2-5개)', '다발성 (6-20개)', '매우 다발성 (20개 이상)']
    },
    'lipoma': {
        'name_ko': '지방종',
        'name_en': 'Lipoma',
        'affected_areas': ['목', '어깨', '등', '팔', '허벅지'],
        'symptoms': [
            '부드럽고 움직이는 덩어리',
            '피부 아래 느껴짐',
            '통증 없음',
            '천천히 성장',
            '압력 시 약간 아플 수 있음',
            '크기 다양 (수 cm)'
        ],
        'description': '지방세포의 양성 종양으로 매우 흔합니다. 대부분 무해하며 치료가 필요 없지만, 크기가 크거나 통증이 있으면 제거할 수 있습니다.',
        'severity_levels': ['소형 (<5cm)', '중형 (5-10cm)', '대형 (>10cm)', '다발성 (여러 개)']
    },
    'keloid': {
        'name_ko': '켈로이드',
        'name_en': 'Keloid',
        'affected_areas': ['가슴', '어깨', '귀', '등', '턱'],
        'symptoms': [
            '상처 범위를 넘어 자라는 흉터',
            '단단하고 융기됨',
            '분홍색, 붉은색 또는 진한 색',
            '가려움증',
            '통증 또는 압통',
            '계속 성장'
        ],
        'description': '상처 치유 과정에서 콜라겐이 과다 생성되어 발생하는 비정상 흉터입니다. 유전적 소인이 있으며 치료 후에도 재발이 흔합니다.',
        'severity_levels': ['작은 켈로이드 (<2cm)', '중간 켈로이드 (2-5cm)', '큰 켈로이드 (>5cm)', '광범위/재발성']
    },
    'hypertrophic_scar': {
        'name_ko': '비후성 반흔',
        'name_en': 'Hypertrophic Scar',
        'affected_areas': ['상처 부위 (전신)'],
        'symptoms': [
            '상처 범위 내에서 융기',
            '붉고 두꺼운 흉터',
            '가려움증',
            '통증 가능',
            '시간이 지나면 호전',
            '켈로이드보다 작음'
        ],
        'description': '상처 치유 과정에서 발생하는 융기된 흉터로 켈로이드와 달리 원래 상처 범위를 벗어나지 않습니다. 시간이 지나면 저절로 호전되는 경향이 있습니다.',
        'severity_levels': ['경증 (약간 융기)', '중등도 (명확한 융기)', '중증 (두꺼운 융기)', '구축 (관절 제한)']
    },
    'lupus_erythematosus': {
        'name_ko': '홍반성 루푸스',
        'name_en': 'Cutaneous Lupus Erythematosus',
        'affected_areas': ['얼굴 (나비 모양)', '두피', '귀', '팔', '손'],
        'symptoms': [
            '나비 모양의 붉은 발진 (뺨과 코)',
            '햇빛 노출 후 악화',
            '원판형 병변 (판상홍반루푸스)',
            '탈모',
            '구강 궤양',
            '광과민성'
        ],
        'description': '자가면역 질환의 일종으로 피부만 침범하거나 전신을 침범할 수 있습니다. 자외선 차단이 중요하며 면역억제 치료가 필요합니다.',
        'severity_levels': ['급성 피부 루푸스', '아급성 피부 루푸스', '만성 원판상 루푸스', '전신 루푸스']
    },
    'scleroderma': {
        'name_ko': '경피증',
        'name_en': 'Scleroderma',
        'affected_areas': ['손가락', '얼굴', '팔', '다리', '몸통'],
        'symptoms': [
            '피부가 두꺼워지고 단단해짐',
            '피부 색소 변화',
            '광택이 나는 피부',
            '레이노 현상 (손가락 색 변화)',
            '관절 운동 제한',
            '피부 궤양'
        ],
        'description': '자가면역 질환으로 피부와 내부 장기의 결합조직이 두꺼워지고 단단해집니다. 국소형과 전신형이 있으며, 전신형은 내부 장기를 침범할 수 있습니다.',
        'severity_levels': ['국소형 (morphea)', '선상 경피증', '제한형 전신 경피증', '미만성 전신 경피증']
    },
    'dermatomyositis': {
        'name_ko': '피부근염',
        'name_en': 'Dermatomyositis',
        'affected_areas': ['얼굴', '눈꺼풀', '손가락 관절', '무릎', '팔꿈치'],
        'symptoms': [
            '눈꺼풀의 보라색 발진 (헬리오트로프 발진)',
            '손가락 관절 위 붉은 구진 (Gottron papules)',
            '근육 약화',
            '광과민성',
            '손톱 주변 모세혈관 확장',
            '피부 가려움증'
        ],
        'description': '자가면역 질환으로 피부와 근육을 침범합니다. 특징적인 피부 증상과 근육 약화가 나타나며, 악성 종양과 연관될 수 있어 정밀 검사가 필요합니다.',
        'severity_levels': ['경증 (피부만)', '중등도 (피부와 근육)', '중증 (전신 침범)', '악성 종양 동반']
    },
    'pemphigus': {
        'name_ko': '천포창',
        'name_en': 'Pemphigus',
        'affected_areas': ['구강 점막', '피부 (전신)', '두피', '생식기'],
        'symptoms': [
            '쉽게 터지는 물집',
            '피부 벗겨짐 (Nikolsky sign 양성)',
            '통증이 있는 미란',
            '구강 내 궤양',
            '딱지 형성',
            '이차 감염'
        ],
        'description': '자가항체가 피부세포 간 결합을 파괴하여 발생하는 자가면역 수포성 질환입니다. 치료하지 않으면 생명을 위협할 수 있어 면역억제 치료가 필요합니다.',
        'severity_levels': ['경증 (제한적 병변)', '중등도 (광범위 병변)', '중증 (전신형)', '생명 위협 (감염, 전해질 불균형)']
    },
    'bullous_pemphigoid': {
        'name_ko': '수포성 유천포창',
        'name_en': 'Bullous Pemphigoid',
        'affected_areas': ['팔', '다리', '복부', '사타구니'],
        'symptoms': [
            '긴장성 큰 물집',
            '터지지 않는 단단한 수포',
            '심한 가려움증',
            '붉은 두드러기 같은 병변',
            '주로 노인에게 발생',
            '구강 침범 드묾'
        ],
        'description': '자가항체가 표피-진피 접합부를 공격하여 발생하는 자가면역 수포성 질환입니다. 천포창보다 예후가 좋으나 치료가 필요합니다.',
        'severity_levels': ['경증 (소수 수포)', '중등도 (다발성 수포)', '중증 (광범위)', '난치성']
    },
    'erythema_multiforme': {
        'name_ko': '다형홍반',
        'name_en': 'Erythema Multiforme',
        'affected_areas': ['손', '발', '팔', '다리', '얼굴'],
        'symptoms': [
            '과녁 모양 병변 (target lesion)',
            '중앙이 어둡고 주변이 밝음',
            '대칭적 분포',
            '가려움증 또는 화끈거림',
            '점막 침범 가능',
            '헤르페스 감염 후 발생 흔함'
        ],
        'description': '감염(특히 헤르페스)이나 약물에 대한 과민 반응으로 발생하는 급성 피부 질환입니다. 경증부터 중증(Stevens-Johnson 증후군)까지 다양합니다.',
        'severity_levels': ['소형 (minor)', '대형 (major)', 'Stevens-Johnson 증후군', '독성 표피 괴사 융해증 (TEN)']
    },
    'pyogenic_granuloma': {
        'name_ko': '화농육아종',
        'name_en': 'Pyogenic Granuloma',
        'affected_areas': ['손가락', '얼굴', '입술', '잇몸', '몸통'],
        'symptoms': [
            '빠르게 자라는 붉은 결절',
            '쉽게 출혈',
            '표면이 축축하거나 딱지',
            '보통 수 주 내에 발생',
            '외상이나 임신 후 발생',
            '통증 없음'
        ],
        'description': '혈관의 빠른 증식으로 발생하는 양성 병변입니다. 외상이나 호르몬 변화 후 발생하며, 자연 소실되지 않아 제거가 필요합니다.',
        'severity_levels': ['작은 병변 (<5mm)', '중간 병변 (5-10mm)', '큰 병변 (>10mm)', '재발성']
    },
    'port_wine_stain': {
        'name_ko': '포도주색 반점 (화염상 모반)',
        'name_en': 'Port-Wine Stain / Nevus Flammeus',
        'affected_areas': ['얼굴', '목', '사지'],
        'symptoms': [
            '선천적 평평한 붉은 반점',
            '경계가 명확',
            '압력으로 창백해지지 않음',
            '나이가 들면서 진해지고 두꺼워짐',
            '대부분 한쪽에만 발생',
            'Sturge-Weber 증후군 동반 가능'
        ],
        'description': '선천적 모세혈관 기형으로 자연 소실되지 않습니다. 레이저 치료로 호전될 수 있으며, 얼굴에 있으면 신경학적 검사가 필요할 수 있습니다.',
        'severity_levels': ['분홍색 (옅음)', '붉은색 (중등도)', '자주색 (진함)', '비후 변화 동반']
    },
    'hemangioma': {
        'name_ko': '혈관종',
        'name_en': 'Infantile Hemangioma',
        'affected_areas': ['얼굴', '두피', '가슴', '등'],
        'symptoms': [
            '생후 수 주 내에 나타나는 붉은 종양',
            '빠르게 자라다가 천천히 소실',
            '부드럽고 압축 가능',
            '표재성 (딸기 모양) 또는 심재성',
            '대부분 5-7세까지 소실',
            '크기와 위치에 따라 합병증 가능'
        ],
        'description': '영아기에 나타나는 가장 흔한 양성 종양입니다. 대부분 자연 소실되나, 위치나 크기에 따라 치료가 필요할 수 있습니다.',
        'severity_levels': ['작은 표재성', '큰 표재성', '심재성', '합병증 (궤양, 기능 장애)']
    },
    'alopecia_areata': {
        'name_ko': '원형 탈모',
        'name_en': 'Alopecia Areata',
        'affected_areas': ['두피', '수염', '눈썹', '속눈썹', '체모'],
        'symptoms': [
            '원형 또는 타원형 탈모 반점',
            '경계가 명확',
            '감탄 부호 모양 털 (exclamation mark hair)',
            '두피 피부는 정상',
            '손발톱 함몰 (pitting)',
            '자연 재생 가능하나 재발 흔함'
        ],
        'description': '자가면역 질환으로 모낭을 공격하여 탈모가 발생합니다. 스트레스가 유발 요인이 될 수 있으며, 범위와 정도가 다양합니다.',
        'severity_levels': ['단발성 원형 탈모', '다발성 원형 탈모', '전두 탈모 (alopecia totalis)', '전신 탈모 (alopecia universalis)']
    },
    'androgenetic_alopecia': {
        'name_ko': '남성형/여성형 탈모',
        'name_en': 'Androgenetic Alopecia',
        'affected_areas': ['두피 (정수리, 앞머리)'],
        'symptoms': [
            '점진적 모발 가늘어짐',
            '남성: M자 탈모, 정수리 탈모',
            '여성: 가르마 부위 넓어짐',
            '모발 밀도 감소',
            '유전적 경향',
            '연령 증가와 함께 진행'
        ],
        'description': '유전과 호르몬(안드로겐)의 영향으로 발생하는 가장 흔한 탈모입니다. 조기 치료가 효과적이며 약물 치료나 모발 이식이 가능합니다.',
        'severity_levels': ['Hamilton I-II (경증)', 'Hamilton III-IV (중등도)', 'Hamilton V-VI (중증)', 'Hamilton VII (최중증)']
    },
    'telogen_effluvium': {
        'name_ko': '휴지기 탈모',
        'name_en': 'Telogen Effluvium',
        'affected_areas': ['두피 전체'],
        'symptoms': [
            '전체적인 모발 가늘어짐',
            '샴푸 시 많은 양의 탈모',
            '스트레스, 출산, 질병 후 발생',
            '2-3개월 뒤 증상 시작',
            '특정 부위 탈모 없음',
            '대부분 6개월 내 회복'
        ],
        'description': '스트레스, 질병, 출산, 급격한 체중 감소 등으로 인해 많은 모발이 동시에 휴지기로 들어가 발생하는 일시적 탈모입니다.',
        'severity_levels': ['경증 (약간 증가된 탈모)', '중등도 (눈에 띄는 가늘어짐)', '중증 (현저한 탈모)', '만성 (6개월 이상)']
    }
}


class DermLIPDiagnosisSystem:
    """DermLIP 기반 피부 질환 진단 시스템"""

    def __init__(self, model_name='hf-hub:redlessone/DermLIP_ViT-B-16', device='cuda'):
        """
        Args:
            model_name: 사용할 DermLIP 모델
            device: 'cuda' 또는 'cpu'
        """
        self.device = device
        self.model_name = model_name

        print(f'\n{"="*70}')
        print(f'DermLIP 피부 질환 진단 시스템')
        print(f'{"="*70}')
        print(f'모델 로드 중: {model_name}')
        print(f'디바이스: {device}\n')

        # 모델 로드
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # 클래스 정보 생성
        self.disease_keys = list(SKIN_DISEASE_DATABASE.keys())
        self.class_descriptions = self._generate_class_descriptions()

        print(f'✓ 모델 로드 완료!')
        print(f'✓ {len(self.disease_keys)}개 피부 질환 진단 가능')
        print(f'{"="*70}\n')

    def _generate_class_descriptions(self):
        """DermLIP 모델용 클래스 설명 생성"""
        descriptions = []
        for key in self.disease_keys:
            info = SKIN_DISEASE_DATABASE[key]
            # 임상적 설명 생성
            desc = f"a clinical dermatological photograph of {info['name_en']}"
            descriptions.append(desc)
        return descriptions

    def diagnose(self, image_path, top_k=3):
        """
        피부 질환 진단 수행

        Args:
            image_path: 진단할 이미지 경로
            top_k: 상위 k개 결과 반환

        Returns:
            diagnosis_results: 진단 결과 리스트
            image: PIL Image 객체
        """
        print(f'📸 이미지 분석 중: {image_path}\n')

        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # 이미지 인코딩
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = F.normalize(image_features, dim=-1)

        # 텍스트 인코딩
        text_tokens = self.tokenizer(self.class_descriptions).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)

        # 유사도 계산
        similarity = (image_features @ text_features.T).squeeze(0)
        probabilities = F.softmax(similarity * 100, dim=0).cpu().numpy()

        # 상위 k개 결과 추출
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        diagnosis_results = []
        for rank, idx in enumerate(top_indices, 1):
            disease_key = self.disease_keys[idx]
            disease_info = SKIN_DISEASE_DATABASE[disease_key]
            confidence = float(probabilities[idx]) * 100

            diagnosis_results.append({
                'rank': rank,
                'confidence': confidence,
                'disease_key': disease_key,
                'disease_info': disease_info
            })

        return diagnosis_results, image

    def print_diagnosis(self, diagnosis_results):
        """진단 결과를 콘솔에 출력"""
        print('\n' + '='*70)
        print('🏥 진단 결과')
        print('='*70 + '\n')

        for result in diagnosis_results:
            rank = result['rank']
            confidence = result['confidence']
            info = result['disease_info']

            print(f'[{rank}위] {info["name_ko"]} ({info["name_en"]})')
            print(f'└─ 신뢰도: {confidence:.1f}%')
            print()

            print(f'📍 주로 영향받는 부위:')
            for area in info['affected_areas']:
                print(f'   • {area}')
            print()

            print(f'🔍 주요 증상:')
            for symptom in info['symptoms']:
                print(f'   • {symptom}')
            print()

            print(f'📋 설명:')
            print(f'   {info["description"]}')
            print()

            print(f'📊 중증도 분류:')
            for severity in info['severity_levels']:
                print(f'   • {severity}')

            if rank < len(diagnosis_results):
                print('\n' + '-'*70 + '\n')

        print('\n' + '='*70)
        print('⚠️  주의사항:')
        print('   이 결과는 AI 기반 예측이며 참고용입니다.')
        print('   정확한 진단과 치료는 반드시 피부과 전문의와 상담하세요.')
        print('='*70 + '\n')

    def save_diagnosis_report(self, image_path, diagnosis_results, output_dir):
        """진단 결과를 JSON 및 텍스트 파일로 저장"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON 리포트
        report_data = {
            'timestamp': timestamp,
            'image_path': image_path,
            'model': self.model_name,
            'diagnoses': [
                {
                    'rank': r['rank'],
                    'confidence': r['confidence'],
                    'disease_name_ko': r['disease_info']['name_ko'],
                    'disease_name_en': r['disease_info']['name_en'],
                    'affected_areas': r['disease_info']['affected_areas'],
                    'symptoms': r['disease_info']['symptoms'],
                    'description': r['disease_info']['description'],
                    'severity_levels': r['disease_info']['severity_levels']
                }
                for r in diagnosis_results
            ]
        }

        json_path = os.path.join(output_dir, f'diagnosis_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # 텍스트 리포트
        txt_path = os.path.join(output_dir, f'diagnosis_{timestamp}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('='*70 + '\n')
            f.write('DermLIP 피부 질환 진단 결과\n')
            f.write('='*70 + '\n\n')
            f.write(f'진단 일시: {datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")}\n')
            f.write(f'이미지: {image_path}\n')
            f.write(f'모델: {self.model_name}\n\n')

            for r in diagnosis_results:
                info = r['disease_info']
                f.write(f'[{r["rank"]}위] {info["name_ko"]} ({info["name_en"]})\n')
                f.write(f'신뢰도: {r["confidence"]:.1f}%\n\n')
                f.write('주로 영향받는 부위:\n')
                for area in info['affected_areas']:
                    f.write(f'  • {area}\n')
                f.write('\n주요 증상:\n')
                for symptom in info['symptoms']:
                    f.write(f'  • {symptom}\n')
                f.write(f'\n설명:\n  {info["description"]}\n\n')
                f.write('중증도 분류:\n')
                for severity in info['severity_levels']:
                    f.write(f'  • {severity}\n')
                if r['rank'] < len(diagnosis_results):
                    f.write('\n' + '-'*70 + '\n\n')

            f.write('\n' + '='*70 + '\n')
            f.write('⚠️  이 결과는 AI 기반 예측이며, 정확한 진단은 전문의와 상담하세요.\n')
            f.write('='*70 + '\n')

        print(f'✓ 진단 리포트 저장 완료:')
        print(f'  - JSON: {json_path}')
        print(f'  - TXT:  {txt_path}')

        return json_path, txt_path

    def visualize_diagnosis(self, image, diagnosis_results, output_path):
        """진단 결과 시각화"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        # 원본 이미지
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('입력 이미지', fontsize=16, fontweight='bold', pad=15)

        # 진단 결과 (신뢰도)
        ax2 = fig.add_subplot(gs[0, 1])
        diseases = [f"{r['rank']}. {r['disease_info']['name_ko']}" for r in diagnosis_results]
        confidences = [r['confidence'] for r in diagnosis_results]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(diseases)))
        bars = ax2.barh(range(len(diseases)), confidences, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(diseases)))
        ax2.set_yticklabels(diseases, fontsize=11)
        ax2.set_xlabel('신뢰도 (%)', fontsize=12)
        ax2.set_title('진단 결과', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlim(0, 100)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')

        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax2.text(conf + 1.5, i, f'{conf:.1f}%', va='center', fontsize=10, fontweight='bold')

        # 1위 질환 상세 정보
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')

        top = diagnosis_results[0]
        info = top['disease_info']

        detail_lines = [
            f"🏥 1위 진단: {info['name_ko']} ({info['name_en']})",
            f"   신뢰도: {top['confidence']:.1f}%",
            "",
            "📍 주로 영향받는 부위:",
        ]
        detail_lines.extend([f"   • {area}" for area in info['affected_areas'][:4]])
        detail_lines.append("")
        detail_lines.append("🔍 주요 증상:")
        detail_lines.extend([f"   • {symptom}" for symptom in info['symptoms'][:4]])
        detail_lines.append("")
        detail_lines.append("📋 설명:")

        # 설명을 적절히 줄바꿈
        desc_lines = []
        words = info['description'].split()
        current_line = "   "
        for word in words:
            if len(current_line) + len(word) + 1 <= 60:
                current_line += word + " "
            else:
                desc_lines.append(current_line.rstrip())
                current_line = "   " + word + " "
        if current_line.strip():
            desc_lines.append(current_line.rstrip())
        detail_lines.extend(desc_lines)

        detail_text = '\n'.join(detail_lines)

        ax3.text(0.05, 0.95, detail_text,
                transform=ax3.transAxes,
                fontsize=10,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', edgecolor='#333', linewidth=2))

        plt.suptitle('DermLIP 피부 질환 진단 시스템', fontsize=18, fontweight='bold', y=0.98)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f'  - 시각화: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='DermLIP 피부 질환 진단 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python dermlip_diagnosis.py --image skin_photo.jpg
  python dermlip_diagnosis.py --image skin_photo.jpg --top_k 5
  python dermlip_diagnosis.py --image skin_photo.jpg --model hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256
        """
    )

    parser.add_argument('--image', type=str, required=True,
                       help='진단할 피부 이미지 경로')
    parser.add_argument('--top_k', type=int, default=3,
                       help='상위 k개 결과 표시 (기본값: 3)')
    parser.add_argument('--model', type=str,
                       default='hf-hub:redlessone/DermLIP_ViT-B-16',
                       choices=['hf-hub:redlessone/DermLIP_ViT-B-16',
                               'hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256'],
                       help='사용할 DermLIP 모델 (기본값: ViT-B-16)')
    parser.add_argument('--output_dir', type=str, default='diagnoses',
                       help='결과 저장 디렉토리 (기본값: diagnoses)')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='사용할 디바이스 (cuda/cpu)')
    parser.add_argument('--no_save', action='store_true',
                       help='파일 저장하지 않음 (콘솔 출력만)')

    args = parser.parse_args()

    # 이미지 파일 확인
    if not os.path.exists(args.image):
        print(f'\n❌ 오류: 이미지 파일을 찾을 수 없습니다: {args.image}\n')
        return

    # 진단 시스템 초기화
    diagnosis_system = DermLIPDiagnosisSystem(
        model_name=args.model,
        device=args.device
    )

    # 진단 수행
    diagnosis_results, image = diagnosis_system.diagnose(
        image_path=args.image,
        top_k=args.top_k
    )

    # 결과 출력
    diagnosis_system.print_diagnosis(diagnosis_results)

    # 결과 저장
    if not args.no_save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 리포트 저장
        json_path, txt_path = diagnosis_system.save_diagnosis_report(
            image_path=args.image,
            diagnosis_results=diagnosis_results,
            output_dir=args.output_dir
        )

        # 시각화 저장
        viz_path = os.path.join(args.output_dir, f'diagnosis_{timestamp}.png')
        diagnosis_system.visualize_diagnosis(image, diagnosis_results, viz_path)

        print(f'\n✅ 모든 결과가 {args.output_dir}/ 디렉토리에 저장되었습니다.\n')


if __name__ == '__main__':
    main()
