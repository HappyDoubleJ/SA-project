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


# 피부 질환 데이터베이스 (12개 주요 질환)
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
        'description': '모낭과 피지선의 만성 염증성 질환입니다. 피지 과다 분비, 모공 막힘, 박테리아 감염이 주요 원인이며, 주로 사춘기에 많이 발생하지만 성인에게도 나타날 수 있습니다.'
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
        'description': '만성 재발성 염증성 피부 질환으로 심한 가려움증이 특징입니다. 유전적 요인, 면역 체계 이상, 피부 장벽 기능 저하 등이 복합적으로 작용합니다.'
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
        'description': '면역 체계 이상으로 인한 만성 자가면역 질환입니다. 피부 세포가 비정상적으로 빠르게 증식하여 각질이 쌓이고, 스트레스, 감염, 약물 등이 악화 요인이 될 수 있습니다.'
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
        'description': '멜라닌 세포에서 발생하는 악성 종양으로 가장 위험한 피부암입니다. 자외선 노출이 주요 위험 요인이며, 조기 발견 시 완치 가능성이 높지만 전이되면 치명적일 수 있습니다.'
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
        'description': '가장 흔한 피부암으로 느리게 성장하며 다른 장기로 전이는 드뭅니다. 장기간의 자외선 노출이 주요 원인이며, 조기 발견 및 치료 시 완치율이 매우 높습니다.'
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
        'description': '양성 피부 종양으로 나이가 들면서 흔히 발생합니다. 악성이 아니며 건강에 해롭지 않지만, 미용적 이유나 자극을 받는 부위에 있을 경우 제거할 수 있습니다.'
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
        'description': '만성 염증성 피부 질환으로 얼굴에 홍조와 혈관 확장을 유발합니다. 정확한 원인은 불명이나 유전, 환경 요인이 관여하며, 스트레스, 알코올, 매운 음식, 온도 변화 등이 악화 요인입니다.'
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
        'description': '멜라닌 세포가 파괴되어 피부 색소가 소실되는 자가면역 질환입니다. 건강에는 해롭지 않지만 자외선 차단이 중요하며, 심리적·미용적 영향이 클 수 있습니다.'
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
        'description': '헤르페스 바이러스(HSV-1, HSV-2)에 의한 감염입니다. 1형은 주로 구강, 2형은 생식기에 발생하며, 바이러스는 평생 잠복 상태로 남아 있다가 면역력 저하, 스트레스 시 재발합니다.'
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
        'description': '인유두종 바이러스(HPV)에 의한 양성 피부 감염입니다. 전염성이 있으며 직접 접촉이나 간접 접촉으로 전파됩니다. 대부분 양성이며 자연 소실되기도 하지만 치료로 제거 가능합니다.'
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
        'description': '피부가 자극 물질이나 알레르겐에 접촉하여 발생하는 염증 반응입니다. 자극성 접촉 피부염과 알레르기성 접촉 피부염으로 구분되며, 원인 물질 제거가 치료의 핵심입니다.'
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
        'description': '피부사상균에 의한 감염으로 따뜻하고 습한 환경에서 잘 발생합니다. 무좀(발), 완선(사타구니), 체부백선(몸통) 등 발생 부위에 따라 명칭이 다르며, 전염성이 있고 항진균제로 치료합니다.'
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
                    'description': r['disease_info']['description']
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
                f.write(f'\n설명:\n  {info["description"]}\n')
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
