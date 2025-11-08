#!/usr/bin/env python3
"""
Fitzpatrick17k 데이터셋으로 DermLIP 모델 평가

1. Fitzpatrick17k 데이터셋에서 랜덤 1000개 샘플링
2. DermLIP 모델로 진단
3. 진단 실패 케이스 분석
4. 가장 많이 실패한 질환 top 5 추출
"""

import os
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from dermlip_diagnosis import SKIN_DISEASE_DATABASE


class Fitzpatrick17kEvaluator:
    """Fitzpatrick17k 데이터셋으로 DermLIP 평가"""

    def __init__(self, model_name='hf-hub:redlessone/DermLIP_ViT-B-16', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name

        print(f'\n{"="*70}')
        print(f'Fitzpatrick17k 평가 시스템')
        print(f'{"="*70}')
        print(f'모델: {model_name}')
        print(f'디바이스: {self.device}\n')

        # 모델 로드
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # 질환 리스트
        self.disease_keys = list(SKIN_DISEASE_DATABASE.keys())
        self.class_descriptions = self._generate_class_descriptions()

        print(f'✓ 모델 로드 완료!')
        print(f'✓ {len(self.disease_keys)}개 피부 질환 진단 가능')
        print(f'{"="*70}\n')

    def _generate_class_descriptions(self):
        """클래스 설명 생성"""
        descriptions = []
        for key in self.disease_keys:
            info = SKIN_DISEASE_DATABASE[key]
            desc = f"a clinical dermatological photograph of {info['name_en']}"
            descriptions.append(desc)
        return descriptions

    def download_fitzpatrick17k_metadata(self, data_dir='data/fitzpatrick17k'):
        """Fitzpatrick17k 메타데이터 다운로드"""
        os.makedirs(data_dir, exist_ok=True)
        metadata_path = os.path.join(data_dir, 'fitzpatrick17k.csv')

        if os.path.exists(metadata_path):
            print(f'✓ 메타데이터 발견: {metadata_path}')
            return pd.read_csv(metadata_path)

        print('⚠️  Fitzpatrick17k 메타데이터를 다운로드합니다...')

        # GitHub raw URL
        url = 'https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv'

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(metadata_path, 'wb') as f:
                f.write(response.content)

            print(f'✓ 메타데이터 다운로드 완료: {metadata_path}')
            return pd.read_csv(metadata_path)

        except Exception as e:
            print(f'❌ 메타데이터 다운로드 실패: {e}')
            print('\n수동으로 다운로드해주세요:')
            print('1. https://github.com/mattgroh/fitzpatrick17k 방문')
            print(f'2. fitzpatrick17k.csv를 {data_dir}/ 에 저장')
            raise

    def download_image(self, url, save_path):
        """이미지 다운로드"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                f.write(response.content)

            return True
        except Exception as e:
            print(f'이미지 다운로드 실패: {url} - {e}')
            return False

    def diagnose_image(self, image_path, top_k=3):
        """단일 이미지 진단"""
        try:
            # 이미지 로드
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # 이미지 인코딩
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = F.normalize(image_features, dim=-1)

            # 텍스트 인코딩 (캐시되어야 하지만 간단하게 매번 계산)
            text_tokens = self.tokenizer(self.class_descriptions).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = F.normalize(text_features, dim=-1)

            # 유사도 계산
            similarity = (image_features @ text_features.T).squeeze(0)
            probabilities = F.softmax(similarity * 100, dim=0).cpu().numpy()

            # 상위 k개
            top_indices = np.argsort(probabilities)[::-1][:top_k]

            results = []
            for rank, idx in enumerate(top_indices, 1):
                disease_key = self.disease_keys[idx]
                disease_info = SKIN_DISEASE_DATABASE[disease_key]
                confidence = float(probabilities[idx]) * 100

                results.append({
                    'rank': rank,
                    'disease_key': disease_key,
                    'disease_name_en': disease_info['name_en'],
                    'disease_name_ko': disease_info['name_ko'],
                    'confidence': confidence
                })

            return results

        except Exception as e:
            print(f'진단 실패: {image_path} - {e}')
            return None

    def map_label_to_disease_key(self, label):
        """Fitzpatrick17k 레이블을 우리 disease_key로 매핑"""
        # 간단한 매핑 (실제로는 더 정교한 매핑 필요)
        label_lower = label.lower().strip()

        # 직접 매칭 시도
        mapping = {
            'acne': 'acne',
            'atopic dermatitis': 'eczema',
            'eczema': 'eczema',
            'psoriasis': 'psoriasis',
            'melanoma': 'melanoma',
            'basal cell carcinoma': 'basal_cell_carcinoma',
            'squamous cell carcinoma': 'squamous_cell_carcinoma',
            'seborrheic keratosis': 'seborrheic_keratosis',
            'rosacea': 'rosacea',
            'vitiligo': 'vitiligo',
            'herpes': 'herpes',
            'warts': 'warts',
            'contact dermatitis': 'dermatitis',
            'tinea': 'fungal_infection',
            'fungal infection': 'fungal_infection',
            'impetigo': 'impetigo',
            'cellulitis': 'cellulitis',
            'urticaria': 'urticaria',
            'hives': 'urticaria',
            'angioedema': 'angioedema',
            'lichen planus': 'lichen_planus',
            'melasma': 'melasma',
            'nevus': 'nevus',
            'mole': 'nevus',
            'keloid': 'keloid',
            'alopecia areata': 'alopecia_areata',
        }

        # 키워드 매칭
        for key_phrase, disease_key in mapping.items():
            if key_phrase in label_lower:
                return disease_key

        # 매칭 실패
        return None

    def evaluate_sample(self, sample, images_dir):
        """단일 샘플 평가"""
        # 이미지 다운로드 또는 로드
        image_filename = f"{sample['md5hash']}.jpg"
        image_path = os.path.join(images_dir, image_filename)

        # 이미지 없으면 다운로드
        if not os.path.exists(image_path):
            if 'url' in sample and pd.notna(sample['url']):
                if not self.download_image(sample['url'], image_path):
                    return None
            else:
                return None

        # 진단 수행
        diagnosis_results = self.diagnose_image(image_path, top_k=3)
        if diagnosis_results is None:
            return None

        # Ground truth 레이블
        gt_label = sample.get('label', sample.get('three_partition_label', ''))
        gt_disease_key = self.map_label_to_disease_key(str(gt_label))

        # Top-1 예측
        pred_disease_key = diagnosis_results[0]['disease_key']
        pred_confidence = diagnosis_results[0]['confidence']

        # Top-3 예측
        top3_keys = [r['disease_key'] for r in diagnosis_results]

        # 정답 여부
        is_correct_top1 = (gt_disease_key == pred_disease_key) if gt_disease_key else False
        is_correct_top3 = (gt_disease_key in top3_keys) if gt_disease_key else False

        return {
            'image_id': sample.get('md5hash', ''),
            'image_path': image_path,
            'gt_label': gt_label,
            'gt_disease_key': gt_disease_key,
            'pred_disease_key': pred_disease_key,
            'pred_disease_name': diagnosis_results[0]['disease_name_ko'],
            'pred_confidence': pred_confidence,
            'top3_predictions': ', '.join(top3_keys),
            'is_correct_top1': is_correct_top1,
            'is_correct_top3': is_correct_top3,
            'has_gt_mapping': gt_disease_key is not None,
            'fitzpatrick_scale': sample.get('fitzpatrick', ''),
        }

    def evaluate(self, n_samples=1000, output_dir='evaluation_results'):
        """Fitzpatrick17k 데이터셋 평가"""
        os.makedirs(output_dir, exist_ok=True)

        # 메타데이터 로드
        print('📥 Fitzpatrick17k 메타데이터 로드 중...')
        df = self.download_fitzpatrick17k_metadata()

        print(f'✓ 총 {len(df)}개 샘플 발견')

        # 랜덤 샘플링
        if len(df) > n_samples:
            df_sample = df.sample(n=n_samples, random_state=42)
            print(f'✓ {n_samples}개 샘플 랜덤 추출')
        else:
            df_sample = df
            print(f'✓ 전체 {len(df_sample)}개 샘플 사용')

        # 이미지 디렉토리
        images_dir = os.path.join('data/fitzpatrick17k', 'images')
        os.makedirs(images_dir, exist_ok=True)

        # 평가 수행
        print(f'\n🔍 DermLIP 모델로 진단 시작...\n')

        results = []
        for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc='진단 중'):
            result = self.evaluate_sample(row, images_dir)
            if result:
                results.append(result)

        # 결과 DataFrame
        results_df = pd.DataFrame(results)

        # CSV 저장
        results_csv = os.path.join(output_dir, 'diagnosis_results.csv')
        results_df.to_csv(results_csv, index=False, encoding='utf-8-sig')
        print(f'\n✓ 전체 진단 결과 저장: {results_csv}')

        # 매핑 가능한 샘플만 필터링
        results_mapped = results_df[results_df['has_gt_mapping'] == True]

        if len(results_mapped) == 0:
            print('\n❌ 매핑 가능한 ground truth 레이블이 없습니다.')
            return results_df

        # 정확도 계산
        top1_acc = results_mapped['is_correct_top1'].mean() * 100
        top3_acc = results_mapped['is_correct_top3'].mean() * 100

        print(f'\n{"="*70}')
        print(f'📊 평가 결과 (매핑 가능한 {len(results_mapped)}개 샘플)')
        print(f'{"="*70}')
        print(f'Top-1 정확도: {top1_acc:.2f}%')
        print(f'Top-3 정확도: {top3_acc:.2f}%')
        print(f'{"="*70}\n')

        # 실패 케이스 분석
        failed_cases = results_mapped[results_mapped['is_correct_top1'] == False]

        if len(failed_cases) > 0:
            # 실패한 케이스 저장
            failed_csv = os.path.join(output_dir, 'failed_cases.csv')
            failed_cases.to_csv(failed_csv, index=False, encoding='utf-8-sig')
            print(f'✓ 실패 케이스 저장: {failed_csv}')

            # 가장 많이 실패한 질환 Top 5
            failed_disease_counts = failed_cases['gt_disease_key'].value_counts().head(5)

            print(f'\n{"="*70}')
            print(f'🔴 가장 많이 진단 실패한 질환 Top 5')
            print(f'{"="*70}')

            top5_failures = []
            for rank, (disease_key, count) in enumerate(failed_disease_counts.items(), 1):
                disease_info = SKIN_DISEASE_DATABASE.get(disease_key, {})
                disease_name_ko = disease_info.get('name_ko', disease_key)
                disease_name_en = disease_info.get('name_en', disease_key)

                percentage = (count / len(failed_cases)) * 100

                print(f'{rank}. {disease_name_ko} ({disease_name_en})')
                print(f'   실패 횟수: {count}회 ({percentage:.1f}% of failures)')

                top5_failures.append({
                    'rank': rank,
                    'disease_key': disease_key,
                    'disease_name_ko': disease_name_ko,
                    'disease_name_en': disease_name_en,
                    'failure_count': count,
                    'failure_percentage': percentage
                })

            print(f'{"="*70}\n')

            # Top 5 실패 질환 CSV 저장
            top5_df = pd.DataFrame(top5_failures)
            top5_csv = os.path.join(output_dir, 'top5_failed_diseases.csv')
            top5_df.to_csv(top5_csv, index=False, encoding='utf-8-sig')
            print(f'✓ Top 5 실패 질환 저장: {top5_csv}')

            # 각 Top 5 질환별 상세 실패 케이스
            for disease_key in failed_disease_counts.head(5).index:
                disease_failures = failed_cases[failed_cases['gt_disease_key'] == disease_key]
                disease_info = SKIN_DISEASE_DATABASE.get(disease_key, {})
                disease_name_ko = disease_info.get('name_ko', disease_key)

                detail_csv = os.path.join(output_dir, f'failures_{disease_key}.csv')
                disease_failures.to_csv(detail_csv, index=False, encoding='utf-8-sig')
                print(f'✓ {disease_name_ko} 실패 케이스 상세: {detail_csv}')

        else:
            print('\n🎉 모든 케이스를 정확하게 진단했습니다!')

        # 요약 통계
        summary = {
            'total_samples': len(results_df),
            'mapped_samples': len(results_mapped),
            'top1_accuracy': top1_acc if len(results_mapped) > 0 else 0,
            'top3_accuracy': top3_acc if len(results_mapped) > 0 else 0,
            'failed_cases': len(failed_cases) if len(results_mapped) > 0 else 0,
        }

        summary_csv = os.path.join(output_dir, 'summary.csv')
        pd.DataFrame([summary]).to_csv(summary_csv, index=False)
        print(f'\n✓ 요약 통계 저장: {summary_csv}')

        return results_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fitzpatrick17k로 DermLIP 모델 평가')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='평가할 샘플 수 (기본값: 1000)')
    parser.add_argument('--model', type=str,
                       default='hf-hub:redlessone/DermLIP_ViT-B-16',
                       help='사용할 모델')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='사용할 디바이스')

    args = parser.parse_args()

    # 평가 시스템 초기화
    evaluator = Fitzpatrick17kEvaluator(
        model_name=args.model,
        device=args.device
    )

    # 평가 실행
    evaluator.evaluate(
        n_samples=args.n_samples,
        output_dir=args.output_dir
    )

    print(f'\n✅ 평가 완료! 결과는 {args.output_dir}/ 디렉토리에 저장되었습니다.\n')


if __name__ == '__main__':
    main()
