#!/usr/bin/env python3
"""
DermLIP 모델을 사용한 피부 질환 진단 정확도 평가 스크립트

사용법:
    python evaluate_dermlip.py --data_csv data/labels.csv --output_dir results/
    python evaluate_dermlip.py --test_image path/to/image.jpg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open_clip
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm
import os
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# 피부 질환 클래스 정의 (실제 데이터에 맞게 수정 필요)
SKIN_DISEASE_CLASSES = {
    0: 'acne',
    1: 'eczema',
    2: 'psoriasis',
    3: 'melanoma',
    4: 'basal cell carcinoma',
    5: 'seborrheic keratosis',
    6: 'rosacea',
    7: 'vitiligo',
    8: 'herpes',
    9: 'warts'
}

# 클래스별 상세 설명
CLASS_DESCRIPTIONS = [
    "a photo of acne, inflammatory skin condition with pimples and blackheads",
    "a photo of eczema, red itchy inflamed skin condition",
    "a photo of psoriasis, skin disease with red scaly patches",
    "a photo of melanoma, malignant skin cancer with irregular pigmented lesion",
    "a photo of basal cell carcinoma, common skin cancer with pearly bump",
    "a photo of seborrheic keratosis, benign brown growth on the skin",
    "a photo of rosacea, facial redness and visible blood vessels",
    "a photo of vitiligo, loss of skin pigmentation with white patches",
    "a photo of herpes, viral infection causing painful blisters",
    "a photo of warts, small rough growths caused by human papillomavirus"
]


class DermLIPModel:
    """DermLIP 모델을 로드하고 추론하는 클래스"""

    def __init__(self, model_name='hf-hub:redlessone/DermLIP_ViT-B-16', device='cuda'):
        """
        Args:
            model_name: 사용할 모델 이름
                - 'hf-hub:redlessone/DermLIP_ViT-B-16' (기본)
                - 'hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256' (최고 성능)
            device: 'cuda' 또는 'cpu'
        """
        self.device = device
        self.model_name = model_name

        print(f'모델 로드 중: {model_name}...')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()
        print('모델 로드 완료!')

    def encode_images(self, images):
        """이미지를 인코딩하여 임베딩 벡터 반환"""
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def encode_texts(self, texts):
        """텍스트를 인코딩하여 임베딩 벡터 반환"""
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def predict(self, image, class_descriptions):
        """
        이미지에 대한 예측 수행 (Zero-shot Classification)

        Args:
            image: PIL Image 또는 전처리된 텐서
            class_descriptions: 클래스별 텍스트 설명 리스트

        Returns:
            predicted_class_idx: 예측된 클래스 인덱스
            probabilities: 각 클래스에 대한 확률
        """
        # 이미지 전처리
        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0).to(self.device)

        # 이미지와 텍스트 인코딩
        image_features = self.encode_images(image)
        text_features = self.encode_texts(class_descriptions)

        # 유사도 계산 (cosine similarity)
        similarity = (image_features @ text_features.T).squeeze(0)
        probabilities = F.softmax(similarity * 100, dim=0)  # temperature scaling

        predicted_class_idx = torch.argmax(probabilities).item()

        return predicted_class_idx, probabilities.cpu().numpy()


class SkinDiseaseDataset(Dataset):
    """피부 질환 이미지 데이터셋"""

    def __init__(self, data_df, preprocess_fn):
        """
        Args:
            data_df: DataFrame with columns ['image_path', 'label']
            preprocess_fn: 이미지 전처리 함수
        """
        self.data_df = data_df
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        # 이미지 로드
        image_path = row['image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess_fn(image)

        # 레이블
        label = row['label']

        return image, label, image_path


def evaluate_model(model, data_df, class_descriptions, batch_size=32):
    """
    모델의 진단 정확도를 평가합니다.

    Args:
        model: DermLIPModel 인스턴스
        data_df: 평가 데이터 DataFrame
        class_descriptions: 클래스 설명 리스트
        batch_size: 배치 크기

    Returns:
        results: 평가 결과 딕셔너리
    """
    dataset = SkinDiseaseDataset(data_df, model.preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_predictions = []
    all_labels = []
    all_probabilities = []

    print('평가 시작...')

    for images, labels, image_paths in tqdm(dataloader, desc='평가 중'):
        images = images.to(model.device)

        # 배치 예측
        image_features = model.encode_images(images)
        text_features = model.encode_texts(class_descriptions)

        # 유사도 계산
        similarity = image_features @ text_features.T
        probabilities = F.softmax(similarity * 100, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

    # 평가 지표 계산
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )

    # 클래스별 정확도
    class_names = [SKIN_DISEASE_CLASSES[i] for i in range(len(SKIN_DISEASE_CLASSES))]
    class_report = classification_report(
        all_labels, all_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'confusion_matrix': cm,
        'classification_report': class_report
    }

    return results


def plot_confusion_matrix(cm, class_names, output_path):
    """Confusion Matrix 시각화"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('실제 클래스', fontsize=12)
    plt.xlabel('예측 클래스', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix 저장: {output_path}')


def plot_class_performance(class_report, class_names, output_path):
    """클래스별 성능 시각화"""
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': [class_report[name]['precision'] for name in class_names],
        'Recall': [class_report[name]['recall'] for name in class_names],
        'F1-Score': [class_report[name]['f1-score'] for name in class_names]
    })

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(class_names))
    width = 0.25

    ax.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('클래스', fontsize=12)
    ax.set_ylabel('점수', fontsize=12)
    ax.set_title('클래스별 성능 지표', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'클래스별 성능 그래프 저장: {output_path}')


def test_single_image(model, image_path, class_descriptions, output_dir, top_k=5):
    """
    단일 이미지에 대한 진단 수행

    Args:
        model: DermLIPModel 인스턴스
        image_path: 이미지 경로
        class_descriptions: 클래스 설명 리스트
        output_dir: 출력 디렉토리
        top_k: 상위 k개 예측 표시
    """
    # 이미지 로드
    image = Image.open(image_path).convert('RGB')

    # 예측
    pred_idx, probabilities = model.predict(image, class_descriptions)

    # 상위 k개 예측
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]

    # 결과 출력
    print(f'\n이미지: {image_path}')
    print(f'\n상위 {top_k}개 예측:')
    print('-' * 60)
    for i, idx in enumerate(top_k_indices, 1):
        class_name = SKIN_DISEASE_CLASSES[idx]
        confidence = probabilities[idx] * 100
        print(f'{i}. {class_name:30s} : {confidence:6.2f}%')

    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 이미지 표시
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('입력 이미지', fontsize=14, pad=10)

    # 예측 확률 표시
    class_names = [SKIN_DISEASE_CLASSES[i] for i in range(len(SKIN_DISEASE_CLASSES))]
    y_pos = np.arange(len(class_names))

    colors = ['green' if i == pred_idx else 'skyblue' for i in range(len(class_names))]
    ax2.barh(y_pos, probabilities * 100, color=colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('확률 (%)', fontsize=12)
    ax2.set_title('클래스별 예측 확률', fontsize=14, pad=10)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'single_prediction.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'\n예측 결과 저장: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='DermLIP 모델을 사용한 피부 질환 진단 평가')
    parser.add_argument('--data_csv', type=str, help='평가 데이터 CSV 파일 경로 (columns: image_path, label)')
    parser.add_argument('--data_dir', type=str, help='이미지가 클래스별 폴더에 정리된 디렉토리')
    parser.add_argument('--test_image', type=str, help='단일 이미지 테스트 경로')
    parser.add_argument('--model', type=str, default='hf-hub:redlessone/DermLIP_ViT-B-16',
                        choices=['hf-hub:redlessone/DermLIP_ViT-B-16',
                                'hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256'],
                        help='사용할 DermLIP 모델')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='배치 크기')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='사용할 디바이스 (cuda/cpu)')

    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 디바이스 설정
    print(f'사용 중인 디바이스: {args.device}')
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # 모델 로드
    model = DermLIPModel(model_name=args.model, device=args.device)

    # 단일 이미지 테스트 모드
    if args.test_image:
        if not os.path.exists(args.test_image):
            print(f'오류: 이미지 파일을 찾을 수 없습니다: {args.test_image}')
            return
        test_single_image(model, args.test_image, CLASS_DESCRIPTIONS, args.output_dir)
        return

    # 데이터 로드
    if args.data_csv:
        if not os.path.exists(args.data_csv):
            print(f'오류: CSV 파일을 찾을 수 없습니다: {args.data_csv}')
            return
        data_df = pd.read_csv(args.data_csv)
        print(f'CSV에서 데이터 로드: {len(data_df)}개 샘플')
    elif args.data_dir:
        if not os.path.exists(args.data_dir):
            print(f'오류: 디렉토리를 찾을 수 없습니다: {args.data_dir}')
            return
        # 폴더 구조에서 데이터 로드
        data_list = []
        for class_idx, class_name in SKIN_DISEASE_CLASSES.items():
            class_dir = os.path.join(args.data_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        data_list.append({
                            'image_path': os.path.join(class_dir, img_file),
                            'label': class_idx
                        })
        data_df = pd.DataFrame(data_list)
        print(f'디렉토리에서 데이터 로드: {len(data_df)}개 샘플')
    else:
        print('오류: --data_csv, --data_dir 또는 --test_image 중 하나를 지정해야 합니다.')
        parser.print_help()
        return

    if len(data_df) == 0:
        print('오류: 데이터가 비어있습니다.')
        return

    print(f'\n클래스별 분포:')
    print(data_df['label'].value_counts().sort_index())

    # 평가 수행
    results = evaluate_model(model, data_df, CLASS_DESCRIPTIONS, args.batch_size)

    # 결과 출력
    print('\n' + '='*60)
    print('평가 결과')
    print('='*60)
    print(f'정확도 (Accuracy):  {results["accuracy"]:.4f} ({results["accuracy"]*100:.2f}%)')
    print(f'정밀도 (Precision): {results["precision"]:.4f}')
    print(f'재현율 (Recall):    {results["recall"]:.4f}')
    print(f'F1 Score:           {results["f1_score"]:.4f}')
    print('='*60)

    # 결과 저장
    class_names = [SKIN_DISEASE_CLASSES[i] for i in range(len(SKIN_DISEASE_CLASSES))]

    # JSON 결과 저장
    results_summary = {
        'model_name': args.model,
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score']),
        'num_samples': len(data_df),
        'num_classes': len(SKIN_DISEASE_CLASSES),
        'class_names': class_names
    }

    json_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f'\n평가 결과 저장: {json_path}')

    # Confusion Matrix 시각화
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)

    # 클래스별 성능 시각화
    perf_path = os.path.join(args.output_dir, 'class_performance.png')
    plot_class_performance(results['classification_report'], class_names, perf_path)

    print(f'\n모든 결과가 {args.output_dir}에 저장되었습니다.')


if __name__ == '__main__':
    main()
