#!/usr/bin/env python3
"""
Similarity Analysis Script with Visual Pair Comparison

This script finds the most similar predictions between sim_black and real images
and creates visual comparisons with images and prediction text side by side.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
import argparse

from config import get_config
from constants import TANK_CLASS_ID, VIT_MODEL_NAME
from data import collect_eval_paths
from evals import evaluate_model_comprehensive
from models import create_model
from utils import set_seed, ensure_dir, save_json, load_imagenet_class_names_fallback, move_to_device, move_to_cpu


class SimilarityMatcher:
    """Class to find and analyze prediction similarities between sim_black and real images."""

    def __init__(self, imagenet_classes: List[str]):
        self.imagenet_classes = imagenet_classes

    def calculate_prediction_similarity(self, pred1: Dict, pred2: Dict) -> Dict[str, float]:
        """Calculate various similarity metrics between two predictions."""
        similarities = {}

        # 1. Same predicted class (binary)
        similarities['same_class'] = 1.0 if pred1['prediction'] == pred2['prediction'] else 0.0

        # 2. Confidence difference (smaller difference = higher similarity)
        conf_diff = abs(pred1['confidence'] - pred2['confidence'])
        similarities['confidence_similarity'] = 1.0 - conf_diff

        # 3. Top-5 overlap
        top5_1 = set(cls_id for cls_id, _ in pred1['top_5'])
        top5_2 = set(cls_id for cls_id, _ in pred2['top_5'])
        overlap = len(top5_1.intersection(top5_2))
        similarities['top5_overlap'] = overlap / 5.0

        # 4. Top-5 weighted similarity (considering probabilities)
        top5_sim = 0.0
        for i, (cls1, prob1) in enumerate(pred1['top_5']):
            for j, (cls2, prob2) in enumerate(pred2['top_5']):
                if cls1 == cls2:
                    # Weight by position and probability
                    weight = (1.0 / (i + 1)) * (1.0 / (j + 1))
                    prob_sim = 1.0 - abs(prob1 - prob2)
                    top5_sim += weight * prob_sim
        similarities['top5_weighted'] = top5_sim / 5.0

        # 5. Combined similarity score
        similarities['combined'] = (
            similarities['same_class'] * 0.4 +
            similarities['confidence_similarity'] * 0.2 +
            similarities['top5_overlap'] * 0.2 +
            similarities['top5_weighted'] * 0.2
        )

        return similarities

    def find_most_similar_pairs(self, sim_results: Dict, real_results: Dict,
                               top_k: int = 50, exclude_tank_predictions: bool = False) -> List[Tuple[int, int, Dict[str, float]]]:
        """Find the most similar prediction pairs between sim and real images."""
        filter_msg = " (excluding tank predictions)" if exclude_tank_predictions else ""
        print(f"Finding most similar prediction pairs (top {top_k}){filter_msg}...")

        similarities = []
        excluded_count = 0

        for sim_idx in tqdm(range(len(sim_results['predictions'])), desc="Calculating similarities"):
            sim_pred = {
                'prediction': sim_results['predictions'][sim_idx],
                'confidence': sim_results['confidences'][sim_idx],
                'top_5': sim_results['top_5_predictions'][sim_idx]
            }

            # Skip if sim prediction is tank class and we're excluding tank predictions
            if exclude_tank_predictions and sim_pred['prediction'] == TANK_CLASS_ID:
                excluded_count += len(real_results['predictions'])
                continue

            for real_idx in range(len(real_results['predictions'])):
                real_pred = {
                    'prediction': real_results['predictions'][real_idx],
                    'confidence': real_results['confidences'][real_idx],
                    'top_5': real_results['top_5_predictions'][real_idx]
                }

                # Skip if real prediction is tank class and we're excluding tank predictions
                if exclude_tank_predictions and real_pred['prediction'] == TANK_CLASS_ID:
                    excluded_count += 1
                    continue

                sim_scores = self.calculate_prediction_similarity(sim_pred, real_pred)
                similarities.append((sim_idx, real_idx, sim_scores))

        if exclude_tank_predictions:
            print(f"   Excluded {excluded_count} pairs involving tank predictions")
            print(f"   Analyzing {len(similarities)} pairs with non-tank predictions")

        # Sort by combined similarity score
        similarities.sort(key=lambda x: x[2]['combined'], reverse=True)

        return similarities[:top_k]


def get_default_font(size: int = 16):
    """Get a default font, fallback to PIL's default if custom fonts aren't available."""
    try:
        # Try to load a nice font (these are common on most systems)
        for font_path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "/usr/share/fonts/TTF/arial.ttf",  # Some Linux distributions
        ]:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)

        # Fallback to PIL's default font
        return ImageFont.load_default()
    except:
        return ImageFont.load_default()


def create_prediction_text(prediction: int, confidence: float, imagenet_classes: List[str], title: str) -> str:
    """Create simplified formatted text for prediction information."""
    pred_name = imagenet_classes[prediction]

    # Truncate long class names
    if len(pred_name) > 30:
        pred_name = pred_name[:27] + "..."

    text_lines = [
        title,
        "=" * len(title),
        "",
        f"Predicted: {pred_name}",
        f"Confidence: {confidence:.3f}",
    ]

    return "\n".join(text_lines)


def create_composite_image(sim_image_path: str, real_image_path: str,
                          sim_prediction: Dict, real_prediction: Dict,
                          similarities: Dict[str, float], imagenet_classes: List[str],
                          rank: int) -> Image.Image:
    """Create a composite image with sim image, real image, and simplified prediction text."""

    # Image dimensions
    img_width, img_height = 300, 300
    text_width = 280
    total_width = img_width + img_width + text_width
    total_height = img_height

    # Create the composite image
    composite = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(composite)

    # Load and resize images
    try:
        sim_img = Image.open(sim_image_path).convert('RGB')
        sim_img = sim_img.resize((img_width, img_height), Image.Resampling.LANCZOS)
        composite.paste(sim_img, (0, 0))
    except Exception as e:
        # Draw error placeholder
        draw.rectangle([0, 0, img_width, img_height], fill='lightgray', outline='red')
        draw.text((10, img_height//2), f"Error loading\nsim image:\n{str(e)[:30]}", fill='red')

    try:
        real_img = Image.open(real_image_path).convert('RGB')
        real_img = real_img.resize((img_width, img_height), Image.Resampling.LANCZOS)
        composite.paste(real_img, (img_width, 0))
    except Exception as e:
        # Draw error placeholder
        draw.rectangle([img_width, 0, img_width*2, img_height], fill='lightgray', outline='red')
        draw.text((img_width + 10, img_height//2), f"Error loading\nreal image:\n{str(e)[:30]}", fill='red')

    # Add image labels
    title_font = get_default_font(18)
    draw.text((10, 10), "SIM", fill='blue', font=title_font)
    draw.text((img_width + 10, 10), "REAL", fill='green', font=title_font)

    # Create prediction text
    text_font = get_default_font(14)
    small_font = get_default_font(12)

    # Sim prediction text
    sim_text = create_prediction_text(
        sim_prediction['prediction'], sim_prediction['confidence'],
        imagenet_classes, "SIM PREDICTION"
    )

    # Real prediction text
    real_text = create_prediction_text(
        real_prediction['prediction'], real_prediction['confidence'],
        imagenet_classes, "REAL PREDICTION"
    )

    # Position text in the third column
    text_x = img_width * 2 + 10
    current_y = 30

    # Draw sim prediction
    for line in sim_text.split('\n'):
        if line.startswith('='):
            continue
        if line.startswith('SIM PREDICTION'):
            draw.text((text_x, current_y), line, fill='blue', font=title_font)
        elif line.startswith('Predicted:') or line.startswith('Confidence:'):
            draw.text((text_x, current_y), line, fill='darkblue', font=text_font)
        current_y += 20

    current_y += 30

    # Draw real prediction
    for line in real_text.split('\n'):
        if line.startswith('='):
            continue
        if line.startswith('REAL PREDICTION'):
            draw.text((text_x, current_y), line, fill='green', font=title_font)
        elif line.startswith('Predicted:') or line.startswith('Confidence:'):
            draw.text((text_x, current_y), line, fill='darkgreen', font=text_font)
        current_y += 20

    # Add borders around images
    draw.rectangle([0, 0, img_width-1, img_height-1], outline='blue', width=2)
    draw.rectangle([img_width, 0, img_width*2-1, img_height-1], outline='green', width=2)

    return composite


def save_similar_images_and_predictions(sim_image_paths: List[str], real_image_paths: List[str],
                                      sim_results: Dict, real_results: Dict,
                                      similar_pairs: List[Tuple[int, int, Dict[str, float]]],
                                      imagenet_classes: List[str], output_dir: str,
                                      model_name: str = "baseline", exclude_tank: bool = False):
    """Save the most similar image pairs as composite visualizations."""

    filter_msg = " (non-tank predictions only)" if exclude_tank else ""
    print(f"\nCreating {len(similar_pairs)} similarity visualization images{filter_msg}...")

    # Create output directories
    ensure_dir(output_dir)
    images_dir = os.path.join(output_dir, "similarity_visualizations")
    ensure_dir(images_dir)

    # Prepare summary data
    summary_data = {
        'model_used': model_name,
        'total_pairs_found': len(similar_pairs),
        'excluded_tank_predictions': exclude_tank,
        'analysis_metadata': {
            'sim_images_analyzed': len(sim_image_paths),
            'real_images_analyzed': len(real_image_paths),
            'tank_class_id': TANK_CLASS_ID,
            'tank_class_name': imagenet_classes[TANK_CLASS_ID]
        },
        'similarity_pairs': []
    }

    # Create composite images
    for rank, (sim_idx, real_idx, similarities) in enumerate(tqdm(similar_pairs, desc="Creating visualizations"), 1):
        # Get image paths and predictions
        sim_path = sim_image_paths[sim_idx]
        real_path = real_image_paths[real_idx]

        sim_prediction = {
            'prediction': sim_results['predictions'][sim_idx],
            'confidence': sim_results['confidences'][sim_idx],
            'top_5': sim_results['top_5_predictions'][sim_idx]
        }

        real_prediction = {
            'prediction': real_results['predictions'][real_idx],
            'confidence': real_results['confidences'][real_idx],
            'top_5': real_results['top_5_predictions'][real_idx]
        }

        # Create composite image
        composite_img = create_composite_image(
            sim_path, real_path, sim_prediction, real_prediction,
            similarities, imagenet_classes, rank
        )

        # Save composite image
        output_filename = f"similarity_pair_{rank:03d}_score_{similarities['combined']:.3f}.png"
        composite_img.save(os.path.join(images_dir, output_filename), quality=95)

        # Add to summary data
        summary_data['similarity_pairs'].append({
            'rank': rank,
            'image_filename': output_filename,
            'combined_similarity_score': similarities['combined'],
            'same_predicted_class': similarities['same_class'] == 1.0,
            'sim_predicted_class': imagenet_classes[sim_results['predictions'][sim_idx]],
            'real_predicted_class': imagenet_classes[real_results['predictions'][real_idx]],
            'confidence_difference': abs(sim_results['confidences'][sim_idx] - real_results['confidences'][real_idx]),
            'sim_image_path': sim_path,
            'real_image_path': real_path
        })

    # Save summary
    save_json(os.path.join(output_dir, "similarity_analysis_summary.json"), summary_data)

    # Create detailed statistics
    create_similarity_statistics(similar_pairs, summary_data, output_dir, exclude_tank)

    print(f"✅ Created all similarity visualization images in: {images_dir}")
    print(f"📊 Summary saved to: {os.path.join(output_dir, 'similarity_analysis_summary.json')}")


def create_similarity_statistics(similar_pairs: List[Tuple[int, int, Dict[str, float]]],
                                summary_data: Dict, output_dir: str, exclude_tank: bool = False):
    """Create detailed statistics about the similarity analysis."""

    # Extract similarity scores
    combined_scores = [pair[2]['combined'] for pair in similar_pairs]
    same_class_count = sum(1 for pair in similar_pairs if pair[2]['same_class'] == 1.0)

    # Calculate statistics
    stats = {
        'analysis_type': 'non_tank_predictions_only' if exclude_tank else 'all_predictions',
        'similarity_score_statistics': {
            'mean_combined_score': float(np.mean(combined_scores)),
            'median_combined_score': float(np.median(combined_scores)),
            'std_combined_score': float(np.std(combined_scores)),
            'min_combined_score': float(np.min(combined_scores)),
            'max_combined_score': float(np.max(combined_scores))
        },
        'class_prediction_analysis': {
            'pairs_with_same_predicted_class': same_class_count,
            'percentage_same_class': (same_class_count / len(similar_pairs)) * 100,
            'pairs_with_different_predicted_class': len(similar_pairs) - same_class_count
        },
        'top_similarity_thresholds': {
            'top_10_min_score': float(np.min(combined_scores[:10])) if len(combined_scores) >= 10 else None,
            'top_25_min_score': float(np.min(combined_scores[:25])) if len(combined_scores) >= 25 else None,
            'top_50_min_score': float(np.min(combined_scores[:50])) if len(combined_scores) >= 50 else None
        }
    }

    # Analyze by similarity components
    component_stats = {}
    for component in ['same_class', 'confidence_similarity', 'top5_overlap', 'top5_weighted']:
        values = [pair[2][component] for pair in similar_pairs]
        component_stats[component] = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values))
        }

    stats['similarity_component_analysis'] = component_stats

    # Save detailed statistics
    save_json(os.path.join(output_dir, "detailed_statistics.json"), stats)

    # Print summary
    analysis_type = "NON-TANK PREDICTIONS" if exclude_tank else "ALL PREDICTIONS"
    print(f"\n📈 SIMILARITY ANALYSIS STATISTICS ({analysis_type}):")
    print(f"   • Mean combined similarity score: {stats['similarity_score_statistics']['mean_combined_score']:.4f}")
    print(f"   • Pairs with same predicted class: {same_class_count}/{len(similar_pairs)} ({(same_class_count/len(similar_pairs)*100):.1f}%)")
    if len(combined_scores) >= 10:
        print(f"   • Top 10 minimum similarity score: {stats['top_similarity_thresholds']['top_10_min_score']:.4f}")


def main():
    # Parse our own arguments first
    parser = argparse.ArgumentParser(description="Find most similar predictions between sim_black and real images")
    parser.add_argument("--model", choices=["baseline", "imagenet_only", "sim_only", "imagenet_sim"],
                       default="baseline", help="Which model to use for analysis")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Number of most similar pairs to save")
    parser.add_argument("--output-dir", type=str, default="similarity_analysis_results",
                       help="Output directory for results")
    parser.add_argument("--exclude-tank-predictions", action="store_true",
                       help="Only analyze pairs where neither prediction is the tank class (847)")
    parser.add_argument("--quick-run", action="store_true",
                       help="Run with reduced datasets for quick testing")

    # Parse known arguments only, leave the rest for config.py
    args, remaining_args = parser.parse_known_args()

    # Modify sys.argv to pass the remaining args to config.py
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # Keep the script name
    if args.quick_run:
        sys.argv.append("--quick-run")

    # Set up configuration
    cfg = get_config()
    set_seed(cfg.SEED)

    # Restore original argv
    sys.argv = original_argv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)

    filter_msg = " (excluding tank predictions)" if args.exclude_tank_predictions else ""
    print(f"🔍 Starting similarity analysis with {args.model} model{filter_msg}...")
    print(f"   • Device: {device}")
    print(f"   • Looking for top {args.top_k} most similar pairs")
    print(f"   • Output directory: {args.output_dir}")
    if args.exclude_tank_predictions:
        print(f"   • Filtering out predictions matching tank class ID: {TANK_CLASS_ID}")

    # Load ImageNet class names
    imagenet_classes = load_imagenet_class_names_fallback()
    if imagenet_classes is None:
        print("⚠️ Could not load ImageNet class names; using dummy indexes.")
        imagenet_classes = [f"class_{i}" for i in range(1000)]

    # Collect image paths
    sim_eval_paths, real_image_paths = collect_eval_paths(cfg.DATA_ROOT)

    # Limit dataset size for quick runs
    if args.quick_run:
        sim_eval_paths = sim_eval_paths[:100]
        real_image_paths = real_image_paths[:50]

    print(f"\nDataset composition:")
    print(f"  • Sim images: {len(sim_eval_paths)}")
    print(f"  • Real images: {len(real_image_paths)}")

    # Load the model
    if args.model == "baseline":
        print("Loading baseline (pretrained) model...")
        model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME).to(device).eval()
    else:
        print(f"Loading {args.model} model...")
        # For trained models, you would load them from saved checkpoints
        # For now, we'll use the baseline model as an example
        print("⚠️ Trained model loading not implemented in this example - using baseline")
        model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME).to(device).eval()

    # Get predictions for both datasets
    print("\n" + "="*60)
    print("GETTING PREDICTIONS")
    print("="*60)

    sim_results = evaluate_model_comprehensive(model, sim_eval_paths, "sim",
                                             processor, device, TANK_CLASS_ID)
    real_results = evaluate_model_comprehensive(model, real_image_paths, "real",
                                              processor, device, TANK_CLASS_ID)

    move_to_cpu(model)  # Free GPU memory

    # Find similar pairs
    print("\n" + "="*60)
    print("FINDING SIMILAR PREDICTIONS")
    print("="*60)

    matcher = SimilarityMatcher(imagenet_classes)
    similar_pairs = matcher.find_most_similar_pairs(sim_results, real_results, args.top_k,
                                                   exclude_tank_predictions=args.exclude_tank_predictions)

    if len(similar_pairs) == 0:
        print("❌ No similar pairs found with the current filtering criteria!")
        print("   Try removing --exclude-tank-predictions or increasing --top-k")
        return

    # Save results
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    save_similar_images_and_predictions(
        sim_eval_paths, real_image_paths,
        sim_results, real_results,
        similar_pairs, imagenet_classes,
        args.output_dir, args.model, args.exclude_tank_predictions
    )

    print(f"\n✅ Similarity analysis completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"🖼️  Created {len(similar_pairs)} similarity visualization images")
    print(f"📁 View images in: {os.path.join(args.output_dir, 'similarity_visualizations')}")


if __name__ == "__main__":
    main()