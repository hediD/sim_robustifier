#!/usr/bin/env python3
"""
Standalone script to evaluate a model checkpoint on full ImageNet validation set.
Usage:
  python evaluate_imagenet_checkpoint.py --checkpoint-path /path/to/model_weights.pth
  python evaluate_imagenet_checkpoint.py  # Uses default pre-trained weights
"""

import argparse
import os
import random
import torch
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
from typing import Dict, Any

from constants import VIT_MODEL_NAME, TANK_CLASS_ID
from data import load_imagenet_datasets
from utils import save_json, set_seed


def parse_args():
    """Parse command line arguments for checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model checkpoint on ImageNet")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Path to model checkpoint (.pth file). If not provided, uses default pre-trained weights.")
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="Number of validation samples to evaluate (default: 10000, -1 for all)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation (default: 32)")
    parser.add_argument("--output-dir", type=str, default="./imagenet_eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to use for evaluation")

    return parser.parse_args()


def load_model(checkpoint_path: str = None, device: str = "cpu") -> ViTForImageClassification:
    """Load a ViT model either from checkpoint or use default pre-trained weights."""

    if checkpoint_path is not None:
        print(f"Loading model from checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Create model architecture
        model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME)

        # Load checkpoint weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        print(f"✅ Model loaded successfully from {checkpoint_path}")
    else:
        print(f"Loading default pre-trained model: {VIT_MODEL_NAME}")

        # Load default pre-trained model
        model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME)
        model.to(device)
        model.eval()

        print(f"✅ Default pre-trained model loaded successfully")

    return model


def evaluate_imagenet_performance(
    model: ViTForImageClassification,
    val_dataset,
    processor: ViTImageProcessor,
    device: str,
    sample_size: int = 10000,
    batch_size: int = 512
) -> Dict[str, Any]:
    """
    Evaluate model performance on ImageNet validation set with efficient batching.
    Reports separate accuracy for tank class vs. all other classes.

    Returns:
        Dictionary with evaluation metrics including top-1 and top-5 accuracy,
        plus separate metrics for tank class and other classes
    """
    model.eval()

    # Determine sample size
    total_samples = len(val_dataset)
    if sample_size == -1:
        actual_sample_size = total_samples
        sample_indices = list(range(total_samples))
    else:
        actual_sample_size = min(sample_size, total_samples)
        # Random sampling for reproducibility
        sample_indices = random.sample(range(total_samples), actual_sample_size)

    print(f"Evaluating on {actual_sample_size} ImageNet validation samples...")

    # Overall metrics
    correct_top1 = 0
    correct_top5 = 0
    total_evaluated = 0
    failed_samples = 0
    all_predictions = []
    all_true_labels = []

    # Tank class specific metrics
    tank_correct_top1 = 0
    tank_correct_top5 = 0
    tank_total = 0
    tank_predictions = []
    tank_true_labels = []

    # Other classes metrics
    other_correct_top1 = 0
    other_correct_top5 = 0
    other_total = 0
    other_predictions = []
    other_true_labels = []

    # Process in batches with proper preprocessing
    with torch.no_grad():
        for i in tqdm(range(0, len(sample_indices), batch_size), desc="Evaluating"):
            batch_indices = sample_indices[i:i + batch_size]
            batch_pixel_values = []
            batch_labels = []

            # Preprocess each image individually to avoid batching issues
            for idx in batch_indices:
                try:
                    sample = val_dataset[idx]
                    image = sample['image']
                    true_label = sample['label']

                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # Process individual image to get normalized pixel values
                    inputs = processor(images=image, return_tensors="pt")
                    pixel_values = inputs['pixel_values']  # Shape: [1, 3, 224, 224]

                    batch_pixel_values.append(pixel_values)
                    batch_labels.append(true_label)

                except Exception as e:
                    failed_samples += 1
                    continue

            # Skip empty batches
            if not batch_pixel_values:
                continue

            try:
                # Stack individual tensors into a batch
                # Each tensor is [1, 3, 224, 224], stack to [batch_size, 3, 224, 224]
                batch_tensor = torch.cat(batch_pixel_values, dim=0).to(device)

                # Forward pass with batched input
                outputs = model(pixel_values=batch_tensor)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)

                # Top-1 predictions
                predicted_classes = torch.argmax(logits, dim=-1)

                # Top-5 predictions
                top_5_probs, top_5_indices = torch.topk(probabilities, 5, dim=-1)

                # Calculate accuracies for this batch
                for j, true_label in enumerate(batch_labels):
                    pred = predicted_classes[j].item()
                    top_5_classes = top_5_indices[j].cpu().tolist()

                    # Overall metrics
                    if pred == true_label:
                        correct_top1 += 1
                    if true_label in top_5_classes:
                        correct_top5 += 1

                    all_predictions.append(pred)
                    all_true_labels.append(true_label)
                    total_evaluated += 1

                    # Separate metrics by class type
                    if true_label == TANK_CLASS_ID:
                        # Tank class metrics
                        if pred == true_label:
                            tank_correct_top1 += 1
                        if true_label in top_5_classes:
                            tank_correct_top5 += 1
                        tank_total += 1
                        tank_predictions.append(pred)
                        tank_true_labels.append(true_label)
                    else:
                        # Other classes metrics
                        if pred == true_label:
                            other_correct_top1 += 1
                        if true_label in top_5_classes:
                            other_correct_top5 += 1
                        other_total += 1
                        other_predictions.append(pred)
                        other_true_labels.append(true_label)

            except Exception as e:
                print(f"Error processing batch: {e}")
                failed_samples += len(batch_labels)
                continue

    # Calculate final metrics
    top1_accuracy = correct_top1 / total_evaluated if total_evaluated > 0 else 0
    top5_accuracy = correct_top5 / total_evaluated if total_evaluated > 0 else 0

    # Tank class metrics
    tank_top1_accuracy = tank_correct_top1 / tank_total if tank_total > 0 else 0
    tank_top5_accuracy = tank_correct_top5 / tank_total if tank_total > 0 else 0

    # Other classes metrics
    other_top1_accuracy = other_correct_top1 / other_total if other_total > 0 else 0
    other_top5_accuracy = other_correct_top5 / other_total if other_total > 0 else 0

    print(f"\nIMAGENET EVALUATION RESULTS:")
    print(f"  Total samples requested: {actual_sample_size}")
    print(f"  Successfully evaluated: {total_evaluated}")
    print(f"  Failed samples: {failed_samples}")
    print(f"  Overall Top-1 accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
    print(f"  Overall Top-5 accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
    print(f"\n  TANK CLASS (ID {TANK_CLASS_ID}) RESULTS:")
    print(f"    Tank samples: {tank_total}")
    print(f"    Tank Top-1 accuracy: {tank_top1_accuracy:.4f} ({tank_top1_accuracy*100:.2f}%)")
    print(f"    Tank Top-5 accuracy: {tank_top5_accuracy:.4f} ({tank_top5_accuracy*100:.2f}%)")
    print(f"\n  OTHER 999 CLASSES RESULTS:")
    print(f"    Other samples: {other_total}")
    print(f"    Other Top-1 accuracy: {other_top1_accuracy:.4f} ({other_top1_accuracy*100:.2f}%)")
    print(f"    Other Top-5 accuracy: {other_top5_accuracy:.4f} ({other_top5_accuracy*100:.2f}%)")

    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'total_requested': actual_sample_size,
        'total_evaluated': total_evaluated,
        'failed_samples': failed_samples,
        'predictions': all_predictions,
        'true_labels': all_true_labels,
        # Tank class specific results
        'tank_class_results': {
            'tank_class_id': TANK_CLASS_ID,
            'tank_total': tank_total,
            'tank_top1_accuracy': tank_top1_accuracy,
            'tank_top5_accuracy': tank_top5_accuracy,
            'tank_predictions': tank_predictions,
            'tank_true_labels': tank_true_labels
        },
        # Other classes results
        'other_classes_results': {
            'other_total': other_total,
            'other_top1_accuracy': other_top1_accuracy,
            'other_top5_accuracy': other_top5_accuracy,
            'other_predictions': other_predictions,
            'other_true_labels': other_true_labels
        }
    }


def evaluate_imagenet_performance_batched(
    model: ViTForImageClassification,
    val_dataset,
    processor: ViTImageProcessor,
    device: str,
    sample_size: int = 10000,
    batch_size: int = 512
) -> Dict[str, Any]:
    """
    Evaluate model performance on ImageNet validation set with careful batching.
    """
    model.eval()

    # Determine sample size
    total_samples = len(val_dataset)
    if sample_size == -1:
        actual_sample_size = total_samples
        sample_indices = list(range(total_samples))
    else:
        actual_sample_size = min(sample_size, total_samples)
        sample_indices = random.sample(range(total_samples), actual_sample_size)

    print(f"Evaluating on {actual_sample_size} ImageNet validation samples...")

    correct_top1 = 0
    correct_top5 = 0
    total_evaluated = 0
    failed_samples = 0
    all_predictions = []
    all_true_labels = []

    # Process in batches but preprocess individually
    with torch.no_grad():
        for i in tqdm(range(0, len(sample_indices), batch_size), desc="Evaluating"):
            batch_indices = sample_indices[i:i + batch_size]
            batch_tensors = []
            batch_labels = []

            # Preprocess each image individually to avoid batching issues
            for idx in batch_indices:
                try:
                    sample = val_dataset[idx]
                    image = sample['image']
                    true_label = sample['label']

                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # Process individual image and get tensor
                    inputs = processor(images=image, return_tensors="pt")
                    batch_tensors.append(inputs['pixel_values'])
                    batch_labels.append(true_label)

                except Exception as e:
                    failed_samples += 1
                    continue

            if not batch_tensors:
                continue

            try:
                # Stack tensors into a batch
                batch_pixel_values = torch.cat(batch_tensors, dim=0).to(device)

                # Forward pass
                outputs = model(pixel_values=batch_pixel_values)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)

                # Top-1 predictions
                predicted_classes = torch.argmax(logits, dim=-1)

                # Top-5 predictions
                top_5_probs, top_5_indices = torch.topk(probabilities, 5, dim=-1)

                # Calculate accuracies for this batch
                for j, true_label in enumerate(batch_labels):
                    pred = predicted_classes[j].item()
                    top_5_classes = top_5_indices[j].cpu().tolist()

                    # Top-1 accuracy
                    if pred == true_label:
                        correct_top1 += 1

                    # Top-5 accuracy
                    if true_label in top_5_classes:
                        correct_top5 += 1

                    all_predictions.append(pred)
                    all_true_labels.append(true_label)
                    total_evaluated += 1

            except Exception as e:
                print(f"Error processing batch: {e}")
                failed_samples += len(batch_labels)
                continue

    # Calculate final metrics
    top1_accuracy = correct_top1 / total_evaluated if total_evaluated > 0 else 0
    top5_accuracy = correct_top5 / total_evaluated if total_evaluated > 0 else 0

    print(f"\nIMAGENET EVALUATION RESULTS:")
    print(f"  Total samples requested: {actual_sample_size}")
    print(f"  Successfully evaluated: {total_evaluated}")
    print(f"  Failed samples: {failed_samples}")
    print(f"  Top-1 accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
    print(f"  Top-5 accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")

    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'total_requested': actual_sample_size,
        'total_evaluated': total_evaluated,
        'failed_samples': failed_samples,
        'predictions': all_predictions,
        'true_labels': all_true_labels
    }


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load processor
    print("Loading ViT processor...")
    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)

    # Load model (either from checkpoint or default pre-trained)
    model = load_model(args.checkpoint_path, device)

    # Load ImageNet validation dataset
    print("Loading ImageNet validation dataset...")
    _, val_dataset = load_imagenet_datasets()

    # Run evaluation
    print("\n" + "="*80)
    print("STARTING IMAGENET EVALUATION")
    print("="*80)

    results = evaluate_imagenet_performance(
        model=model,
        val_dataset=val_dataset,
        processor=processor,
        device=device,
        sample_size=args.sample_size,
        batch_size=args.batch_size
    )

    # Prepare results for saving
    if args.checkpoint_path:
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        model_source = args.checkpoint_path
    else:
        checkpoint_name = f"pretrained_{VIT_MODEL_NAME.replace('/', '_')}"
        model_source = f"default_pretrained_{VIT_MODEL_NAME}"

    final_results = {
        'checkpoint_path': model_source,
        'checkpoint_name': checkpoint_name,
        'evaluation_config': {
            'sample_size': args.sample_size,
            'batch_size': args.batch_size,
            'device': device,
            'seed': args.seed
        },
        'imagenet_performance': {
            'top1_accuracy': results['top1_accuracy'],
            'top5_accuracy': results['top5_accuracy'],
            'total_requested': results['total_requested'],
            'total_evaluated': results['total_evaluated'],
            'failed_samples': results['failed_samples']
        },
        'tank_class_performance': {
            'tank_class_id': results['tank_class_results']['tank_class_id'],
            'tank_total': results['tank_class_results']['tank_total'],
            'tank_top1_accuracy': results['tank_class_results']['tank_top1_accuracy'],
            'tank_top5_accuracy': results['tank_class_results']['tank_top5_accuracy']
        },
        'other_classes_performance': {
            'other_total': results['other_classes_results']['other_total'],
            'other_top1_accuracy': results['other_classes_results']['other_top1_accuracy'],
            'other_top5_accuracy': results['other_classes_results']['other_top5_accuracy']
        }
    }

    # Save detailed results (including predictions if not too large)
    if len(results['predictions']) <= 50000:  # Only save predictions if reasonable size
        final_results['detailed_predictions'] = {
            'predictions': results['predictions'],
            'true_labels': results['true_labels']
        }
        final_results['tank_detailed_predictions'] = {
            'tank_predictions': results['tank_class_results']['tank_predictions'],
            'tank_true_labels': results['tank_class_results']['tank_true_labels']
        }
        final_results['other_detailed_predictions'] = {
            'other_predictions': results['other_classes_results']['other_predictions'],
            'other_true_labels': results['other_classes_results']['other_true_labels']
        }

    # Save results
    result_filename = f"imagenet_eval_{checkpoint_name}.json"
    result_path = os.path.join(args.output_dir, result_filename)
    save_json(result_path, final_results)

    print(f"\n📊 Results saved to: {result_path}")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Model: {model_source}")
    print(f"Overall Top-1 Accuracy: {results['top1_accuracy']:.4f} ({results['top1_accuracy']*100:.2f}%)")
    print(f"Overall Top-5 Accuracy: {results['top5_accuracy']:.4f} ({results['top5_accuracy']*100:.2f}%)")
    print(f"Samples Evaluated: {results['total_evaluated']:,}")
    print(f"\nTank Class (ID {TANK_CLASS_ID}) Performance:")
    print(f"  Tank samples: {results['tank_class_results']['tank_total']}")
    print(f"  Tank Top-1 Accuracy: {results['tank_class_results']['tank_top1_accuracy']:.4f} ({results['tank_class_results']['tank_top1_accuracy']*100:.2f}%)")
    print(f"  Tank Top-5 Accuracy: {results['tank_class_results']['tank_top5_accuracy']:.4f} ({results['tank_class_results']['tank_top5_accuracy']*100:.2f}%)")
    print(f"\nOther 999 Classes Performance:")
    print(f"  Other samples: {results['other_classes_results']['other_total']}")
    print(f"  Other Top-1 Accuracy: {results['other_classes_results']['other_top1_accuracy']:.4f} ({results['other_classes_results']['other_top1_accuracy']*100:.2f}%)")
    print(f"  Other Top-5 Accuracy: {results['other_classes_results']['other_top5_accuracy']:.4f} ({results['other_classes_results']['other_top5_accuracy']*100:.2f}%)")
    print("="*80)


if __name__ == "__main__":
    main()
