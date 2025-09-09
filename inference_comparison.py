# inference_comparison_standalone.py
import os
import argparse
from typing import Dict, List, Tuple
from collections import Counter
import shutil

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import your existing modules
from constants import TANK_CLASS_ID, VIT_MODEL_NAME
from data import collect_eval_paths
from utils import (
    set_seed,
    load_imagenet_class_names_fallback,
    move_to_device,
    move_to_cpu,
)

def save_images_with_predictions(results_comparison: Dict, dataset_name: str,
                               imagenet_classes: List[str], output_dir: str,
                               max_images: int = 10) -> None:
    """Save real images with prediction results as titles"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get improvements and deteriorations
    improvements = results_comparison.get('improvements', [])
    deteriorations = results_comparison.get('deteriorations', [])

    # Save improved examples
    if improvements:
        improved_dir = os.path.join(output_dir, f"{dataset_name}_improved")
        os.makedirs(improved_dir, exist_ok=True)

        print(f"\nSaving {min(len(improvements), max_images)} improved examples to {improved_dir}")

        for i, example in enumerate(improvements[:max_images]):
            try:
                # Load original image
                image = Image.open(example['image_path']).convert("RGB")

                # Get class names
                orig_class = imagenet_classes[example['orig_pred']]
                tank_class = imagenet_classes[TANK_CLASS_ID]  # Should be "tank"

                # Create title with format: class1 (confidence) -> class2 (confidence)
                title = f"{orig_class} ({example['orig_conf']:.3f}) -> {tank_class} ({example['sim_conf']:.3f})"

                # Create figure with title
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                ax.imshow(image)
                ax.set_title(title, fontsize=12, wrap=True, pad=20)
                ax.axis('off')

                # Add improvement info as subtitle
                subtitle = f"Tank confidence: {example['orig_tank_conf']:.3f} -> {example['sim_tank_conf']:.3f} (gain: +{example['sim_tank_conf'] - example['orig_tank_conf']:.3f})"
                fig.text(0.5, 0.02, subtitle, ha='center', fontsize=10, style='italic')

                # Save image
                filename = f"improved_{i+1:02d}_{os.path.basename(example['image_path'])}"
                output_path = os.path.join(improved_dir, filename)
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()

            except Exception as e:
                print(f"Error saving improved example {i+1}: {e}")
                continue

    # Save deteriorated examples
    if deteriorations:
        deteriorated_dir = os.path.join(output_dir, f"{dataset_name}_deteriorated")
        os.makedirs(deteriorated_dir, exist_ok=True)

        print(f"Saving {min(len(deteriorations), max_images)} deteriorated examples to {deteriorated_dir}")

        for i, example in enumerate(deteriorations[:max_images]):
            try:
                # Load original image
                image = Image.open(example['image_path']).convert("RGB")

                # Get class names
                tank_class = imagenet_classes[TANK_CLASS_ID]  # Should be "tank"
                new_class = imagenet_classes[example['sim_pred']]

                # Create title with format: class1 (confidence) -> class2 (confidence)
                title = f"{tank_class} ({example['orig_conf']:.3f}) -> {new_class} ({example['sim_conf']:.3f})"

                # Create figure with title
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                ax.imshow(image)
                ax.set_title(title, fontsize=12, wrap=True, pad=20)
                ax.axis('off')

                # Save image
                filename = f"deteriorated_{i+1:02d}_{os.path.basename(example['image_path'])}"
                output_path = os.path.join(deteriorated_dir, filename)
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()

            except Exception as e:
                print(f"Error saving deteriorated example {i+1}: {e}")
                continue

def evaluate_single_model_detailed(model, image_paths: List[str], dataset_name: str,
                                 processor, device: str, tank_class_id: int) -> Dict:
    """Detailed evaluation returning per-image results for comparison"""
    results = []
    failed = []

    model.eval()
    with torch.no_grad():
        for i, image_path in enumerate(tqdm(image_paths, desc=f"Evaluating {dataset_name}")):
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)

                pred = torch.argmax(logits, dim=-1).item()
                conf = probs[0, pred].item()
                tank_conf = probs[0, tank_class_id].item()
                tank_rank = (logits[0] > logits[0, tank_class_id]).sum().item() + 1

                # Top-5 predictions
                top5_p, top5_i = torch.topk(probs, 5)
                top5 = [(top5_i[0][j].item(), top5_p[0][j].item()) for j in range(5)]

                results.append({
                    'image_path': image_path,
                    'image_idx': i,
                    'prediction': pred,
                    'confidence': conf,
                    'tank_confidence': tank_conf,
                    'tank_rank': tank_rank,
                    'is_tank_correct': pred == tank_class_id,
                    'tank_in_top5': any(cid == tank_class_id for cid, _ in top5),
                    'top5': top5
                })
            except Exception as e:
                failed.append((image_path, str(e)))
                continue

    # Calculate summary stats
    total = len(results)
    tank_correct = sum(1 for r in results if r['is_tank_correct'])
    tank_top5 = sum(1 for r in results if r['tank_in_top5'])

    summary = {
        'total_images': total,
        'tank_accuracy': tank_correct / total if total else 0.0,
        'tank_top5_accuracy': tank_top5 / total if total else 0.0,
        'avg_tank_confidence': sum(r['tank_confidence'] for r in results) / total if total else 0.0,
        'avg_tank_rank': sum(r['tank_rank'] for r in results) / total if total else 0.0,
        'failed_count': len(failed)
    }

    return {
        'summary': summary,
        'detailed_results': results,
        'failed': failed
    }

def compare_models_and_find_improvements(original_results: Dict, sim_results: Dict,
                                       dataset_name: str, imagenet_classes: List[str]) -> Dict:
    """Compare two model results and find examples that improved"""

    orig_details = original_results['detailed_results']
    sim_details = sim_results['detailed_results']

    print(f"\n{'='*80}")
    print(f"COMPARISON: {dataset_name.upper()} DATASET")
    print(f"{'='*80}")

    # Overall accuracy comparison
    orig_acc = original_results['summary']['tank_accuracy']
    sim_acc = sim_results['summary']['tank_accuracy']
    orig_top5 = original_results['summary']['tank_top5_accuracy']
    sim_top5 = sim_results['summary']['tank_top5_accuracy']

    print(f"Original Model:")
    print(f"  Tank Top-1 Accuracy: {orig_acc:.4f} ({orig_acc*100:.2f}%)")
    print(f"  Tank Top-5 Accuracy: {orig_top5:.4f} ({orig_top5*100:.2f}%)")
    print(f"  Avg Tank Confidence: {original_results['summary']['avg_tank_confidence']:.4f}")
    print(f"  Avg Tank Rank: {original_results['summary']['avg_tank_rank']:.1f}")

    print(f"\nSim-Trained Model:")
    print(f"  Tank Top-1 Accuracy: {sim_acc:.4f} ({sim_acc*100:.2f}%)")
    print(f"  Tank Top-5 Accuracy: {sim_top5:.4f} ({sim_top5*100:.2f}%)")
    print(f"  Avg Tank Confidence: {sim_results['summary']['avg_tank_confidence']:.4f}")
    print(f"  Avg Tank Rank: {sim_results['summary']['avg_tank_rank']:.1f}")

    improvement = sim_acc - orig_acc
    print(f"\nImprovement: {improvement:+.4f} ({improvement*100:+.2f} percentage points)")

    # Find specific examples that improved
    improvements = []
    deteriorations = []

    for i, (orig, sim) in enumerate(zip(orig_details, sim_details)):
        assert orig['image_path'] == sim['image_path'], "Image order mismatch!"

        orig_correct = orig['is_tank_correct']
        sim_correct = sim['is_tank_correct']

        if not orig_correct and sim_correct:
            # Improved: was wrong, now correct
            improvements.append({
                'image_path': orig['image_path'],
                'orig_pred': orig['prediction'],
                'orig_conf': orig['confidence'],
                'orig_tank_conf': orig['tank_confidence'],
                'orig_tank_rank': orig['tank_rank'],
                'sim_pred': sim['prediction'],
                'sim_conf': sim['confidence'],
                'sim_tank_conf': sim['tank_confidence'],
                'sim_tank_rank': sim['tank_rank'],
            })
        elif orig_correct and not sim_correct:
            # Deteriorated: was correct, now wrong
            deteriorations.append({
                'image_path': orig['image_path'],
                'orig_pred': orig['prediction'],
                'orig_conf': orig['confidence'],
                'sim_pred': sim['prediction'],
                'sim_conf': sim['confidence'],
            })

    print(f"\nIMPROVED EXAMPLES (was wrong → now correct): {len(improvements)}")
    if improvements:
        print("\nTop improved examples:")
        # Sort by improvement in tank confidence
        improvements.sort(key=lambda x: x['sim_tank_conf'] - x['orig_tank_conf'], reverse=True)

        for i, example in enumerate(improvements[:5]):  # Show top 5
            orig_pred_name = imagenet_classes[example['orig_pred']]
            print(f"\n{i+1}. {os.path.basename(example['image_path'])}")
            print(f"   Original: {orig_pred_name[:40]}... (conf: {example['orig_conf']:.3f})")
            print(f"   Original tank conf: {example['orig_tank_conf']:.3f} (rank #{example['orig_tank_rank']})")
            print(f"   Sim-trained: TANK (conf: {example['sim_conf']:.3f})")
            print(f"   Tank confidence gain: {example['sim_tank_conf'] - example['orig_tank_conf']:+.3f}")

    print(f"\nDETERIORATED EXAMPLES (was correct → now wrong): {len(deteriorations)}")
    if deteriorations:
        print("\nTop deteriorated examples:")
        for i, example in enumerate(deteriorations[:3]):  # Show top 3
            sim_pred_name = imagenet_classes[example['sim_pred']]
            print(f"\n{i+1}. {os.path.basename(example['image_path'])}")
            print(f"   Original: TANK (conf: {example['orig_conf']:.3f})")
            print(f"   Sim-trained: {sim_pred_name[:40]}... (conf: {example['sim_conf']:.3f})")

    # Return comparison results for image saving
    return {
        'improvements': improvements,
        'deteriorations': deteriorations,
        'summary': {
            'orig_acc': orig_acc,
            'sim_acc': sim_acc,
            'improvement': improvement
        }
    }

def create_mock_config(quick_run=False, sim_data_folder="sim_tank", sim_eval_subfolder="sim_black"):
    """Create a mock config object without using the conflicting parser"""
    class MockConfig:
        def __init__(self):
            DATA_ROOT = "/mnt/xfs/home/hdriss/sim2real/data"
            self.DATA_ROOT = DATA_ROOT
            self.IMAGENET_SAVE_DIR = f"{DATA_ROOT}/imagenet_samples"
            self.SEED = 42
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            if quick_run:
                self.mode = "quick"
                self.IMAGES_PER_FOLDER = 50
                self.TOTAL_SIM_IMAGES = 200
            else:
                self.mode = "full"
                self.IMAGES_PER_FOLDER = 400
                self.TOTAL_SIM_IMAGES = 1600

            # Required for collect_eval_paths
            self.sim_data_folder = sim_data_folder
            self.sim_eval_subfolder = sim_eval_subfolder
            self.sim_train_subfolders = ["sim_autumn", "sim_farm", "sim_interior", "sim_resting"]

    return MockConfig()

def main():
    parser = argparse.ArgumentParser(description="Compare original vs sim-trained model inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to sim-trained model weights")
    parser.add_argument("--quick_run", action="store_true",
                       help="Quick mode (fewer samples) for testing")
    parser.add_argument("--real_only", action="store_true",
                       help="Only evaluate on real data")
    parser.add_argument("--sim_only", action="store_true",
                       help="Only evaluate on simulated data")
    parser.add_argument("--save_images", action="store_true",
                       help="Save images with prediction results")
    parser.add_argument("--output_dir", type=str, default="./prediction_results",
                       help="Directory to save images with predictions")
    parser.add_argument("--max_images", type=int, default=10,
                       help="Maximum number of images to save per category")

    args = parser.parse_args()

    # Setup without conflicting with your config.py
    cfg = create_mock_config(quick_run=args.quick_run)
    set_seed(cfg.SEED)
    device = cfg.DEVICE

    # Load processor and class names
    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
    imagenet_classes = load_imagenet_class_names_fallback()

    # Collect evaluation data - CORRECTED LINE
    sim_eval_paths, real_image_paths = collect_eval_paths(cfg.DATA_ROOT, cfg.sim_data_folder, cfg.sim_eval_subfolder)

    if cfg.mode == "quick":
        real_image_paths = real_image_paths[:20]  # Limit for quick testing
        sim_eval_paths = sim_eval_paths[:20]

    print(f"Real images: {len(real_image_paths)}")
    print(f"Sim images: {len(sim_eval_paths)}")

    # Load original model
    print("Loading original pretrained model...")
    original_model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME)

    # Load sim-trained model
    print(f"Loading sim-trained model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return

    sim_model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME)
    sim_model.load_state_dict(torch.load(args.model_path, map_location='cpu'))

    # Run comparisons
    datasets_to_evaluate = []
    if not args.sim_only:
        datasets_to_evaluate.append(("real", real_image_paths))
    if not args.real_only:
        datasets_to_evaluate.append(("simulated", sim_eval_paths))

    for dataset_name, image_paths in datasets_to_evaluate:
        print(f"\n{'='*60}")
        print(f"EVALUATING {dataset_name.upper()} DATA")
        print(f"{'='*60}")

        # Evaluate original model
        move_to_device(original_model, device)
        original_results = evaluate_single_model_detailed(
            original_model, image_paths, f"{dataset_name}_original",
            processor, device, TANK_CLASS_ID
        )
        move_to_cpu(original_model)

        # Evaluate sim-trained model
        move_to_device(sim_model, device)
        sim_results = evaluate_single_model_detailed(
            sim_model, image_paths, f"{dataset_name}_sim",
            processor, device, TANK_CLASS_ID
        )
        move_to_cpu(sim_model)

        # Compare and analyze
        comparison_results = compare_models_and_find_improvements(
            original_results, sim_results, dataset_name, imagenet_classes
        )

        # Save images with predictions if requested
        if args.save_images:
            save_images_with_predictions(
                comparison_results, dataset_name, imagenet_classes,
                args.output_dir, args.max_images
            )

    print(f"\n{'='*80}")
    print("INFERENCE COMPARISON COMPLETED")
    if args.save_images:
        print(f"Images with predictions saved to: {args.output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()