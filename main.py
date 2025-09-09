"""
Sim2Real Tank Classification Experiment.

This module implements a comprehensive evaluation of sim-to-real transfer learning
for tank classification using Vision Transformers.
"""

import os
import logging
import traceback
from collections import Counter
from typing import Dict, List, Tuple

import torch
from transformers import ViTForImageClassification, ViTImageProcessor

from analysis import summarize_core_models, key_insights
from config import get_config
from constants import TANK_CLASS_ID, VIT_MODEL_NAME
from data import (
    collect_sim_images,
    collect_eval_paths,
    load_imagenet_datasets,
    collect_imagenet_samples_for_classes,
    check_imagenet_availability_for_classes,
)
from datasets_local import FinetuneDataset
from evals import evaluate_model_comprehensive
from models import create_model, train_model
from utils import (
    setup_logging,
    set_seed,
    ensure_dir,
    save_json,
    load_imagenet_class_names_fallback,
    move_to_device,
    move_to_cpu,
)


def analyze_sim_predictions(
    predictions: List[int],
    imagenet_classes: List[str],
    tank_class_id: int,
    k: int,
    min_pct: float
) -> List[Tuple[int, str, float]]:
    """
    Analyze predictions on simulated data to identify frequent classes.

    Args:
        predictions: List of predicted class IDs
        imagenet_classes: List of ImageNet class names
        tank_class_id: Target tank class ID
        k: Number of top classes to consider
        min_pct: Minimum percentage threshold for including classes

    Returns:
        List of tuples (class_id, class_name, percentage) for frequent classes
    """
    try:
        counts = Counter(predictions)
        total = sum(counts.values()) if counts else 1
        top_k = counts.most_common(k)
        frequent = [(tank_class_id, imagenet_classes[tank_class_id], 0.0)]  # always include tank

        logging.info(f"Analyzing top {k} most frequent predicted classes on simulated data")
        print(f"\nTop {k} most frequent predicted classes on simulated data:")

        for rank, (cid, count) in enumerate(top_k, 1):
            cname = imagenet_classes[cid] if cid < len(imagenet_classes) else f"class_{cid}"
            pct = (count / total) * 100
            print(f"{rank:2d}. Class {cid:3d}: {cname:40s} - {count:4d} times ({pct:5.1f}%)")

            if cid != tank_class_id and pct > min_pct:
                frequent.append((cid, cname, pct))

        logging.info(f"Training will focus on sim-to-real transfer for tank class with negative examples from {len(frequent)-1} other classes")
        print(f"\nTraining will focus on sim-to-real transfer for tank class with negative examples from {len(frequent)-1} other classes")

        return frequent

    except Exception as e:
        logging.error(f"Error analyzing sim predictions: {e}")
        raise


def build_training_configs(
    imagenet_tank_paths: List[str],
    sim_train_paths: List[str],
    negative_paths_by_class: Dict[int, List[str]],
    tank_class_id: int,
) -> Dict[str, Dict]:
    """
    Build balanced training configurations for the core models.

    Args:
        imagenet_tank_paths: Paths to ImageNet tank images
        sim_train_paths: Paths to simulated training images
        negative_paths_by_class: Dictionary mapping class IDs to image paths
        tank_class_id: Target tank class ID

    Returns:
        Dictionary containing training configurations for each model
    """
    try:
        configs: Dict[str, Dict] = {}
        negative_class_ids = [cid for cid in negative_paths_by_class.keys() if cid != tank_class_id]

        def balanced_negatives(n_tanks: int) -> List[Tuple[str, int]]:
            """Return negatives balanced across all negative classes."""
            per_class = n_tanks
            negs: List[Tuple[str, int]] = []
            for cid in negative_class_ids:
                samples = negative_paths_by_class.get(cid, [])[:per_class]
                negs.extend([(p, cid) for p in samples])
            return negs

        # ImageNet only configuration
        n_tanks = len(imagenet_tank_paths)
        negs = balanced_negatives(n_tanks)
        configs["imagenet_only"] = {
            "data": [(p, tank_class_id) for p in imagenet_tank_paths] + negs,
            "description": f"{n_tanks} ImageNet tanks + {len(negs)} negatives ({n_tanks} per class × {len(negative_class_ids)})"
        }

        # Simulation only configuration
        n_tanks = len(sim_train_paths)
        negs = balanced_negatives(n_tanks)
        configs["sim_only"] = {
            "data": [(p, tank_class_id) for p in sim_train_paths] + negs,
            "description": f"{n_tanks} sim tanks + {len(negs)} negatives ({n_tanks} per class × {len(negative_class_ids)})"
        }

        # Combined ImageNet + simulation configuration
        n_tanks = len(sim_train_paths) + len(imagenet_tank_paths)
        negs = balanced_negatives(n_tanks)
        configs["imagenet_sim"] = {
            "data": ([(p, tank_class_id) for p in sim_train_paths] +
                     [(p, tank_class_id) for p in imagenet_tank_paths] +
                     negs),
            "description": f"{len(sim_train_paths)} sim + {len(imagenet_tank_paths)} ImageNet tanks + {len(negs)} negatives ({n_tanks} per class × {len(negative_class_ids)})"
        }

        logging.info(f"Built {len(configs)} training configurations")
        return configs

    except Exception as e:
        logging.error(f"Error building training configurations: {e}")
        raise


def save_model(model: torch.nn.Module, model_name: str, save_dir: str = "saved_models") -> str:
    """
    Save model weights to disk.

    Args:
        model: PyTorch model to save
        model_name: Name for the saved model
        save_dir: Directory to save the model

    Returns:
        Path to the saved model file
    """
    try:
        ensure_dir(save_dir)
        model_path = os.path.join(save_dir, f"{model_name}_weights.pth")
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model weights saved: {model_path}")
        print(f"💾 Model weights saved: {model_path}")
        return model_path
    except Exception as e:
        logging.error(f"Error saving model {model_name}: {e}")
        raise


def main():
    """Main experiment execution function."""
    try:
        # Setup logging and configuration
        setup_logging()
        logging.info("Starting Sim2Real Tank Classification Experiment")

        cfg = get_config()
        set_seed(cfg.SEED)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        # Initialize model processor
        try:
            processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
            logging.info(f"Loaded image processor for {VIT_MODEL_NAME}")
        except Exception as e:
            logging.error(f"Failed to load image processor: {e}")
            raise

        # Create output directories
        sim_specific_save_dir = f"saved_models_{cfg.sim_data_folder}"
        ensure_dir(sim_specific_save_dir)

        # Load ImageNet class names
        logging.info("Loading ImageNet class names")
        imagenet_classes = load_imagenet_class_names_fallback()
        if imagenet_classes is None:
            logging.warning("Failed to load ImageNet class names from URL, using dataset fallback")
            try:
                _, valds = load_imagenet_datasets()
                imagenet_classes = valds.features["label"].names
                logging.info("Successfully loaded class names from dataset")
            except Exception as e:
                logging.error(f"Failed to load class names from dataset: {e}")
                imagenet_classes = [f"class_{i}" for i in range(1000)]
                logging.warning("Using dummy class names")

        # Load baseline model
        logging.info(f"Loading baseline ViT model: {VIT_MODEL_NAME}")
        try:
            baseline_model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME).to(device).eval()
            logging.info("Successfully loaded baseline model")
        except Exception as e:
            logging.error(f"Failed to load baseline model: {e}")
            raise

        # Gather data paths
        logging.info("Collecting dataset paths")
        try:
            sim_train_paths = collect_sim_images(
                cfg.DATA_ROOT, cfg.sim_data_folder, cfg.sim_train_subfolders, cfg.TOTAL_SIM_IMAGES
            )
            sim_eval_paths, real_image_paths = collect_eval_paths(
                cfg.DATA_ROOT, cfg.sim_data_folder, cfg.sim_eval_subfolder
            )

            logging.info(f"Collected {len(sim_train_paths)} sim training images")
            logging.info(f"Collected {len(sim_eval_paths)} sim evaluation images")
            logging.info(f"Collected {len(real_image_paths)} real test images")

        except Exception as e:
            logging.error(f"Failed to collect dataset paths: {e}")
            raise

        print("\nDataset composition:")
        print(f"  Sim data folder: {cfg.sim_data_folder}")
        print(f"  Training:  {len(sim_train_paths)} images from {len(cfg.sim_train_subfolders)} sim folders: {cfg.sim_train_subfolders}")
        print(f"  Evaluation: {len(sim_eval_paths)} {cfg.sim_eval_subfolder} images")
        print(f"  Real test:  {len(real_image_paths)} real tank images")

        # Baseline evaluations
        print("\n" + "="*80)
        print("BASELINE EVALUATION ON SIMULATED/REAL")
        print("="*80)

        logging.info("Running baseline evaluations")
        try:
            sim_baseline = evaluate_model_comprehensive(
                baseline_model, sim_train_paths, "simulated_baseline",
                processor, device, TANK_CLASS_ID
            )
            real_baseline = evaluate_model_comprehensive(
                baseline_model, real_image_paths, "real_baseline",
                processor, device, TANK_CLASS_ID
            )
            logging.info("Completed baseline evaluations")
        except Exception as e:
            logging.error(f"Failed during baseline evaluation: {e}")
            raise

        # Analyze predictions for negative class selection
        frequent_classes = analyze_sim_predictions(
            predictions=sim_baseline["predictions"],
            imagenet_classes=imagenet_classes,
            tank_class_id=TANK_CLASS_ID,
            k=cfg.TOPK_PRED_CLASSES,
            min_pct=cfg.FILTER_MIN_PCT
        )
        frequent_class_ids = [cid for cid, _, _ in frequent_classes]

        # ImageNet data mining
        print("\nChecking ImageNet availability and creating balanced dataset...")
        logging.info("Loading ImageNet datasets for mining")

        try:
            train_ds, val_ds = load_imagenet_datasets()
            availability = check_imagenet_availability_for_classes(train_ds, val_ds, frequent_class_ids)
            logging.info("Completed ImageNet availability check")
        except Exception as e:
            logging.error(f"Failed to load ImageNet datasets: {e}")
            raise

        # Calculate sample targets
        num_environments = len(cfg.sim_train_subfolders)
        sim_per_environment = cfg.TOTAL_SIM_IMAGES // num_environments

        TARGET_IMAGENET_TANK = cfg.TANK_IMAGENET_SAMPLES
        TARGET_NEGATIVE_PER_CLASS = cfg.NEGATIVE_SAMPLES_PER_CLASS

        num_sim_images = len(sim_train_paths)
        negative_class_ids = [cid for cid in frequent_class_ids if cid != TANK_CLASS_ID]

        # Check availability and adjust targets
        tank_imagenet_available = availability[TANK_CLASS_ID]["train"]
        min_negative_available = min(availability[cid]["train"] for cid in negative_class_ids)

        actual_tank_imagenet = min(TARGET_IMAGENET_TANK, tank_imagenet_available)
        actual_negative_per_class = min(TARGET_NEGATIVE_PER_CLASS, min_negative_available)

        total_tank_samples = num_sim_images + actual_tank_imagenet

        print(f"\nBalanced training plan:")
        print(f"  Sim environments: {num_environments} × {sim_per_environment} = {num_sim_images} sim images")
        print(f"  Tank class ({TANK_CLASS_ID}): {num_sim_images} sim + {actual_tank_imagenet} ImageNet = {total_tank_samples} total")

        for cid in negative_class_ids:
            cname = imagenet_classes[cid] if cid < len(imagenet_classes) else f"class_{cid}"
            print(f"  Class {cid} ({cname}): {actual_negative_per_class} ImageNet samples")

        if total_tank_samples > 0 and actual_negative_per_class > 0:
            print(f"  Ratio: Tank class has {total_tank_samples/actual_negative_per_class:.1f}x samples vs negative classes")

        if actual_tank_imagenet < TARGET_IMAGENET_TANK:
            logging.warning(f"Tank ImageNet reduced to {actual_tank_imagenet} (only {tank_imagenet_available} available)")
            print(f"⚠️  Tank ImageNet reduced to {actual_tank_imagenet} (only {tank_imagenet_available} available)")

        if actual_negative_per_class < TARGET_NEGATIVE_PER_CLASS:
            logging.warning(f"Negative classes reduced to {actual_negative_per_class} (min available: {min_negative_available})")
            print(f"⚠️  Negative classes reduced to {actual_negative_per_class} (min available: {min_negative_available})")

        # Collect ImageNet samples
        targets_per_class = {TANK_CLASS_ID: actual_tank_imagenet}
        for cid in negative_class_ids:
            targets_per_class[cid] = actual_negative_per_class

        ensure_dir(cfg.IMAGENET_SAVE_DIR)

        try:
            class_train_paths, class_val_paths = collect_imagenet_samples_for_classes(
                imagenet_save_dir=cfg.IMAGENET_SAVE_DIR,
                train_dataset=train_ds,
                val_dataset=val_ds,
                class_ids=frequent_class_ids,
                targets_per_class=targets_per_class,
                max_val_per_class=cfg.MAX_VAL_SAMPLES_PER_CLASS,
            )
            logging.info("Successfully collected ImageNet samples")
        except Exception as e:
            logging.error(f"Failed to collect ImageNet samples: {e}")
            raise

        # Build negative class paths
        negative_paths_by_class: Dict[int, List[str]] = {}
        for cid in frequent_class_ids:
            if cid != TANK_CLASS_ID:
                negative_paths_by_class[cid] = class_train_paths.get(cid, [])

        total_negatives = sum(len(v) for v in negative_paths_by_class.values())
        logging.info(f"Collected {total_negatives} negative samples from ImageNet")
        print(f"Collected negative samples from ImageNet: {total_negatives} total")

        imagenet_tank_paths = class_train_paths.get(TANK_CLASS_ID, [])

        # Build training configurations
        training_configs = build_training_configs(
            imagenet_tank_paths=imagenet_tank_paths,
            sim_train_paths=sim_train_paths,
            negative_paths_by_class=negative_paths_by_class,
            tank_class_id=TANK_CLASS_ID,
        )

        print(f"\nGenerated {len(training_configs)} training configurations:")
        for k, v in training_configs.items():
            print(f"  {k}: {v['description']}")

        # Train models
        trained_models: Dict[str, torch.nn.Module] = {}
        core_models = ["imagenet_only", "sim_only", "imagenet_sim"]
        eval_sample_size = 50 if cfg.mode == "quick" else 200

        def train_one(name: str, data: List[Tuple[str, int]], enable_eval: bool):
            """Train a single model configuration."""
            logging.info(f"Starting training for {name}")
            print(f"\n[TRAIN] {name}: {training_configs[name]['description']}")

            try:
                model = create_model(VIT_MODEL_NAME, device)

                # Create DataLoader
                from torch.utils.data import DataLoader
                ds = FinetuneDataset(data, processor)
                dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)

                model = train_model(
                    model=model,
                    dataloader=dl,
                    device=device,
                    epochs=cfg.NUM_EPOCHS,
                    processor=processor,
                    tank_class_id=TANK_CLASS_ID,
                    real_image_paths=real_image_paths,
                    sim_eval_paths=sim_eval_paths,
                    eval_sample_size=eval_sample_size,
                    enable_eval=enable_eval,
                )

                # Save model weights before moving to CPU
                save_model(model, name, sim_specific_save_dir)

                # Move off GPU to free memory
                move_to_cpu(model)
                trained_models[name] = model

                logging.info(f"Successfully completed training for {name}")

            except Exception as e:
                logging.error(f"Training failed for {name}: {e}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                trained_models[name] = None
                raise

        # Train all core models
        for name in core_models:
            try:
                train_one(name, training_configs[name]["data"], enable_eval=True)
                print(f"✅ {name} training successful")
            except Exception as e:
                print(f"❌ {name} training failed: {e}")
                trained_models[name] = None

        trained_ok = len([m for m in trained_models.values() if m is not None])
        logging.info(f"Completed training {trained_ok}/{len(trained_models)} models")
        print(f"\n✅ Completed training {trained_ok}/{len(trained_models)} models")

        # Comprehensive evaluation
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)

        logging.info("Starting comprehensive model evaluation")

        # Evaluate baseline
        print("Evaluating baseline (original pretrained) model...")
        move_to_device(baseline_model, device)

        try:
            original_real = evaluate_model_comprehensive(
                baseline_model, real_image_paths, "original", processor, device, TANK_CLASS_ID
            )
            original_sim = evaluate_model_comprehensive(
                baseline_model, sim_eval_paths, "original", processor, device, TANK_CLASS_ID
            )
        except Exception as e:
            logging.error(f"Failed during baseline evaluation: {e}")
            raise
        finally:
            move_to_cpu(baseline_model)

        results: Dict[str, Dict] = {
            "original": {"real": original_real, "sim": original_sim, "model": None}
        }

        # Evaluate all trained models
        for name, model in trained_models.items():
            if model is None:
                continue

            print(f"Evaluating {name}...")
            logging.info(f"Evaluating {name}")

            try:
                move_to_device(model, device)
                r_real = evaluate_model_comprehensive(
                    model, real_image_paths, name, processor, device, TANK_CLASS_ID
                )
                r_sim = evaluate_model_comprehensive(
                    model, sim_eval_paths, name, processor, device, TANK_CLASS_ID
                )
                results[name] = {"real": r_real, "sim": r_sim, "model": None}
                logging.info(f"Successfully evaluated {name}")
            except Exception as e:
                logging.error(f"Failed to evaluate {name}: {e}")
                results[name] = {"real": None, "sim": None, "model": None}
            finally:
                move_to_cpu(model)

        # Results analysis
        print("\n" + "="*80)
        print("RESULTS ANALYSIS")
        print("="*80)

        try:
            summarize_core_models(results)
            key_insights(results)
            logging.info("Completed results analysis")
        except Exception as e:
            logging.error(f"Error during results analysis: {e}")

        # Save final results
        try:
            final_obj = {
                "experiment_type": "sim2real_transfer_clean",
                "training_configurations": {k: v["description"] for k, v in training_configs.items()},
                "model_results": {
                    name: {
                        "real_accuracy": res["real"]["tank_accuracy"] if res["real"] else 0.0,
                        "sim_accuracy": res["sim"]["tank_accuracy"] if res["sim"] else 0.0
                    }
                    for name, res in results.items()
                },
                "baseline_accuracy": results["original"]["real"]["tank_accuracy"] if results["original"]["real"] else 0.0,
            }

            results_file = "sim2real_clean_results.json"
            save_json(results_file, final_obj)
            logging.info(f"Results saved to {results_file}")
            print(f"📊 Results saved to '{results_file}'")

        except Exception as e:
            logging.error(f"Failed to save results: {e}")

        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED")
        print("="*80)
        logging.info("Experiment completed successfully")

    except Exception as e:
        logging.error(f"Experiment failed: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        print(f"❌ Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
