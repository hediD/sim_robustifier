"""
Configuration management for Sim2Real Tank Classification Experiment.

This module handles command-line arguments and environment variable configuration
for the sim-to-real transfer learning experiment.
"""

import os
import argparse
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    """Configuration dataclass for the experiment."""
    mode: str
    IMAGES_PER_FOLDER: int
    TOTAL_SIM_IMAGES: int
    MAX_VAL_SAMPLES_PER_CLASS: int
    NUM_EPOCHS: int
    IMAGENET_EVAL_SAMPLES: int
    BATCH_SIZE: int
    DATA_ROOT: str
    IMAGENET_SAVE_DIR: str
    SEED: int = 42
    TOPK_PRED_CLASSES: int = 10
    FILTER_MIN_PCT: float = 2.0  # % threshold to include frequent negative classes
    sim_data_folder: str = "sim_tank"
    sim_train_subfolders: Optional[List[str]] = None
    sim_eval_subfolder: str = "sim_black"
    TANK_IMAGENET_SAMPLES: int = 1200
    NEGATIVE_SAMPLES_PER_CLASS: int = 1200


def get_config() -> Config:
    """
    Parse command line arguments and environment variables to create configuration.

    Environment Variables:
        SIM2REAL_DATA_ROOT: Root directory for data (default: ./data)
        SIM2REAL_OUTPUT_DIR: Directory for outputs (default: ./outputs)

    Returns:
        Config: Configuration object with all experiment parameters
    """
    parser = argparse.ArgumentParser(
        description="Sim2Real Tank Classification Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment mode
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help="Run a quick experiment with reduced data and epochs for testing"
    )

    # Data configuration
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.getenv("SIM2REAL_DATA_ROOT", "./data"),
        help="Root directory containing the dataset"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("SIM2REAL_OUTPUT_DIR", "./outputs"),
        help="Directory for saving outputs"
    )

    parser.add_argument(
        "--sim-data-folder",
        type=str,
        default="sim_tank",
        choices=["sim_tank", "sim_tank_leopard"],
        help="Which sim data folder to use"
    )

    parser.add_argument(
        "--sim-train-subfolders",
        type=str,
        nargs="+",
        default=["sim_autumn", "sim_farm", "sim_interior", "sim_resting"],
        help="Subfolders to use for training (default: all except sim_black)"
    )

    parser.add_argument(
        "--sim-eval-subfolder",
        type=str,
        default="sim_black",
        help="Subfolder to use for sim evaluation"
    )

    # Model configuration
    parser.add_argument(
        "--topk-negative-classes",
        type=int,
        default=10,
        help="Maximum number of top frequent negative classes to select"
    )

    parser.add_argument(
        "--min-freq-pct",
        type=float,
        default=5.0,
        help="Minimum frequency percentage threshold for negative class selection"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Validate paths
    data_root = os.path.abspath(args.data_root)
    if not os.path.exists(data_root):
        raise ValueError(f"Data root directory does not exist: {data_root}")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    imagenet_save_dir = os.path.join(output_dir, "imagenet_samples")

    if args.quick_run:
        print("🚀 QUICK RUN MODE ENABLED - Using reduced datasets and epochs")
        cfg = Config(
            mode="quick",
            IMAGES_PER_FOLDER=50,
            TOTAL_SIM_IMAGES=200,
            MAX_VAL_SAMPLES_PER_CLASS=25,
            NUM_EPOCHS=2,
            IMAGENET_EVAL_SAMPLES=1000,
            BATCH_SIZE=8,
            DATA_ROOT=data_root,
            IMAGENET_SAVE_DIR=imagenet_save_dir,
            sim_data_folder=args.sim_data_folder,
            sim_train_subfolders=args.sim_train_subfolders,
            sim_eval_subfolder=args.sim_eval_subfolder,
            TOPK_PRED_CLASSES=args.topk_negative_classes,
            FILTER_MIN_PCT=args.min_freq_pct,
            SEED=args.seed,
        )
    else:
        print("🔬 FULL EXPERIMENT MODE")
        cfg = Config(
            mode="full",
            IMAGES_PER_FOLDER=400,
            TOTAL_SIM_IMAGES=1600,     # across sim folders
            MAX_VAL_SAMPLES_PER_CLASS=-1,
            NUM_EPOCHS=8,
            IMAGENET_EVAL_SAMPLES=10000,
            BATCH_SIZE=16,
            DATA_ROOT=data_root,
            IMAGENET_SAVE_DIR=imagenet_save_dir,
            sim_data_folder=args.sim_data_folder,
            sim_train_subfolders=args.sim_train_subfolders,
            sim_eval_subfolder=args.sim_eval_subfolder,
            TOPK_PRED_CLASSES=args.topk_negative_classes,
            FILTER_MIN_PCT=args.min_freq_pct,
            SEED=args.seed,
        )

    print(f"📁 Data root: {cfg.DATA_ROOT}")
    print(f"📁 Using sim data folder: {cfg.sim_data_folder}")
    print(f"🏋️ Training subfolders: {cfg.sim_train_subfolders}")
    print(f"🧪 Evaluation subfolder: {cfg.sim_eval_subfolder}")
    print(f"🎯 Negative class selection: top-{cfg.TOPK_PRED_CLASSES} classes above {cfg.FILTER_MIN_PCT}% frequency")

    return cfg
