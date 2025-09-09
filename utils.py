"""
Utility functions for the Sim2Real Tank Classification Experiment.

This module provides common utility functions for logging, reproducibility,
file operations, and model management.
"""

import json
import os
import random
import logging
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
import torch
import requests

from constants import IMAGENET_CLASSES_URL


def setup_logging(level: int = logging.INFO, log_file: str = "experiment.log") -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (default: experiment.log)
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(log_file).parent
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w')  # Overwrite previous log
        ]
    )

    logging.info("Logging initialized")


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Additional settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logging.info(f"Random seed set to {seed}")


def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create
    """
    try:
        os.makedirs(path, exist_ok=True)
        logging.debug(f"Directory ensured: {path}")
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        raise


def save_json(path: str, obj: Any, indent: int = 2) -> None:
    """
    Save object as JSON file with error handling.

    Args:
        path: File path to save to
        obj: Object to serialize
        indent: JSON indentation (default: 2)
    """
    try:
        # Ensure parent directory exists
        parent_dir = os.path.dirname(path)
        if parent_dir:
            ensure_dir(parent_dir)

        with open(path, "w", encoding='utf-8') as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)

        logging.info(f"JSON saved successfully: {path}")

    except Exception as e:
        logging.error(f"Failed to save JSON to {path}: {e}")
        raise


def load_json(path: str) -> Any:
    """
    Load JSON file with error handling.

    Args:
        path: File path to load from

    Returns:
        Loaded JSON object
    """
    try:
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)

        logging.info(f"JSON loaded successfully: {path}")
        return data

    except FileNotFoundError:
        logging.error(f"JSON file not found: {path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in file {path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to load JSON from {path}: {e}")
        raise


def load_imagenet_class_names_fallback() -> Optional[List[str]]:
    """
    Try to download the standard ImageNet class list with robust error handling.

    Returns:
        List of class names or None on failure
    """
    try:
        logging.info(f"Attempting to download ImageNet class names from {IMAGENET_CLASSES_URL}")

        response = requests.get(IMAGENET_CLASSES_URL, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes

        class_names = response.text.strip().split("\n")

        # Validate the response
        if len(class_names) != 1000:
            logging.warning(f"Expected 1000 classes, got {len(class_names)}")

        logging.info("Successfully downloaded ImageNet class names")
        return class_names

    except requests.exceptions.Timeout:
        logging.warning("Timeout while downloading ImageNet class names")
    except requests.exceptions.ConnectionError:
        logging.warning("Connection error while downloading ImageNet class names")
    except requests.exceptions.HTTPError as e:
        logging.warning(f"HTTP error while downloading ImageNet class names: {e}")
    except Exception as e:
        logging.warning(f"Unexpected error downloading ImageNet class names: {e}")

    return None


def move_to_device(model: torch.nn.Module, device: str) -> None:
    """
    Move model to specified device with error handling.

    Args:
        model: PyTorch model to move
        device: Target device ('cuda' or 'cpu')
    """
    if model is not None:
        try:
            model.to(device)
            logging.debug(f"Model moved to {device}")
        except Exception as e:
            logging.error(f"Failed to move model to {device}: {e}")
            raise


def move_to_cpu(model: torch.nn.Module) -> None:
    """
    Move model to CPU and clear GPU cache.

    Args:
        model: PyTorch model to move
    """
    if model is not None:
        try:
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.debug("Model moved to CPU and GPU cache cleared")
        except Exception as e:
            logging.error(f"Failed to move model to CPU: {e}")
            raise


def validate_paths(*paths: str) -> None:
    """
    Validate that all provided paths exist.

    Args:
        *paths: Variable number of paths to validate

    Raises:
        FileNotFoundError: If any path doesn't exist
    """
    for path in paths:
        if not os.path.exists(path):
            error_msg = f"Path does not exist: {path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory information if CUDA is available.

    Returns:
        Dictionary with memory information or empty dict if no CUDA
    """
    if not torch.cuda.is_available():
        return {}

    try:
        device_count = torch.cuda.device_count()
        info = {}

        for i in range(device_count):
            total = torch.cuda.get_device_properties(i).total_memory
            allocated = torch.cuda.memory_allocated(i)
            cached = torch.cuda.memory_reserved(i)

            info[f"gpu_{i}"] = {
                "total_memory_gb": total / (1024**3),
                "allocated_memory_gb": allocated / (1024**3),
                "cached_memory_gb": cached / (1024**3),
                "free_memory_gb": (total - allocated) / (1024**3)
            }

        return info

    except Exception as e:
        logging.warning(f"Failed to get GPU memory info: {e}")
        return {}