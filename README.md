# Sim2Real Tank Classification

A comprehensive evaluation framework for sim-to-real transfer learning in tank classification using Vision Transformers.

## Overview

This project implements a systematic study of sim-to-real domain transfer for military vehicle classification. It compares three training strategies:
- **ImageNet-only**: Fine-tuning on ImageNet tank images
- **Simulation-only**: Training exclusively on simulated tank data
- **Combined**: Leveraging both ImageNet and simulated data

The framework uses Vision Transformers (ViT) and evaluates performance on both simulated and real tank images.



## Setup

1. Clone the repository:
```bash
git clone https://github.com/hediD/sim_robustifier.git
cd sim_robustifier 
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your data directory structure:
The repository expects the following structure for simulated and real tank images used for training and evaluation:
```
data/
├── sim_tank/                    # Simulated tank data
│   ├── sim_autumn/              # Training environment 1 (see config.py: --sim_train_subfolders)
│   ├── sim_farm/                # Training environment 2
│   ├── sim_interior/            # Training environment 3
│   ├── sim_resting/             # Training environment 4
│   └── sim_black/               # Evaluation environment (see config.py: --sim_eval_subfolder)
└── real/                        # Real tank images for testing (see constants.py)
```

## Usage

```bash
python main.py --quick-run --data-root ./data --sim-data-folder sim_tank
```

### Full Experiment

Run the complete experiment:

```bash
python main.py --data-root ./data
```

### Configuration Options

```bash
python main.py \
    --data-root ./data \
    --output-dir ./outputs \
    --sim-data-folder sim_tank \
    --sim-train-subfolders sim_autumn sim_farm sim_interior sim_resting \
    --sim-eval-subfolder sim_black \
    --topk-negative-classes 10 \
    --min-freq-pct 2.0 \
    --seed 42
```

### Environment Variables

You can also configure paths using environment variables:

```bash
export SIM2REAL_DATA_ROOT="./data"
export SIM2REAL_OUTPUT_DIR="./outputs"
python main.py
```

## Configuration

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--quick-run` | Run quick experiment with reduced data | False |
| `--data-root` | Root directory containing datasets | `./data` |
| `--output-dir` | Directory for saving outputs | `./outputs` |
| `--sim-data-folder` | Simulated data folder name | `sim_tank` |
| `--sim-train-subfolders` | Training environment folders | `[sim_autumn, sim_farm, sim_interior, sim_resting]` |
| `--sim-eval-subfolder` | Evaluation environment folder | `sim_black` |
| `--topk-negative-classes` | Max number of negative classes | 10 |
| `--min-freq-pct` | Min frequency threshold for negative classes | 5.0 |
| `--seed` | Random seed for reproducibility | 42 |

### Experiment Modes

#### Quick Mode (`--quick-run`)
- 200 total sim images
- 2 training epochs
- 8 batch size
- Suitable for testing and development

#### Full Mode (default)
- 1600 total sim images
- 8 training epochs
- 16 batch size
- Complete experimental evaluation

## Output Structure

The experiment generates several outputs:

```
outputs/
├── imagenet_samples/           # Downloaded ImageNet samples
├── saved_models_sim_tank/      # Trained model weights
├── sim2real_clean_results.json # Final results summary
└── experiment.log             # Detailed execution log
```

## Results Analysis

The framework provides comprehensive analysis including:

- **Tank Classification Accuracy**: Performance on real tank images
- **Transfer Learning Effectiveness**: Comparison across training strategies
- **Simulation Quality**: How well sim data transfers to real images
- **Model Insights**: Detailed breakdowns and recommendations

Example output:
```
CORE MODEL COMPARISON:
Model                Real Tank Acc   Sim Tank Acc    Real Improvement
----------------------------------------------------------------------
original             0.2150          0.1250          +0.0000
imagenet_only        0.8420          0.7650          +0.6270
sim_only             0.7890          0.9230          +0.5740
imagenet_sim         0.8650          0.8940          +0.6500

KEY INSIGHTS:
  • Sim-only vs ImageNet-only: -0.0530 (-5.30%)
    ➡️ Simulated and ImageNet data perform similarly
  • Adding ImageNet to sim data: +0.0760 (+7.60%)
    ✅ ImageNet + Sim combination is beneficial!
```

## Architecture


- `main.py`: Main experiment orchestration
- `config.py`: Configs/Arguments
- `data.py`: Dataset collection and preprocessing
- `models.py`: Model creation and training
- `evals.py`: Evaluation and metrics
- `analysis.py`: Results analysis and insights
- `utils.py`: Utility functions and helpers
- `constants.py`: Global constants
