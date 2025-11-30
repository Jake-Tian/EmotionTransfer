# EmotionTransfer

Fine-tuning Qwen2.5-0.5B-Instruct for emotion transfer tasks using supervised fine-tuning (SFT), direct preference optimization (DPO), Kahneman-Tversky optimization (KTO), and odds ratio preference optimization (ORPO).

## Overview

This project fine-tunes language models to transfer emotions in text while preserving semantic content. The pipeline includes dataset preparation, multi-stage training (SFT → DPO/KTO/ORPO), inference, and evaluation.

## Training Pipeline

1. **SFT (Supervised Fine-Tuning)**: Initial fine-tuning on emotion transfer pairs
   - Script: `qwen_train/sft/sft_qwen2.5.sh`
   - Output: `models/Qwen2.5-0.5B-Instruct-sft`

2. **Preference Optimization**: Fine-tuning using preference pairs
   - DPO Script: `qwen_train/dpo/dpo_qwen2.5.sh`
   - KTO Script: `qwen_train/kto/kto_qwen2.5.sh`
   - ORPO Script: `qwen_train/orpo/orpo_qwen2.5.sh`
   - Output: `models/Qwen2.5-0.5B-Instruct-dpo` / `models/Qwen2.5-0.5B-Instruct-kto` / `models/Qwen2.5-0.5B-Instruct-orpo`

## Project Structure

```
EmotionTransfer/
├── dataset/              # Dataset processing and cleaning scripts
├── data_processing/      # DPO data filtering and conversion
├── qwen_train/
│   ├── sft/             # SFT training scripts
│   ├── dpo/             # DPO training scripts
│   ├── kto/             # KTO training scripts
│   ├── orpo/            # ORPO training scripts
│   ├── inference/       # Inference scripts
│   └── evaluate/        # Evaluation scripts
├── LLaMA-Factory/       # Training framework (gitignored)
└── models/              # Model checkpoints (gitignored)
```

## Key Features

- **Multi-stage Training**: SFT followed by preference-based optimization (DPO/KTO/ORPO)
- **Evaluation Metrics**: BERTScore for content preservation, emotion classification for style transfer
- **Parallel Inference**: Multi-sample generation with parallel processing
- **Data Quality Control**: Automated dataset cleaning and validation

## Usage

### Training

```bash
# SFT
bash qwen_train/sft/sft_qwen2.5.sh

# Preference Optimization (DPO/KTO/ORPO)
bash qwen_train/dpo/dpo_qwen2.5.sh
bash qwen_train/kto/kto_qwen2.5.sh
bash qwen_train/orpo/orpo_qwen2.5.sh
```

### Inference

```bash
bash qwen_train/inference/scripts/run_inference_multiple_samples.sh
```

### Evaluation

```bash
bash qwen_train/evaluate/run_evaluate_samples.sh
```

## Requirements

- PyTorch
- LLaMA-Factory
- Transformers
- BERTScore (for evaluation)

## Evaluation Results
| Method | Content Score | Emotion Score | Joint Score |
|--------|---------------|---------------|-------------|
| SFT    | 71.8          | 97.0          | 69.6        |
| DPO    | 74.3          | 98.0          | 72.7        |
| KTO    | **74.8**          | **98.8**          | **73.8**        |
| ORPO   | 72.8          | 97.2          | 70.8        |

## Note

This project uses LLaMA-Factory for training. Ensure you have the framework set up and configured before running training scripts.

