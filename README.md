# Sentiment Analysis of Customer Service Conversations

**Course:** DI 725 - Transformers and Attention-Based Deep Networks
**Student Number:** 2786028

## Task

3-class sentiment classification (positive, negative, neutral) on customer service conversations using a fine-tuned RoBERTa-base model.

## Approach

- **Model:** RoBERTa-base (125M parameters)
- **Metadata prepend:** Categorical features (issue area, complexity, product category, agent level) prepended as text
- **Head+Tail truncation:** First 128 + last 382 tokens to capture both problem and resolution
- **Class-weighted loss:** Handles severe class imbalance (positive ~1.7%)
- **Training:** AdamW optimizer, linear warmup scheduler, early stopping (patience=2), gradient clipping

## Experiments

6 experiments with greedy hyperparameter search:

| # | Change | LR | Batch | Truncation | Metadata |
|---|--------|----|-------|------------|----------|
| 1 | Baseline | 2e-5 | 16 | head-only | Yes |
| 2 | Truncation | 2e-5 | 16 | head+tail | Yes |
| 3 | LR low | 1e-5 | 16 | best | Yes |
| 4 | LR high | 5e-5 | 16 | best | Yes |
| 5 | Batch size | best LR | 8 | best | Yes |
| 6 | Ablation | best LR | best batch | best | No |

## Project Structure

```
├── Assignment - 1 Dataset/
│   ├── train.csv
│   └── test.csv
├── sentiment-analysis.ipynb
├── report.pdf
└── README.md
```

## Tracking

- **WANDB:** [Project Dashboard](https://wandb.ai/sceran/sentiment-analysis-transformer?nw=nwusersceran) (public)
- **Evaluation metric:** F1-macro (primary), accuracy, F1-weighted
