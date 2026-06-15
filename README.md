# hinglish-sentiment-analysis
SemEval-2020 Task 9 — Sentiment Analysis for Hinglish code-mixed text
# Hinglish Sentiment Analysis — SemEval-2020 Task 9

## Research Question
<!-- TODO: What is the core question this project investigates? -->
> Placeholder: How can transformer-based models effectively capture sentiment in Hinglish (Hindi-English) code-mixed social media text?

## Dataset
<!-- TODO: Describe the dataset in detail -->
- **Source:** SemEval-2020 Task 9 — Sentiment Analysis for Code-Mixed Social Media Text (Hinglish subtask)
- **Splits:** Train / Dev / Test
- **Labels:** Positive, Negative, Neutral
- **Size:** ~TBD examples

## Method
<!-- TODO: Describe your approach -->
- Baseline: Placeholder (e.g., TF-IDF + Logistic Regression)
- Proposed: Placeholder (e.g., MuRIL / XLM-R fine-tuning)
- Evaluation metric: Weighted F1

## Results
<!-- TODO: Fill after experiments -->
| Model | Weighted F1 |
|-------|-------------|
| Baseline | — |
| Proposed | — |

## Repo Structure

├── data/           # Raw & processed datasets

├── notebooks/      # EDA and experiments

├── src/            # Source code

├── results/        # Outputs and logs

└── README.md

## Setup

```bash
pip install -r requirements.txt
```

## Citation

@inproceedings{patwa-etal-2020-sentimix,

title     = {SentiMix 2020: Hinglish Sentiment Analysis},

author    = {Patwa, Parth and others},

booktitle = {Proceedings of SemEval-2020},

year      = {2020}

}