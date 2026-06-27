# Which Language Drives Emotion? Explainable Multilingual Models for Hinglish Emotion Detection

[![arXiv](https://img.shields.io/badge/arXiv-paper-red)](YOUR_ARXIV_LINK)
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/Gek524/hinglish-emotion-xlmr)
[![HuggingFace Demo](https://img.shields.io/badge/🤗-Demo-blue)](https://huggingface.co/spaces/Gek524/hinglish-emotion-detector)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

This project investigates **which language's tokens drive emotion classification** in code-mixed Hinglish (Hindi-English) social media text. We fine-tune three multilingual transformer models and apply SHAP-based token attribution analysis to reveal systematic language asymmetry across emotion classes.

**Central Finding:** Anger classification is predominantly driven by Hindi tokens (attribution ratio = 0.527), while joy is the only emotion driven more by English tokens (0.397). This challenges the assumption that multilingual models learn language-neutral representations.

---

## Live Demo

Try the model live on HuggingFace Spaces:
👉 [hinglish-emotion-detector](https://huggingface.co/spaces/Gek524/hinglish-emotion-detector)

---

## Results

### Sentiment Classification (3-class)

| Model | Macro F1 | Neg F1 | Neu F1 | Pos F1 |
|-------|----------|--------|--------|--------|
| TF-IDF + LR | 0.592 | 0.63 | 0.51 | 0.64 |
| mBERT | 0.614 | 0.63 | 0.56 | 0.65 |
| XLM-R | 0.644 | 0.66 | 0.57 | 0.70 |
| MuRIL | 0.641 | 0.64 | 0.57 | 0.71 |

### Emotion Classification (4-class)

| Model | Macro F1 | Anger F1 | Joy F1 | Sadness F1 | Trust F1 |
|-------|----------|----------|--------|------------|----------|
| XLM-R | **0.681** | 0.85 | 0.86 | 0.49 | 0.52 |
| MuRIL | 0.611 | 0.85 | 0.82 | 0.32 | 0.45 |

### SHAP Language Attribution

| Emotion | Hindi Ratio | English Ratio |
|---------|-------------|---------------|
| Anger | **0.527** | 0.256 |
| Joy | 0.356 | **0.397** |
| Sadness | 0.389 | 0.368 |
| Trust | 0.401 | 0.364 |

---

## Dataset

- **Base:** SemEval-2020 Task 9 Hinglish dataset (15,480 tweets with gold CoNLL language tags)
- **Custom split:** 85/15 stratified (13,158 train / 2,322 dev, zero overlap)
- **Emotion labels:** Silver-standard GPT-4o-mini annotation → 5,980 tweets, 4 classes
- **Validation:** Cohen's κ = 0.667 (200 manually labeled tweets)

| Emotion | Train | Dev |
|---------|-------|-----|
| Anger | 2,375 | 419 |
| Joy | 1,990 | 351 |
| Sadness | 427 | 75 |
| Trust | 291 | 52 |

---

## Project Structure

```
hinglish-sentiment-analysis/
├── data/
│   ├── train_custom.csv
│   ├── dev_custom.csv
│   ├── emotion_train_final.csv
│   ├── emotion_dev_final.csv
│   └── emotion_merged.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Baseline.ipynb
│   ├── 03_mBERTv2.ipynb
│   ├── 04_XlmRv2_and_MuRilv2.ipynb
│   ├── 06_emotion_relabeling.ipynb
│   ├── 07_emotion_label_train.ipynb
│   ├── 08_emotion_xlmr.ipynb
│   ├── 09_emotion_MuRIL.ipynb
│   ├── 10_ablation_muril.ipynb
│   └── 11_shap_analysis.ipynb
├── results/
│   ├── shap_attribution_figure3.png
│   ├── shap_attribution_ratios.csv
│   └── muril_ablation_curve.png
├── demo/
│   ├── app.py
│   └── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/HarshAggarwal524/hinglish-sentiment-analysis
cd hinglish-sentiment-analysis
pip install transformers torch shap gradio pandas scikit-learn
```

### Run demo locally

```bash
cd demo
pip install -r requirements.txt
python app.py
```

### Load model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("Gek524/hinglish-emotion-xlmr")
model = AutoModelForSequenceClassification.from_pretrained("Gek524/hinglish-emotion-xlmr")
model.eval()

text = "Yaar yeh bahut bura hua, I can't believe this"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)

labels = ['anger', 'joy', 'sadness', 'trust']
print({labels[i]: round(float(probs[0][i]), 3) for i in range(4)})
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{aggarwal2026hinglish,
  title={Which Language Drives Emotion? Explainable Multilingual Models for Hinglish Emotion Detection},
  author={Harsh Aggarwal},
  year={2026},
  eprint={YOUR_ARXIV_ID},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

---

## Model

The fine-tuned XLM-RoBERTa model is available on HuggingFace Hub:
👉 [Gek524/hinglish-emotion-xlmr](https://huggingface.co/Gek524/hinglish-emotion-xlmr)

---

## Author

**Harsh Aggarwal**
Guru Gobind Singh Indraprastha University, New Delhi
harshaggarwalofficial1@gmail.com
