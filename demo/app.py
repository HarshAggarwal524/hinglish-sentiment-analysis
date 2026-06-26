import gradio as gr
import torch
import shap
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Gek524/hinglish-emotion-xlmr"
LABELS = ['anger', 'joy', 'sadness', 'trust']

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def predict_proba(texts):
    inputs = tokenizer(list(texts), max_length=128, padding=True,
                       truncation=True, return_tensors='pt')
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)
    return probs.numpy()

def analyze(text):
    probs = predict_proba([text])[0]
    pred_idx = np.argmax(probs)
    label_probs = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return label_probs

examples = [
    ["Yaar yeh log bilkul bakwaas kar rahe hain, harami kahin ke"],
    ["It's a classic hope you enjoy this amazing day!"],
    ["Sir please help karo halat bahut buri hai"],
    ["Inshallah sab theek ho jayega, Allah par bharosa rakho"]
]

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Enter Hinglish text", placeholder="e.g. Yaar yeh bahut bura hua..."),
    outputs=gr.Label(label="Emotion Probabilities"),
    title="🎭 Hinglish Emotion Detector",
    description="XLM-RoBERTa fine-tuned on Hinglish code-mixed text for emotion detection.",
    examples=examples
)

demo.launch()