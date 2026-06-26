import streamlit as st
import torch
import shap
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup
MODEL_NAME = "Gek524/hinglish-emotion-xlmr"
LABELS = ['anger', 'joy', 'sadness', 'trust']
COLORS = {'anger': '#FF4B4B', 'joy': '#FFD700', 'sadness': '#4B9CD3', 'trust': '#50C878'}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def predict_proba(texts):
    inputs = tokenizer(list(texts), max_length=128, padding=True,
                       truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.numpy()

def get_shap_values(text):
    explainer = shap.Explainer(predict_proba, tokenizer,
                                output_names=LABELS)
    shap_values = explainer([text])
    return shap_values

def highlight_tokens(tokens, shap_vals, lang_tags=None):
    html = ""
    max_val = max(abs(v) for v in shap_vals) + 1e-9
    for token, val in zip(tokens, shap_vals):
        token = token.strip()
        if not token:
            continue
        intensity = abs(val) / max_val
        if val > 0:
            color = f"rgba(255, 75, 75, {intensity:.2f})"
        else:
            color = f"rgba(75, 153, 211, {intensity:.2f})"
        html += f'<span style="background-color: {color}; padding: 2px 4px; margin: 2px; border-radius: 3px;">{token}</span> '
    return html

# UI
st.set_page_config(page_title="Hinglish Emotion Detector", page_icon="🎭", layout="centered")

st.title("🎭 Hinglish Emotion Detector")
st.markdown("**Which language drives emotion in code-mixed Hinglish text?**")
st.markdown("Enter a Hinglish tweet to detect its emotion and see which tokens drive the prediction.")

# Examples
st.markdown("**Try an example:**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("😡 Anger"):
        st.session_state.input_text = "Yaar yeh log bilkul bakwaas kar rahe hain, harami kahin ke"
with col2:
    if st.button("😊 Joy"):
        st.session_state.input_text = "It's a classic hope you enjoy this amazing day!"
with col3:
    if st.button("😢 Sadness"):
        st.session_state.input_text = "Sir please help karo halat bahut buri hai yahan pollution bahut zyada hai"
with col4:
    if st.button("🤝 Trust"):
        st.session_state.input_text = "Inshallah sab theek ho jayega, Allah par bharosa rakho"

text_input = st.text_area(
    "Enter Hinglish text:",
    value=st.session_state.get('input_text', ''),
    height=100,
    placeholder="e.g. Yaar yeh bahut bura hua, I can't believe this happened..."
)

if st.button("Analyze", type="primary") and text_input:
    with st.spinner("Analyzing..."):
        # Prediction
        probs = predict_proba([text_input])[0]
        pred_idx = np.argmax(probs)
        pred_emotion = LABELS[pred_idx]
        confidence = probs[pred_idx]

        # Results
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### Predicted Emotion")
            st.markdown(f"<h2 style='color: {COLORS[pred_emotion]}'>{pred_emotion.upper()}</h2>",
                       unsafe_allow_html=True)
        with col2:
            st.markdown(f"### Confidence")
            st.progress(float(confidence))
            st.markdown(f"**{confidence:.1%}**")

        # All class probabilities
        st.markdown("### Probability Distribution")
        for label, prob in zip(LABELS, probs):
            st.markdown(f"**{label}**")
            st.progress(float(prob))
            st.markdown(f"{prob:.1%}")

        # SHAP token attribution
        st.markdown("### Token Attribution")
        st.markdown("*Red = positive attribution (pushes toward predicted emotion), Blue = negative*")

        with st.spinner("Computing token attributions..."):
            shap_vals = get_shap_values(text_input)
            tokens = shap_vals[0].data
            values = shap_vals[0].values[:, pred_idx]
            html = highlight_tokens(tokens, values)
            st.markdown(html, unsafe_allow_html=True)

st.markdown("---")
st.markdown("*Model: XLM-RoBERTa fine-tuned on Hinglish emotion dataset · [GitHub](https://github.com/HarshAggarwal524/hinglish-sentiment-analysis) · [Paper]()*")