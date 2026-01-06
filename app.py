import streamlit as st
import torch
import torch.nn.functional as F
from src.preprocess import clean_text
from src.features import PerplexityFeature
from src.model import load_model_and_tokenizer
import os

# Page Config
st.set_page_config(page_title="AI Misinformation Detector", layout="wide")

st.title("🕵️ Misinformation Detection System")
st.markdown("Enter a news article below to check if it's **Human-Written** or **AI-Generated**.")

# Load Resources (Cached)
@st.cache_resource
def get_model():
    # If a fine-tuned model exists, load it. Else load base.
    model_path = "./saved_model" if os.path.exists("./saved_model") else "roberta-base"
    model, tokenizer = load_model_and_tokenizer(model_path)
    return model, tokenizer

@st.cache_resource
def get_ppl_feature():
    return PerplexityFeature()

with st.spinner("Loading models..."):
    model, tokenizer = get_model()
    ppl_calc = get_ppl_feature()

# Input
user_input = st.text_area("Paste Article Text Here:", height=300)

if st.button("Analyze Article"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # 1. Preprocess
            clean_input = clean_text(user_input)
            
            # 2. Features - Perplexity
            ppl_score = ppl_calc.calculate_perplexity(clean_input)
            
            # 3. Model Inference
            inputs = tokenizer(clean_input, return_tensors="pt", truncation=True, max_length=512, padding=True)
            # Move inputs to same device as model
            device = model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1)
                
            # Assuming Label 1 = Fake (AI), Label 0 = Real (Human) - Typical for these datasets
            # BUT verify training mapping! 
            # In WELFake: 1 is Fake, 0 is Real.
            fake_prob = probs[0][1].item()
            real_prob = probs[0][0].item()
            
            # Classify
            is_fake = fake_prob > 0.5
            label = "AI-Generated / Fake" if is_fake else "Human-Written / Real"
            confidence = fake_prob if is_fake else real_prob

        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Result")
            color = "red" if is_fake else "green"
            st.markdown(f"<h2 style='color: {color};'>{label}</h2>", unsafe_allow_html=True)
            st.metric("Confidence Score", f"{confidence:.2%}")
            
        with col2:
            st.subheader("Secondary Signals")
            st.metric("Perplexity (GPT-2)", f"{ppl_score:.2f}", 
                      help="Lower perplexity can sometimes indicate AI generation (more predictable text).")
            
        # Debug/Raw info
        with st.expander("See Preprocessed Text"):
            st.write(clean_input)
