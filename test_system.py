
import sys
import os
import torch

# Ensure src is in path or run from root
sys.path.append(os.getcwd())

from src.preprocess import clean_text
from src.features import PerplexityFeature
from src.model import load_model_and_tokenizer

def test_preprocess():
    print("Testing Preprocessing...")
    raw = "Example with HTML <br> and emoji 🐍"
    clean = clean_text(raw)
    print(f"Original: {raw}")
    print(f"Cleaned: {clean}")
    assert "<br>" not in clean
    assert ":snake:" in clean
    print("Preprocessing OK.\n")

def test_features():
    print("Testing Perplexity Feature (GPT-2)...")
    ppl_calc = PerplexityFeature()
    text = "This is a simple sentence to test perplexity."
    ppl = ppl_calc.calculate_perplexity(text)
    print(f"Text: {text}")
    print(f"Perplexity: {ppl}")
    assert ppl > 0
    print("Perplexity Feature OK.\n")

def test_model_loading():
    print("Testing Model Loading (RoBERTa)...")
    # This might take a moment to download if not cached
    model, tokenizer = load_model_and_tokenizer()
    text = "Test sentence."
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    print("Model inference output shape:", outputs.logits.shape)
    assert outputs.logits.shape[-1] == 2
    print("Model Loading & Inference OK.\n")

if __name__ == "__main__":
    try:
        test_preprocess()
        test_model_loading()
        # Features might take longer to load GPT2, put last
        test_features()
        print("ALL TESTS PASSED.")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
