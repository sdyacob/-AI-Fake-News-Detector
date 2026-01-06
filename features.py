import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math

class PerplexityFeature:
    def __init__(self, model_id='gpt2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculates the perplexity of the text using GPT-2.
        Lower perplexity often indicates text that is more 'predictable' to the model 
        (sometimes associated with AI generation, though high-quality human text can also be low PPL).
        """
        if not text or not text.strip():
            return 0.0

        encodings = self.tokenizer(text, return_tensors='pt')
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        
        # Stride window approach if text is long, but for simplicity here strictly following 
        # a basic PPL calculation for the snippet. If it's too long, we truncate or stride.
        # Let's do a simple truncation to 1024 tokens for now to avoid errors.
        max_length = self.model.config.n_positions
        if encodings['input_ids'].size(1) > max_length:
             encodings['input_ids'] = encodings['input_ids'][:, :max_length]
             encodings['attention_mask'] = encodings['attention_mask'][:, :max_length]

        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings['input_ids'])
            loss = outputs.loss
            ppl = torch.exp(loss)

        return ppl.item()
