from transformers import RobertaForSequenceClassification, RobertaTokenizer

def load_model_and_tokenizer(model_path='roberta-base', num_labels=2):
    """
    Loads the RoBERTa model and tokenizer.
    If model_path is a directory, it loads the fine-tuned model.
    Otherwise it loads the base model from generic tag.
    """
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    return model, tokenizer
