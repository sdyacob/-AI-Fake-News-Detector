import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from src.model import load_model_and_tokenizer
from src.preprocess import clean_text
import os

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train(dataset_path='WELFake.csv'):
    print(f"Loading dataset from {dataset_path}...")
    
    # Heuristic to load common datasets. 
    # WELFake usually has: title, text, label (1=fake, 0=real) - verify this mapping!
    # ISOT has separate files often, but here we assume a merged CSV for simplicity as per plan.
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Please place the CSV file there.")
        return

    df = pd.read_csv(dataset_path)
    
    # Basic data preparation (Adjust column names as needed based on actual CSV)
    # Trying to auto-detect text column
    text_col = 'text' if 'text' in df.columns else df.columns[1] 
    label_col = 'label' if 'label' in df.columns else df.columns[-1]

    print(f"Using text column: {text_col}, label column: {label_col}")

    # Fill NaNs
    df[text_col] = df[text_col].fillna('')
    
    # Preprocess a subset for speed in demo/first run, or all if full training
    # For robust training on full dataset, remove slicer.
    print("Preprocessing text...")
    df['clean_text'] = df[text_col].apply(clean_text)

    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['clean_text'].tolist(), df[label_col].tolist(), test_size=0.2, random_state=42
    )

    # Load Model & Tokenizer
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    # Tokenize
    print("Tokenizing...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Adjust based on VRAM
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")
    print("Model saved to ./saved_model")

if __name__ == "__main__":
    # You can change the path to your dataset here
    train(dataset_path='WELFake_Dataset.csv') 
