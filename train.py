import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from models.classifier import ResumeClassifier

# Simple Tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.word2idx = self.vocab
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        
    def fit(self, texts):
        idx = 2
        for text in texts:
            words = str(text).lower().split()
            for word in words:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1
                    
    def encode(self, text, max_len=100):
        words = str(text).lower().split()
        tokens = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
        if len(tokens) < max_len:
            tokens.extend([self.vocab["<PAD>"]] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
        return tokens
        
    def __len__(self):
        return len(self.word2idx)

class ResumeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=100):
        self.texts = df['text'].tolist()
        self.labels = df['category'].astype('category').cat.codes.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = len(df['category'].unique())
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.encode(text, self.max_len)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def train_model():
    print("Loading Data...")
    DATA_PATH = "data/synthetic_resumes.csv"
    if not os.path.exists(DATA_PATH):
        print("Data not found. Run generate_data.py first.")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    tokenizer = SimpleTokenizer()
    tokenizer.fit(df['text'])
    
    dataset = ResumeDataset(df, tokenizer, max_len=50)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    vocab_size = len(tokenizer)
    d_model = 32
    num_heads = 4
    num_classes = dataset.num_classes
    
    print(f"Initializing ResumeClassifier (Vocab: {vocab_size}, Classes: {num_classes})")
    model = ResumeClassifier(vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, num_classes=num_classes, max_seq_len=50)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    print("Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_tokens, batch_labels in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            logits, attn_weights = model(batch_tokens)
            
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
    # Save the model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/resume_classifier.pth')
    print("Training Complete! Model saved to models/resume_classifier.pth")

if __name__ == "__main__":
    train_model()
