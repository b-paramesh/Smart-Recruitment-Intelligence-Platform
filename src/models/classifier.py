import torch
import torch.nn as nn
from .positional import PositionalEncoding
from .attention import MultiHeadAttention

class ResumeClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_classes: int, max_seq_len: int = 500):
        super().__init__()
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # 3. MultiHeadAttention Layer
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # 4. Feed Forward / Classification Layer
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, seq_len) containing token IDs
        """
        # Embed and scale
        embedded = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)
        
        # Add positional encoding
        encoded = self.pos_encoder(embedded)
        
        # Self-Attention
        attn_out, attn_weights = self.attention(encoded, encoded, encoded)
        
        # Pooling (Average Pooling over seq_len)
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        logits = self.fc(pooled)
        
        return logits, attn_weights
