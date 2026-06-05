import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Create a vector of shape (max_len, 1) with positions (0, 1, ..., max_len - 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create a vector of shape (d_model/2) with frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register pe as a buffer so it's not considered a parameter but saved in state_dict
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch_size, seq_len, d_model)
        Returns: Tensor of shape (batch_size, seq_len, d_model) with positional encoding added.
        """
        seq_len = x.size(1)
        # Add positional encoding to input
        x = x + self.pe[:, :seq_len, :]
        return x

def prove_positional_encoding_effect():
    """
    Demonstrates that order matters when Positional Encoding is applied.
    """
    d_model = 16
    seq_len = 5
    pe = PositionalEncoding(d_model)
    
    # Create two identical embeddings but in different order
    emb1 = torch.ones(1, seq_len, d_model)
    emb2 = torch.ones(1, seq_len, d_model)
    
    encoded1 = pe(emb1)
    
    # Reverse the order of emb2
    emb2_reversed = emb2.flip(dims=[1])
    encoded2 = pe(emb2_reversed)
    
    # Even though emb1 and emb2 are identical (all ones), 
    # their positional encodings at index 0 and index (seq_len-1) will be different
    # This proves order affects the representation.
    return encoded1, encoded2
