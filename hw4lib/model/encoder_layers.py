import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer

'''
TODO: Implement this Module.

This file contains the encoder layer implementation used in transformer architectures:

SelfAttentionEncoderLayer: Used in encoder part of transformers
- Contains self-attention and feed-forward sublayers
- Unlike decoder, does not use causal masking (can attend to all positions)
- Used for tasks like encoding input sequences where bidirectional context is needed

The layer follows a Pre-LN (Layer Normalization) architecture where:
- Layer normalization is applied before each sublayer operation
- Residual connections wrap around each sublayer

Implementation Steps:
1. Initialize the required sublayers in __init__:
   - SelfAttentionLayer for self-attention (no causal mask needed)
   - FeedForwardLayer for position-wise processing

2. Implement the forward pass to:
   - Apply sublayers in the correct order
   - Pass appropriate padding masks (no causal mask needed)
   - Return both outputs and attention weights
'''

class SelfAttentionEncoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
      
        super().__init__()
        # TODO: Implement _init_

        # TODO: Initialize the sublayers   
        self.self_attn =  SelfAttentionLayer(d_model=d_model,num_heads=num_heads,dropout=dropout)
        self.ffn = FeedForwardLayer(d_model=d_model,d_ff=d_ff,dropout=dropout)
        

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
     
        # TODO: Implement forward: Follow the figure in the writeup

        x, mha_attn_weights = self.self_attn(x=x,key_padding_mask=key_padding_mask,attn_mask=None)
        x = self.ffn(x=x)

        return x, mha_attn_weights
