# import numpy as np
# from .activation import Softmax

# class ScaledDotProductAttention:
#     """
#     Scaled Dot Product Attention
#     """ 
#     def __init__(self):
#         '''
#         Initialize the ScaledDotProductAttention class.
#         '''
#         # Initialize your softmax layer
#         # What dimension should you pass to the softmax constructor?
#         self.eps = 1e10 # DO NOT MODIFY
#         self.softmax = NotImplementedError
        
    
#     def forward(self, Q, K, V, mask=None):
#         """
#         :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
#         :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
#         :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
#         :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
#         :return: Output matrix of shape (N, ..., H, L, Ev)
#         """
#         # TODO: Implement forward pass
        
#         # Calculate attention scores: (N, ..., H, L, S)
#         # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
#         scaled_dot_product = NotImplementedError
        
#         # Apply mask before softmax if provided
#         # If mask is not None, add -self.eps to the attention scores for positions to ignore
#         if mask is not None:
#             scaled_dot_product = NotImplementedError

#         # Compute attention scores: Apply softmax along S dimension (N, ..., H, L, S)
#         self.attention_scores = NotImplementedError

#         # Calculate output: (N, ..., H, L, Ev)
#         # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
#         output = NotImplementedError

#         # Return output
#         raise NotImplementedError
    
#     def backward(self, d_output):
#         """
#         :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
#         :return: Gradient of loss wrt input Q, K, V
#         """
#         # TODO: Implement backward pass

#         # Calculate gradients for V: (N, ..., H, S, Ev)
#         # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
#         # Use the transpose of stored softmax output to swap last two dimensions   
#         d_V = NotImplementedError
        
#         # Calculate gradients for attention scores
#         # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
#         d_attention_scores = NotImplementedError
#         d_scaled_dot_product = NotImplementedError
        
#         # Scale gradients by sqrt(d_k)
#         d_scaled_dot_product = NotImplementedError
        
#         # Calculate gradients for Q and K
#         # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)   
#         d_Q = NotImplementedError
#         # (N, ..., H, L, S) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
#         d_K = NotImplementedError
        
#         # Return gradients for Q, K, V
#         raise NotImplementedError



import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    def __init__(self):
        self.eps = 1e10
        self.softmax = Softmax(dim=-1)  # Softmax along the last dimension (S)
    
    def forward(self, Q, K, V, mask=None):
        # Store shapes and inputs for backward pass
        self.Q = Q
        self.K = K
        self.V = V
        
        # Calculate attention scores
        d_k = Q.shape[-1]  # Get embedding dimension using .shape instead of .size()
        scaled_dot_product = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scaled_dot_product = np.where(mask, -self.eps, scaled_dot_product)
        
        # Compute attention scores
        self.attention_scores = self.softmax.forward(scaled_dot_product)
        
        # Calculate output
        output = np.matmul(self.attention_scores, V)
        
        return output
    
    def backward(self, d_output):
        # Gradient for V
        d_V = np.matmul(np.swapaxes(self.attention_scores, -2, -1), d_output)
        
        # Gradient for attention scores
        d_attention_scores = np.matmul(d_output, np.swapaxes(self.V, -2, -1))
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        # Scale gradients by sqrt(d_k)
        d_k = self.Q.shape[-1]
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(d_k)
        
        # Gradient for Q
        d_Q = np.matmul(d_scaled_dot_product, self.K)
        
        # Gradient for K
        d_K = np.matmul(np.swapaxes(d_scaled_dot_product, -2, -1), self.Q)
        
        return d_Q, d_K, d_V