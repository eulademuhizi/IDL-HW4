import torch

# ''' 
# TODO: Implement this function.

# Specification:
# - Function should create a padding mask that identifies padded positions in the input
# - Mask should be a boolean tensor of shape (N, T) where:
#   * N = batch size from padded_input
#   * T = sequence length from padded_input
# - True values indicate padding positions that should be masked
# - False values indicate valid positions that should not be masked
# - Padding is assumed to be on the right side of sequences
# - Each sequence in the batch may have different valid lengths
# - Mask should be on same device as input tensor
# '''
# def PadMask(padded_input, input_lengths):
#     """ 
#     Create a mask to identify non-padding positions. 
#     Args:
#         padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
#         input_lengths: The actual lengths of each sequence before padding, shape (N,).
#     Returns:
#         A boolean mask tensor with shape (N, T), where: 
#             - padding positions are marked with True 
#             - non-padding positions are marked with False.
#     """
#     # TODO: Implement PadMask
#     raise NotImplementedError # Remove once implemented

# ''' 
# TODO: Implement this function.

# Specification:
# - Function should create a causal mask for self-attention
# - Mask should be a boolean tensor of shape (T, T) where T is sequence length
# - True values indicate positions that should not attend to each other
# - False values indicate positions that can attend to each other
# - Causal means each position can only attend to itself and previous positions
# - Mask should be on same device as input tensor
# - Mask should be upper triangular (excluding diagonal)
# '''
# def CausalMask(padded_input):
#     """ 
#     Create a mask to identify non-causal positions. 
#     Args:
#         padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
    
#     Returns:
#         A boolean mask tensor with shape (T, T), where: 
#             - non-causal positions (don't attend to) are marked with True 
#             - causal positions (can attend to) are marked with False.
#     """
#     # TODO: Implement CausalMask
#     raise NotImplementedError # Remove once implemented



def PadMask(padded_input, input_lengths):
    """Create padding mask for transformer inputs."""
    # Get sequence length from input shape
    seq_len = padded_input.size(1)
    
    # Create range tensor [0, 1, 2, ..., seq_len-1]
    range_tensor = torch.arange(seq_len, device=padded_input.device)
    
    # Compare each position with input lengths to create mask
    # True where position >= input_length (padding positions)
    mask = range_tensor.expand(len(input_lengths), seq_len) >= input_lengths.unsqueeze(1)
    
    return mask

def CausalMask(padded_input):
    """Create causal mask for autoregressive attention."""
    # Get sequence length from input shape
    seq_len = padded_input.size(1)
    
    # Create upper triangular matrix with diagonal offset 1
    # True values will be masked (not allowed to attend)
    mask = torch.triu(torch.ones(seq_len, seq_len, 
                                dtype=torch.bool,
                                device=padded_input.device), 
                      diagonal=1)
    
    return mask