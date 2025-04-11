# import numpy as np

# class Linear:
#     def __init__(self, in_features, out_features):
#         """
#         Initialize the weights and biases with zeros
#         W shape: (out_features, in_features)
#         b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
#         """
#         # DO NOT MODIFY
#         self.W = np.zeros((out_features, in_features))
#         self.b = np.zeros(out_features)


#     def init_weights(self, W, b):
#         """
#         Initialize the weights and biases with the given values.
#         """
#         # DO NOT MODIFY
#         self.W = W
#         self.b = b

#     def forward(self, A):
#         """
#         :param A: Input to the linear layer with shape (*, in_features)
#         :return: Output Z with shape (*, out_features)
        
#         Handles arbitrary batch dimensions like PyTorch
#         """
#         # TODO: Implement forward pass
        
#         # Store input for backward pass
#         self.A = A
        
#         raise NotImplementedError

#     def backward(self, dLdZ):
#         """
#         :param dLdZ: Gradient of loss wrt output Z (*, out_features)
#         :return: Gradient of loss wrt input A (*, in_features)
#         """
#         # TODO: Implement backward pass

#         # Compute gradients (refer to the equations in the writeup)
#         self.dLdA = NotImplementedError
#         self.dLdW = NotImplementedError
#         self.dLdb = NotImplementedError
#         self.dLdA = NotImplementedError
        
#         # Return gradient of loss wrt input
#         raise NotImplementedError




import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)

    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # Store input shape and flatten for matrix multiplication
        self.input_shape = A.shape
        batch_size = np.prod(A.shape[:-1])
        A_flat = A.reshape(batch_size, -1)  # Shape: (batch_size, in_features)
        
        # Store input for backward pass
        self.A = A_flat
        
        # Compute linear transformation
        Z_flat = np.dot(A_flat, self.W.T) + self.b  # Shape: (batch_size, out_features)
        
        # Reshape output to match input dimensions (except last dimension)
        output_shape = self.input_shape[:-1] + (self.W.shape[0],)
        Z = Z_flat.reshape(output_shape)
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # Flatten the gradient to match stored input shape
        batch_size = np.prod(dLdZ.shape[:-1])
        dLdZ_flat = dLdZ.reshape(batch_size, -1)  # Shape: (batch_size, out_features)
        
        # Compute gradients (refer to the equations in the writeup)
        # Gradient of loss wrt input A
        self.dLdA_flat = np.dot(dLdZ_flat, self.W)  # Shape: (batch_size, in_features)
        
        # Gradient of loss wrt weights W
        self.dLdW = np.dot(dLdZ_flat.T, self.A)  # Shape: (out_features, in_features)
        
        # Gradient of loss wrt bias b
        self.dLdb = np.sum(dLdZ_flat, axis=0)  # Shape: (out_features,)
        
        # Reshape dLdA to match original input shape
        dLdA = self.dLdA_flat.reshape(self.input_shape)
        
        return dLdA