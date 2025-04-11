# import numpy as np


# class Softmax:
#     """
#     A generic Softmax activation function that can be used for any dimension.
#     """
#     def __init__(self, dim=-1):
#         """
#         :param dim: Dimension along which to compute softmax (default: -1, last dimension)
#         DO NOT MODIFY
#         """
#         self.dim = dim

#     def forward(self, Z):
#         """
#         :param Z: Data Z (*) to apply activation function to input Z.
#         :return: Output returns the computed output A (*).
#         """
#         if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
#             raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
#         # TODO: Implement forward pass
#         # Compute the softmax in a numerically stable way
#         # Apply it to the dimension specified by the `dim` parameter
#         self.A = NotImplementedError
#         raise NotImplementedError

#     def backward(self, dLdA):
#         """
#         :param dLdA: Gradient of loss wrt output
#         :return: Gradient of loss with respect to activation input
#         """
#         # TODO: Implement backward pass
        
#         # Get the shape of the input
#         shape = self.A.shape
#         # Find the dimension along which softmax was applied
#         C = shape[self.dim]
           
#         # Reshape input to 2D
#         if len(shape) > 2:
#             self.A = NotImplementedError
#             dLdA = NotImplementedError

#         # Reshape back to original dimensions if necessary
#         if len(shape) > 2:
#             # Restore shapes to original
#             self.A = NotImplementedError
#             dLdZ = NotImplementedError

#         raise NotImplementedError
 

    

import numpy as np

class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # Store input for backward pass
        self.Z = Z
        
        # For numerical stability, subtract the max along the specified dimension
        max_Z = np.max(Z, axis=self.dim, keepdims=True)
        shifted_Z = Z - max_Z
        
        # Compute softmax
        exp_Z = np.exp(shifted_Z)
        sum_exp_Z = np.sum(exp_Z, axis=self.dim, keepdims=True)
        A = exp_Z / sum_exp_Z
        
        # Store output for backward pass
        self.A = A
        
        return A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # Get the shape of the input
        shape = self.A.shape
        
        # Move softmax dimension to last position for easier computation
        if self.dim != -1 and len(shape) > 1:
            # Move the softmax dimension to last position
            A = np.moveaxis(self.A, self.dim, -1)
            dLdA_moved = np.moveaxis(dLdA, self.dim, -1)
        else:
            A = self.A
            dLdA_moved = dLdA
        
        # Flatten all dimensions except the softmax dimension
        if len(shape) > 2:
            original_shape = A.shape
            batch_size = np.prod(original_shape[:-1])
            A = A.reshape(batch_size, -1)  # Shape: (batch_size, C)
            dLdA_moved = dLdA_moved.reshape(batch_size, -1)  # Shape: (batch_size, C)
        
        # Compute Jacobian matrix for each sample in the batch
        batch_size = A.shape[0]
        C = A.shape[1]
        
        # Initialize gradient
        dLdZ = np.zeros_like(A)
        
        # Compute gradient for each sample in the batch
        for i in range(batch_size):
            a = A[i]
            # Create Jacobian matrix: J = diag(a) - a.T @ a
            J = np.diag(a) - np.outer(a, a)
            # Compute gradient: dLdZ = dLdA @ J
            dLdZ[i] = np.dot(dLdA_moved[i], J)
        
        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            dLdZ = dLdZ.reshape(original_shape)
        
        # Move softmax dimension back to original position
        if self.dim != -1 and len(shape) > 1:
            dLdZ = np.moveaxis(dLdZ, -1, self.dim)
        
        return dLdZ