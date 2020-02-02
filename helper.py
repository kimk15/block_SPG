from numpy.linalg import pinv, norm
import numpy as np
import time
import random
import tensorly.tenalg as tl_alg
import tensorly.base as tl_base
import tensorly.kruskal_tensor as tl_kruskal

"""
Randomly generates numpy matrix G with entries [0,1)
"""
def rand_init(dim, rank):
	return np.random.rand(dim, rank)

"""
Uniformly samples from {0....N-1}
"""
def sample(N):
	return random.randint(0, N-1)

"""
Unfolds a tensors following the Kolda and Bader definition
Source: https://stackoverflow.com/questions/49970141/using-numpy-reshape-to-perform-3rd-rank-tensor-unfold-operation
"""
def unfold(tensor, mode=0):
	return tl_base.unfold(tensor, mode)

"""
Generates sketching indices
"""
def generate_sketch_indices(B, total_col):
	return np.random.choice(range(total_col), size=B, replace=False, p=None)

"""
Update a single factor matrix via gradient update
"""
def update_factor(X, A, idx, n, rank, alpha):
	X_n = ((unfold(X, mode=n)).T)[idx, :]
	H_n = (tl_alg.khatri_rao(A, skip_matrix=n))[idx, :]
	A[n] = A[n] - alpha/rank * (A[n] @ H_n.T @ H_n - X_n.T @ H_n)



"""
Computes residual error
"""
def residual_error(X, A):
	X_bar = tl_kruskal.kruskal_to_tensor(A)
	return norm(X-X_bar)/norm(X)

"""
Computes norm of a tensor
"""
def norm(X):
	return np.linalg.norm(X)
	

