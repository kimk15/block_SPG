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
	return np.random.choice(total_col, size=B, replace=False, p=None)

"""
Update a single factor matrix via gradient update
"""
def update_factor_bras(X_unfold, A, idx, n, rank, alpha):
	H_n = (tl_alg.khatri_rao(A, skip_matrix=n))[idx, :]
	A[n] -= alpha/rank * (A[n] @ H_n.T @ H_n - X_unfold[n][:, idx] @ H_n)

	# Apply proximal regularization constraint (A > 0)
	A[n] = np.clip(A[n], a_min=0, a_max=None)

"""
Update a single factor matrix via gradient update
"""
def update_factor_ada(X_unfold, A, G, idx, n, rank, eta, b, eps):
	# Compute Learning Rate
	lr = compute_learning_rate(G[n], b, eta, eps)
	# lr = 0.1

	# Compute Grad
	H_n = (tl_alg.khatri_rao(A, skip_matrix=n))[idx, :]
	grad = 1/rank * (A[n] @ H_n.T @ H_n - X_unfold[n][:, idx] @ H_n)
	A[n] -= lr*grad

	# Apply proximal regularization constraint (A > 0)
	A[n] = np.clip(A[n], a_min=0, a_max=None)

	# Accumulate gradient information.
	G[n] += grad**(2)

"""
Computes ada learning rate
"""
def compute_learning_rate(G_n, b, eta, eps):
	return eta/(G_n + b)**((1/2) + eps)


"""
Computes Normalized MSE between the factor matrices:
"""
def MSE(F, F_norm, A):
	mse = 0
	for i in range(len(A)):
		A_true_norm = F[i] / F_norm[i]
		A_norm = A[i] / norm(A[i])
		mse += norm(A_true_norm - A_norm)**2
	return mse/len(A)

"""
Computes residual error
"""
def residual_error(X0, norm_x, A, B, C):
	X_bar = A @ (tl_alg.khatri_rao([A, B, C], skip_matrix=0).T)
	return norm(X0-X_bar)/norm_x


"""
Computes sketched residual error
"""
def sketched_residual_error(X, norm_x,  idx, A, m):
	X_bar = tl_kruskal.kruskal_to_tensor(A)
	return norm(unfold(X, mode=m)[:,idx]-unfold(X_bar, mode=m)[:,idx])/norm_x


"""
Computes norm of a tensor
"""
def norm(X):
	return np.linalg.norm(X)
	

