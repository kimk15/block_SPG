from helper import rand_init
from helper import sample
from helper import generate_sketch_indices
from helper import update_factor_bras
from helper import update_factor_ada
from helper import residual_error
from helper import norm
from helper import MSE
import numpy as np

"""
Computes brasCPD for 3 dimensional tensors.
Returns A,B,C
"""
def bras_CPD(A_true, X, rank, B, alpha, beta, num_iterations=100):
	# Keep residual errors
	cost = []
	mse = []

	# Randomly initialize A,B,C
	dim_1, dim_2, dim_3 = X.shape
	A = [rand_init(dim_1, rank), rand_init(dim_2, rank), rand_init(dim_3, rank)]
	total_col = {0:dim_2*dim_3, 1:dim_1*dim_3, 2:dim_1*dim_2}
	cost.append(residual_error(X, A))
	mse.append(MSE(A_true, A))
	# Run bras_CPD
	for r in range(num_iterations):
		if (r+1)%1000 == 0:
			print("Iteration:", r)
		# Randomly select mode n to update.
		n = sample(3)

		# Generate sketching indices
		idx = generate_sketch_indices(B, total_col[n])

		# Update Factor matrix
		update_factor_bras(X, A, idx, n, rank, alpha)

		# Update learning rate
		alpha /= (r+1)**beta

		# Append error
		cost.append(residual_error(X, A))
		mse.append(MSE(A_true, A))
	return A, np.array(cost), np.array(mse)

"""
Computes AdaCPD for 3 dimensional tensors.
Returns A,B,C
"""
def ada_CPD(A_true, X, rank, B, eta, b, eps, num_iterations=100):
	# Keep residual errors
	cost = []
	mse = []

	# Randomly initialize A,B,C
	dim_1, dim_2, dim_3 = X.shape
	A = [rand_init(dim_1, rank), rand_init(dim_2, rank), rand_init(dim_3, rank)]
	G = [np.zeros((dim_1, rank), dtype=float)+ 1,np.zeros((dim_2, rank), dtype=float)+ 1,np.zeros((dim_3, rank), dtype=float)+ 1]
	total_col = {0:dim_2*dim_3, 1:dim_1*dim_3, 2:dim_1*dim_2}
	cost.append(residual_error(X, A))
	mse.append(MSE(A_true, A))

	# Run bras_CPD
	for r in range(num_iterations):
		if (r+1)%1000 == 0:
			print("Iteration:", r)
		# Randomly select mode n to update.
		n = sample(3)

		# Generate sketching indices
		idx = generate_sketch_indices(B, total_col[n])

		# Update factor matrix
		update_factor_ada(X, A, G, idx, n, rank, eta, b, eps)

		# Append error
		cost.append(residual_error(X, A))
		mse.append(MSE(A_true, A))
	return A, np.array(cost), np.array(mse)