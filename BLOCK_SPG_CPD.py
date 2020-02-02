from helper import rand_init
from helper import sample
from helper import generate_sketch_indices
from helper import update_factor
from helper import residual_error
from helper import norm

import numpy as np

"""
Computes brasCPD for 3 dimensional tensors.
Returns A,B,C
"""
def bras_CPD(X, rank, B, alpha, beta, num_iterations=100):
	# Keep residual errors
	error = []

	# Randomly initialize A,B,C
	dim_1, dim_2, dim_3 = X.shape
	A = [rand_init(dim_1, rank), rand_init(dim_2, rank), rand_init(dim_3, rank)]
	total_col = {0:dim_2*dim_3, 1:dim_1*dim_3, 2:dim_1*dim_2}

	# Run bras_CPD
	for r in range(num_iterations):
		# Randomly select mode n to update.
		n = sample(3)

		# Generate sketching indices
		idx = generate_sketch_indices(B, total_col[n])

		# Update Factor matrix
		update_factor(X, A, idx, n, rank, alpha)

		# Update learning rate
		alpha /= (r+1)**beta

		# Append error
		error.append(residual_error(X, A))
	return A, np.array(error)

"""
Computes AdaCPD for 3 dimensional tensors.
Returns A,B,C
"""
def ada_CPD(X, rank, B, alpha, beta, num_iterations=100):
	# Keep residual errors
	error = []

	# Randomly initialize A,B,C
	dim_1, dim_2, dim_3 = X.shape
	A = [rand_init(dim_1, rank), rand_init(dim_2, rank), rand_init(dim_3, rank)]
	total_col = {0:dim_2*dim_3, 1:dim_1*dim_3, 2:dim_1*dim_2}

	# Run bras_CPD
	for r in range(num_iterations):
		# Randomly select mode n to update.
		n = sample(3)

		# Generate sketching indices
		idx = generate_sketch_indices(B, total_col[n])

		# Update Factor matrix
		update_factor(X, A, idx, n, rank, alpha)

		# Update learning rate
		alpha /= (r+1)**beta

		# Append error
		error.append(residual_error(X, A))
	return A, np.array(error)