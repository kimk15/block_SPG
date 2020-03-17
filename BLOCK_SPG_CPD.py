from helper import rand_init
from helper import sample
from helper import generate_sketch_indices
from helper import update_factor_bras
from helper import update_factor_ada
from helper import residual_error
from helper import norm
from helper import MSE
from helper import sketched_residual_error
from helper import unfold
from timeit import default_timer as timer
import numpy as np

"""
Computes brasCPD for 3 dimensional tensors.
Returns A,B,C
"""
def bras_CPD(F, X, rank, B, alpha, beta, num_iterations=100, max_time=None):
	# bookkeeping
	total_time = 0
	res_error = []
	time = []

	# Cache norm
	start = timer()
	norm_x = norm(X)
	F_norm = [norm(F[0]), norm(F[1]), norm(F[2])]

	# Randomly initialize A,B,C
	dim_1, dim_2, dim_3 = X.shape
	A = [rand_init(dim_1, rank), rand_init(dim_2, rank), rand_init(dim_3, rank)]
	total_col = {0:dim_2*dim_3, 1:dim_1*dim_3, 2:dim_1*dim_2}

	# Cache Unfoldings
	X_unfold = [unfold(X, mode=0), unfold(X, mode=1), unfold(X, mode=2)]

	# Finish timing initialization step
	end = timer()
	total_time += end - start

	# Append initialization residual error
	res_error.append(residual_error(X_unfold[0], norm_x, A[0], A[1], A[2]))
	time.append(total_time)

	# Run bras_CPD
	if max_time == None:
		for r in range(num_iterations):
			if (r+1) % 5000 == 0:
				print("iteration:", r)

			# Time start step
			start = timer()
			# Randomly select mode n time update.
			n = sample(3)
			# Generate sketching indices
			idx = generate_sketch_indices(B, total_col[n])
			# Update Factor matrix
			update_factor_bras(X_unfold, A, idx, n, rank, alpha)
			# Update learning rate
			alpha /= (r+1)**beta
			# Time iteration step
			end = timer()
			total_time += end - start
			# Append error
			res_error.append(residual_error(X_unfold[0], norm_x, A[0], A[1], A[2]))
	else:
		r = 1
		while total_time < max_time:
			# Time start step
			start = timer()
			# Randomly select mode n time update.
			n = sample(3)
			# Generate sketching indices
			idx = generate_sketch_indices(B, total_col[n])
			# Update Factor matrix
			update_factor_bras(X_unfold, A, idx, n, rank, alpha)
			# Update learning rate
			alpha /= (r+1)**beta
			r += 1
			# Time iteration step
			end = timer()
			total_time += end - start
			# Append error
			res_error.append(residual_error(X_unfold[0], norm_x, A[0], A[1], A[2]))
			time.append(total_time)



	return total_time, res_error, time

"""
Computes AdaCPD for 3 dimensional tensors.
Returns A,B,C
"""
def ada_CPD(F, X, rank, B, eta, b, eps, num_iterations=100, max_time=None):
	# bookkeeping
	total_time = 0
	res_error = []
	time = []

	# Cache norm
	start = timer()
	norm_x = norm(X)
	F_norm = [norm(F[0]), norm(F[1]), norm(F[2])]

	# Randomly initialize A,B,C
	dim_1, dim_2, dim_3 = X.shape
	A = [rand_init(dim_1, rank), rand_init(dim_2, rank), rand_init(dim_3, rank)]
	G = [np.zeros((dim_1, rank), dtype=float),np.zeros((dim_2, rank), dtype=float),np.zeros((dim_3, rank), dtype=float)]
	total_col = {0:dim_2*dim_3, 1:dim_1*dim_3, 2:dim_1*dim_2}

	# Cache Unfoldings
	X_unfold = [unfold(X, mode=0), unfold(X, mode=1), unfold(X, mode=2)]

	# Finish timing initialization step
	end = timer()
	total_time += end - start

	# Append initialization residual error
	res_error.append(residual_error(X_unfold[0], norm_x, A[0], A[1], A[2]))
	time.append(total_time)
	if max_time == None:
		# Run ada_CPD
		for r in range(num_iterations):
			if (r+1) % 5000 == 0:
				print("iteration:", r)

			# Time start step
			start = timer()
			# Randomly select mode n to update.
			n = sample(3)
			# Generate sketching indices
			idx = generate_sketch_indices(B, total_col[n])
			# Update factor matrix
			update_factor_ada(X_unfold, A, G, idx, n, rank, eta, b, eps)
			# Time iteration step
			end = timer()
			total_time += end - start
			# Append error
			res_error.append(residual_error(X_unfold[0], norm_x, A[0], A[1], A[2]))
	else:
		while total_time < max_time:
			# Time start step
			start = timer()
			# Randomly select mode n to update.
			n = sample(3)
			# Generate sketching indices
			idx = generate_sketch_indices(B, total_col[n])
			# Update factor matrix
			update_factor_ada(X_unfold, A, G, idx, n, rank, eta, b, eps)
			# Time iteration step
			end = timer()
			total_time += end - start

			# Append error
			res_error.append(residual_error(X_unfold[0], norm_x, A[0], A[1], A[2]))
			time.append(total_time)
	return total_time, res_error, time