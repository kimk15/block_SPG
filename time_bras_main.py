import matplotlib.pyplot as plt
import tensorly.kruskal_tensor as tl_kruskal
import tensorly.random as tl_rand
import numpy as np
from timeit import default_timer as timer

from BLOCK_SPG_CPD import bras_CPD
from BLOCK_SPG_CPD import ada_CPD

# Set up
for i in range(10):
	rank = 100

	# Each entry of the factor matrix is uniformly sampled from (0,1) and kruskal_tensor is used to from X
	F = tl_rand.random_kruskal((300,300,300), rank, full=False, random_state=np.random.RandomState(seed=i))
	X = tl_kruskal.kruskal_to_tensor(F)

	# Ill conditioned matrix

	# Hetero noise

	# Homo noise
	
	alphas = [0.1, 0.05, 0.01]
	B = 18
	beta = 10**-6
	num_iterations = 0
	max_time = 600

	# Run bras_cpd update
	alpha = 0.1
	total_time, res_error, time = bras_CPD(F, X, rank, B, alpha, beta, num_iterations, max_time)

	# Save data
	s = "timed_rand_error_bras_" + str(i) + ".txt"
	t = "timed_rand_time_bras_" + str(i) + ".txt"
	np.savetxt(s, res_error)
	np.savetxt(t, time)
	print(total_time)