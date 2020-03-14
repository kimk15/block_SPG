import matplotlib.pyplot as plt
import tensorly.kruskal_tensor as tl_kruskal
import tensorly.random as tl_rand
import numpy as np
from timeit import default_timer as timer

from BLOCK_SPG_CPD import bras_CPD
from BLOCK_SPG_CPD import ada_CPD

# Set up
for i in range(2):
	rank = 100
	F = tl_rand.random_kruskal((300,300,300), rank, full=False, random_state=np.random.RandomState(seed=i))
	X = tl_kruskal.kruskal_to_tensor(F)
	alphas = [0.1, 0.05, 0.01]
	B = 18
	beta = 10**-6
	num_iterations = 100

	# Run bras_cpd update
	alpha = 0.1
	time, res_error = bras_CPD(F, X, rank, B, alpha, beta, num_iterations)

	# Save data
	np.savetxt("fixed_rand_error_bras_0.txt", res_error)
	print(time)