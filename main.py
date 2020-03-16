import matplotlib.pyplot as plt
import tensorly.kruskal_tensor as tl_kruskal
import tensorly.random as tl_rand
import numpy as np
from timeit import default_timer as timer

from BLOCK_SPG_CPD import bras_CPD
from BLOCK_SPG_CPD import ada_CPD

# # Set up
# for i in range(2):
# 	rank = 100

# 	# Each entry of the factor matrix is uniformly sampled from (0,1) and kruskal_tensor is used to from X
# 	F = tl_rand.random_kruskal((300,300,300), rank, full=False, random_state=np.random.RandomState(seed=i))
# 	X = tl_kruskal.kruskal_to_tensor(F)

# 	# Ill conditioned matrix

# 	# Hetero noise

# 	# Homo noise
	
# 	alphas = [0.1, 0.05, 0.01]
# 	B = 18
# 	beta = 10**-6
# 	num_iterations = 150000

# 	# Run bras_cpd update
# 	alpha = 0.1
# 	time, res_error = bras_CPD(F, X, rank, B, alpha, beta, num_iterations)

# 	# Save data
# 	s = "fixed_rand_error_bras_" + str(i) + ".txt"
# 	np.savetxt(s, res_error)
# 	print(time)

# Set up
for i in range(2):
    # Each entry of the factor matrix is uniformly sampled from (0,1) and kruskal_tensor is used to from X
    rank = 100
    F = tl_rand.random_kruskal((300,300,300), rank, full=False, random_state=np.random.RandomState(seed=i))
    X = tl_kruskal.kruskal_to_tensor(F)

    # Ill conditioned matrix

    # Hetero noise

    # Homo noise

    # Parameters
    B = 18
    eta = 1
    b = 10**-6
    eps = 0
    num_iterations = 75000

    # Run ada_cpd update
    time, res_error = ada_CPD(F, X, rank, B, eta, b, eps, num_iterations)

    # Save data
    s = "fixed_rand_error_ada_" + str(i) + ".txt"
    np.savetxt(s, res_error)
    print(time)