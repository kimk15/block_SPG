import matplotlib.pyplot as plt
import tensorly.kruskal_tensor as tl_kruskal
import tensorly.random as tl_rand
import numpy as np
import time

from BLOCK_SPG_CPD import bras_CPD
from BLOCK_SPG_CPD import ada_CPD

# Set up
rank = 100
A_true = tl_rand.random_kruskal((300,300,300), rank, full=False)
X = tl_kruskal.kruskal_to_tensor(A_true)
B = 20
b = 10**-6
eps = 0
eta = 1
num_iterations = 30*5000

# Run ada_CPD update
A, error, mse = ada_CPD(A_true, X, rank, B, eta, b, eps, num_iterations)

# Save data
np.save("cost_ada.txt", cost)
np.save("mse_ada.txt", mse)





# Set up
rank = 100
A_true = tl_rand.random_kruskal((300,300,300), rank, full=False)
X = tl_kruskal.kruskal_to_tensor(A_true)
alphas = [0.1, 0.05, 0.01]
B = 20
beta = 10**-6
num_iterations = 30*5000


# Run bras_cpd update
costs = []
mses = []
for alpha in alphas:
    A, error, mse = bras_CPD(A_true, X, rank, B, alpha, beta, num_iterations)
    costs.append(error)
    mses.append(mse)

# Save error/mse/A
np.save("cost_bras.txt", np.array(costs))
np.save("mse_bras.txt", np.array(mses))


