import matplotlib.pyplot as plt
import tensorly.random as tl_rand
import numpy as np
import time

from BLOCK_SPG_CPD import bras_CPD
from BLOCK_SPG_CPD import ada_CPD

# Set up
rank = 10
X = tl_rand.random_kruskal((366,366,100), rank, full=True)
alphas = [0.1, 0.05, 0.01]
B = 20
beta = 10**-6
num_iterations = 100

# Run bras_cpd update
errors = []
for alpha in alphas:
    A, error = bras_CPD(X, rank, B, alpha, beta, num_iterations)
    errors.append(error)

# Plot out error
x = [i for i in range(len(error))]
plt.title('Residual error vs # iterations')
plt.xlabel('Iterations')
plt.ylabel('Residual Error')
plt.plot(x, no_sketch_error, label='No sketch + weight update')
plt.plot(x, sketch_no_weight_update_error, label='Sketch + w/o weight update')
plt.plot(x, sketch_with_weight_update_error, label='Sketch + weight update')
plt.legend(loc="upper left")
plt.figure()