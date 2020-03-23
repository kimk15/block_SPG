import pathlib
import matplotlib.pyplot as plt
import tensorly.kruskal_tensor as tl_kruskal
import tensorly.random as tl_rand
import numpy as np
import time

from BLOCK_SPG_CPD import bras_CPD
from BLOCK_SPG_CPD import ada_CPD


# Set up
for i in range(10):
    # Load in factor matrices
    a = r"C:\Users\Public\generated_data\ill_conditioned\collinear_matrix_A_"  + str(i) + ".txt"
    b = r"C:\Users\Public\generated_data\ill_conditioned\collinear_matrix_B_"  + str(i) + ".txt"
    c = r"C:\Users\Public\generated_data\ill_conditioned\collinear_matrix_C_"  + str(i) + ".txt"

    A = np.loadtxt(pathlib.Path(a))
    B = np.loadtxt(pathlib.Path(b))
    C = np.loadtxt(pathlib.Path(c))

    F = [A,B,C]
    X = tl_kruskal.kruskal_to_tensor(F)

    # Parameters
    B = 18
    eta = 1
    b = 10**-6
    eps = 0
    num_iterations = 0
    max_time = 600
    rank = 5

    # Run ada_cpd update
    total_time, res_error, time = ada_CPD(F, X, rank, B, eta, b, eps, num_iterations, max_time)

    # Save data
    res_file = "timed_ill_error_ada_" + str(i) + ".txt"
    time_file = "timed_ill_time_ada_" + str(i) + ".txt"

    np.savetxt(res_file, res_error)
    np.savetxt(time_file, time)
    print(total_time)