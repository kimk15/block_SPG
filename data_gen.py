import numpy as np
from numpy.linalg import norm

# """
# Generate data depending on parameter given:
# """
# def generate_data(shape, rank, t, seed, congruence=None):
# 	F, X = None
# 	if t == 'rand':
# 		F = tl_rand.random_kruskal(shape, rank, full=False, random_state=np.random.RandomState(seed=seed))
# 		X = tl_kruskal.kruskal_to_tensor(F)
# 	elif t == 'collinear':
# 		F = generate_collinear_factors(shape, rank, seed, congruence)

"""
Generates collinear factors
"""
def generate_collinear_factors(shape, rank, congruence):
	# Grab tensor shape
	I, J, K = shape

	# Generate A, B, C
	A = generate_collinear_matrix(I, rank, congruence)
	B = generate_collinear_matrix(J, rank, congruence)
	C = generate_collinear_matrix(K, rank, congruence)

	return [A, B, C]

"""
Generates a collinear factor matrix
"""
def generate_collinear_matrix(I,R,congruence):
    # Generate F matrix using cholesky
    F = generate_F((R,R), congruence)
    # Generate P
    P = generate_P((I, R))

    return P @ F

"""
Generates A: Entries are generated from standard normal distribution
"""
def generate_A(size, seed):
	return np.random.standard_normal(size)	

"""
Generate F: F is generated as stated in PARAFAC2 - PART I by kiers, berge, bro.
"""
def generate_F(size, congruence):
	# Fill an array with congruence
	A = np.ones(size) * congruence
	# Fill with one
	np.fill_diagonal(A, 1)
	# Compute choelesky decomposition
	F = np.linalg.cholesky(A)

	return F.T

"""
Generates P:
	Sample from standard normal -> orthnormalize column using QR
"""
def generate_P(size):
	# Sample from standard normal distribution
	P = np.random.standard_normal(size)
	#Orthnormalize columns
	Q, _ = np.linalg.qr(np.random.standard_normal(size))

	return Q

"""
congruence(): computes congruence between two vectors
"""
def congruence(x, y):
	return np.dot(x,y)/(norm(x)*norm(y))
