import numpy as np

def power_iteration(A, iter=100):
    b = np.random.rand(A.shape[1])

    for _ in range(iter):
        b_k1 = np.dot(A, b)
        b_k1_norm = 0
        for e in b_k1:
            b_k1_norm += e**2

        b_k1_norm = np.sqrt(b_k1_norm)
        b = b_k1 / b_k1_norm

    return np.dot(np.dot(A, b), b) / np.dot(b, b), b

def deflation(A, iter=100):
    n = A.shape[0]
    eig_vals = np.zeros(n)
    eig_vecs = np.zeros((n, n))

    for i in range(n):
        eig_val, eig_vec = power_iteration(A, iter)
        eig_vals[i] = eig_val
        eig_vecs[:, i] = eig_vec

        A = A - eig_val * np.outer(eig_vec, eig_vec)

    return eig_vals, eig_vecs

def svd(A):
    AT = A.T
    ATA = AT.dot(A)
    eig_vals, eig_vecs = deflation(ATA)

    sorted_indices = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_indices]
    eig_vecs = eig_vecs[:, sorted_indices]

    s = np.sqrt(eig_vals)

    V = eig_vecs

    U = A.dot(V) / s

    return U, s, V.T