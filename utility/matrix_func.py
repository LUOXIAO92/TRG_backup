import cupy as cp

def expm(A, is_hermitian = True):
    if is_hermitian:
        e, u = cp.linalg.eigh()