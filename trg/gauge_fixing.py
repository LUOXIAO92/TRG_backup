import cupy as cp
import numpy as np
#import cupyx.scipy.linalg as LA
import scipy.linalg as LA
import opt_einsum as oe

def exphm(A:np.ndarray):
    """
    A: hermit matrix
    """
    e, u = np.linalg.eigh(A)
    e = np.diag(np.exp(e))
    expA = u @ (e @ np.conj(u.T))
    return expA

def reduced_density_matrix_2d(T:np.ndarray):
    """
    ρ00: ρ_{x,1}
    ρ01: ρ_{x,2}
    ρ10: ρ_{y,1}
    ρ11: ρ_{y,2}
    """
    rho = [[0 for j in range(2)] for i in range(2)]

    rho[0][0] = oe.contract("aYXy,bYXy->ab", T, np.conj(T))
    rho[0][1] = oe.contract("xYay,xYby->ab", T, np.conj(T))
    rho[1][0] = oe.contract("xaXy,xbXy->ab", T, np.conj(T))
    rho[1][1] = oe.contract("xYXa,xYXb->ab", T, np.conj(T))

    return rho

def apply_gauge_transformation_2d(T:np.ndarray, g:list):
    gx, gy = g
    gx_inv = np.linalg.inv(gx)
    gy_inv = np.linalg.inv(gy)
    Tnew = oe.contract("ijkl,xi,Yj,kX,ly", T, gx, gy, gx_inv, gy_inv)
    return Tnew

def optimization_error(rho:list, dim:int):
    Trrho = np.abs(np.trace(rho[0][0]))

    err_sqr = 0.0
    for k in range(dim):
        err_sqr += np.linalg.norm(rho[k][0] - rho[k][1].T)**2
    err_sqr /= Trrho

    return np.sqrt(err_sqr)

def gauge_optimization(T, learning_rate=None, max_iter=2000, eps=1e-8):
    legs_num = len(T.shape)
    assert legs_num % 2 == 0

    if legs_num == 4:
        rdm = reduced_density_matrix_2d
        apply = apply_gauge_transformation_2d
    else:
        raise ValueError(f"not support for {legs_num/2} dimension trg scheme")

    dim = legs_num // 2
    chis = T.shape[:dim]
    g = [np.diag(np.ones(chi)) for chi in chis]

    err_old = 1.0
    eta = 1 / (4*dim) if learning_rate == None else learning_rate
    for t in range(max_iter):
        T_t = apply(T, g)
        rho = rdm(T_t)

        err_new = optimization_error(rho, dim)

        if ((t % 20) == 0) and (t > 0):
            print(f"gauge fixing, iteration:{t}, error= {err_new:.6e}")

        if err_new < eps:
            print(f"gauge fixing, iteration:{t}, error= {err_new:.6e}")
            return g
        if (np.abs(err_old - err_new)/err_old < 1e-3) or (err_old < err_new):
            print(f"gauge fixing, iteration:{t-1}, error= {err_old:.6e}")
            print(f"gauge fixing, iteration:{t}, error= {err_new:.6e}")
            return g
        
        Trrho = np.trace(rho[0][0])
        for k in range(dim):
            h = -eta * (rho[k][0] - rho[k][1].T) / Trrho
            h = LA.expm(h)
            g[k] = h @ g[k]

        err_old = err_new


def gauge_fixing_2d(T:np.ndarray, learning_rate=None, max_iter=2000, eps=1e-8):
    """
    tensor T should be transposed to T_{x,y',x',y}
    """

    T = T.get()
    g = gauge_optimization(T, learning_rate, max_iter, eps)
    T = apply_gauge_transformation_2d(T, g)
    T = cp.asarray(T)
    return T
