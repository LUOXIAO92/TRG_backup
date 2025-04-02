import os
import time
import numpy as np
import cupy as cp
import opt_einsum as oe
from opt_einsum import contract

import sys
import configparser

sys.path.append('../')
OUTPUT_DIR = os.environ["OUTPUT_DIR"]


def init_tensor(beta, h, Dcut, rgscheme):
    s = cp.asarray([1,-1])
    I = cp.asarray([1,1])
    M = beta * cp.einsum("i,j->ij", s, s) #+ 0.5 * h * (cp.einsum("i,j->ij", s, I) + cp.einsum("i,j->ij", I, s))
    site = cp.exp(h * s)
    M = cp.exp(M)
    print(M)
    #from utility.truncated_svd import svd
    #us, ss, svh = svd(M, shape=[[0], [1]], k=8, split=True)
    u, s, vh = cp.linalg.svd(M)
    us = oe.contract("ia,a->ia", u, cp.sqrt(s))
    svh = oe.contract("a,aj->aj", cp.sqrt(s), vh)
    T = contract("a,xa,aX,ya,aY,ta,aT->xXyYtT", site, svh, us, svh, us, svh, us)
    T = T.astype(complex)

    if rgscheme == "hotrg":
        return T
    elif rgscheme == "atrg":
        T = cp.transpose(T, (5,1,3,4,0,2))
        T = cp.reshape(T, (2*2*2, 2*2*2))
        u, s ,vh = cp.linalg.svd(T)
        u = cp.reshape(u, (2,2,2,8))
        vh = cp.reshape(vh, (8,2,2,2))

        from tensor_class.tensor_class import ATRG_Tensor as Tensor
        T = Tensor(u, s, vh, Dcut, 3, False, {})
        return T

def init_tensor_imp(beta, h, Dcut, rgscheme):
    σ = cp.asarray([1,-1])
    I = cp.asarray([1,1])
    M = beta * cp.einsum("i,j->ij", σ, σ) #+ 0.5 * h * (cp.einsum("i,j->ij", σ, I) + cp.einsum("i,j->ij", I, σ))
    M = cp.exp(M)
    site = cp.exp(h * σ)
    print(M)
    
    
    u, s, vh = cp.linalg.svd(M)
    us = oe.contract("ia,a->ia", u, cp.sqrt(s))
    svh = oe.contract("a,aj->aj", cp.sqrt(s), vh)

    T = contract("a,xa,aX,ya,aY,ta,aT->xXyYtT", site, svh, us, svh, us, svh, us)
    T = T.astype(complex)

    Timp = contract("a,a,xa,aX,ya,aY,ta,aT->xXyYtT", σ, site, svh, us, svh, us, svh, us)
    Timp = Timp.astype(complex)

    if rgscheme == "hotrg":
        return T, Timp
    elif rgscheme == "atrg":
        from tensor_class.tensor_class import ATRG_Tensor as Tensor
        T = cp.transpose(T, (5,1,3,4,0,2))
        T = cp.reshape(T, (2*2*2, 2*2*2))
        u, s ,vh = cp.linalg.svd(T)
        u = cp.reshape(u, (2,2,2,8))
        vh = cp.reshape(vh, (8,2,2,2))
        T = Tensor(u, s, vh, Dcut, 3, False, {})

        Timp = cp.transpose(Timp, (5,1,3,4,0,2))
        Timp = cp.reshape(Timp, (2*2*2, 2*2*2))
        uu, ss ,vhh = cp.linalg.svd(Timp)
        uu = cp.reshape(uu, (2,2,2,8))
        vhh = cp.reshape(vhh, (8,2,2,2))
        Timp = Tensor(uu, ss, vhh, Dcut, 3, True, {"T":0, "X":0, "Y":0})
        return T, Timp

def init_tensor2(beta, h):
    s = cp.asarray([1,-1])
    I = cp.asarray([1,1])

    E = - cp.einsum("i,j,k,l->ijkl", s,s,I,I) - cp.einsum("i,j,k,l->ijkl", I,s,s,I) \
        - cp.einsum("i,j,k,l->ijkl", I,I,s,s) - cp.einsum("i,j,k,l->ijkl", s,I,I,s)
    E *= 0.5
    
    T = np.zeros((2,2,2,2,
                  2,2,2,2))
    
    from itertools import product
    iter = product(range(2), range(2), range(2), range(2))
    for i,j,k,l in iter:
        T[i,j,j,k,k,l,l,i] = E[i,j,k,l]
    T = T.reshape((4,4,4,4))
    T = cp.exp(- beta * T)
    T = T.astype(complex)
    return T

def ln_Z_over_V(beta, h, Dcut:int, XLOOPS:int, YLOOPS:int, TLOOPS:int, gilt_eps=0):
    
    rgscheme = os.environ['RGSCHEME']
    if rgscheme == 'hotrg':
        import trg.HOTRG_3d_QR as gilthotrg
        T = init_tensor(beta, h, Dcut, rgscheme)
        T = {'Tensor': T, 'loc':{}, 'factor':{}}
        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "Z": TLOOPS}

        T = gilthotrg.pure_tensor(T, Dcut, TOT_RGSTEPS)
        ln_normfact = cp.asarray(list(T["factor"].values()))
        trace = oe.contract("xxyytt", T["Tensor"])
        
    elif rgscheme == 'atrg':
        import trg.ATRG_3d_new as atrg
        T = init_tensor(beta, h, Dcut, rgscheme)

        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "T": TLOOPS}
        T = atrg.pure_tensor(T, Dcut, TOT_RGSTEPS)

        trace = T.trace()
        ln_normfact = T.get_normalization_const()

    else:
        raise ValueError("Only support hotrg")

    del T

    print()
    print("lnc_i/V=", ln_normfact)
    print()


    V = 2**(XLOOPS+YLOOPS+TLOOPS)
    #print(ln_normfact)
    #print(trace)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    return ln_ZoverV


def magnetization(beta, h, Dcut:int, XLOOPS:int, YLOOPS:int, TLOOPS:int, gilt_eps=0):
    rgscheme = os.environ['RGSCHEME']

    if rgscheme == 'hotrg':
        import trg.HOTRG_3d_QR as gilthotrg
        Tpure, Timpure = init_tensor_imp(beta, h, Dcut, rgscheme)

        T  = {'Tensor': Tpure.copy(),   'loc':{}, 'factor':{}}
        T0 = {"Tensor": Timpure.copy(), 'loc'   : {"X":0, "Y":0, "Z":0}, 'factor': {}}

        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "Z": TLOOPS}
        T, T0 = gilthotrg.one_point_function(T, T0, Dcut, TOT_RGSTEPS)
        
        TrT  = oe.contract("xxyyzz", T["Tensor"])
        TrT00 = oe.contract("xxyyzz", T0["Tensor"])
        TrT01 = oe.contract("xxyyzz", T0["Tensor"])

        Tfactor  = cp.asarray(list(T["factor"].values()))
        T00factor = cp.asarray(list(T0["factor"].values()))
        T01factor = cp.asarray(list(T0["factor"].values()))
        fact00 = cp.exp(cp.sum(T00factor))
        P = cp.abs(fact00*TrT00/TrT)
        
    elif rgscheme == 'atrg':
        import trg.ATRG_3d_new as atrg
        T, Timp0 = init_tensor_imp(beta, h, Dcut, rgscheme)
        T, Timp1 = init_tensor_imp(beta, h, Dcut, rgscheme)

        T0 = [Timp0, Timp1]
        T0[0].loc = {"X":0, "Y":0, "T":0}
        T0[1].loc = {"X":0, "Y":0, "T":0}
        
        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "T": TLOOPS}
        T, T0 = atrg.one_point_function(T, T0, Dcut, TOT_RGSTEPS)
        #T0 = [T0, Timp1]

        TrT = T.trace()
        TrT00 = T0[0].trace()
        TrT01 = T0[1].trace()

        Tfactor   = T.get_normalization_const()
        T00factor = T0[0].get_normalization_const()
        T01factor = T0[1].get_normalization_const()
        fact00 = cp.exp(cp.sum(T00factor))
        fact01 = cp.exp(cp.sum(T01factor))
        P = cp.abs(fact00*TrT00/TrT)
        Pdag = cp.abs(fact01*TrT01/TrT)

    else:
        raise ValueError("Only support hotrg")

    del T

    print()
    print("factor of T=",  Tfactor )
    print("factor of T0[0]=", T00factor)
    print("factor of T0[1]=", T01factor)
    print()
    print(f"TrT=  {TrT:.12e}")
    print(f"TrT0[0]= {TrT00:.12e}")
    print(f"TrT0[1]= {TrT01:.12e}")
    print("<σ>=", P)

    V = 2**(XLOOPS+YLOOPS+TLOOPS)
    ln_ZoverV = cp.sum(Tfactor) + cp.log(TrT) / V

    return ln_ZoverV, P



def internal_energy(beta, h, Dcut:int, XLOOPS:int, YLOOPS:int, TLOOPS:int, gilt_eps=0):
    rgscheme = os.environ['RGSCHEME']

    if rgscheme == 'hotrg':
        import trg.HOTRG_3d_QR as gilthotrg
        Tpure, Timpure = init_tensor_imp(beta, h, Dcut, rgscheme)

        T  = {'Tensor': Tpure.copy(),   'loc':{}, 'factor':{}}
        T0 = {"Tensor": Timpure.copy(), 'loc'   : {"X":0, "Y":0, "Z":0}, 'factor': {}}
        Tn = {"Tensor": Timpure.copy(), 'loc'   : {"X":0, "Y":0, "Z":1}, 'factor': {}}

        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "Z": TLOOPS}
        T, T0 = gilthotrg.two_point_function(T, T0, Tn, Dcut, TOT_RGSTEPS)
        
        TrT  = oe.contract("xxyyzz", T["Tensor"])
        TrT0 = oe.contract("xxyyzz", T0["Tensor"])

        Tfactor  = cp.asarray(list(T["factor"].values()))
        T0factor = cp.asarray(list(T0["factor"].values()))
        fact0 = cp.exp(cp.sum(T0factor))
        twopointfunc = fact0*TrT0/TrT
        
    elif rgscheme == 'atrg':
        import trg.ATRG_3d_new as atrg
        T, T0 = init_tensor_imp(beta, h, Dcut, rgscheme)
        T, Tn = init_tensor_imp(beta, h, Dcut, rgscheme)

        T0.loc = {"X":0, "Y":0, "T":0}
        Tn.loc = {"X":0, "Y":0, "T":1}
        
        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "T": TLOOPS}
        T, T0 = atrg.two_point_function(T, T0, Tn, Dcut, TOT_RGSTEPS)

        TrT = T.trace()
        TrT0 = T0.trace()

        Tfactor  = T.get_normalization_const()
        T0factor = T0.get_normalization_const()
        fact0 = cp.exp(cp.sum(T0factor))
        twopointfunc = fact0*TrT0/TrT

    else:
        raise ValueError("Only support hotrg")

    del T

    print()
    print("factor of T=",  Tfactor )
    print("factor of T0[0]=", T0factor)
    print()
    print(f"TrT=  {TrT:.12e}")
    print(f"TrT0[0]= {TrT0:.12e}")
    print("<σ_0σ_1>=", twopointfunc)

    V = 2**(XLOOPS+YLOOPS+TLOOPS)
    ln_ZoverV = cp.sum(Tfactor) + cp.log(TrT) / V

    return ln_ZoverV, twopointfunc
