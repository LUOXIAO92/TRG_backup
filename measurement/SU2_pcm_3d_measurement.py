import time
import math
import cupy as cp
import opt_einsum as oe
from itertools import product

import sys
sys.path.append('../')

from tensor_init.SU2_pcm import SU2_pcm_initialize as SU2_pcm

import os
OUTPUT_DIR = os.environ['OUTPUT_DIR']


def ln_Z_over_V(su2pcm:SU2_pcm, XLOOPS:int, YLOOPS:int, TLOOPS:int, gilt_eps=0.0):
    Dcut = su2pcm.Dcut
    
    rgscheme = os.environ['RGSCHEME']
    if rgscheme == 'hotrg':
        import trg.HOTRG_3d_QR as gilthotrg
        T = su2pcm.cal_init_tensor(Dinit=Dcut)
        T, ln_normfact = gilthotrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS, TLOOPS)
        trace = oe.contract("xxyytt", T)

    elif rgscheme == 'atrg':
        import trg.ATRG_3d_new as atrg
        T = su2pcm.cal_ATRG_init_tensor(Dinit=Dcut, 
                                        k=Dcut, 
                                        p=Dcut, 
                                        q=Dcut,
                                        seed=12345)
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


def internal_energy(su2pcm:SU2_pcm, XLOOPS:int, YLOOPS:int, TLOOPS:int, gilt_eps=0.0):
    Dcut = su2pcm.Dcut
    
    rgscheme = os.environ['RGSCHEME']
    if rgscheme == 'hotrg':
        import trg.HOTRG_3d_QR as hotrg
        Tpure, J, w, I, Bs, As, Bt, At, U = su2pcm.cal_init_tensor(Dinit=Dcut, impure=True)
        
        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "Z": TLOOPS}
        two_point_func = cp.zeros(shape=(2,2), dtype=complex)
        for i,j in product(range(2), range(2)):

            T  = {"Tensor": Tpure.copy(), 
                  "factor": {}}

            T0 = {"Tensor": su2pcm.form_hotrg_tensor(J, w, U[i,j], Bs, As, Bt, At),
                  "loc"   : {"X":0, "Y":0, "Z":0},
                  "factor": {}}

            Tn = {"Tensor": su2pcm.form_hotrg_tensor(J, w, cp.conj(U[i,j]), Bs, As, Bt, At), 
                  "loc"   : {"X":0, "Y":0, "Z":1}}

            T, T0 = hotrg.two_point_function(T, T0, Tn, Dcut, TOT_RGSTEPS)

            print("factor of T0=", T0["factor"].values())

            TrT  = oe.contract("xxyyzz", T["Tensor"])
            TrT0 = oe.contract("xxyyzz", T0["Tensor"])

            Tfactor  = cp.asarray(list(T["factor"].values()))
            T0factor = cp.asarray(list(T0["factor"].values()))
            fact0 = cp.exp(cp.sum(T0factor))
            two_point_func[i,j] = fact0*TrT0/TrT

    elif rgscheme == 'atrg':
        import trg.ATRG_3d_new as atrg
        Tpure, J, w, U, I, Bs, As, Bt, At = su2pcm.cal_ATRG_init_tensor(Dinit=Dcut, 
                                                                        k=Dcut, 
                                                                        p=Dcut, 
                                                                        q=Dcut,
                                                                        seed=12345)
        
        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "Z": TLOOPS}
        two_point_func = cp.zeros(shape=(2,2), dtype=complex)
        for i,j in product(range(2), range(2)):

            T  = {"Tensor": Tpure.copy(), 
                  "factor": {}}

            T0 = {"Tensor": su2pcm.form_ATRG_tensor(J, w, U[i,j], Bs, As, Bt, At),
                  "loc"   : {"X":0, "Y":0, "Z":0},
                  "factor": {}}

            Tn = {"Tensor": su2pcm.form_tensor(J, w, cp.conj(U[i,j]), Bs, As, Bt, At), 
                  "loc"   : {"X":0, "Y":0, "Z":1}}

            T, T0 = atrg.two_point_function(T, T0, Tn, Dcut, TOT_RGSTEPS)

            print("factor of T0=", T0["factor"].values())

            TrT  = T.trace()
            TrT0 = T0.trace()

            Tfactor  = T.get_normalization_const()
            T0factor = T0.get_normalization_const()
            fact0 = cp.exp(cp.sum(T0factor))
            two_point_func[i,j] = fact0*TrT0/TrT

        else:
            raise ValueError("Only support hotrg")
    
    
    
    print()
    print("factor of T=", Tfactor)
    print()
    print("components of nearest 2-point function", two_point_func)

    V = 2**(XLOOPS+YLOOPS+TLOOPS)
    
    ln_ZoverV = cp.sum(Tfactor) + cp.log(TrT) / V
    e = cp.sum(two_point_func) / 2

    return ln_ZoverV, e