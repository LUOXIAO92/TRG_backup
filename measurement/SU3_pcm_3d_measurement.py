import time
import cupy as cp
import opt_einsum as oe

import sys
sys.path.append('../')

from tensor_init.SU3_pcm import SU3_pcm_initialize as SU3_pcm

import os
OUTPUT_DIR = os.environ['OUTPUT_DIR']


def ln_Z_over_V(su3pcm:SU3_pcm, XLOOPS:int, YLOOPS:int, TLOOPS:int, gilt_eps=0.0):
    Dcut = su3pcm.Dcut
    
    rgscheme = os.environ['RGSCHEME']
    if rgscheme == 'hotrg':
        import trg.HOTRG_3d as gilthotrg
        T = su3pcm.cal_init_tensor(Dinit=Dcut, legs="xXyYtT")
        T, ln_normfact = gilthotrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS, TLOOPS)
        trace = oe.contract("xxyytt", T)
    elif rgscheme == 'atrg':
        import trg.ATRG_3d_new as atrg
        T = su3pcm.cal_ATRG_init_tensor(Dinit=Dcut, 
                                        k=Dcut, 
                                        p=Dcut, 
                                        q=Dcut,
                                        seed=12345)
        T, ln_normfact = atrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS, TLOOPS)
        trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
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