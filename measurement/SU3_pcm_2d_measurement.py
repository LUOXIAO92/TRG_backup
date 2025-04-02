import time
import cupy as cp
import opt_einsum as oe

import sys
sys.path.append('../')

from tensor_init.SU3_pcm import SU3_pcm_initialize as SU3_pcm

import os
OUTPUT_DIR = os.environ['OUTPUT_DIR']


def ln_Z_over_V(su3pcm:SU3_pcm, XLOOPS:int, YLOOPS:int, gilt_eps=0.0):
    Dcut = su3pcm.Dcut

    T = su3pcm.cal_init_tensor(Dinit=Dcut)
    
    rgscheme = os.environ['RGSCHEME']
    if rgscheme == 'trg':
        import trg.TRG_2d_gilt as trg
        T = oe.contract("xXyY->xyXY", T)
        T, ln_normfact = trg.pure_tensor_renorm(T, Dcut, gilt_eps, XLOOPS, YLOOPS)
        T = oe.contract("xyXY->xXyY", T)

    elif rgscheme == 'hotrg':
        import trg.gilt_HOTRG_2d_QR as gilthotrg
        T, ln_normfact = gilthotrg.pure_tensor_renorm(T, Dcut, gilt_eps, XLOOPS, YLOOPS)
        
    trace = oe.contract("aabb", T)
    del T

    V = 2**(XLOOPS+YLOOPS)
    #print(ln_normfact)
    #print(trace)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    return ln_ZoverV