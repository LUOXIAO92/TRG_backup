import time
import cupy as cp
import opt_einsum as oe
from mpi4py import MPI

import sys
sys.path.append('../')

from tensor_init.SU2_pure_gauge import SU2_pure_gauge as SU2_gauge

import os
OUTPUT_DIR = os.environ['OUTPUT_DIR']


def ln_Z_over_V(su2gauge:SU2_gauge, XLOOPS:int, YLOOPS:int, comm:MPI.Intercomm, gilt_eps=0.0):
    rank = comm.Get_rank()
    Dcut = su2gauge.Dcut
    N = su2gauge.Kt * su2gauge.Ka * su2gauge.Kb
    K = su2gauge.Ka
    T = su2gauge.plaquette_tensor(Dcut, chunk=(N,K,K), legs_to_hosvd=[0])
    
    if rank == 0:
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
    else:
        ln_ZoverV = None


    return ln_ZoverV