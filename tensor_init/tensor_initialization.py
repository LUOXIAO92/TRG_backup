import numpy as np
import cupy as cp
import opt_einsum as oe

from trg.gilt_HOTRG_2d_QR import __squeezer__, coarse_graining, __tensor_normalization__
from trg.TRG_2d_gilt import TRG_pure_tensor

def calinit(A:cp.ndarray, B:cp.ndarray, Dcut:int):
    """
    >>>                      j\ /k
    >>> input: A(B)_{ijkl} =   A(B)
    >>>                      i/ \l 
    >>>                       
    >>>                      |j
    >>> output: T_{ijkl} = i-T-k
    >>>                     l|
    >>> return: T
    """

    #ln_normal_fact = cp.zeros(2)
    #A, c = __tensor_normalization__(A)
    #ln_normal_fact[0] = cp.log(c) / 2**(count_totloop)

    #full rg: step1~step3

    #half rg: step1~step2
    #step1:
    #   |   |              |        |    
    #  -B---A-           /-B-\    /-A-\         |   |
    #   |   |   --->  --P0 | P1--P2 | P3--  =  -C---D-
    #  -A---B-           \-A-/    \-B-/         |   |
    #   |   |              |        |  
    A = cp.transpose(A, axes=(0,2,3,1))
    B = cp.transpose(B, axes=(0,2,3,1))
    P1, P2 = __squeezer__(A, B, A, B)
    P3, P0 = __squeezer__(B, A, B, A)
    C = coarse_graining(A, B, P0, P1)
    C = C.get()
    D = coarse_graining(B, A, P2, P3)
    D = D.get()
    del A, B, P0, P1, P2, P3
    C = cp.asarray(C)
    D = cp.asarray(D)

    #step2:
    #                |
    # |   |         /P0\       |
    #-C---D- ---> -C----D- = --T--
    # |   |         \P1/       |
    #                |
    C = cp.transpose(C, (2,3,0,1))
    D = cp.transpose(D, (2,3,0,1))
    P0, P1 = __squeezer__(C, D, D, C)
    T = coarse_graining(C, D)
    del C, D, P0, P1
    T = cp.transpose(T, (2,1,3,0))

    #half rg: step3
    #step3: 
    T = TRG_pure_tensor(T, Dcut, renormal_loop=2, gilt_eps=0.0)

    return T