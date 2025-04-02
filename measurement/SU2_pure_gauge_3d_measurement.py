import time
import cupy as cp
import opt_einsum as oe
from itertools import product
from functools import reduce
from operator import mul

from mpi4py import MPI

import sys
sys.path.append('../')

from tensor_init.SU2_pure_gauge import SU2_pure_gauge as SU2_gauge
from tensor_class.tensor_class import ATRG_Tensor as Tensor

import os
OUTPUT_DIR = os.environ['OUTPUT_DIR']


def ln_Z_over_V(su2gauge:SU2_gauge, XLOOPS:int, YLOOPS:int, TLOOPS:int, comm:MPI.Intercomm, gilt_eps=0.0):
    Dcut = su2gauge.Dcut
    
    rgscheme = os.environ['RGSCHEME']

    if rgscheme == 'hotrg':
        #import trg.HOTRG_3d_QR as gilthotrg
        #T = su2gauge.cal_hotrg_init_tensor(Dinit=Dcut)
        #T  = {"Tensor": T, "factor": {}}
        #
        #TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "Z": TLOOPS}
        #
        #T = gilthotrg.pure_tensor(T, Dcut, TOT_RGSTEPS)
        #ln_normfact = cp.asarray(list(T["factor"].values()))
        #trace = oe.contract("xxyytt", T["Tensor"])
        raise ValueError("Only support atrg!")
        
    elif rgscheme == 'atrg':
        import trg.ATRG_3d_new as atrg
        chi_plaquette     = Dcut
        chi_atrgtensor    = Dcut
        chunk_plaquette   = (reduce(mul, su2gauge.Ks), su2gauge.Ks[1], su2gauge.Ks[1])
        chunk_environment = (1, )
        legs_to_hosvd     = [0]
        truncate_eps      = 1e-10
        degeneracy_eps    = 0
        chunk_atrg_rsvd   = (4, 4)
        n_oversample      = 0
        n_iters           = 0
        seed_atrg_rsvd    = 1234
        verbose           = True
        U, S, VH = su2gauge.atrg_tensor(chi_plaquette     = chi_plaquette    ,
                                        chi_atrgtensor    = chi_atrgtensor   ,
                                        chunk_plaquette   = chunk_plaquette  ,
                                        chunk_environment = chunk_environment,
                                        legs_to_hosvd     = legs_to_hosvd    ,
                                        truncate_eps      = truncate_eps     ,
                                        degeneracy_eps    = degeneracy_eps   ,
                                        chunk_atrg_rsvd   = chunk_atrg_rsvd  ,
                                        n_oversample      = n_oversample     ,
                                        n_iters           = n_iters          ,
                                        seed_atrg_rsvd    = seed_atrg_rsvd   ,
                                        verbose           = verbose           )
        
        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "T": TLOOPS}
        
        T = None
        if comm.Get_rank() == 0:
            T = Tensor(U, S, VH, Dcut=Dcut, dim=3, is_impure=False, loc={})
            T = atrg.pure_tensor(T, Dcut, TOT_RGSTEPS)
            trace = T.trace()
            ln_normfact = T.get_normalization_const()

        del U, S, VH
    else:
        raise ValueError("Only support hotrg")

    del T

    ln_ZoverV = None
    if comm.Get_rank() == 0:
        print()
        print("lnc_i/V=", ln_normfact)
        print()


        V = 2**(XLOOPS+YLOOPS+TLOOPS)
        #print(ln_normfact)
        #print(trace)
        ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    return ln_ZoverV


def internal_energy(su3sm:SU2_gauge, XLOOPS:int, YLOOPS:int, TLOOPS:int, gilt_eps=0.0):
    Dcut = su3sm.Dcut
    
    LOCN = [{"X":0, "Y":0, "Z":1}, {"X":1, "Y":0, "Z":0}, {"X":0, "Y":1, "Z":0}]

    
    rgscheme = os.environ['RGSCHEME']
    if rgscheme == 'hotrg':
        import trg.HOTRG_3d_QR as trg
        import trg.HOTRG_3d_QR as hotrg
        Tpure, J, w, I, Bs, As, Bt, At, TrU = su3sm.cal_hotrg_init_tensor(Dinit=Dcut, impure=True)
        
        T  = {"Tensor": Tpure.copy(), 
          "factor": {}}

        T0 = {"Tensor": su3sm.form_hotrg_tensor(J, w, TrU, Bs, As, Bt, At),
              "loc"   : {"X":0, "Y":0, "Z":0},
              "factor": {}}

        Tn = {"Tensor": su3sm.form_hotrg_tensor(J, w, cp.conj(TrU), Bs, As, Bt, At), 
              "loc"   : {"X":0, "Y":0, "Z":1}}

        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "Z": TLOOPS}
        T, T0 = hotrg.two_point_function(T, T0, Tn, Dcut, TOT_RGSTEPS)


        TrT  = oe.contract("xxyyzz", T["Tensor"])
        TrT0 = oe.contract("xxyyzz", T0["Tensor"])

        Tfactor  = cp.asarray(list(T["factor"].values()))
        T0factor = cp.asarray(list(T0["factor"].values()))
        fact0 = cp.exp(cp.sum(T0factor))
        two_point_func = fact0*TrT0/TrT

    elif rgscheme == 'atrg':
        import trg.ATRG_3d_new as atrg
        T ,J, w, TrU, I, Bs, As, Bt, At = su3sm.cal_ATRG_init_tensor(Dinit=Dcut, k=Dcut, impure=True, p=Dcut, q=Dcut, seed=12345)

        T0 = su3sm.form_ATRG_tensor(J, w, TrU, Bs, As, Bt, At, k=Dcut, 
                                    is_impure=True, p=Dcut, q=Dcut, seed=45678)
        T0.loc = {"X":0, "Y":0, "T":0}
        
        Tn = su3sm.form_ATRG_tensor(J, w, cp.conj(TrU), Bs, As, Bt, At, k=Dcut, 
                                    is_impure=True, p=Dcut, q=Dcut, seed=78910)
        Tn.loc = {"X":0, "Y":0, "T":1}

        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "T": TLOOPS}
        T, T0 = atrg.two_point_function(T, T0, Tn, Dcut, TOT_RGSTEPS)

        TrT = T.trace()
        TrT0 = T0.trace()

        Tfactor  = T.get_normalization_const()
        T0factor = T0.get_normalization_const()
        fact0 = cp.exp(cp.sum(T0factor))
        two_point_func = fact0*TrT0/TrT

        #Xt, Xx, Xy = atrg.cal_X(T, save=False)

    else:
        raise ValueError("Only support hotrg and atrg")
    
        
    print()
    print("factor of T=",  Tfactor )
    print("factor of T0=", T0factor)
    print()
    print(f"TrT=  {TrT:.12e}")
    print(f"TrT0= {TrT0:.12e}")
    print("<TrU(0,0,0)TrU†(0,0,1)>=", two_point_func)

    V = 2**(XLOOPS+YLOOPS+TLOOPS)
    ln_ZoverV = cp.sum(Tfactor) + cp.log(TrT) / V
    e = cp.sum(two_point_func)

    return ln_ZoverV, e

def field_expected_value(su3sm:SU2_gauge, XLOOPS:int, YLOOPS:int, TLOOPS:int, gilt_eps=0.0):
    Dcut = su3sm.Dcut
    
    LOCN = [{"X":0, "Y":0, "Z":1}, {"X":1, "Y":0, "Z":0}, {"X":0, "Y":1, "Z":0}]

    
    rgscheme = os.environ['RGSCHEME']
    if rgscheme == 'hotrg':
        import trg.HOTRG_3d_QR as trg
        import trg.HOTRG_3d_QR as hotrg
        Tpure, J, w, I, Bs, As, Bt, At, TrU = su3sm.cal_hotrg_init_tensor(Dinit=Dcut, impure=True)
        
        T  = {"Tensor": Tpure.copy(), 
          "factor": {}}

        T0 = {"Tensor": su3sm.form_hotrg_tensor(J, w, TrU, Bs, As, Bt, At),
              "loc"   : {"X":0, "Y":0, "Z":0},
              "factor": {}}


        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "Z": TLOOPS}
        T, T0 = hotrg.one_point_function(T, T0, Dcut, TOT_RGSTEPS)

        TrT  = oe.contract("xxyyzz", T["Tensor"])
        TrT0 = oe.contract("xxyyzz", T0["Tensor"])

        Tfactor  = cp.asarray(list(T["factor"].values()))
        T0factor = cp.asarray(list(T0["factor"].values()))
        fact0 = cp.exp(cp.sum(T0factor))
        P = fact0*TrT0/TrT
        Pdag = 0.0

    elif rgscheme == 'atrg':
        import trg.ATRG_3d_new as atrg
        T ,J, w, TrU, I, Bs, As, Bt, At = su3sm.cal_ATRG_init_tensor(Dinit=Dcut, k=Dcut, impure=True, p=Dcut, q=Dcut, seed=12345)

        T0 = [su3sm.form_ATRG_tensor(J, w, TrU, Bs, As, Bt, At, k=Dcut, 
                                     is_impure=True, p=Dcut, q=Dcut, seed=45678), 
              su3sm.form_ATRG_tensor(J, w, cp.conj(TrU), Bs, As, Bt, At, k=Dcut, 
                                     is_impure=True, p=Dcut, q=Dcut, seed=56789)]
        T0[0].loc = {"X":0, "Y":0, "T":0}
        T0[1].loc = {"X":0, "Y":0, "T":0}
        
        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "T": TLOOPS}
        T, T0 = atrg.one_point_function(T, T0, Dcut, TOT_RGSTEPS)

        TrT = T.trace()
        TrT00 = T0[0].trace()
        TrT01 = T0[1].trace()

        Tfactor   = T.get_normalization_const()
        T00factor = T0[0].get_normalization_const()
        T01factor = T0[1].get_normalization_const()
        fact00 = cp.exp(cp.sum(T00factor))
        fact01 = cp.exp(cp.sum(T01factor))
        P = fact00*TrT00/TrT
        Pdag = fact01*TrT01/TrT

    else:
        raise ValueError("Only support hotrg and atrg")
    
        
    print()
    print("factor of T=",  Tfactor )
    print("factor of T0[0]=", T00factor)
    print("factor of T0[1]=", T00factor)
    print()
    print(f"TrT=  {TrT:.12e}")
    print(f"TrT0[0]= {TrT00:.12e}")
    print(f"TrT0[1]= {TrT01:.12e}")
    print("<TrU(0,0,0)>=", P)
    print("<TrU†(0,0,0)>=", Pdag)

    V = 2**(XLOOPS+YLOOPS+TLOOPS)
    ln_ZoverV = cp.sum(Tfactor) + cp.log(TrT) / V

    return ln_ZoverV, P, Pdag

def number_density(su3sm:SU2_gauge, XLOOPS:int, YLOOPS:int, TLOOPS:int, gilt_eps=0.0):
    Dcut = su3sm.Dcut
    
    LOCN = [{"X":0, "Y":0, "Z":1}, {"X":1, "Y":0, "Z":0}, {"X":0, "Y":1, "Z":0}]

    
    rgscheme = os.environ['RGSCHEME']
    if rgscheme == 'hotrg':
        import trg.HOTRG_3d_QR as trg
        import trg.HOTRG_3d_QR as hotrg
        Tpure, J, w, I, Bs, As, Bt, At, TrU = su3sm.cal_hotrg_init_tensor(Dinit=Dcut, impure=True)
        
        T  = {"Tensor": Tpure.copy(), 
          "factor": {}}

        T0 = {"Tensor": su3sm.form_hotrg_tensor(J, w, TrU, Bs, As, Bt, At),
              "loc"   : {"X":0, "Y":0, "Z":0},
              "factor": {}}

        Tn = {"Tensor": su3sm.form_hotrg_tensor(J, w, cp.conj(TrU), Bs, As, Bt, At), 
              "loc"   : {"X":0, "Y":0, "Z":1}}

        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "Z": TLOOPS}
        T, T0 = hotrg.two_point_function(T, T0, Tn, Dcut, TOT_RGSTEPS)


        TrT  = oe.contract("xxyyzz", T["Tensor"])
        TrT0 = oe.contract("xxyyzz", T0["Tensor"])

        Tfactor  = cp.asarray(list(T["factor"].values()))
        T0factor = cp.asarray(list(T0["factor"].values()))
        fact0 = cp.exp(cp.sum(T0factor))
        two_point_func = fact0*TrT0/TrT

    elif rgscheme == 'atrg':
        import trg.ATRG_3d_new as atrg
        T ,J, w, TrU, I, Bs, As, Bt, At = su3sm.cal_ATRG_init_tensor(Dinit=Dcut, k=Dcut, impure=True, p=Dcut, q=Dcut, seed=12345)

        T0 = su3sm.form_ATRG_tensor(J, w, TrU, Bs, As, Bt, At, k=Dcut, 
                                    is_impure=True, p=Dcut, q=Dcut, seed=45678)
        T0.loc = {"X":0, "Y":0, "T":0}
        
        Tn = su3sm.form_ATRG_tensor(J, w, cp.conj(TrU), Bs, As, Bt, At, k=Dcut, 
                                    is_impure=True, p=Dcut, q=Dcut, seed=78910)
        Tn.loc = {"X":0, "Y":0, "T":1}

        TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "T": TLOOPS}
        T, T0 = atrg.two_point_function(T, T0, Tn, Dcut, TOT_RGSTEPS)

        TrT = T.trace()
        TrT0 = T0.trace()

        Tfactor  = T.get_normalization_const()
        T0factor = T0.get_normalization_const()
        fact0 = cp.exp(cp.sum(T0factor))
        two_point_func = fact0*TrT0/TrT

        #Xt, Xx, Xy = atrg.cal_X(T, save=False)

    else:
        raise ValueError("Only support hotrg and atrg")
    
        
    print()
    print("factor of T=",  Tfactor )
    print("factor of T0=", T0factor)
    print()
    print(f"TrT=  {TrT:.12e}")
    print(f"TrT0= {TrT0:.12e}")
    print("<TrU(0,0,0)TrU†(0,0,1)>=", two_point_func)

    V = 2**(XLOOPS+YLOOPS+TLOOPS)
    ln_ZoverV = cp.sum(Tfactor) + cp.log(TrT) / V
    e = cp.sum(two_point_func)

    return ln_ZoverV, e