import os
import time
import numpy as np
import cupy as cp
import opt_einsum as oe
from opt_einsum import contract

import sys
import configparser

sys.path.append('../')

from tensor_init.SU2_2d_principle_chiral_moedl import gauss_legendre_quadrature as ti
import trg.HOTRG_2d_QR as hotrg

def ln_Z_over_V(beta, mu1, mu2, Dcut:int, XLOOPS:int, YLOOPS:int):
    U, VH, _, _, w = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    #U, VH, _, _, w = ti().__init_tensor_component_parts_finit_density_test__(beta, mu1, mu2, Dcut)
    
    T = ti().__init_pure_tensor__(w, U[0], VH[0], U[1], VH[1])
    T, ln_normfact = hotrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS)

    trace = contract("aabb", T)
    del T

    V = 2**(XLOOPS+YLOOPS)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V
    print(beta,mu1,mu2)
    return ln_ZoverV

from tensor_init.SU2_pcm import SU2_pcm_initialize as SU2_pcm
def ln_Z_over_V2(su2pcm:SU2_pcm, XLOOPS:int, YLOOPS:int, gilt_eps=0.0):
    Dcut = su2pcm.Dcut

    T = su2pcm.cal_init_tensor(Dinit=Dcut)
    
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


def lnZoV(K, beta, mu1, mu2, Dcut:int, XLOOPS:int, YLOOPS:int, seed:int):
    from tensor_init.SU2_2d_principle_chiral_moedl import Randomized_init
    import trg.gilt_HOTRG_2d_QR as gilthotrg
    SU2pcm = Randomized_init(K, beta, mu1, mu2, Dcut, seed)
    T = SU2pcm.initial_tensor()

    gilteps = 0.0
    T, ln_normfact = gilthotrg.pure_tensor_renorm(T, Dcut, gilteps, XLOOPS, YLOOPS)

    trace = contract("aabb", T)
    del T

    V = 2**(XLOOPS+YLOOPS)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V
    return ln_ZoverV


def renyi_entropy2(beta, mu1, mu2, Dcut:int, lA:int, lB:int, lt:int):
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    U, VH, _, _, w = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    T = ti().__init_pure_tensor__(w, U[0], VH[0], U[1], VH[1])
    
    #TA, ln_normfact_A = hotrg.renyi_entropy_renorm(T, Dcut, lA, lt)
    TA, ln_normfact_A = hotrg.pure_tensor_renorm(T, Dcut, lA, lt)
    #TB, ln_normfact_B = hotrg.renyi_entropy_renorm(T, Dcut, lB, lt)
    TB, ln_normfact_B = TA, ln_normfact_A 

    TrTBTA = contract("ijaa,jibb", TB, TA)
    print("contract, TrTBTA",TrTBTA)
    Dens_Mat = contract("ijab,jicd->acbd", TB, TA) / TrTBTA
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(Dcut*Dcut, Dcut*Dcut))
    u_Dens_Mat, e_Dens_Mat, _ = cp.linalg.svd(Dens_Mat)
    print("hermit err",cp.linalg.norm(Dens_Mat-cp.conj(Dens_Mat.T))/cp.linalg.norm(Dens_Mat))

    #M = contract("ijab,jicd->acbd", TB, TA) #/TrTBTA
    #M = cp.reshape(M, newshape=(Dcut*Dcut, Dcut*Dcut))
    #u, e, _ = cp.linalg.svd(M)
    #TrTBTA = cp.sum(e)
    #print("sum(e), TrTBTA",TrTBTA)
    ##print("e",e[:20])
    #print("hermit err",cp.linalg.norm(M-cp.conj(M.T))/cp.linalg.norm(M))
    
    sv_dir, get_sv = hotrg.__singularvalue_dir__(Dcut, lA, lt, get_sv=True)
    if get_sv == True:
        with open("{:}/densitymatrix_sv".format(sv_dir), "w") as svout:
            for ee in Dens_Mat:
               svout.write("{:.12e}\n".format(ee))
        

    #lnZ/V
    VA = Nx_A*Nt
    VB = Nx_B*Nt
    ln_ZoverV = (VA*cp.sum(ln_normfact_A) + VB*cp.sum(ln_normfact_B) + cp.log(TrTBTA))/(VA+VB)

    #STE
    #STE = - cp.sum( e_Dens_Mat * cp.log(e_Dens_Mat))/cp.sum(e_Dens_Mat) + cp.log(cp.sum(e_Dens_Mat))
    STE = - cp.sum( e_Dens_Mat * cp.log(e_Dens_Mat))
    #STE = - cp.sum( e * cp.log(e))/TrTBTA + cp.log(TrTBTA)

    #SEE
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(Dcut, Dcut, Dcut, Dcut))
    rho_A = contract("aiaj->ij", Dens_Mat)
    _, e_A, _ = cp.linalg.svd(rho_A)
    SEE = -cp.sum(e_A * cp.log(e_A))

    #M = cp.reshape(M, newshape=(Dcut, Dcut, Dcut, Dcut))
    #M_A = contract("aiaj->ij", M)
    #_, e_A, _ = cp.linalg.svd(M_A)
    ##e_A[e_A < 1e-100] = 1e-100
    #SEE = -cp.sum(e_A * cp.log(e_A))/TrTBTA + cp.log(TrTBTA)
    ##print("e_A",e_A[:20])

    if get_sv == True:
        with open("{:}/densitymatrix_A_sv".format(sv_dir), "w") as svout:
            for ee in e_A:
               svout.write("{:.12e}\n".format(ee))

    #renyi
    N = 2
    Sn = cp.zeros(N, dtype=cp.complex128)
    n  = cp.array([i for i in range(2,N+2)], dtype=cp.int64)
    for i in range(0,N):
        #print("rho_A^{:}".format(i+2))
        #Sn[i] = cp.trace(cp.linalg.matrix_power(rho_A, i+2)) 
        Sn[i] = cp.sum(e_A**(i+2))
    Sn = cp.log(Sn)
    Sn = Sn / (1-n)
    Sn = Sn.real.get()
    #from scipy import optimize
    #def f(x,a,b):
    #    return a*x+b
    #n = n.get()
    #popt, pcov = optimize.curve_fit(f,n,Sn)
    #S1[lT] = f(1, popt[0], popt[1])
    S2 = Sn[0]

    #Energy
    #E = -cp.sum(e * cp.log(e))/TrTBTA - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = -cp.sum(e_Dens_Mat * cp.log(e_Dens_Mat)) - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = E / (VA+VB)

    return ln_ZoverV, STE, SEE, S2, E