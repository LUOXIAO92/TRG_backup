import gc
import sys
import time
import numpy as np
import cupy as cp

import numpy as np
import cupy as cp
import opt_einsum as oe

#sys.path.append('../')
from tensor_init.Phi4_model_real import gauss_hermite_quadrature as phi4_init

import trg.HOTRG_2d_QR as hotrg
import trg.TRG_2d as trg
import trg.ATRG_2d as atrg2d
from tensor_class.tensor_class import Tensor

def Phi4_2d_puretensor(U, VH, w, Dcut):
    T = oe.contract("a,ia,aj,ka,al->ijkl", w, VH[0], U[0], VH[1], U[1])
    del U, VH, w
    return T

def Phi4_2d_puretensor_atrg(s, w, U, VH, Dcut:int)->Tensor:
    T = oe.contract("a,a,ia,aj,ka,al->ijkl", s, w, VH[0], U[0], VH[1], U[1])
    
    T = cp.transpose(T, (3,0,2,1))
    T = cp.reshape(T, (Dcut*Dcut, Dcut*Dcut))
    U, s, VH = cp.linalg.svd(T)
    del T

    U  = U[:,:Dcut]
    s  = s[:Dcut]
    VH = VH[:Dcut,:]

    U  = cp.reshape(U , (Dcut,Dcut,Dcut))
    VH = cp.reshape(VH, (Dcut,Dcut,Dcut))
    T = Tensor(U, s, VH)
    del U, s, VH

    return T

def ln_Z_over_V(nth, mu02, lam, h, Dcut:int, XLOOPS:int, YLOOPS:int):
    phi4 = phi4_init(nth, mu02, lam, h, Dcut)
    U, VH, w, _ = phi4.phi4_init_tensor_component()

    #T = Phi4_2d_puretensor(U, VH, w, Dcut)
    ##T = cp.transpose(T    , axes=(0,2,1,3))
    ##T = oe.contract("a,ia,aj,ka,al->ijkl", exp, VH[0], U[0], VH[1], U[1])
    ##T, T0, ln_normfact, ln_normalized_factor_imp = trg.nearest_two_point_func_renorm(T, T, T, T, T, Dcut, XLOOPS, YLOOPS)
    #T, ln_normfact = hotrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS)
    
    #trace = oe.contract("aabb", T)
    #del T

    I = cp.ones_like(w)
    T = Phi4_2d_puretensor_atrg(I, w, U, VH, Dcut)
    T, ln_normfact = atrg2d.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS)
    trace = oe.contract("ija,a,aij", T.U, T.s, T.VH)

    I = oe.contract("iyx,yxj->ij", T.VH, T.U)
    print("norm(I)^2= {:} , Tr(I)= {:} , TrT= {:}".format(cp.linalg.norm(I)**2, cp.trace(I), trace))
    print("TrT={:}".format(trace.real))

    V = 2**(XLOOPS+YLOOPS)

    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V
    del ln_normfact

    print("lnZ={:.12e}".format((ln_ZoverV*V).real))

    return ln_ZoverV

def entanglement_entropy(nth, mu02, lam, h, Dcut:int, lA:int, lB:int, lt:int):
    #import trg.HOTRG_2d as hotrg
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    phi4 = phi4_init(nth, mu02, lam, h, Dcut)
    U, VH, w, _ = phi4.phi4_init_tensor_component()

    #T = Phi4_2d_puretensor(U, VH, w, Dcut)
    #TA, ln_normfact_A = hotrg.pure_tensor_renorm(T, Dcut, lA, lt)
    #TB, ln_normfact_B = TA, ln_normfact_A 

    I = cp.ones_like(w)
    T = Phi4_2d_puretensor_atrg(I, w, U, VH, Dcut)
    T, ln_normfact = atrg2d.pure_tensor_renorm(T, Dcut, lA, lt)
    trace = oe.contract("ija,a,aij", T.U, T.s, T.VH)
    I = oe.contract("iyx,yxj->ij", T.VH, T.U)
    print("norm(I)^2= {:} , Tr(I)= {:} , TrT= {:}".format(cp.linalg.norm(I)**2, cp.trace(I), trace))
    print("TrT={:}".format(trace.real))

    TB = oe.contract("Yxi,i,iyX->xXyY", T.U, T.s, T.VH)
    TA = TB
    ln_normfact_A = ln_normfact_B = ln_normfact

    TrTBTA = oe.contract("ijaa,jibb", TB, TA)
    print("contract, TrTBTA",TrTBTA)
    Dens_Mat = oe.contract("ijab,jicd->acbd", TB, TA) / TrTBTA
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(Dcut*Dcut, Dcut*Dcut))
    u_Dens_Mat, e_Dens_Mat, _ = cp.linalg.svd(Dens_Mat)
    print("dens_mat hermit err",cp.linalg.norm(Dens_Mat-cp.conj(Dens_Mat.T))/cp.linalg.norm(Dens_Mat))

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
    rho_A = oe.contract("aiaj->ij", Dens_Mat)
    _, e_A, _ = cp.linalg.svd(rho_A)
    SEE = -cp.sum(e_A * cp.log(e_A))
    print("rho_A hermit err",cp.linalg.norm(rho_A-cp.conj(rho_A.T))/cp.linalg.norm(rho_A))
    print("Tr(ρA)=", oe.contract("ii", rho_A))

    if get_sv == True:
        with open("{:}/densitymatrix_A_sv".format(sv_dir), "w") as svout:
            for ee in e_A:
               svout.write("{:.12e}\n".format(ee))

    
    #renyi
    N = 11
    Sn = cp.zeros(N, dtype=cp.complex128)
    n  = cp.array([i for i in range(1,N+1)], dtype=cp.float64)
    n[0] = 1/2
    for i in range(len(n)):
        print("rho_A^{:}".format(str(n[i])))
        Sn[i] = cp.sum(e_A**(n[i]))
    Sn = cp.log(Sn)
    Sn = Sn / (1-n)
    Sn = Sn.real
    
    sn_str = ''
    for sn in Sn:
        sn_str += "{:.12e} ".format(sn)
    sn_str += "\n" 
    print(sn_str)
    
    #Energy
    #E = -cp.sum(e * cp.log(e))/TrTBTA - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = -cp.sum(e_Dens_Mat * cp.log(e_Dens_Mat)) - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = E / (VA+VB)

    return ln_ZoverV, STE, SEE, E, sn_str