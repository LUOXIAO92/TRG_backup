import os
import time
import numpy as np
import cupy as cp
from opt_einsum import contract

import sys
import configparser

sys.path.append('../')
from tensor_init.Potts_model import infinite_density as potts_model

import trg.HOTRG_2d_QR as hotrg

OUTPUT_DIR = os.environ['OUTPUT_DIR']
    

def ln_Z_over_V(q, k, h, Dcut:int, XLOOPS:int, YLOOPS:int, gilt_eps=0):
    Potts = potts_model(q, k, h, Dcut)
    u, vh, _ = Potts.tensor_component_2d_classical()
    #u  = cp.pad( u, ((0,0), (0,Dcut-q)))
    #vh = cp.pad(vh, ((0,Dcut-q), (0,0)))
    T = contract("ia,aj,ka,al->ijkl", vh, u, vh, u)

    #print(f"gilt_eps={gilt_eps}")
    #import trg.TRG_2d_gilt as trg
    #T = contract("xXyY->xyXY", T)
    #T, ln_normfact = trg.pure_tensor_renorm(T, Dcut, gilt_eps, XLOOPS, YLOOPS)
    #T = contract("xyXY->xXyY", T)

    import trg.gilt_HOTRG_2d_QR as gilt_hotrg
    T, ln_normfact = gilt_hotrg.pure_tensor_renorm(T, Dcut, gilt_eps, XLOOPS, YLOOPS)

    trace = contract("aabb", T)
    del T
    print("TrT=",trace)

    V = 2**(XLOOPS+YLOOPS)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    return ln_ZoverV

def entanglement_entropy(q, k, h, Dcut:int, lA:int, lB:int, lt:int, gilt_eps=0.0):
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    Potts = potts_model(q, k, h, Dcut)
    u, vh, _ = Potts.tensor_component_2d()
    #u  = cp.pad( u, ((0,0), (0,Dcut-q)), 'constant', constant_values=1e-16)
    #vh = cp.pad(vh, ((0,Dcut-q), (0,0)), 'constant', constant_values=1e-16)
    T = contract("ia,aj,ka,al->ijkl", vh, u, vh, u)

    rgscheme = os.environ['RGSCHEME']
    if rgscheme == 'trg':
        import trg.TRG_2d_gilt as trg
        T = contract("xXyY->xyXY", T)
        TA, ln_normfact_A = trg.pure_tensor_renorm(T, Dcut, gilt_eps, lA, lt)
        del T
        TA = contract("xyXY->xXyY", TA)
        TB, ln_normfact_B = TA, ln_normfact_A 

    elif rgscheme == 'hotrg':
        import trg.gilt_HOTRG_2d_QR as gilthotrg
        TA, ln_normfact_A = gilthotrg.pure_tensor_renorm(T, Dcut, gilt_eps, lA, lt)
        TB, ln_normfact_B = TA, ln_normfact_A

    chiA_x, chiA_X, chiA_y, chiA_Y = TA.shape
    chiB_x, chiB_X, chiB_y, chiB_Y = TB.shape

    TrTBTA = contract("ijaa,jibb", TB, TA)
    print("contract, TrTBTA",TrTBTA)
    Dens_Mat = contract("ijab,jicd->acbd", TB, TA) / TrTBTA
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(chiB_Y*chiA_Y, chiB_y*chiB_y))
    u_Dens_Mat, e_Dens_Mat, _ = cp.linalg.svd(Dens_Mat)
    print("ρ hermit err", cp.linalg.norm(Dens_Mat-cp.conj(Dens_Mat.T))/cp.linalg.norm(Dens_Mat))

    with open("{:}/densitymatrix.dat".format(OUTPUT_DIR), "w") as svout:
        for ee in e_Dens_Mat:
           svout.write("{:.12e}\n".format(ee))
        

    #lnZ/V
    VA = Nx_A*Nt
    VB = Nx_B*Nt
    ln_ZoverV = (VA*cp.sum(ln_normfact_A) + VB*cp.sum(ln_normfact_B) + cp.log(TrTBTA))/(VA+VB)

    #STE
    STE = - cp.sum( e_Dens_Mat * cp.log(e_Dens_Mat))

    #SEE
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(chiB_Y, chiA_Y, chiB_y, chiB_y))
    rho_A = contract("aiaj->ij", Dens_Mat)
    _, e_A, _ = cp.linalg.svd(rho_A)
    #e_A[e_A < 1e-100] = 1e-100
    SEE = -cp.sum(e_A * cp.log(e_A))
    print("ρA hermit err", cp.linalg.norm(rho_A-cp.conj(rho_A.T))/cp.linalg.norm(rho_A))

    with open("{:}/densitymatrix_A.dat".format(OUTPUT_DIR), "w") as svout:
        for ee in e_A:
           svout.write("{:.12e}\n".format(ee))

    #renyi
    N = 11
    Sn = cp.zeros(N, dtype=cp.complex128)
    n  = cp.array([i for i in range(1,N+1)], dtype=cp.float64)
    n[0] = 1/2
    for i in range(len(n)):
        Sn[i] = cp.sum(e_A**(n[i]))
    Sn = cp.log(Sn)
    Sn = Sn / (1-n)
    Sn = Sn.real
    
    sn_str = ''
    for sn in Sn:
        sn_str += "{:.12e} ".format(sn)
    sn_str += "\n" 
    print(sn_str)
    S2 = Sn[1]

    #Energy
    #E = -cp.sum(e * cp.log(e))/TrTBTA - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = -cp.sum(e_Dens_Mat * cp.log(e_Dens_Mat)) - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = E / (VA+VB)

    return ln_ZoverV, STE, SEE, E, sn_str

def entanglement_entropy_ATRG(q, k, h, Dcut:int, lA:int, lB:int, lt:int):
    
    from tensor_class.tensor_class import Tensor
    def to_tensor_gl(U1, VH1, U2, VH2, Dcut)->Tensor:
        T = (cp.pi / 8) * contract("ia,aj,ka,al->ijkl", VH1, U1, VH2, U2)
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
    
    import trg.ATRG_2d as atrg
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    Potts = potts_model(q, k, h, Dcut)
    u, vh, _ = Potts.tensor_component_2d()
    u  = cp.pad( u, ((0,0), (0,Dcut-q)))
    vh = cp.pad(vh, ((0,Dcut-q), (0,0)))
    T = to_tensor_gl(u, vh, u, vh, Dcut)

    TA, ln_normfact_A = atrg.pure_tensor_renorm(T, Dcut, lA, lt)
    TB, ln_normfact_B = TA, ln_normfact_A 

    #T = U_(xy'i) s_i VH_(ix'y)
    #Dens_Mat_(yB,yA,y'B,y'A) = U_(x y'B i) s_i VH_(i x' yB) U_(x' y'A j) s_j VH_(j x yA) 
    #        _(i  j  k   l)        a  k  α,   α,    α b  i,     b   l  β,   β,    β a j
    Dens_Mat = contract("akα,α,αbi,blβ,β,βaj->ijkl", TB.U, TB.s, TB.VH , TA.U, TA.s, TA.VH)
    TrTBTA = contract("iijj", Dens_Mat)
    print("contract, TrTBTA", TrTBTA)
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(Dcut*Dcut, Dcut*Dcut)) / TrTBTA
    u_Dens_Mat, e_Dens_Mat, _ = cp.linalg.svd(Dens_Mat)
    print("dens_mat hermit err",cp.linalg.norm(Dens_Mat-cp.conj(Dens_Mat.T))/cp.linalg.norm(Dens_Mat))

    with open("{:}/densitymatrix.dat".format(OUTPUT_DIR), "w") as svout:
        for ee in e_Dens_Mat:
           svout.write("{:.12e}\n".format(ee))
    
    #e_Dens_Mat[e_Dens_Mat<1e-100] = 1e-100
    
    
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
    #e_A[e_A < 1e-100] = 1e-100
    SEE = -cp.sum(e_A * cp.log(e_A))
    print("rho_A hermit err",cp.linalg.norm(rho_A-cp.conj(rho_A.T))/cp.linalg.norm(rho_A))

    with open("{:}/densitymatrix_A.dat".format(OUTPUT_DIR), "w") as svout:
        for ee in e_A:
           svout.write("{:.12e}\n".format(ee))

    
    #renyi
    N = 11
    Sn = cp.zeros(N, dtype=cp.complex128)
    n  = cp.array([i for i in range(1,N+1)], dtype=cp.float64)
    n[0] = 1/2
    for i in range(len(n)):
        Sn[i] = cp.sum(e_A**(n[i]))
    Sn = cp.log(Sn)
    Sn = Sn / (1-n)
    Sn = Sn.real
    
    sn_str = ''
    for sn in Sn:
        sn_str += "{:.12e} ".format(sn)
    sn_str += "\n" 
    print(sn_str)
    S2 = Sn[1]

    #Energy
    #E = -cp.sum(e * cp.log(e))/TrTBTA - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = -cp.sum(e_Dens_Mat * cp.log(e_Dens_Mat)) - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = E / (VA+VB)

    return ln_ZoverV, STE, SEE, E, sn_str

