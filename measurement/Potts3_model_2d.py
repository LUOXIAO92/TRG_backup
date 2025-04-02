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

from tensor_init.Potts_model import infinite_density as Potts

def ln_Z_over_V(q, k, h, mu, Dcut:int, XLOOPS:int, YLOOPS:int, gilt_eps=0):
    
    potts = Potts(dim=2, Dcut=Dcut, q=q, k=k, h=h, mu=mu)

    rgscheme = os.environ['RGSCHEME']
    if rgscheme == 'hotrg':
        import trg.gilt_HOTRG_2d_QR as gilthotrg
        T = potts.cal_HOTRG_init_tensor()
        
        T, ln_normfact = gilthotrg.pure_tensor_renorm(T, Dcut, gilt_eps, XLOOPS, YLOOPS)
        trace = oe.contract("xxyy", T)

    elif rgscheme == 'atrg':
        import trg.ATRG_2d as atrg
        T = potts.cal_ATRG_init_tensor()
        T, ln_normfact = atrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS)
        trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
    else:
        raise ValueError("Only support hotrg")

    del T

    print()
    print("lnc_i/V=", ln_normfact)
    print()


    V = 2**(XLOOPS+YLOOPS)
    #print(ln_normfact)
    #print(trace)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    return ln_ZoverV



def entanglement_entropy(beta, h, Dcut:int, lA:int, lB:int, lt:int, gilt_eps=0):
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    T = init_tensor(beta, h)

    #import trg.TRG_2d_gilt as trg
    #T = contract("xXyY->xyXY", T)
    #TA, ln_normfact_A = trg.pure_tensor_renorm(T, Dcut, gilt_eps, lA, lt)
    #TA = contract("xyXY->xXyY", TA)
    #TB, ln_normfact_B = TA, ln_normfact_A 

    import trg.gilt_HOTRG_2d_QR as gilt_hotrg
    TA, ln_normfact_A = gilt_hotrg.pure_tensor_renorm(T, Dcut, gilt_eps, lA, lt)
    TB, ln_normfact_B = TA, ln_normfact_A 

    TrTBTA = contract("ijaa,jibb", TB, TA)
    print("contract, TrTBTA",TrTBTA)
    Dens_Mat = contract("ijab,jicd->acbd", TB, TA) / TrTBTA
    chi_1, chi_2, chi_3, chi_4 = Dens_Mat.shape[0], Dens_Mat.shape[1], Dens_Mat.shape[2], Dens_Mat.shape[3]
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(chi_1*chi_2, chi_3*chi_4))
    u_Dens_Mat, e_Dens_Mat, _ = cp.linalg.svd(Dens_Mat)
    print("dens_mat hermit err",cp.linalg.norm(Dens_Mat-cp.conj(Dens_Mat.T))/cp.linalg.norm(Dens_Mat))

    with open("{:}/densitymatrix.dat".format(OUTPUT_DIR), "w") as svout:
        emax = cp.max(e_Dens_Mat)
        e_dens_mat = e_Dens_Mat / emax
        svout.write("#ρmax={:.12e}\n".format(emax))
        for ee in e_dens_mat:
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
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(chi_1, chi_2, chi_3, chi_4))
    rho_A = contract("aiaj->ij", Dens_Mat)
    _, e_A, _ = cp.linalg.svd(rho_A)
    #e_A[e_A < 1e-100] = 1e-100
    SEE = -cp.sum(e_A * cp.log(e_A))
    print("rho_A hermit err",cp.linalg.norm(rho_A-cp.conj(rho_A.T))/cp.linalg.norm(rho_A))

    with open("{:}/densitymatrix_A.dat".format(OUTPUT_DIR), "w") as svout:
        emax = cp.max(e_A)
        e_a = e_A / emax
        svout.write("#emax={:.12e}\n".format(emax))
        for ee in e_a:
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


def entanglement_entropy_transferMatrix_method(beta, h, Dcut:int, lA:int, lB:int, lt:int, gilt_eps=0):
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    T = init_tensor(beta, h)

    import trg.gilt_HOTRG_2d_QR as gilt_hotrg
    TA, ln_normfact_A = gilt_hotrg.transfermatrix_renorm(T, Dcut, gilt_eps, lA, lt)
    TB, ln_normfact_B = TA, ln_normfact_A 

    TrTBTA = contract("ijaa,jibb", TB, TA)
    print("contract, TrTBTA",TrTBTA)
    Dens_Mat = contract("ijab,jicd->acbd", TB, TA) / TrTBTA
    chi_1, chi_2, chi_3, chi_4 = Dens_Mat.shape[0], Dens_Mat.shape[1], Dens_Mat.shape[2], Dens_Mat.shape[3]
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(chi_1*chi_2, chi_3*chi_4))
    u_Dens_Mat, e_Dens_Mat, _ = cp.linalg.svd(Dens_Mat)
    print("dens_mat hermit err",cp.linalg.norm(Dens_Mat-cp.conj(Dens_Mat.T))/cp.linalg.norm(Dens_Mat))

    with open("{:}/densitymatrix.dat".format(OUTPUT_DIR), "w") as svout:
        emax = cp.max(e_Dens_Mat)
        e_dens_mat = e_Dens_Mat / emax
        svout.write("#ρmax={:.12e}\n".format(emax))
        for ee in e_dens_mat:
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
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(chi_1, chi_2, chi_3, chi_4))
    rho_A = contract("aiaj->ij", Dens_Mat)
    _, e_A, _ = cp.linalg.svd(rho_A)
    #e_A[e_A < 1e-100] = 1e-100
    SEE = -cp.sum(e_A * cp.log(e_A))
    print("rho_A hermit err",cp.linalg.norm(rho_A-cp.conj(rho_A.T))/cp.linalg.norm(rho_A))

    with open("{:}/densitymatrix_A.dat".format(OUTPUT_DIR), "w") as svout:
        emax = cp.max(e_A)
        e_a = e_A / emax
        svout.write("#emax={:.12e}\n".format(emax))
        for ee in e_a:
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