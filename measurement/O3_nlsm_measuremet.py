import os
import time
import numpy as np
import cupy as cp
from opt_einsum import contract

import sys
import configparser

sys.path.append('../')
from tensor_init.O3_2d_nonlinear_sigma_model import gauss_legendre_quadrature as ti

import trg.gilt_HOTRG_2d_QR as hotrg

OUTPUT_DIR = os.environ["OUTPUT_DIR"]
    

def ln_Z_over_V(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    #T = ti.init_pure_tensor_finit_density(beta, mu1, mu2, Dcut)
    #U, VH, _, _, w = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    U1, VH1, U2, VH2, sin_theta, _, _, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)

    gilt_eps = float(os.environ['GILT_EPS'])
    rgscheme = os.environ['RGSCHEME']
    if rgscheme == 'trg':
        import trg.TRG_2d_gilt as trg
        T = contract("xXyY->xyXY", T)
        T, ln_normfact = trg.pure_tensor_renorm(T, Dcut, gilt_eps, XLOOPS, YLOOPS)
        T = contract("xyXY->xXyY", T)

    elif rgscheme == 'hotrg':
        import trg.gilt_HOTRG_2d_QR as gilt_hotrg
        T, ln_normfact = gilt_hotrg.pure_tensor_renorm(T, Dcut, gilt_eps, XLOOPS, YLOOPS)

    #T, ln_normfact = hotrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS)

    trace = contract("aabb", T)
    del T
    print("TrT=",trace)

    V = 2**(XLOOPS+YLOOPS)
    #print(ln_normfact)
    #print(trace)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    return ln_ZoverV

def entanglement_entropy(beta, mu, Dcut:int, lA:int, lB:int, lt:int, gilt_eps=1e-7):
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    U1, VH1, U2, VH2, sin_theta, _, _, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)

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

def entanglement_entropy_Y2X1(beta, mu, Dcut:int, lA:int, lB:int, lt:int, gilt_eps=1e-7):
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    U1, VH1, U2, VH2, sin_theta, _, _, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)

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
        TA, ln_normfact_A = gilthotrg.pure_tensor_renorm_Y2X1(T, Dcut, gilt_eps, lA, lt)
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

def entanglement_entropy_transferMatrix_method(beta, mu, Dcut:int, lA:int, lB:int, lt:int, gilt_eps=1e-7):
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    U1, VH1, U2, VH2, sin_theta, _, _, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)

    rgscheme = os.environ['RGSCHEME']
    if rgscheme == 'hotrg':
        import trg.gilt_HOTRG_2d_QR as gilthotrg
        TA, ln_normfact_A = gilthotrg.transfermatrix_renorm(T, Dcut, gilt_eps, lA, lt)
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

def correlation_length(beta, mu, Dcut:int, lx:int, lt:int):
    Nx = int(2**lx)
    Nt = int(2**lt)
    V = Nx * Nt

    U1, VH1, U2, VH2, sin_theta, _, _, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)

    import trg.gilt_HOTRG_2d_QR as gilthotrg
    T, ln_normfact = gilthotrg.pure_tensor_renorm(T, Dcut, lx, lt)

    TrT = contract("iijj", T)
    print("contract, TrT",TrT)

    print("normalization factor:")
    for lnc in ln_normfact:
        print("{:.12e}".format(lnc))
    print()

    #lnZ/V
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(TrT) / V

    #x direction
    Dens_Mat_x = contract("xXyy->xX", T) / TrT
    u_Dens_Mat_x, e_Dens_Mat_x, _ = cp.linalg.svd(Dens_Mat_x)
    print("dens_mat_x hermit err",cp.linalg.norm(Dens_Mat_x-cp.conj(Dens_Mat_x.T))/cp.linalg.norm(Dens_Mat_x))

    with open("{:}/densitymatrix_x.dat".format(OUTPUT_DIR), "w") as svout:
        for ee in e_Dens_Mat_x:
           svout.write("{:.12e}\n".format(ee))

    correlation_length_x = Nx / cp.log((e_Dens_Mat_x[0]/e_Dens_Mat_x[1]))

    #STE_x
    STE_x = - cp.sum( e_Dens_Mat_x * cp.log(e_Dens_Mat_x))

    #t direction
    Dens_Mat_t = contract("xxyY->yY", T) / TrT
    u_Dens_Mat_t, e_Dens_Mat_t, _ = cp.linalg.svd(Dens_Mat_t)
    print("dens_mat_y hermit err",cp.linalg.norm(Dens_Mat_t-cp.conj(Dens_Mat_t.T))/cp.linalg.norm(Dens_Mat_t))

    with open("{:}/densitymatrix_t.dat".format(OUTPUT_DIR), "w") as svout:
        for ee in e_Dens_Mat_t:
           svout.write("{:.12e}\n".format(ee))
    
    correlation_length_t = Nt / cp.log((e_Dens_Mat_t[0]/e_Dens_Mat_t[1]))

    #STE_t
    STE_t = - cp.sum( e_Dens_Mat_t * cp.log(e_Dens_Mat_t))

    #E0
    E0 =  - cp.sum(ln_normfact) - cp.log(e_Dens_Mat_t[0]) / V

    return ln_ZoverV, STE_x, correlation_length_x, STE_t, correlation_length_t, E0

def entanglement_entropy_and_correlation_length(beta, mu, Dcut:int, lA:int, lB:int, lt:int, gilt_eps=1e-7):
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    U1, VH1, U2, VH2, sin_theta, _, _, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)

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
    #print(sn_str)
    S2 = Sn[1]

    #Energy
    #E = -cp.sum(e * cp.log(e))/TrTBTA - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = -cp.sum(e_Dens_Mat * cp.log(e_Dens_Mat)) - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = E / (VA+VB)

    #E0
    E0 = - cp.log(e_Dens_Mat[0]) - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E0 = E0 / (VA+VB)

    print("len(e_Dens_Mat)", len(e_Dens_Mat))
    #correlation length
    xi = Nt / cp.log((e_Dens_Mat[0]/e_Dens_Mat[1]))

    return ln_ZoverV, STE, SEE, E0, E, xi, sn_str

def correlation_length_TRG(beta, mu, Dcut:int, lx:int, lt:int):
    import trg.TRG_2d as trg

    Nx = int(2**lx)
    Nt = int(2**lt)
    V = Nx * Nt

    U1, VH1, U2, VH2, sin_theta, _, _, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
    T = cp.transpose(T, axes=(0,2,1,3))
    T, ln_normfact = trg.pure_tensor_renorm(T, Dcut, lx, lt)
    T = cp.transpose(T, axes=(0,2,1,3))

    TrT = contract("iijj", T)
    print("contract, TrT",TrT)

    print("normalization factor:")
    for lnc in ln_normfact:
        print("{:.12e}".format(lnc))
    print()

    #lnZ/V
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(TrT) / V

    #x direction
    Dens_Mat_x = contract("xXyy->xX", T) / TrT
    u_Dens_Mat_x, e_Dens_Mat_x, _ = cp.linalg.svd(Dens_Mat_x)
    print("dens_mat_x hermit err",cp.linalg.norm(Dens_Mat_x-cp.conj(Dens_Mat_x.T))/cp.linalg.norm(Dens_Mat_x))

    with open("{:}/densitymatrix_x.dat".format(OUTPUT_DIR), "w") as svout:
        for ee in e_Dens_Mat_x:
           svout.write("{:.12e}\n".format(ee))

    correlation_length_x = Nx / cp.log((e_Dens_Mat_x[0]/e_Dens_Mat_x[1]))

    #STE_x
    STE_x = - cp.sum( e_Dens_Mat_x * cp.log(e_Dens_Mat_x))

    #t direction
    Dens_Mat_t = contract("xxyY->yY", T) / TrT
    u_Dens_Mat_t, e_Dens_Mat_t, _ = cp.linalg.svd(Dens_Mat_t)
    print("dens_mat_y hermit err",cp.linalg.norm(Dens_Mat_t-cp.conj(Dens_Mat_t.T))/cp.linalg.norm(Dens_Mat_t))

    with open("{:}/densitymatrix_t.dat".format(OUTPUT_DIR), "w") as svout:
        for ee in e_Dens_Mat_t:
           svout.write("{:.12e}\n".format(ee))
    
    correlation_length_t = Nt / cp.log((e_Dens_Mat_t[0]/e_Dens_Mat_t[1]))

    #STE_t
    STE_t = - cp.sum( e_Dens_Mat_t * cp.log(e_Dens_Mat_t))

    #E0
    E0 = - cp.sum(ln_normfact) - cp.log(e_Dens_Mat_t[0]) / V

    return ln_ZoverV, STE_x, correlation_length_x, STE_t, correlation_length_t, E0

def entanglement_entropy_and_correlation_length_TRG(beta, mu, Dcut:int, lA:int, lB:int, lt:int):
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    U1, VH1, U2, VH2, sin_theta, _, _, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    TA = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)

    #import trg.gilt_HOTRG_2d_QR as gilthotrg
    import trg.TRG_2d as trg
    TA = cp.transpose(TA, axes=(0,2,1,3))
    TA, ln_normfact_A = trg.pure_tensor_renorm(TA, Dcut, lA, lt)
    TA = cp.transpose(TA, axes=(0,2,1,3))
    TB, ln_normfact_B = TA, ln_normfact_A 

    TrTBTA = contract("ijaa,jibb", TB, TA)
    print("contract, TrTBTA",TrTBTA)
    Dens_Mat = contract("ijab,jicd->acbd", TB, TA) / TrTBTA
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(Dcut*Dcut, Dcut*Dcut))
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
    #print(sn_str)
    S2 = Sn[1]

    #Energy
    #E = -cp.sum(e * cp.log(e))/TrTBTA - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = -cp.sum(e_Dens_Mat * cp.log(e_Dens_Mat)) - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = E / (VA+VB)

    #E0
    E0 = - cp.log(e_Dens_Mat[0]) - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E0 = E0 / (VA+VB)

    #correlation length
    xi = Nt / cp.log((e_Dens_Mat[0]/e_Dens_Mat[1]))

    return ln_ZoverV, STE, SEE, E0, E, xi, sn_str

def entanglement_entropy_ATRG(beta, mu, Dcut:int, lA:int, lB:int, lt:int):
    
    from tensor_class.tensor_class import Tensor
    def to_tensor_gl(U1, VH1, U2, VH2, sin_theta, wa, wb, Dcut)->Tensor:
        T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
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

    U1, VH1, U2, VH2, sin_theta, _, _, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = to_tensor_gl(U1, VH1, U2, VH2, sin_theta, wa, wb, Dcut)

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

    #sv_dir, get_sv = hotrg.__singularvalue_dir__(Dcut, lA, lt, get_sv=True)
    #if get_sv == True:
    #    with open("{:}/densitymatrix_sv".format(sv_dir), "w") as svout:
    #        for ee in Dens_Mat:
    #           svout.write("{:.12e}\n".format(ee))
    
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

    #M = cp.reshape(M, newshape=(Dcut, Dcut, Dcut, Dcut))
    #M_A = contract("aiaj->ij", M)
    #_, e_A, _ = cp.linalg.svd(M_A)
    ##e_A[e_A < 1e-100] = 1e-100
    #SEE = -cp.sum(e_A * cp.log(e_A))/TrTBTA + cp.log(TrTBTA)
    ##print("e_A",e_A[:20])

    #if get_sv == True:
    #    with open("{:}/densitymatrix_A_sv".format(sv_dir), "w") as svout:
    #        for ee in e_A:
    #           svout.write("{:.12e}\n".format(ee))

    
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
    #from scipy import optimize
    #def f(x,a,b):
    #    return a*x+b
    #n = n.get()
    #popt, pcov = optimize.curve_fit(f,n,Sn)
    #S1[lT] = f(1, popt[0], popt[1])
    S2 = Sn[1]

    #Energy
    #E = -cp.sum(e * cp.log(e))/TrTBTA - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = -cp.sum(e_Dens_Mat * cp.log(e_Dens_Mat)) - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = E / (VA+VB)

    return ln_ZoverV, STE, SEE, E, sn_str


def entanglement_entropy2(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    U1, VH1, U2, VH2, sin_theta, _, _, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
    
    ln_ZoV, SEE, ln_emax = hotrg.entanglement_entropy_renorm(T, Dcut, XLOOPS, YLOOPS)

    return ln_ZoV, SEE, ln_emax


def particle_number(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    U1, VH1, U2, VH2, sin_theta, _, phi, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    import trg.gilt_HOTRG_2d_QR as hotrg

    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
    Timp0 = (cp.pi / 8) * contract("a,a,b,b,iab,abj,kab,abl->ijkl", wa, sin_theta**2, wb, exp_piphi, VH1, U1, VH2, U2)
    Timp1 = (cp.pi / 8) * contract("a,a,b,b,iab,abj,kab,abl->ijkl", wa, sin_theta**2, wb, exp_miphi, VH1, U1, VH2, U2)
    #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
    trace1 = contract("aabb", T)
    trace_imp1 = contract("aabb", Timp0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
    Timp0 = (cp.pi / 8) * contract("a,a,b,b,iab,abj,kab,abl->ijkl", wa, sin_theta**2, wb, exp_miphi, VH1, U1, VH2, U2)
    Timp1 = (cp.pi / 8) * contract("a,a,b,b,iab,abj,kab,abl->ijkl", wa, sin_theta**2, wb, exp_piphi, VH1, U1, VH2, U2)
    #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
    trace2 = contract("aabb", T)
    trace_imp2 = contract("aabb", Timp0)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace2) / V
    n = (beta/2) * (cp.exp(mu)*normfact_imp1*trace_imp1/trace1 - cp.exp(-mu)*normfact_imp2*trace_imp2/trace2)

    return lnZoV, n

def particle_number_TRG(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d as trg
    U1, VH1, U2, VH2, sin_theta, _, phi, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    #exp_piphi = cp.sin(phi)
    #exp_miphi = cp.cos(phi+1j*mu)

    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
    Timp0 = (cp.pi / 8) * contract("a,a,b,b,iab,abj,kab,abl->ijkl", wa, sin_theta**2, wb, exp_piphi, VH1, U1, VH2, U2)
    Timp1 = (cp.pi / 8) * contract("a,a,b,b,iab,abj,kab,abl->ijkl", wa, sin_theta**2, wb, exp_miphi, VH1, U1, VH2, U2)

    T     = cp.transpose(T    , axes=(0,2,1,3))
    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
    
    trace1 = contract("abab", T)
    trace_imp1 = contract("abab", Timp0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    #exp_miphi = cp.cos(phi)
    #exp_piphi = cp.sin(phi+1j*mu)

    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
    Timp0 = (cp.pi / 8) * contract("a,a,b,b,iab,abj,kab,abl->ijkl", wa, sin_theta**2, wb, exp_miphi, VH1, U1, VH2, U2)
    Timp1 = (cp.pi / 8) * contract("a,a,b,b,iab,abj,kab,abl->ijkl", wa, sin_theta**2, wb, exp_piphi, VH1, U1, VH2, U2)
    
    T     = cp.transpose(T    , axes=(0,2,1,3))
    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
    
    trace2 = contract("abab", T)
    trace_imp2 = contract("abab", Timp0)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace2) / V
    n = (beta/2) * (cp.exp(mu)*normfact_imp1*trace_imp1/trace1 - cp.exp(-mu)*normfact_imp2*trace_imp2/trace2)
    #n = (beta*1j) * (normfact_imp1*trace_imp1/trace1 - normfact_imp2*trace_imp2/trace2) #test

    return lnZoV, n

def internal_energy_0(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    U1, VH1, U2, VH2, sin_theta, cos_theta, phi, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, 0, Dcut)

    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    s0 = [cp.ndarray]*3
    s1 = [cp.ndarray]*3

    I = cp.ones(len(exp_piphi), dtype=cp.float64)
    s0[0] = contract("a,b->ab", cos_theta, I)
    s1[0] = contract("a,b->ab", cos_theta, I)
    s0[1] = contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    s1[1] = contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s0[2] = contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s1[2] = contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)

    #exp_ptheta = cos_theta + 1j*sin_theta
    #exp_mtheta = cos_theta - 1j*sin_theta
    #s0[0] = contract("a,b->ab", exp_ptheta, I) / cp.sqrt(2)
    #s1[0] = contract("a,b->ab", exp_mtheta, I) / cp.sqrt(2)
    #s0[1] = contract("a,b->ab", exp_mtheta, exp_piphi) / 2
    #s1[1] = contract("a,b->ab", exp_ptheta, exp_miphi) / 2
    #s0[2] = contract("a,b->ab", exp_ptheta, exp_miphi) / 2
    #s1[2] = contract("a,b->ab", exp_mtheta, exp_piphi) / 2
    
    two_point_y = 0.0
    #part=cp.zeros(3, dtype=cp.complex128)
    for i in range(3):
        T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
        Timp0 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s0[i], wb, VH1, U1, VH2, U2)
        Timp1 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s1[i], wb, VH1, U1, VH2, U2)
        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    
        trace = contract("aabb", T)
        trace_imp = contract("aabb", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))

        #part[i] = normfact_imp * trace_imp / trace
        two_point_y += normfact_imp * trace_imp / trace

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    energy = 4-2*(2*two_point_y)

    return lnZoV, energy#, part

def internal_energy_0_TRG(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d_2 as trg
    U1, VH1, U2, VH2, sin_theta, cos_theta, phi, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, 0, Dcut)
    
    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    s0 = [cp.ndarray]*3
    s1 = [cp.ndarray]*3

    I = cp.ones(len(exp_piphi), dtype=cp.float64)
    s0[0] = contract("a,b->ab", cos_theta, I)
    s1[0] = contract("a,b->ab", cos_theta, I)
    s0[1] = contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    s1[1] = contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s0[2] = contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s1[2] = contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    print(cp.sum(s0[0]))

    two_point_y = 0.0
    part=cp.zeros(3, dtype=cp.complex128)
    for i in range(3):
        T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
        Timp0 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s0[i], wb, VH1, U1, VH2, U2)
        Timp1 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s1[i], wb, VH1, U1, VH2, U2)
    
        T     = cp.transpose(T    , axes=(0,2,1,3))
        Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
        Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))
    
        T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, T, T, Timp1, Dcut, XLOOPS, YLOOPS)
    
        trace = contract("abab", T)
        trace_imp = contract("abab", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    
        part[i] = normfact_imp * trace_imp / trace
        two_point_y += normfact_imp * trace_imp / trace
    
    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    energy = 4-2*(2*two_point_y)

    #exp_ptheta = cos_theta + 1j*sin_theta
    #exp_mtheta = cos_theta - 1j*sin_theta
    #s0[0] = contract("a,b->ab", exp_ptheta, I) / cp.sqrt(2)
    #s1[0] = contract("a,b->ab", exp_mtheta, I) / cp.sqrt(2)
    #s0[1] = contract("a,b->ab", exp_mtheta, exp_piphi) / 2
    #s1[1] = contract("a,b->ab", exp_ptheta, exp_miphi) / 2
    #s0[2] = contract("a,b->ab", exp_mtheta, exp_miphi) / 2
    #s1[2] = contract("a,b->ab", exp_ptheta, exp_piphi) / 2
    #
    #two_point_y = 0.0
    #for i in range(3):
    #    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
    #    Timp0 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s0[i], wb, VH1, U1, VH2, U2)
    #    Timp1 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s1[i], wb, VH1, U1, VH2, U2)
    #
    #    T     = cp.transpose(T    , axes=(0,2,1,3))
    #    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
    #    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))
    #
    #    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
    #
    #    trace = contract("abab", T)
    #    trace_imp = contract("abab", Timp0)
    #    normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    #
    #    two_point_y += normfact_imp * trace_imp / trace
    #
    #V = 2**(XLOOPS+YLOOPS)
    #lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    #energy = -2*(two_point_y)

    return lnZoV, energy

def internal_energy(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    U1, VH1, U2, VH2, sin_theta, cos_theta, phi, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    s0 = [cp.ndarray]*3
    s1 = [cp.ndarray]*3

    I = cp.ones(len(exp_piphi), dtype=cp.float64)
    s0[0] = contract("a,b->ab", cos_theta, I)
    s1[0] = contract("a,b->ab", cos_theta, I)
    s0[1] = cp.exp(mu/2)*contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    s1[1] = cp.exp(mu/2)*contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s0[2] = cp.exp(-mu/2)*contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s1[2] = cp.exp(-mu/2)*contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    two_point_y = 0.0
    for i in range(3):
        T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
        Timp0 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s0[i], wb, VH1, U1, VH2, U2)
        Timp1 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s1[i], wb, VH1, U1, VH2, U2)
        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    
        trace = contract("aabb", T)
        trace_imp = contract("aabb", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    
        two_point_y += normfact_imp * trace_imp / trace

    s0[1] = contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    s1[1] = contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s0[2] = contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s1[2] = contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    two_point_x = 0.0
    for i in range(3):
        T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
        Timp0 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s0[i], wb, VH1, U1, VH2, U2)
        Timp1 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s1[i], wb, VH1, U1, VH2, U2)
        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 1, 0, Dcut, XLOOPS, YLOOPS)
    
        trace = contract("aabb", T)
        trace_imp = contract("aabb", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    
        two_point_x += normfact_imp * trace_imp / trace

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    energy = 4-2*(two_point_y+two_point_x)

    return lnZoV, energy

def internal_energy_TRG(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d as trg
    U1, VH1, U2, VH2, sin_theta, cos_theta, phi, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    s0 = [cp.ndarray]*3
    s1 = [cp.ndarray]*3

    I = cp.ones(len(exp_piphi), dtype=cp.float64)
    s0[0] = contract("a,b->ab", cos_theta, I)
    s1[0] = contract("a,b->ab", cos_theta, I)
    s0[1] = cp.exp(mu/2)*contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    s1[1] = cp.exp(mu/2)*contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s0[2] = cp.exp(-mu/2)*contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s1[2] = cp.exp(-mu/2)*contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    two_point_y = 0.0
    for i in range(3):
        T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
        Timp0 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s0[i], wb, VH1, U1, VH2, U2)
        Timp1 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s1[i], wb, VH1, U1, VH2, U2)

        T     = cp.transpose(T    , axes=(0,2,1,3))
        Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
        Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

        T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
        #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    
        trace = contract("abab", T)
        trace_imp = contract("abab", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    
        two_point_y += normfact_imp * trace_imp / trace


    s0[1] = contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    s1[1] = contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s0[2] = contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s1[2] = contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    two_point_x = 0.0
    for i in range(3):
        T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
        Timp0 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s0[i], wb, VH1, U1, VH2, U2)
        Timp1 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s1[i], wb, VH1, U1, VH2, U2)

        T     = cp.transpose(T    , axes=(0,2,1,3))
        Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
        Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

        T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, T, T, Timp1, Dcut, XLOOPS, YLOOPS)
        #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 1, 0, Dcut, XLOOPS, YLOOPS)
    
        trace = contract("abab", T)
        trace_imp = contract("abab", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    
        two_point_x += normfact_imp * trace_imp / trace

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    energy = 4-2*(two_point_y+two_point_x)

    return lnZoV, energy


def two_point_function(beta, mu, x, y, Dcut:int, XLOOPS:int, YLOOPS:int):
    U1, VH1, U2, VH2, sin_theta, cos_theta, phi, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    s0 = [cp.ndarray]*3
    s1 = [cp.ndarray]*3

    I = cp.ones(len(exp_piphi), dtype=cp.float64)
    s0[0] = contract("a,b->ab", cos_theta, I)
    s1[0] = contract("a,b->ab", cos_theta, I)
    s0[1] = contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    s1[1] = contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s0[2] = contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s1[2] = contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    s0sn = 0.0
    for i in range(3):
        T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
        Timp0 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s0[i], wb, VH1, U1, VH2, U2)
        Timp1 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s1[i], wb, VH1, U1, VH2, U2)
        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp1, Timp0, x, y, Dcut, XLOOPS, YLOOPS)
    
        trace = contract("aabb", T)
        trace_imp = contract("aabb", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    
        s0sn += normfact_imp * trace_imp / trace

    #s0s0 = 0.0
    #for i in range(3):
    #    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
    #    Timp0 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s0[i], wb, VH1, U1, VH2, U2)
    #    Timp1 = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
    #    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, x, y, Dcut, XLOOPS, YLOOPS)
    #
    #    trace = contract("aabb", T)
    #    trace_imp = contract("aabb", Timp0)
    #    normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    #
    #    s0s0 += normfact_imp * trace_imp / trace

    two_point_func = s0sn #- s0s0**2

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    return lnZoV, two_point_func


from tensor_class.tensor_class import Tensor
def to_tensor_gl2(U1, VH1, U2, VH2, sin_theta, wa, wb, Dcut, si=None)->Tensor:

    if si is None:
        T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
    else:
        T = (cp.pi / 8) * contract("a,a,b,ab,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, si, VH1, U1, VH2, U2)
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

    return T
    
def to_tensor_gl(U1, VH1, U2, VH2, sin_theta, wa, wb, Dcut, si=None)->Tensor:

    #pure tensor
    T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
    TrT = contract("xxyy", T)
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

    if si is None:
        return T

    else:
        Timp = (cp.pi / 8) * contract("a,a,b,ab,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, si, VH1, U1, VH2, U2)
        TrTimp = contract("xxyy", Timp)
        Timp = cp.transpose(Timp, (3,0,2,1))
        simp = contract("YxyX,Yxi,jyX->ij", Timp, cp.conj(U), cp.conj(VH))
        Timp = Tensor(U, simp, VH)
        
        print(f"TrT={TrT}, TrTimp={TrTimp}")
        norms = cp.linalg.norm(s)
        normsimp = cp.linalg.norm(simp)
        print(f"||s||={norms}, ||simp||={normsimp}")
        return T, Timp

def internal_energy_0_TRG2(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d as trg
    import trg.ATRG_2d as atrg
    U1, VH1, U2, VH2, sin_theta, cos_theta, phi, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, 0, Dcut)

    exp_pithe = cos_theta+1j*sin_theta
    exp_mithe = cos_theta-1j*sin_theta
    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    s0 = cp.zeros((6,len(wa),len(wa)), dtype=cp.complex128)
    s1 = cp.zeros((6,len(wa),len(wa)), dtype=cp.complex128)

    I = cp.ones(len(wa), dtype=cp.complex128)
    s1[0] = contract("a,b->ab", exp_pithe, I)
    s1[1] = contract("a,b->ab", exp_mithe, I)
    s1[2] = contract("a,b->ab", exp_pithe, exp_miphi)
    s1[3] = contract("a,b->ab", exp_mithe, exp_miphi)
    s1[4] = contract("a,b->ab", exp_pithe, exp_piphi)
    s1[5] = contract("a,b->ab", exp_mithe, exp_piphi)
    s0 = cp.conj(s1)

    D = cp.zeros((3,6),dtype=cp.complex128)
    D[0] = cp.asarray([1/2,1/2,0,0,0,0])
    D[1] = cp.asarray([0,0,-1j/2,1j/2,0,0]) / cp.sqrt(2)
    D[2] = cp.asarray([0,0,0,0,-1j/2,1j/2]) / cp.sqrt(2)
    D = cp.conj(D.T) @ D

    two_point_y = 0.0
    #part=cp.zeros(3, dtype=cp.complex128)
    for i in range(6):
        for j in range(6):
            if cp.abs(D[i,j]) > 1e-6:
                print(f"i={i}, j={j}")
                #T     = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
                #Timp0 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s0[i], wb, VH1, U1, VH2, U2)
                #Timp1 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s1[j], wb, VH1, U1, VH2, U2)
                #
                #T     = cp.transpose(T    , axes=(0,2,1,3))
                #Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
                #Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

                #T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, T, T, Timp1, Dcut, XLOOPS, YLOOPS)
                #trace = contract("abab", T)
                #trace_imp = contract("abab", Timp0)

                #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
                #trace = contract("aabb", T)
                #trace_imp = contract("aabb", Timp0)

                #T = to_tensor_gl(U1, VH1, U2, VH2, sin_theta, wa, wb, Dcut)
                T, Timp0 = to_tensor_gl(U1, VH1, U2, VH2, sin_theta, wa, wb, Dcut, s0[i])
                T, Timp1 = to_tensor_gl(U1, VH1, U2, VH2, sin_theta, wa, wb, Dcut, s1[j])
                T, Timp0, ln_normfact, ln_normfact_imp = atrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
                trace = contract("yxi,i,iyx", T.U, T.s, T.VH)
                trace_imp = contract("yxi,ij,jyx", Timp0.U, Timp0.s, Timp0.VH)

                normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
                #part[i] = normfact_imp * trace_imp / trace
                two_point_y += D[i,j] * normfact_imp * trace_imp / trace

                #print(f"i={i}, j={j}")
                #print("Timp1")
                #T, Timp0 = to_tensor_gl(U1, VH1, U2, VH2, sin_theta, wa, wb, Dcut, s0[i])
                #print()
                #print("Timp2")
                #T, Timp1 = to_tensor_gl(U1, VH1, U2, VH2, sin_theta, wa, wb, Dcut, s1[j])
    

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    energy = 4-2*(2*two_point_y)

    return lnZoV, energy


def internal_energy_TRG2(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    #mu = 0.0
    import trg.TRG_2d as trg
    import trg.ATRG_2d as atrg
    U1, VH1, U2, VH2, sin_theta, cos_theta, phi, wa, wb = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    s0 = [cp.ndarray]*3
    s1 = [cp.ndarray]*3

    I = cp.ones(len(exp_piphi), dtype=cp.float64)
    s0[0] = contract("a,b->ab", cos_theta, I)
    s1[0] = contract("a,b->ab", cos_theta, I)
    s0[1] = cp.exp(mu/2)*contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    s1[1] = cp.exp(mu/2)*contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s0[2] = cp.exp(-mu/2)*contract("a,b->ab", sin_theta, exp_miphi) / cp.sqrt(2)
    s1[2] = cp.exp(-mu/2)*contract("a,b->ab", sin_theta, exp_piphi) / cp.sqrt(2)
    two_point_y = 0.0
    for i in range(3):
        #T = (cp.pi / 8) * contract("a,a,b,iab,abj,kab,abl->ijkl", wa, sin_theta, wb, VH1, U1, VH2, U2)
        #Timp0 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s0[i], wb, VH1, U1, VH2, U2)
        #Timp1 = (cp.pi / 8) * contract("a,a,ab,b,iab,abj,kab,abl->ijkl", wa, sin_theta, s1[i], wb, VH1, U1, VH2, U2)

        #T     = cp.transpose(T    , axes=(0,2,1,3))
        #Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
        #Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))
        #T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
        #trace = contract("abab", T)
        #trace_imp = contract("abab", Timp0)

        #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
        #trace = contract("aabb", T)
        #trace_imp = contract("aabb", Timp0)

        I = cp.ones_like(s0[i])
        print(f"impure tensor {i} start")
        T = to_tensor_gl(U1, VH1, U2, VH2, sin_theta, wa, wb, Dcut)
        #Timp0 = T
        #Timp1 = T
        print("Timp1")
        T, Timp0 = to_tensor_gl(U1, VH1, U2, VH2, sin_theta, wa, wb, Dcut, s0[i])
        print("Timp2")
        T, Timp1 = to_tensor_gl(U1, VH1, U2, VH2, sin_theta, wa, wb, Dcut, s1[i])

        T, Timp0, ln_normfact, ln_normfact_imp = atrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
        trace = contract("yxi,i,iyx", T.U, T.s, T.VH)
        trace_imp = contract("yxi,ij,jyx", Timp0.U, Timp0.s, Timp0.VH)

        print("rg factors pure:",ln_normfact)
        print("rg factors impure:",ln_normfact_imp)
        
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
        
        two_point_y += normfact_imp * trace_imp / trace

        print(f"impure tensor {i} finished")


    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    energy = 4-4*(two_point_y)

    return lnZoV, energy