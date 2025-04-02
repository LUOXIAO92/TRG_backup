import os
import time
import copy
import numpy as np
import cupy as cp
from opt_einsum import contract

import sys
import configparser

sys.path.append('../')
from tensor_init.XY_2d_nonlinear_sigma_model import gauss_legendre_quadrature as ti

import trg.HOTRG_2d_QR as hotrg
import trg.gilt_HOTRG_2d_QR as gilt_hotrg

OUTPUT_DIR = os.environ['OUTPUT_DIR']

#HOTRG
def ln_Z_over_V(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int, gilt_eps=0.0):
    #T = ti.init_pure_tensor_finit_density(beta, mu1, mu2, Dcut)
    #U, VH, _, _, w = ti().__init_tensor_component_parts_finit_density__(beta, mu1, mu2, Dcut)
    U1, VH1, U2, VH2, _, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
    #T, ln_normfact = gilt_hotrg.pure_tensor_renorm(T, Dcut, gilt_eps, XLOOPS, YLOOPS)
    
    rgscheme = os.environ['RGSCHEME']
    if rgscheme == 'trg':
        import trg.TRG_2d_gilt as trg
        T = contract("xXyY->xyXY", T)
        T, ln_normfact = trg.pure_tensor_renorm(T, Dcut, gilt_eps, XLOOPS, YLOOPS)
        T = contract("xyXY->xXyY", T)

    elif rgscheme == 'hotrg':
        import trg.gilt_HOTRG_2d_QR as gilthotrg
        T, ln_normfact = gilthotrg.pure_tensor_renorm_Y2X1(T, Dcut, gilt_eps, XLOOPS, YLOOPS)

    trace = contract("aabb", T)
    del T

    V = 2**(XLOOPS+YLOOPS)
    #print(ln_normfact)
    #print(trace)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    return ln_ZoverV

def ln_Z_over_V_reweighting(beta, mu, beta_t, mu_t, Dcut:int, XLOOPS:int, YLOOPS:int):
    """
    beta, mu: Compute physics quantities under this parameters
    beta_t, mu_t: Reweighting parameters. Compute lnZ(β_t,μ_t)/V, β_t=beta_t, μ_t=mu_t, under parameter beta, mu
    """
    U1, VH1, U2, VH2, _, w = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    u1, vh1, u2, vh2, _, w = ti().__init_tensor_component_parts_finit_density__(beta_t-beta, mu_t-mu, Dcut)

    def compress_component(A, B, Dcut, direction:str):
        """
        A: associate to beta, mu
        B: associate to beta_t, mu_t
        """    
        if direction == "+":
            a = contract("ai,aj->aij", A, B)
            a = cp.reshape(a, newshape=(A.shape[0], A.shape[1]*B.shape[1]))
            aa = cp.conj(a.T) @ a

        elif direction == "-":
            a = contract("ia,ja->ija", A, B)
            a = cp.reshape(a, newshape=(A.shape[0]*B.shape[0], A.shape[1]))
            aa = a @ cp.conj(a.T)

        e, u = cp.linalg.eigh(aa)

        if direction == "+":
            u = cp.reshape(u, newshape=(A.shape[1], B.shape[1], u.shape[1]))
            a = contract("ai,aj,ijk->ak", A, B, u)

        elif direction == "-":
            u = cp.reshape(u, newshape=(A.shape[0], B.shape[0], u.shape[1]))
            a = contract("ia,ja,ijk->", A, B, cp.conj(u))

        return a

    u1 = compress_component(U1, u1, Dcut, "+")
    u2 = compress_component(U2, u2, Dcut, "+")
    vh1 = compress_component(VH1, vh1, Dcut, "-")
    vh2 = compress_component(VH2, vh2, Dcut, "-")

    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", w, VH1, U1, VH2, U2)
    #T, ln_normfact = hotrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS)
    
    import trg.TRG_2d as trg
    T = contract("xXyY->xyXY", T)
    T, _, ln_normfact, _ = trg.nearest_two_point_func_renorm(T, T, T, T, T, Dcut, XLOOPS, YLOOPS)
    T = contract("xyXY->xXyY", T)
        
    trace = contract("aabb", T)
    del T

    V = 2**(XLOOPS+YLOOPS)
    #print(ln_normfact)
    #print(trace)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    return ln_ZoverV


def ln_Z_over_V_ce(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    from tensor_init.XY_2d_nonlinear_sigma_model import character_expansion as ti
    T, _, _ = ti().init_impure_tensor_particle_number_2d(beta, mu, Dcut, part=1)

    T, ln_normfact = hotrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS)
    
        
    trace = contract("aabb", T)
    del T

    V = 2**(XLOOPS+YLOOPS)
    #print(ln_normfact)
    #print(trace)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    return ln_ZoverV

def entanglement_entropy(beta, mu, Dcut:int, lA:int, lB:int, lt:int, gilt_eps=0.0):
    import trg.HOTRG_2d as hotrg
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    U1, VH1, U2, VH2, _, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)

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

def entanglement_entropy_transferMatrix_method(beta, mu, Dcut:int, lA:int, lB:int, lt:int, gilt_eps=0.0):
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    U1, VH1, U2, VH2, _, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)

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

def renyi_entropy(beta, mu, Dcut:int, lA:int, lB:int, lt:int):
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    U1, VH1, U2, VH2, _, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)

    #from tensor_init.XY_2d_nonlinear_sigma_model import character_expansion as ti
    #T, _, _ = ti().init_impure_tensor_particle_number_2d(beta, mu, Dcut, part=1)
    #import trg.HOTRG_2d as hotrg

    TA, ln_normfact_A = hotrg.pure_tensor_renorm(T, Dcut, lA, 0)
    #TB, ln_normfact_B = hotrg.pure_tensor_renorm(T, Dcut, lA, 0)
    TB, ln_normfact_B = TA, ln_normfact_A 

    Trans_Mat = contract("ijab,jicd->acbd", TB, TA)
    Trans_Mat = cp.reshape(Trans_Mat, (Dcut*Dcut, Dcut*Dcut))
    print("Trans_Mat hermit err",cp.linalg.norm(Trans_Mat-cp.conj(Trans_Mat.T))/cp.linalg.norm(Trans_Mat))
    
    u, e, _ = cp.linalg.svd(Trans_Mat)
    e_max = cp.max(cp.abs(e))
    e = e / e_max
    #e[e < 1e-100] = 1e-100

    sv_dir, get_sv = hotrg.__singularvalue_dir__(Dcut, lA, 0, get_sv=True)
    if get_sv == True:
        with open("{:}/densitymatrix_sv".format(sv_dir), "w") as svout:
            for ee in e*e_max:
               svout.write("{:.12e}\n".format(ee))
    
    ln_ZoverV = cp.zeros(lt, dtype=cp.float64)
    STE = cp.zeros(lt, dtype=cp.float64)
    SEE = cp.zeros(lt, dtype=cp.float64)
    E   = cp.zeros(lt, dtype=cp.float64)
    S2  = cp.zeros(lt, dtype=cp.float64)
    ln_E0 = cp.zeros(lt, dtype=cp.float64)
    for lT in range(0,lt,1):
        Nt = int(2**(lT+1))

        #lnZ/V
        TrTBTA = cp.sum(e**Nt)
        VA = Nx_A*Nt
        VB = Nx_B*Nt
        ln_ZoverV[lT] = (VA*cp.sum(ln_normfact_A) + VB*cp.sum(ln_normfact_B) + cp.log(TrTBTA) + Nt*cp.log(e_max))/(VA+VB)

        #SEE
        Dens_Mat = contract("ij,j,kj->ik", u, e**Nt, cp.conj(u)) / cp.sum(e**Nt)
        Dens_Mat = cp.reshape(Dens_Mat, (Dcut, Dcut, Dcut, Dcut))
        rho_A = contract("aiaj->ij", Dens_Mat)
        _, e_A, _ = cp.linalg.svd(rho_A)
        #e_A[e_A < 1e-100] = 1e-100
        SEE[lT] = -cp.sum(e_A * cp.log(e_A))

        if get_sv == True:
            with open("{:}/densitymatrix_A_sv_Nt2^{:}".format(sv_dir, lT+1), "w") as svout:
                for ea in e_A:
                   svout.write("{:.12e}\n".format(ea))

        #STE
        Z = cp.sum(e**Nt) 
        STE[lT] = - Nt * cp.sum( (e**Nt) * cp.log(e))/Z + np.log(Z)

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
        S2[lT] = Sn[0]

        #Internal energy
        Ei = -cp.log(e*e_max) - Nx_A*cp.sum(ln_normfact_A) - Nx_B*cp.sum(ln_normfact_B)
        E[lT] = cp.sum((e**Nt) * Ei)/Z
        E[lT] = E[lT] / (Nx_A+Nx_B)

        e_pow_Nt = e**Nt
        ln_e0 = (VA*cp.sum(ln_normfact_A) + VB*cp.sum(ln_normfact_B) + Nt*cp.log(cp.max(e_pow_Nt)) + Nt*cp.log(e_max))/(VA+VB)
        ln_E0[lT] = ln_e0

    #return ln_ZoverV, STE, SEE, SEE_ext, SEE_replica1, SEE_replica2, SEE_replica3
    return ln_ZoverV, STE, SEE, S2, E, ln_E0

def renyi_entropy2(beta, mu, Dcut:int, lA:int, lB:int, lt:int):
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    U1, VH1, U2, VH2, _, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
    
    TA, ln_normfact_A = hotrg.renyi_entropy_renorm(T, Dcut, lA, lt)
    #TB, ln_normfact_B = hotrg.renyi_entropy_renorm(T, Dcut, lB, lt)
    TB, ln_normfact_B = TA, ln_normfact_A 

    TrTBTA = contract("ijaa,jibb", TB, TA)
    print("contract, TrTBTA",TrTBTA)
    Dens_Mat = contract("ijab,jicd->acbd", TB, TA)/TrTBTA
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(Dcut*Dcut, Dcut*Dcut))
    u_Dens_Mat, e_Dens_Mat, _ = cp.linalg.svd(Dens_Mat)
    #e_Dens_Mat, u_Dens_Mat = cp.linalg.eigh(Dens_Mat)
    #e_Dens_Mat = e_Dens_Mat.astype(cp.complex128)

    print("Dens_Mat hermit error", cp.linalg.norm(Dens_Mat-cp.conj(Dens_Mat.T))/cp.linalg.norm(Dens_Mat))
    
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
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(Dcut, Dcut, Dcut, Dcut))
    rho_A = contract("aiaj->ij", Dens_Mat)
    _, e_A, _ = cp.linalg.svd(rho_A)
    #e_A, _ = cp.linalg.eigh(rho_A)
    #e_A = e_A.astype(cp.complex128)

    SEE = -cp.sum(e_A * cp.log(e_A))
    print("ρ_A hermit error", cp.linalg.norm(rho_A-cp.conj(rho_A.T))/cp.linalg.norm(rho_A))

    with open("{:}/densitymatrix_A.dat".format(OUTPUT_DIR), "w") as svout:
        for ee in e_A:
           svout.write("{:.12e}\n".format(ee))

    
    #renyi
    N = 2
    Sn = cp.zeros(N, dtype=cp.complex128)
    n  = cp.array([i for i in range(2,N+2)], dtype=cp.int64)
    for i in range(0,N):
        Sn[i] = cp.sum(e_A**(i+2))
    Sn = cp.log(Sn)
    Sn = Sn / (1-n)
    Sn = Sn.real
    S2 = Sn[0]

    #Energy
    E = -cp.sum(e_Dens_Mat * cp.log(e_Dens_Mat)) - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = E / (VA+VB)

    return ln_ZoverV, STE, SEE, S2, E

def renyi_entropy3(beta, mu, Dcut:int, lA:int, lB:int, lt:int):
    Nx_A = int(2**lA)
    Nx_B = int(2**lB)
    Nt   = int(2**lt)

    U1, VH1, U2, VH2, _, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
    
    TA, ln_normfact_A = hotrg.renyi_entropy_renorm(T, Dcut, lA, lt)
    #TB, ln_normfact_B = hotrg.renyi_entropy_renorm(T, Dcut, lB, lt)
    TB, ln_normfact_B = TA, ln_normfact_A 

    TrTBTA = contract("ijaa,jibb", TB, TA)
    print("contract, TrTBTA",TrTBTA)

    M = contract("ijab,jicd->acbd", TB, TA)
    M = cp.reshape(M, newshape=(Dcut*Dcut, Dcut*Dcut))
    #u, e, _ = cp.linalg.svd(M)
    e, u = cp.linalg.eigh(M)
    e = e.astype(cp.complex128)
    TrTBTA = cp.sum(e)
    print("sum(e), TrTBTA",TrTBTA)
    #print("e",e[:20])
    print("M hermit err",cp.linalg.norm(M-cp.conj(M.T))/cp.linalg.norm(M))
    
    sv_dir, get_sv = hotrg.__singularvalue_dir__(Dcut, lA, lt, get_sv=True)
    if get_sv == True:
        with open("{:}/densitymatrix_sv".format(sv_dir), "w") as svout:
            for ee in e:
               svout.write("{:.12e}\n".format(ee))
    

    #lnZ/V
    VA = Nx_A*Nt
    VB = Nx_B*Nt
    ln_ZoverV = (VA*cp.sum(ln_normfact_A) + VB*cp.sum(ln_normfact_B) + cp.log(TrTBTA))/(VA+VB)

    #STE
    STE = - cp.sum( e * cp.log(e))/TrTBTA + cp.log(TrTBTA)

    #SEE
    M = cp.reshape(M, newshape=(Dcut, Dcut, Dcut, Dcut))
    M_A = contract("aiaj->ij", M)
    #_, e_A, _ = cp.linalg.svd(M_A)
    e_A, _ = cp.linalg.eigh(M_A)
    e_A = e_A.astype(cp.complex128)
    SEE = -cp.sum(e_A * cp.log(e_A))/TrTBTA + cp.log(TrTBTA)
    print("M_A hermit error", cp.linalg.norm(M-cp.conj(M.T))/cp.linalg.norm(M))

    if get_sv == True:
        with open("{:}/densitymatrix_A_sv".format(sv_dir), "w") as svout:
            for ee in e_A:
               svout.write("{:.12e}\n".format(ee))

    #renyi
    N = 2
    Sn = cp.zeros(N, dtype=cp.complex128)
    n  = cp.array([i for i in range(2,N+2)], dtype=cp.int64)
    for i in range(0,N):
        Sn[i] = cp.sum(e_A**(i+2))/(TrTBTA**(i+2))
    Sn = cp.log(Sn)
    Sn = Sn / (1-n)
    Sn = Sn.real
    S2 = Sn[0]

    #Energy
    E = -cp.sum(e * cp.log(e))/TrTBTA - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = E / (VA+VB)

    return ln_ZoverV, STE, SEE, S2, E

def entanglement_entropy2(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    U1, VH1, U2, VH2, _, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
    
    ln_ZoV, SEE, ln_emax = hotrg.entanglement_entropy_renorm(T, Dcut, XLOOPS, YLOOPS)
    return ln_ZoV, SEE, ln_emax


def particle_number(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    #a = phi - 1j*mu/2
    #b = phi + 1j*mu/2

    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
    Timp0 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", wphi, exp_piphi, VH1, U1, VH2, U2)
    Timp1 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", wphi, exp_miphi, VH1, U1, VH2, U2)
    #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
    trace1 = contract("aabb", T)
    trace_imp1 = contract("aabb", Timp0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
    Timp0 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", wphi, exp_miphi, VH1, U1, VH2, U2)
    Timp1 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", wphi, exp_piphi, VH1, U1, VH2, U2)
    #T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
    trace2 = contract("aabb", T)
    trace_imp2 = contract("aabb", Timp0)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace2) / V
    n = (beta/2) * (cp.exp(mu)*normfact_imp1*trace_imp1/trace1 - cp.exp(-mu)*normfact_imp2*trace_imp2/trace2)
    #n = 1j * beta * (normfact_imp1*trace_imp1/trace1 - normfact_imp2*trace_imp2/trace2)
    #print(normfact_imp1*trace_imp1/trace1, normfact_imp2*trace_imp2/trace2)
    return lnZoV, n

def particle_number_ce(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.HOTRG_2d as hotrg
    from tensor_init.XY_2d_nonlinear_sigma_model import character_expansion as ti
    T, Timp0, Timp1 = ti().init_impure_tensor_particle_number_2d(beta, mu, Dcut, part=1)
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
    trace1 = contract("aabb", T)
    trace_imp1 = contract("aabb", Timp0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    T, Timp0, Timp1 = ti().init_impure_tensor_particle_number_2d(beta, mu, Dcut, part=2)
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
    trace2 = contract("aabb", T)
    trace_imp2 = contract("aabb", Timp0)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace2) / V
    n = (beta/2) * (cp.exp(mu)*normfact_imp1*trace_imp1/trace1 - cp.exp(-mu)*normfact_imp2*trace_imp2/trace2)
    #print(normfact_imp1*trace_imp1/trace1, normfact_imp2*trace_imp2/trace2)
    return lnZoV, n

def internal_energy_0(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, 0, Dcut)

    s0 = [cp.ndarray]*3
    s1 = [cp.ndarray]*3

    s0[0] = cp.cos(phi)
    s1[0] = cp.cos(phi)
    s0[1] = cp.sin(phi)
    s1[1] = cp.sin(phi)
    two_point_y = 0.0
    for i in range(2):
        T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
        Timp0 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", s0[i], wphi, VH1, U1, VH2, U2)
        Timp1 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", s1[i], wphi, VH1, U1, VH2, U2)
        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    
        trace = contract("aabb", T)
        trace_imp = contract("aabb", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    
        two_point_y += normfact_imp * trace_imp / trace

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    energy = -2*two_point_y

    return lnZoV, energy

def internal_energy_0_HOTRG_test(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, 0, Dcut)

    s0 = [cp.ndarray]*3
    s1 = [cp.ndarray]*3

    s0[0] = cp.exp(1j*phi)
    s1[0] = cp.exp(-1j*phi)
    s0[1] = cp.exp(-1j*phi)
    s1[1] = cp.exp(1j*phi)
    two_point_y = 0.0
    a = cp.zeros(2,dtype=cp.complex128)
    for i in range(2):
        T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
        Timp0 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", s0[i], wphi, VH1, U1, VH2, U2)
        Timp1 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", s1[i], wphi, VH1, U1, VH2, U2)
        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    
        trace = contract("aabb", T)
        trace_imp = contract("aabb", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
        a[i] = normfact_imp * trace_imp / trace
        two_point_y += normfact_imp * trace_imp / trace

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    energy = -two_point_y#-cp.conj(two_point_y)
    print("part1= {:.12e} part2 = {:.12e}".format(a[0],a[1]))
    return lnZoV, energy

def particle_number_HOTRG_test(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    #a = phi - 1j*mu/2
    #b = phi + 1j*mu/2

    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
    Timp0 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", wphi, exp_piphi, VH1, U1, VH2, U2)
    Timp1 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", wphi, exp_miphi, VH1, U1, VH2, U2)
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
    trace1 = contract("aabb", T)
    trace_imp1 = contract("aabb", Timp0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
    Timp0 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", wphi, exp_miphi, VH1, U1, VH2, U2)
    Timp1 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", wphi, exp_piphi, VH1, U1, VH2, U2)
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS)
    trace2 = contract("aabb", T)
    trace_imp2 = contract("aabb", Timp0)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace2) / V
    n = (beta/2) * (cp.exp(mu)*normfact_imp1*trace_imp1/trace1 - cp.exp(-mu)*normfact_imp2*trace_imp2/trace2)
    #n = 1j * beta * (normfact_imp1*trace_imp1/trace1 - normfact_imp2*trace_imp2/trace2)
    print("part1= {:.12e}  part2= {:.12e}".format(normfact_imp1*trace_imp1/trace1, normfact_imp2*trace_imp2/trace2))
    return lnZoV, n

def internal_energy(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

    s0 = [cp.ndarray]*3
    s1 = [cp.ndarray]*3

    s0[0] = cp.cos(phi)
    s1[0] = cp.cos(phi+1j*mu)
    s0[1] = cp.sin(phi)
    s1[1] = cp.sin(phi+1j*mu)
    two_point_y = 0.0
    for i in range(2):
        T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
        Timp0 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", s0[i], wphi, VH1, U1, VH2, U2)
        Timp1 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", s1[i], wphi, VH1, U1, VH2, U2)
        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    
        trace = contract("aabb", T)
        trace_imp = contract("aabb", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    
        two_point_y += normfact_imp * trace_imp / trace

    s0[0] = cp.cos(phi)
    s1[0] = cp.cos(phi)
    s0[1] = cp.sin(phi)
    s1[1] = cp.sin(phi)
    two_point_x = 0.0
    for i in range(3):
        T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
        Timp0 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", s0[i], wphi, VH1, U1, VH2, U2)
        Timp1 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", s1[i], wphi, VH1, U1, VH2, U2)
        T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 1, 0, Dcut, XLOOPS, YLOOPS)
    
        trace = contract("aabb", T)
        trace_imp = contract("aabb", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    
        two_point_x += normfact_imp * trace_imp / trace

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    energy = -two_point_y-two_point_x

    return lnZoV, energy

def internal_energy_ce(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    from tensor_init.XY_2d_nonlinear_sigma_model import character_expansion as ti

    Tpure, Timpp, Timpm = ti().init_impure_tensor_internal_energy_2d(beta, mu, Dcut)

    #Y
    T = Tpure
    Timp0 = Timpp
    Timp1 = Timpm
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    trace1 = contract("aabb", T)
    trace_imp1 = contract("aabb", Timp0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))
    del T, Timp0, Timp1

    T = Tpure
    Timp0 = Timpm
    Timp1 = Timpp
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    trace2 = contract("aabb", T)
    trace_imp2 = contract("aabb", Timp0)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))
    del T, Timp0, Timp1

    #X
    T = Tpure
    Timp0 = Timpp
    Timp1 = Timpm
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 1, 0, Dcut, XLOOPS, YLOOPS)
    trace3 = contract("aabb", T)
    trace_imp3 = contract("aabb", Timp0)
    normfact_imp3 = cp.exp(cp.sum(ln_normfact_imp))
    del T, Timp0, Timp1

    T = Tpure
    Timp0 = Timpm
    Timp1 = Timpp
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 1, 0, Dcut, XLOOPS, YLOOPS)
    trace4 = contract("aabb", T)
    trace_imp4 = contract("aabb", Timp0)
    normfact_imp4 = cp.exp(cp.sum(ln_normfact_imp))
    del T, Timp0, Timp1

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace1) / V
    e = -(normfact_imp3*trace_imp3/trace3 + normfact_imp4*trace_imp4/trace4 
        + cp.exp(mu)*normfact_imp1*trace_imp1/trace1 + cp.exp(-mu)*normfact_imp2*trace_imp2/trace2) / 2

    return lnZoV, e

def internal_energy_0_ce(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    from tensor_init.XY_2d_nonlinear_sigma_model import character_expansion as ti

    Tpure, Timpp, Timpm = ti().init_impure_tensor_internal_energy_2d(beta, 0, Dcut)

    #Y
    T = Tpure
    Timp0 = Timpp
    Timp1 = Timpm
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    trace1 = contract("aabb", T)
    trace_imp1 = contract("aabb", Timp0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))
    del T, Timp0, Timp1

    T = Tpure
    Timp0 = Timpm
    Timp1 = Timpp
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg.two_point_func_renorm(T, Timp0, Timp1, 0, 1, Dcut, XLOOPS, YLOOPS)
    trace2 = contract("aabb", T)
    trace_imp2 = contract("aabb", Timp0)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))
    del T, Timp0, Timp1

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace1) / V
    e = -normfact_imp1*trace_imp1/trace1 - normfact_imp2*trace_imp2/trace2

    return lnZoV, e

#TRG
def particle_number_TRG(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d as trg
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)

    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    #exp_piphi = cp.sin(phi)
    #exp_miphi = cp.cos(phi+1j*mu)
    
    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
    Timp0 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", wphi, exp_piphi, VH1, U1, VH2, U2)
    Timp1 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", wphi, exp_miphi, VH1, U1, VH2, U2)

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

    T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
    Timp0 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", wphi, exp_miphi, VH1, U1, VH2, U2)
    Timp1 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", wphi, exp_piphi, VH1, U1, VH2, U2)

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
    print("part1= {:.12e}  part2= {:.12e}".format(normfact_imp1*trace_imp1/trace1, normfact_imp2*trace_imp2/trace2))

    #n = (beta*1j) * (normfact_imp1*trace_imp1/trace1 - normfact_imp2*trace_imp2/trace2) #test

    return lnZoV, n

def particle_number_ce_TRG(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d as trg
    from tensor_init.XY_2d_nonlinear_sigma_model import character_expansion as ti

    T, Timp0, Timp1 = ti().init_impure_tensor_particle_number_2d(beta, mu, Dcut, part=1)
    T     = cp.transpose(T    , axes=(0,2,1,3))
    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))

    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
    trace1 = contract("abab", T)
    trace_imp1 = contract("abab", Timp0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    T, Timp0, Timp1 = ti().init_impure_tensor_particle_number_2d(beta, mu, Dcut, part=2)
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
    #print(normfact_imp1*trace_imp1/trace1, normfact_imp2*trace_imp2/trace2)
    return lnZoV, n

def internal_energy_0_TRG(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d as trg
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, 0, Dcut)

    s0 = [cp.ndarray]*3
    s1 = [cp.ndarray]*3

    #s0[0] = cp.exp(1j*phi)
    #s1[0] = cp.exp(-1j*phi)
    #s0[1] = cp.exp(-1j*phi)
    #s1[1] = cp.exp(1j*phi)

    s0[0] = cp.cos(phi)
    s1[0] = cp.cos(phi)
    s0[1] = cp.sin(phi)
    s1[1] = cp.sin(phi)

    two_point_y = 0.0
    for i in range(2):
        T = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
        Timp0 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", s0[i], wphi, VH1, U1, VH2, U2)
        Timp1 = 0.5 * contract("a,a,ia,aj,ka,al->ijkl", s1[i], wphi, VH1, U1, VH2, U2)
        Timp2 = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)
        Timp3 = 0.5 * contract("a,ia,aj,ka,al->ijkl", wphi, VH1, U1, VH2, U2)


        T     = cp.transpose(T    , axes=(0,2,1,3))
        Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
        Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))
        Timp2 = cp.transpose(Timp2, axes=(0,2,1,3))
        Timp3 = cp.transpose(Timp3, axes=(0,2,1,3))

        T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, Timp2, Timp3, Dcut, XLOOPS, YLOOPS)
        trace = contract("abab", T)
        trace_imp = contract("abab", Timp0)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    
        two_point_y += normfact_imp * trace_imp / trace

        #sys.exit(0)

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    energy = -2*two_point_y

    return lnZoV, energy

def internal_energy_ce_TRG(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d as trg
    from tensor_init.XY_2d_nonlinear_sigma_model import character_expansion as ti

    Tpure, Timpp, Timpm = ti().init_impure_tensor_internal_energy_2d(beta, mu, Dcut)

    #Y
    T = Tpure
    Timp0 = Timpp
    Timp1 = Timpm
    T     = cp.transpose(T    , axes=(0,2,1,3))
    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))
    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
    trace1 = contract("abab", T)
    trace_imp1 = contract("abab", Timp0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))
    del T, Timp0, Timp1

    T = Tpure
    Timp0 = Timpm
    Timp1 = Timpp
    T     = cp.transpose(T    , axes=(0,2,1,3))
    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))
    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
    trace2 = contract("abab", T)
    trace_imp2 = contract("abab", Timp0)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))
    del T, Timp0, Timp1

    #X
    T = Tpure
    Timp0 = Timpp
    Timp1 = Timpm
    T     = cp.transpose(T    , axes=(0,2,1,3))
    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))
    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, T, T, Timp1, Dcut, XLOOPS, YLOOPS)
    trace3 = contract("abab", T)
    trace_imp3 = contract("abab", Timp0)
    normfact_imp3 = cp.exp(cp.sum(ln_normfact_imp))
    del T, Timp0, Timp1

    T = Tpure
    Timp0 = Timpm
    Timp1 = Timpp
    T     = cp.transpose(T    , axes=(0,2,1,3))
    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))
    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, T, T, Timp1, Dcut, XLOOPS, YLOOPS)
    trace4 = contract("abab", T)
    trace_imp4 = contract("abab", Timp0)
    normfact_imp4 = cp.exp(cp.sum(ln_normfact_imp))
    del T, Timp0, Timp1

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace1) / V
    e = (normfact_imp3*trace_imp3/trace3 + normfact_imp4*trace_imp4/trace4 
        + cp.exp(mu)*normfact_imp1*trace_imp1/trace1 + cp.exp(-mu)*normfact_imp2*trace_imp2/trace2) / 2

    return lnZoV, e

def internal_energy_0_ce_TRG(beta, Dcut:int, XLOOPS:int, YLOOPS:int):
    import trg.TRG_2d as trg
    from tensor_init.XY_2d_nonlinear_sigma_model import character_expansion as ti

    Tpure, Timpp, Timpm = ti().init_impure_tensor_internal_energy_2d(beta, 0, Dcut)

    #Y
    T = Tpure
    Timp0 = Timpp
    Timp1 = Timpm
    T     = cp.transpose(T    , axes=(0,2,1,3))
    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))
    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
    trace1 = contract("abab", T)
    trace_imp1 = contract("abab", Timp0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))
    del T, Timp0, Timp1

    T = Tpure
    Timp0 = Timpm
    Timp1 = Timpp
    T     = cp.transpose(T    , axes=(0,2,1,3))
    Timp0 = cp.transpose(Timp0, axes=(0,2,1,3))
    Timp1 = cp.transpose(Timp1, axes=(0,2,1,3))
    T, Timp0, ln_normfact, ln_normfact_imp = trg.nearest_two_point_func_renorm(T, Timp0, Timp1, T, T, Dcut, XLOOPS, YLOOPS)
    trace2 = contract("abab", T)
    trace_imp2 = contract("abab", Timp0)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))
    del T, Timp0, Timp1

    V = 2**(XLOOPS+YLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace1) / V
    e = -(normfact_imp1*trace_imp1/trace1 + normfact_imp2*trace_imp2/trace2)

    return lnZoV, e