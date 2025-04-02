import gc
import time
import numpy as np
import cupy as cp
import opt_einsum as oe
#from cuquantum import contract

import sys
import configparser

sys.path.append('../')
from tensor_init.XY_2d_nonlinear_sigma_model import gauss_legendre_quadrature as ti

from tensor_class.tensor_class import ATRG_Tensor as Tensor
K = ti().SAMPLE_NUM
def to_tensor_gl(U, VH, w, Dcut:int, S)->Tensor:
    from utility.randomized_svd import rsvd_for_3dATRG_tensor_init
    A = 0.5 * oe.contract("a,a,ai,ja,ak->ijka", w, S, U[1], VH[0], U[0])
    B = oe.contract("la,am,na->almn", VH[1], U[0], VH[0])
    
    A = cp.reshape(A, (Dcut,Dcut,Dcut,K))
    B = cp.reshape(B, (K,Dcut,Dcut,Dcut))

    rs=cp.random.RandomState(1234)
    u, s, vh = rsvd_for_3dATRG_tensor_init(A, B, Dcut, n_oversamples=2*Dcut, n_power_iter=Dcut,seed=rs)
    
    T = Tensor(u, s, vh)
    del u, s, vh, A, B

    return T

def to_tensor_gl2(U, VH, w, Dcut:int, S)->Tensor:
    from utility.randomized_svd import rsvd_for_3dATRG_tensor_init
    A = 0.5 * oe.contract("a,a,ai,aj,ak->ijka", w, S, U[1], U[0], U[0])
    B = oe.contract("la,ma,na->almn", VH[1], VH[0], VH[0])
    
    A = cp.reshape(A, (Dcut,Dcut,Dcut,K))
    B = cp.reshape(B, (K,Dcut,Dcut,Dcut))

    rs=cp.random.RandomState(1234)
    u, s, vh = rsvd_for_3dATRG_tensor_init(A, B, Dcut, n_oversamples=2*Dcut, n_power_iter=Dcut,seed=rs)
    print("s_init", s)
    
    T = Tensor(u, s, vh)
    del u, s, vh, A, B

    return T

def to_tensor_gl3(U, VH, w, Dcut:int, S)->Tensor:
    from tensor_init.ATRG_init import initial_tensor_for_ATRG as init
    
    J = cp.ones_like(w) * 0.5
    u, s, vh = init(dim=3, J=J, w=w, 
                    As=U[0], Bs=VH[0],
                    At=U[1], Bt=VH[1], 
                    k=Dcut, p=2*Dcut, q=Dcut, seed=12345)
    print("s_init", s)
    T = Tensor(u, s, vh, dim=3, is_impure=False, loc={})
    del u, s, vh

    return T

def ln_Z_over_V(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    U  = [U1, U2]
    VH = [VH1, VH2]

    #import trg.HOTRG_3d as hotrg
    #T = 0.5* oe.contract("a,aZ,xa,aY,za,aX,ya->xXyYzZ", wphi, U[1], VH[0], U[0], VH[1], U[0], VH[0])
    #T, ln_normfact = hotrg.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS, ZLOOPS)
    #
    #T1 = oe.contract("xXyYzZ->ZxYzXy", T)
    #T1 = cp.reshape(T1, newshape=(int(Dcut**3), int(Dcut**3)))
    ##u, ss, vh = cp.linalg.svd(T1)
    #ee, u = cp.linalg.eigh(T1)
    #vh = cp.conj(u.T)
    ##print(ss[:Dcut])
    #print((ee[::-1])[:Dcut])
    #u  = cp.reshape(u , (Dcut,Dcut,Dcut,int(Dcut**3)))
    #vh = cp.reshape(vh, (int(Dcut**3),Dcut,Dcut,Dcut))
    #I = oe.contract("izxy,zxyj->ij", vh, u)
    #print("Full: ||V^†U||^2= {:.2f} , Tr(V^†U)= {:.2f}".format(cp.linalg.norm(I)**2, cp.trace(I)))
    #
    #u  = u[:,:,:,:Dcut]
    #vh = vh[:Dcut,:,:,:]
    #I = oe.contract("izxy,zxyj->ij", vh, u)
    #print("Dcut: ||V^†U||^2= {:.2f} , Tr(V^†U)= {:.2f}".format(cp.linalg.norm(I)**2, cp.trace(I)))
    #
    #trace = oe.contract("xxyyzz", T)
    #V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    #lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    #T = 0.5* oe.contract("a,aZ,xa,aY,za,aX,ya->xXyYzZ", wphi, U[1], VH[0], U[0], VH[1], U[0], VH[0])
    #T = oe.contract("xXyYzZ->ZxYzXy", T)
    #T = cp.reshape(T, newshape=(int(Dcut**3), int(Dcut**3)))
    #u, s, vh = cp.linalg.svd(T)
    #del T
    #u  = cp.reshape(u , (Dcut,Dcut,Dcut,int(Dcut**3)))
    #vh = cp.reshape(vh, (int(Dcut**3),Dcut,Dcut,Dcut))
    #u  = u[:,:,:,:Dcut]
    #s  = s[:Dcut]
    #vh = vh[:Dcut,:,:,:]
    #T = Tensor(u, s, vh)
    import trg.ATRG_3d_new as atrg3d
    I = cp.ones(len(phi), dtype=cp.complex128)
    T = to_tensor_gl3(U, VH, wphi, Dcut, I)
    T, ln_normfact = atrg3d.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS, ZLOOPS)
    trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
    V = 2**(XLOOPS+YLOOPS+ZLOOPS)

    print("factors", ln_normfact)
    print("trace", trace)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    return lnZoV

def ln_Z_over_V_hotrg(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    import trg.HOTRG_3d as hotrg3d
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    U  = [U1, U2]
    VH = [VH1, VH2]

    I = cp.ones_like(phi)
    T = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, I, U[1], VH[0], U[0], VH[1], U[0], VH[0])
    T, ln_normfact = hotrg3d.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS, ZLOOPS)

    trace = oe.contract("iijjkk", T)
    #ln_normfact = cp.exp(cp.sum(ln_normfact))

    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    return lnZoV


def entanglement_entropy(beta, mu, Dcut:int, lxA:int, lyA:int, lxB:int, lyB:int, lt:int):
    import trg.HOTRG_3d as hotrg
    Nx_A = int(2**lxA)
    Ny_A = int(2**lyA)
    Nx_B = int(2**lxB)
    Ny_B = int(2**lyB)
    Nt   = int(2**lt)

    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    U  = [U1, U2]
    VH = [VH1, VH2]
    #A = 0.5 * oe.contract("a,ai,ja,ak->ijka", wphi, U[1], VH[0], U[0])
    #B = oe.contract("la,am,na->almn", VH[1], U[0], VH[0])
    T = 0.5* oe.contract("a,aZ,xa,aY,za,aX,ya->xXyYzZ", wphi, U[1], VH[0], U[0], VH[1], U[0], VH[0])

    TA, ln_normfact_A = hotrg.pure_tensor_renorm(T, Dcut, lxA, lyA, lt)
    TB, ln_normfact_B = TA, ln_normfact_A 

    TrTBTA = oe.contract("ijkkaa,jillbb", TB, TA)
    print("TrTBTA=",TrTBTA)
    Dens_Mat = oe.contract("ijkkac,jillbd->abcd", TB, TA) / TrTBTA
    TrDens_Mat = oe.contract("ijij", Dens_Mat)
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(Dcut*Dcut, Dcut*Dcut))
    u_Dens_Mat, e_Dens_Mat, _ = cp.linalg.svd(Dens_Mat)
    print("dens_mat hermit err",cp.linalg.norm(Dens_Mat-cp.conj(Dens_Mat.T))/cp.linalg.norm(Dens_Mat))
    print("Tr(ρ)=",TrDens_Mat)

    #with open("XY_3d_nlsm_SEE_D{:}_b_{:}_mu_{:}_Ns_2^{:}_Ny_2^{:d}_Nt_2^{:d}_densM_sv.dat".format(Dcut, beta, mu, lxA+1, lyA, lt), "w") as out:
    #    for e in e_Dens_Mat:
    #        out.write("{:.12e}\n".format(e))

    T = oe.contract("xXyYzZ->ZxYzXy", TA)
    T = cp.reshape(T, newshape=(int(Dcut**3), int(Dcut**3)))
    _, ss, _ = cp.linalg.svd(T)
    ee, _ = cp.linalg.eigh(T)
    print(ss)
    print(ee[::-1])
    #with open("XY_3d_nlsm_SEE_D{:}_b_{:}_mu_{:}_Ns_2^{:}_Ny_2^{:d}_Nt_2^{:d}_tensor_sv.dat".format(Dcut, beta, mu, lxA+1, lyA, lt), "w") as out:
    #    for ss in s:
    #        out.write("{:.12e}\n".format(ss))

    TA_conj = cp.conj(TA)
    M = oe.contract("aefghm,bijkml,cefghn,dijknl->abcd", TA, TA, TA_conj, TA_conj)
    M = cp.reshape(M, newshape=(Dcut*Dcut, Dcut*Dcut))
    _, eM, _ = cp.linalg.svd(M)
    with open("XY_3d_nlsm_SEE_D{:}_b_{:}_mu_{:}_L_2^{:d}_M_sv.dat".format(Dcut, beta, mu, lt), "w") as out:
        for ss in eM:
            out.write("{:.12e}\n".format(ss))
    eM2 = eM**2
    eM2 = eM2 / cp.sum(eM2)
    see = - cp.sum(eM2 * cp.log(eM2))
    print("CTM SEE= {:.12e}".format(see))
    print(cp.sum(eM2))
    print("M hermit err",cp.linalg.norm(M-cp.conj(M.T))/cp.linalg.norm(M))
    

    #lnZ/V
    VA = Nx_A*Ny_A*Nt
    VB = Nx_B*Ny_B*Nt
    ln_ZoverV = (VA*cp.sum(ln_normfact_A) + VB*cp.sum(ln_normfact_A) + cp.log(TrTBTA))/(VA+VB)
    print("lnZ/V= {:12e}".format(ln_ZoverV.real))

    #STE
    #STE = - cp.sum( e_Dens_Mat * cp.log(e_Dens_Mat))/cp.sum(e_Dens_Mat) + cp.log(cp.sum(e_Dens_Mat))
    STE = - cp.sum( e_Dens_Mat * cp.log(e_Dens_Mat))
    #STE = - cp.sum( e * cp.log(e))/TrTBTA + cp.log(TrTBTA)

    #SEE
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(Dcut, Dcut, Dcut, Dcut))
    rho_A = oe.contract("aiaj->ij", Dens_Mat)
    _, e_A, _ = cp.linalg.svd(rho_A)
    #e_A[e_A < 1e-100] = 1e-100
    SEE = -cp.sum(e_A * cp.log(e_A))
    print("rho_A hermit err",cp.linalg.norm(rho_A-cp.conj(rho_A.T))/cp.linalg.norm(rho_A))
    TrArho_A = oe.contract("ii",rho_A)
    print("Tr_A(ρ_A)=",TrArho_A)

    #with open("XY_3d_nlsm_SEE_D{:}_b_{:}_mu_{:}_Ns_2^{:}_Ny_2^{:d}_Nt_2^{:d}_rhoA_sv.dat".format(Dcut, beta, mu, lxA+1, lyA, lt), "w") as out:
    #    for e in e_A:
    #        out.write("{:.12e}\n".format(e))

    #M = cp.reshape(M, newshape=(Dcut, Dcut, Dcut, Dcut))
    #M_A = oe.contract("aiaj->ij", M)
    #_, e_A, _ = cp.linalg.svd(M_A)
    ##e_A[e_A < 1e-100] = 1e-100
    #SEE = -cp.sum(e_A * cp.log(e_A))/TrTBTA + cp.log(TrTBTA)
    ##print("e_A",e_A[:20])

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
    print("S1,S0.5,S2,...,S11=",SEE, sn_str)
    
    #Energy
    #E = -cp.sum(e * cp.log(e))/TrTBTA - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = -cp.sum(e_Dens_Mat * cp.log(e_Dens_Mat)) - VA*cp.sum(ln_normfact_A) - VB*cp.sum(ln_normfact_B)
    E = E / (VA+VB)

    return ln_ZoverV, STE, SEE, E, sn_str


def internal_energy_zero_density(beta, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    import trg.ATRG_3d as atrg3d
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, 0.0, Dcut)
    U  = [U1, U2]
    VH = [VH1, VH2]
    s0 = [cp.ndarray]*3
    s1 = [cp.ndarray]*3

    I = cp.ones(len(phi))
    s0[0] = cp.exp(1j*phi)
    s1[0] = cp.exp(-1j*phi)
    s0[1] = cp.exp(-1j*phi)
    s1[1] = cp.exp(1j*phi)
    two_point_y = 0.0
    a = cp.zeros(2,dtype=cp.complex128)
    for i in range(2):
        T     = to_tensor_gl(U, VH, wphi, Dcut,  I)
        Timp0 = to_tensor_gl(U, VH, wphi, Dcut, s0[i])
        Timp1 = to_tensor_gl(U, VH, wphi, Dcut, s1[i])

        #trace = cp.einsum("ijka,a,aijk", T.U, T.s, T.VH)
        #trace_imp0 = cp.einsum("ijka,a,aijk", Timp0.U, Timp0.s, Timp0.VH)
        #trace_imp1 = cp.einsum("ijka,a,aijk", Timp1.U, Timp1.s, Timp1.VH)
        #print("trace",trace)
        #print("trace_imp0",trace_imp0)
        #print("trace_imp1",trace_imp1)
        
        T, Timp0, ln_normfact, ln_normfact_imp = atrg3d.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS, ZLOOPS)
    
        trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
        trace_imp = oe.contract("ijka,a,aijk", Timp0.U, Timp0.s, Timp0.VH)
        normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
        two_point_y += normfact_imp * trace_imp / trace

        #print("Timp0.s",Timp0.s)
        #print("Timp1.s",Timp1.s)
        #print("two_point_y[i]" ,normfact_imp * trace_imp / trace)
        #print("trace_imp",trace_imp)
        #print("normfact_imp",normfact_imp)
        #print("ln_normfact_imp[i]",ln_normfact_imp)


    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V
    energy = -1.5*two_point_y
    return lnZoV, energy

def particle_number(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    import trg.ATRG_3d as atrg3d
    
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    U  = [U1, U2]
    VH = [VH1, VH2]

    I = cp.ones(len(phi), dtype=cp.complex128)
    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    T     = to_tensor_gl(U, VH, wphi, Dcut,  I)
    Timp0 = to_tensor_gl(U, VH, wphi, Dcut, exp_piphi)
    Timp1 = to_tensor_gl(U, VH, wphi, Dcut, exp_miphi)
    print("T s:",T.s)
    print("Timp0 s:",Timp0.s)
    print("Timp1 s:",Timp1.s)
    
    T, Timp0, ln_normfact, ln_normfact_imp = atrg3d.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS, ZLOOPS)
    trace1 = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
    trace_imp1 = oe.contract("ijka,a,aijk", Timp0.U, Timp0.s, Timp0.VH)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    T     = to_tensor_gl(U, VH, wphi, Dcut,  I)
    Timp0 = to_tensor_gl(U, VH, wphi, Dcut, exp_miphi)
    Timp1 = to_tensor_gl(U, VH, wphi, Dcut, exp_piphi)
    print("T s:",T.s)
    print("Timp0 s:",Timp0.s)
    print("Timp1 s:",Timp1.s)
    
    T, Timp0, ln_normfact, ln_normfact_imp = atrg3d.ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut, XLOOPS, YLOOPS, ZLOOPS)
    trace2 = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
    trace_imp2 = oe.contract("ijka,a,aijk", Timp0.U, Timp0.s, Timp0.VH)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))

    del T, Timp0, Timp1

    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace2) / V
    n = (beta/2) * (cp.exp(mu)*normfact_imp1*trace_imp1/trace1 - cp.exp(-mu)*normfact_imp2*trace_imp2/trace2)

    return lnZoV, n


def particle_number_hotrg(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    import trg.HOTRG_3d as hotrg3d
    
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    U  = [U1, U2]
    VH = [VH1, VH2]

    I = cp.ones(len(phi), dtype=cp.complex128)
    exp_piphi = cp.exp(1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    T     = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, I, U[1], VH[0], U[0], VH[1], U[0], VH[0])
    T0 = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, exp_piphi, U[1], VH[0], U[0], VH[1], U[0], VH[0])
    Tn = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, exp_miphi, U[1], VH[0], U[0], VH[1], U[0], VH[0])
    
    T, T0, ln_normfact, ln_normfact_imp = hotrg3d.ynearest_two_point_func_renorm(T, T0, Tn, Dcut, XLOOPS, YLOOPS, ZLOOPS)
    trace1 = oe.contract("iijjkk", T)
    trace_imp1 = oe.contract("iijjkk", T0)
    normfact_imp1 = cp.exp(cp.sum(ln_normfact_imp))

    del T, T0

    T     = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, I, U[1], VH[0], U[0], VH[1], U[0], VH[0])
    T0 = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, exp_miphi, U[1], VH[0], U[0], VH[1], U[0], VH[0])
    Tn = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, exp_piphi, U[1], VH[0], U[0], VH[1], U[0], VH[0])
    
    T, Timp0, ln_normfact, ln_normfact_imp = hotrg3d.ynearest_two_point_func_renorm(T, T0, Tn, Dcut, XLOOPS, YLOOPS, ZLOOPS)
    trace2 = oe.contract("iijjkk", T)
    trace_imp2 = oe.contract("iijjkk", Timp0)
    normfact_imp2 = cp.exp(cp.sum(ln_normfact_imp))

    del T, T0

    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace2) / V
    n = (beta/2) * (cp.exp(mu)*normfact_imp1*trace_imp1/trace1 - cp.exp(-mu)*normfact_imp2*trace_imp2/trace2)

    return lnZoV, n


def internal_energy_hotrg(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    import trg.HOTRG_3d_QR as hotrg
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    U  = [U1, U2]
    VH = [VH1, VH2]
    s0 = [cp.ndarray]*2
    s1 = [cp.ndarray]*2

    I = cp.ones(len(phi))
    s0[0] = cp.exp(1j*phi)
    s1[0] = cp.exp(-1j*phi)
    s0[1] = cp.exp(-1j*phi)
    s1[1] = cp.exp(1j*phi)

    Tpure = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, I, U[1], VH[0], U[0], VH[1], U[0], VH[0])

    #a = cp.zeros(2,dtype=cp.complex128)

    rangei = [0,1,2]
    deltamu = [mu, 0, 0]
    locn = [{"X":0, "Y":0, "Z":1}, {"X":1, "Y":0, "Z":0}, {"X":0, "Y":1, "Z":0}]
    TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "Z": ZLOOPS}

    two_point_func = 0.0
    for i,dm,loc in zip(rangei, deltamu, locn):
        #term1
        T0 = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, s0[0], U[1], VH[0], U[0], VH[1], U[0], VH[0])
        Tn = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, s1[0], U[1], VH[0], U[0], VH[1], U[0], VH[0])

        T  = {"Tensor": Tpure.copy(), 
          "factor": {}}
        
        T0 = {"Tensor": T0,
              "loc"   : {"X":0, "Y":0, "Z":0},
              "factor": {}}

        Tn = {"Tensor": Tn, 
              "loc"   : loc}

        T, T0 = hotrg.znearest_two_point_func_renorm(T, T0, Tn, Dcut, TOT_RGSTEPS)

        TrT  = oe.contract("xxyyzz", T["Tensor"])
        TrT0 = oe.contract("xxyyzz", T0["Tensor"])

        T0factor = cp.asarray(list(T0["factor"].values()))
        fact0 = cp.exp(cp.sum(T0factor))
        two_point_func += 0.5*cp.exp(dm)*fact0*TrT0/TrT

        #term2
        T0 = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, s0[1], U[1], VH[0], U[0], VH[1], U[0], VH[0])
        Tn = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, s1[1], U[1], VH[0], U[0], VH[1], U[0], VH[0])

        T  = {"Tensor": Tpure.copy(), 
          "factor": {}}
        
        T0 = {"Tensor": T0,
              "loc"   : {"X":0, "Y":0, "Z":0},
              "factor": {}}

        Tn = {"Tensor": Tn, 
              "loc"   : loc}

        T, T0 = hotrg.znearest_two_point_func_renorm(T, T0, Tn, Dcut, TOT_RGSTEPS)

        TrT  = oe.contract("xxyyzz", T["Tensor"])
        TrT0 = oe.contract("xxyyzz", T0["Tensor"])

        T0factor = cp.asarray(list(T0["factor"].values()))
        fact0 = cp.exp(cp.sum(T0factor))
        two_point_func += 0.5*cp.exp(-dm)*fact0*TrT0/TrT


    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    Tfactor = cp.asarray(list(T["factor"].values()))
    lnZoV = cp.sum(Tfactor) + cp.log(TrT) / V
    energy = -two_point_func/3
    return lnZoV, energy


def particle_number_hotrg2(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    import trg.HOTRG_3d_QR as hotrg
    
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    U  = [U1, U2]
    VH = [VH1, VH2]

    I = cp.ones(len(phi), dtype=cp.complex128)
    exp_piphi = cp.exp( 1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    Tpure = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, I, U[1], VH[0], U[0], VH[1], U[0], VH[0])
    TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "Z": ZLOOPS}

    #part1
    T0 = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, exp_piphi, U[1], VH[0], U[0], VH[1], U[0], VH[0])
    Tn = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, exp_miphi, U[1], VH[0], U[0], VH[1], U[0], VH[0])

    T  = {"Tensor": Tpure.copy(), 
          "factor": {}}
        
    T0 = {"Tensor": T0,
          "loc"   : {"X":0, "Y":0, "Z":0},
          "factor": {}}

    Tn = {"Tensor": Tn, 
          "loc"   : {"X":0, "Y":0, "Z":1}}

    T, T0 = hotrg.two_point_function(T, T0, Tn, Dcut, TOT_RGSTEPS)

    TrT  = oe.contract("xxyyzz", T["Tensor"])
    TrT0 = oe.contract("xxyyzz", T0["Tensor"])

    T0factor = cp.asarray(list(T0["factor"].values()))
    fact0 = cp.exp(cp.sum(T0factor))
    part1 = cp.exp( mu)*fact0*TrT0/TrT
    exp1 = fact0*TrT0/TrT
    
    #part2
    T0 = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, exp_miphi, U[1], VH[0], U[0], VH[1], U[0], VH[0])
    Tn = 0.5 * oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", wphi, exp_piphi, U[1], VH[0], U[0], VH[1], U[0], VH[0])

    T  = {"Tensor": Tpure.copy(), 
          "factor": {}}
        
    T0 = {"Tensor": T0,
          "loc"   : {"X":0, "Y":0, "Z":0},
          "factor": {}}

    Tn = {"Tensor": Tn, 
          "loc"   : {"X":0, "Y":0, "Z":1}}
    
    T, T0 = hotrg.two_point_function(T, T0, Tn, Dcut, TOT_RGSTEPS)

    TrT  = oe.contract("xxyyzz", T["Tensor"])
    TrT0 = oe.contract("xxyyzz", T0["Tensor"])

    T0factor = cp.asarray(list(T0["factor"].values()))
    fact0 = cp.exp(cp.sum(T0factor))
    part2 = cp.exp(-mu)*fact0*TrT0/TrT
    exp2  = fact0*TrT0/TrT


    print("<exp{{ i(θ(0,0,0)-θ(0,0,1))}}>=", exp1)
    print("<exp{{-i(θ(0,0,0)-θ(0,0,1))}}>=", exp2)

    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    Tfactor = cp.asarray(list(T["factor"].values()))
    lnZoV = cp.sum(Tfactor) + cp.log(TrT) / V

    n = (beta/2) * (part1 - part2)

    return lnZoV, n


def particle_number_atrg2(beta, mu, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):

    from tensor_class.tensor_class import ATRG_Tensor as Tensor
    def to_tensor_gl(U, VH, w, Dcut:int, S)->Tensor:
        from utility.randomized_svd import rsvd_for_3dATRG_tensor_init
        A = 0.5 * oe.contract("a,a,ai,ja,ak->ijka", w, S, U[1], VH[0], U[0])
        B = oe.contract("la,am,na->almn", VH[1], U[0], VH[0])

        A = cp.reshape(A, (Dcut,Dcut,Dcut,K))
        B = cp.reshape(B, (K,Dcut,Dcut,Dcut))

        rs=cp.random.RandomState(1234)
        u, s, vh = rsvd_for_3dATRG_tensor_init(A, B, Dcut, n_oversamples=2*Dcut, n_power_iter=Dcut,seed=rs)
        return u,s,vh
    
    import trg.ATRG_3d_new as atrg
    
    U1, VH1, U2, VH2, phi, wphi = ti().__init_tensor_component_parts_finit_density__(beta, mu, Dcut)
    U  = [U1, U2]
    VH = [VH1, VH2]

    I = cp.ones(len(phi), dtype=cp.complex128)
    exp_piphi = cp.exp( 1j*phi)
    exp_miphi = cp.exp(-1j*phi)

    TOT_RGSTEPS = {"X": XLOOPS, "Y": YLOOPS, "T": ZLOOPS}

    #part1
    u0, s0, vh0 = to_tensor_gl(U, VH, wphi, Dcut,  I)
    T  = Tensor(u0, s0, vh0, 3, False, {})
        
    u0, s0, vh0 = to_tensor_gl(U, VH, wphi, Dcut, exp_piphi)
    T0 = Tensor(u0, s0, vh0, 3, True, {"X":0, "Y":0, "T":0})

    u0, s0, vh0 = to_tensor_gl(U, VH, wphi, Dcut, exp_miphi)
    Tn = Tensor(u0, s0, vh0, 3, True, {"X":0, "Y":0, "T":1})

    T, T0 = atrg.two_point_function(T, T0, Tn, Dcut, TOT_RGSTEPS)

    TrT = T.trace()
    TrT0 = T0.trace()

    Tfactor  = T.get_normalization_const()
    T0factor = T0.get_normalization_const()

    T0factor = cp.asarray(T0factor)
    fact0 = cp.exp(cp.sum(T0factor))
    part1 = cp.exp( mu)*fact0*TrT0/TrT
    exp1 = fact0*TrT0/TrT
    
    #part2
    u0, s0, vh0 = to_tensor_gl(U, VH, wphi, Dcut,  I)
    T  = Tensor(u0, s0, vh0, 3, False, {})
        
    u0, s0, vh0 = to_tensor_gl(U, VH, wphi, Dcut, exp_miphi)
    T0 = Tensor(u0, s0, vh0, 3, True, {"X":0, "Y":0, "T":0})

    u0, s0, vh0 = to_tensor_gl(U, VH, wphi, Dcut, exp_piphi)
    Tn = Tensor(u0, s0, vh0, 3, True, {"X":0, "Y":0, "T":1})

    T, T0 = atrg.two_point_function(T, T0, Tn, Dcut, TOT_RGSTEPS)

    TrT = T.trace()
    TrT0 = T0.trace()

    Tfactor  = T.get_normalization_const()
    T0factor = T0.get_normalization_const()

    T0factor = cp.asarray(T0factor)
    fact0 = cp.exp(cp.sum(T0factor))
    part2 = cp.exp(-mu)*fact0*TrT0/TrT
    exp2  = fact0*TrT0/TrT


    print("<exp{{ i(θ(0,0,0)-θ(0,0,1))}}>=", exp1)
    print("<exp{{-i(θ(0,0,0)-θ(0,0,1))}}>=", exp2)

    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(Tfactor) + cp.log(TrT) / V

    n = (beta/2) * (part1 - part2)

    return lnZoV, n