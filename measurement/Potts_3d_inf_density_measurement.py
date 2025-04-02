import sys
import gc
import time
import zarr

import numpy as np
import cupy as cp
import opt_einsum as oe

from operator import mul
from functools import reduce

sys.path.append('../')
from tensor_init.Potts_model import infinite_density as potts_model

import trg.ATRG_3d as atrg3d
import trg.HOTRG_3d as hotrg3d
from tensor_class.tensor_class import Tensor
from utility.randomized_svd import rsvd, rsvd_for_3dATRG_tensor_init


OUTPUT_DIR = sys.argv[8]

def Potts_model_tensor(U, VH, S, w, Dcut:int):
    from math import ceil

    A = oe.contract("a,a,ai,ja,ak->ijka", w, S, U[1], VH[0], U[0])
    B = oe.contract("la,am,na->almn", VH[1], U[0], VH[0])
    
    #print("tensor rsvd init start")
    t0 = time.time()
    rs=cp.random.RandomState(1234)
    u, s, vh = rsvd_for_3dATRG_tensor_init(A, B, Dcut, n_oversamples=2*Dcut, n_power_iter=Dcut,seed=rs)
    T = Tensor(u, s, vh)
    del u, s, vh

    #q = len(S)
    #T = oe.contract("ijka,almn->ijklmn", A, B)
    #T_shape = T.shape
    #T = cp.reshape(T, newshape=(reduce(mul, T_shape[:3]), reduce(mul, T_shape[3:])))
    #u, s, vh = cp.linalg.svd(T)
    #u  = u[:,:q]
    #s  = s[:q]
    #vh = vh[:q,:]
    #u  = cp.reshape(u , newshape=(q,q,q,q)).astype(cp.complex128)
    #vh = cp.reshape(vh, newshape=(q,q,q,q)).astype(cp.complex128)

    #u = cp.pad(u , ((0,Dcut-q), (0,Dcut-q), (0,Dcut-q), (0,Dcut-q)))
    #s = cp.pad(s , (0,Dcut-q))
    #vh= cp.pad(vh, ((0,Dcut-q), (0,Dcut-q), (0,Dcut-q), (0,Dcut-q)))

    #T = Tensor(u, s, vh)
    #del u, s, vh, A, B
    t1 = time.time()
    #print("tensor rsvd init finished, time= {:.2f} s".format(t1-t0))

    return T


def ln_Z_over_V(q, k, h, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    Potts = potts_model(q, k, h, Dcut)
    u, vh, _ = Potts.tensor_component_3d_classical()
    u  = cp.pad( u, ((0,0), (0,Dcut-q)))
    vh = cp.pad(vh, ((0,Dcut-q), (0,0)))
    
    w = cp.ones(Potts.q)
    I = cp.ones_like(w)
    U = [u, u]
    VH = [vh, vh]
    T = Potts_model_tensor(U, VH, I, w, Dcut)
    del U, VH, I, w


    #T.U  = cp.pad(T.U, ((0,Dcut-q), (0,Dcut-q), (0,Dcut-q), (0,Dcut-q)))
    #T.s  = cp.pad(T.s, (0,Dcut-q))
    #T.VH = cp.pad(T.VH, ((0,Dcut-q), (0,Dcut-q), (0,Dcut-q), (0,Dcut-q)))

    T, ln_normfact = atrg3d.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS, ZLOOPS)

    trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    lnZoV = lnZoV
    
    return lnZoV

def ln_Z_over_V_hotrg(q, k, h, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    Potts = potts_model(q, k, h, Dcut)
    u, vh, _ = Potts.tensor_component_3d_classical()
    w = cp.ones(Potts.q)
    I = cp.ones_like(w)
    u  = cp.pad( u, ((0,0), (0,Dcut-q)))
    vh = cp.pad(vh, ((0,Dcut-q), (0,0)))

    w = cp.ones(Potts.q)
    I = cp.ones_like(w)
    T = oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", w, I, u, vh, u, vh, u, vh)

    T, ln_normfact = hotrg3d.pure_tensor_renorm(T, Dcut, XLOOPS, YLOOPS, ZLOOPS)

    trace = oe.contract("iijjkk", T)
    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    lnZoV = lnZoV
    
    return lnZoV


def entanglement_entropy(q, k, h, Dcut:int, lxA:int, lyA:int, lxB:int, lyB:int, lt:int):
    Nx_A = int(2**lxA)
    Ny_A = int(2**lyA)
    Nx_B = int(2**lxB)
    Ny_B = int(2**lyB)
    Nt   = int(2**lt)

    Potts = potts_model(q, k, h, Dcut)
    u, vh, _ = Potts.tensor_component_3d()
    u  = cp.pad( u, ((0,0), (0,Dcut-q)))
    vh = cp.pad(vh, ((0,Dcut-q), (0,0)))
    
    w = cp.ones(Potts.q)
    I = cp.ones_like(w)
    U = [u, u]
    VH = [vh, vh]
    T = Potts_model_tensor(U, VH, I, w, Dcut)
    del U, VH, I, w

    from trg import gilt_ATRG_3d as giltatrg3d
    T, ln_normfact = atrg3d.pure_tensor_renorm(T, Dcut, lxA, lyA, lt)

    trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
    print("TrT", trace)
    print("normalization factors ln(c_i)/2^i:", ln_normfact)

    TrTBTA = oe.contract("icbf,f,fiab,laeg,g,glce", T.U, T.s, T.VH, T.U, T.s, T.VH)
    print("TrTBTA=UsVHUsVH",TrTBTA)
    Dens_Mat = oe.contract("kcbf,f,fiab,laeg,g,gjce->ijkl", T.U, T.s, T.VH, T.U, T.s, T.VH) / TrTBTA
    TrDens_Mat = oe.contract("ijij", Dens_Mat)
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(Dcut*Dcut, Dcut*Dcut))
    _, e_Dens_Mat, _ = cp.linalg.svd(Dens_Mat)
    print("dens_mat hermit err",(cp.linalg.norm(Dens_Mat-cp.conj(Dens_Mat.T))/cp.linalg.norm(Dens_Mat)))
    print("Tr(ρ)=",TrDens_Mat)

    with open(OUTPUT_DIR+"/densM_sv.dat", "w") as out:
        for e in e_Dens_Mat:
            out.write("{:.12e}\n".format(e))

    #lnZ/V
    VA = Nx_A*Ny_A*Nt
    VB = Nx_B*Ny_B*Nt
    ln_ZoverV = (VA*cp.sum(ln_normfact) + VB*cp.sum(ln_normfact) + cp.log(TrTBTA))/(VA+VB)
    ln_ZoverV = ln_ZoverV

    #STE
    STE = - cp.sum( e_Dens_Mat * cp.log(e_Dens_Mat))

    #SEE
    rho_A = oe.contract("icbf,f,fiab,laeg,g,gkce->kl", T.U, T.s, T.VH, T.U, T.s, T.VH) / TrTBTA
    TrArho_A = oe.contract("ii", rho_A)
    print("Tr_A(ρ_A)=", TrArho_A)    
    _, e_A, _ = cp.linalg.svd(rho_A)
    e_A = e_A
    SEE = -cp.sum(e_A * cp.log(e_A))
    SEE = SEE
    print("rho_A hermit err",(cp.linalg.norm(rho_A-cp.conj(rho_A.T))/cp.linalg.norm(rho_A)))
    print("SEE=",SEE)

    with open(OUTPUT_DIR+"/rhoA_sv.dat", "w") as out:
        for e in e_A:
            out.write("{:.12e}\n".format(e))

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
    ln_coV = ln_normfact
    E = -cp.sum(e_Dens_Mat * cp.log(e_Dens_Mat)) - VA*cp.sum(ln_coV) - VB*cp.sum(ln_coV)
    E = E / (VA+VB)

    #T.U  =  T.U.get()
    #T.s  =  T.s.get()
    #T.VH = T.VH.get()
    #ln_normfact = ln_normfact.get()
    #zarr.save(T.U , OUTPUT_DIR+"/tensor.zarr", "/U", overwrite=True)
    #zarr.save(T.s , OUTPUT_DIR+"/tensor.zarr", "/s", overwrite=True)
    #zarr.save(T.VH, OUTPUT_DIR+"/tensor.zarr", "/VH", overwrite=True)
    #zarr.save(ln_normfact, OUTPUT_DIR+"/tensor.zarr", "/lnCoV", overwrite=True)
    del T

    return ln_ZoverV, STE, SEE, E, sn_str

def entanglement_entropy_hotrg(q, k, h, Dcut:int, lxA:int, lyA:int, lxB:int, lyB:int, lt:int):
    import trg.HOTRG_3d as hotrg
    Nx_A = int(2**lxA)
    Ny_A = int(2**lyA)
    Nx_B = int(2**lxB)
    Ny_B = int(2**lyB)
    Nt   = int(2**lt)

    Potts = potts_model(q, k, h, Dcut)
    u, vh, _ = Potts.tensor_component_3d()
    w = cp.ones(Potts.q)
    I = cp.ones_like(w)
    u  = cp.pad( u, ((0,0), (0,Dcut-q)))
    vh = cp.pad(vh, ((0,Dcut-q), (0,0)))

    w = cp.ones(Potts.q)
    I = cp.ones_like(w)
    T = oe.contract("a,a,aT,xa,aY,ta,aX,ya->xXyYtT", w, I, u, vh, u, vh, u, vh)

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

    with open(OUTPUT_DIR+"/densM_sv.dat", "w") as out:
        for e in e_Dens_Mat:
            out.write("{:.12e}\n".format(e))
    

    #lnZ/V
    VA = Nx_A*Ny_A*Nt
    VB = Nx_B*Ny_B*Nt
    ln_ZoverV = (VA*cp.sum(ln_normfact_A) + VB*cp.sum(ln_normfact_A) + cp.log(TrTBTA))/(VA+VB)
    print("lnZ/V= {:12e}".format(ln_ZoverV.real))

    #STE
    #STE = - cp.sum( e_Dens_Mat * cp.log(e_Dens_Mat))/cp.sum(e_Dens_Mat) + cp.log(cp.sum(e_Dens_Mat))
    print("Eigv_ρ=",e_Dens_Mat)
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

    with open(OUTPUT_DIR+"/rhoA_sv.dat", "w") as out:
        for e in e_A:
            out.write("{:.12e}\n".format(e))

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

def internal_energy(q, k, h, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    Potts = potts_model(q, k, h, Dcut)
    #u, vh, _ = Potts.tensor_component_3d_classical()
    u, vh, _ = Potts.tensor_component_3d()

    def impure_tensor(u, vh, Dcut:int):
        q = u.shape[0]

        T0 = oe.contract("ΦT,xΦ,ΦX,yΦ,ΦY,tΦ->ΦTxXyYt", u, vh, u, vh, u, vh)
        T0_shape = T0.shape
        T0 = cp.reshape(T0, newshape=(reduce(mul, T0_shape[:2]), reduce(mul, T0_shape[2:])))
        u0, s0, v0h = cp.linalg.svd(T0)
        u0 = u0[:,:q]
        s0 = s0[:q]
        v0h = v0h[:q,:]
        del T0

        T3 = oe.contract("tΦ,xΦ,ΦX,yΦ,ΦY,ΦT->ΦtxXyYT", vh, vh, u, vh, u, u)
        T3_shape = T3.shape
        T3 = cp.reshape(T3, newshape=(reduce(mul, T3_shape[:2]), reduce(mul, T3_shape[2:])))
        u3, s3, v3h = cp.linalg.svd(T3)
        u3 = u3[:,:q]
        s3 = s3[:q]
        v3h = v3h[:q,:]
        del T3

        M = oe.contract("Φα,α,Φβ,β->αβ", u0, s0, u3, s3)
        uM, sM, vMh = cp.linalg.svd(M)
        del M

        v0h = cp.reshape(v0h, newshape=(v0h.shape[0], T0_shape[2], T0_shape[3], T0_shape[4], T0_shape[5], T0_shape[6]))
        T0 = oe.contract("T,αT,αxXyYt->TxYtXy", cp.sqrt(sM), vMh, v0h)
        del u0, s0, v0h
        
        T0 = cp.reshape(T0, newshape=(reduce(mul, T0.shape[:3]), reduce(mul, T0.shape[3:])))
        k = min(T0.shape[0], Dcut)
        U0, S0, V0H = rsvd(T0, k, seed=1234)
        U0  = cp.reshape( U0, newshape=(T0_shape[6], T0_shape[2], T0_shape[5], S0.shape[0]))
        V0H = cp.reshape(V0H, newshape=(S0.shape[0], T0_shape[1], T0_shape[3], T0_shape[4]))
        T0imp = Tensor(U0, S0, V0H)
        del U0, S0, V0H, T0

        v3h = cp.reshape(v3h, newshape=(v3h.shape[0], T3_shape[2], T3_shape[3], T3_shape[4], T3_shape[5], T3_shape[6]))
        T3 = oe.contract("t,βt,βxXyYT->TxYtXy", cp.sqrt(sM), vMh, v3h)
        del u3, s3, v3h
        
        T3 = cp.reshape(T3, newshape=(reduce(mul, T3.shape[:3]), reduce(mul, T3.shape[3:])))
        k = min(T3.shape[0], Dcut)
        U3, S3, V3H = rsvd(T3, k, seed=1234)
        U3  = cp.reshape( U3, newshape=(T3_shape[6], T3_shape[2], T3_shape[5], S3.shape[0]))
        V3H = cp.reshape(V3H, newshape=(S3.shape[0], T3_shape[1], T3_shape[3], T3_shape[4]))
        T3imp = Tensor(U3, S3, V3H)
        del U3, S3, V3H, T3

        p = T0imp.s.shape[0]
        if p > Dcut:
            T0imp.U  = cp.pad(T0imp.U , ((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, 0)))
            T0imp.VH = cp.pad(T0imp.VH, ((0, 0), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
            T3imp.U  = cp.pad(T3imp.U , ((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, 0)))
            T3imp.VH = cp.pad(T3imp.VH, ((0, 0), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        else:
            T0imp.s  = cp.pad(T0imp.s, (0, Dcut-p))
            T0imp.U  = cp.pad(T0imp.U , ((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-p)))
            T0imp.VH = cp.pad(T0imp.VH, ((0, Dcut-p), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))

            T3imp.s  = cp.pad(T3imp.s, (0, Dcut-p))
            T3imp.U  = cp.pad(T3imp.U , ((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-p)))
            T3imp.VH = cp.pad(T3imp.VH, ((0, Dcut-p), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))

        return T0imp, T3imp

    def impure_tensor2(u, vh, Dcut:int):
        q = u.shape[0]
        M = oe.contract("aΦ,Φb,cΦ,Φd,eΦ,Φz,fΦ,Φg,hΦ,Φi,zΦ,Φj->abcdefghij",
                        vh, u, vh, u, vh, u, 
                        vh, u, vh, u, vh, u)
        M_shape = M.shape
        M = cp.reshape(M, newshape=(reduce(mul, M_shape[:5]), reduce(mul, M_shape[5:])))
        uM, sM, vMh = cp.linalg.svd(M)
        del M

        print("sM", sM[:Dcut])

        uM  = uM[:,:q]
        sM  = sM[:q]
        vMh = vMh[:q,:]
        uM  = cp.reshape(uM , newshape=(q,q,q,q,q,q))
        vMh = cp.reshape(vMh, newshape=(q,q,q,q,q,q))

        T0 = oe.contract("xXyYtT,T->TxYtXy", uM, cp.sqrt(sM))
        T0_shape = T0.shape
        T0 = cp.reshape(T0, newshape=(reduce(mul, T0_shape[:3]), reduce(mul, T0_shape[3:])))
        U0, s0, V0H = cp.linalg.svd(T0)
        print("s0", s0)
        del T0
        U0 = U0[:,:q]
        s0 = s0[:q]
        V0H = V0H[:q,:]
        U0 = cp.reshape(U0, newshape=(q,q,q,q))
        V0H = cp.reshape(V0H, newshape=(q,q,q,q))
        U0 = cp.pad(U0, pad_width=((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        s0 = cp.pad(s0, pad_width=(0, Dcut-q))
        V0H = cp.pad(V0H, pad_width=((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        T0 = Tensor(U0, s0, V0H)
        
        T3 = oe.contract("t,txXyYT->TxYtXy", cp.sqrt(sM), vMh)
        T3_shape = T3.shape
        T3 = cp.reshape(T3, newshape=(reduce(mul, T3_shape[:3]), reduce(mul, T3_shape[3:])))
        U3, s3, V3H = cp.linalg.svd(T3)
        print("s3", s3)
        del T3
        U3 = U3[:,:q]
        s3 = s3[:q]
        V3H = V3H[:q,:]
        U3 = cp.reshape(U3, newshape=(q,q,q,q))
        V3H = cp.reshape(V3H, newshape=(q,q,q,q))
        U3 = cp.pad(U3, pad_width=((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        s3 = cp.pad(s3, pad_width=(0, Dcut-q))
        V3H = cp.pad(V3H, pad_width=((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        T3 = Tensor(U3, s3, V3H)

        return T0, T3
        

    T0imp, T3imp = impure_tensor2(u, vh, Dcut)

    u  = cp.pad( u, ((0,0), (0,Dcut-q)))
    vh = cp.pad(vh, ((0,Dcut-q), (0,0)))
    
    w = cp.ones(Potts.q)
    I = cp.ones_like(w)
    U = [u, u]
    VH = [vh, vh]
    T = Potts_model_tensor(U, VH, I, w, Dcut)
    del U, VH, I, w

    T, Timp0, ln_normfact, ln_normfact_imp = atrg3d.ynearest_two_point_func_renorm(T, T0imp, T3imp, Dcut, XLOOPS, YLOOPS, ZLOOPS)

    trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
    ln_normfact = ln_normfact

    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    trace_imp = oe.contract("ijka,a,aijk", Timp0.U, Timp0.s, Timp0.VH)
    normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    two_point_z = normfact_imp * trace_imp / trace
    E = -3*two_point_z
    
    return lnZoV, E

def internal_energy_hotrg(q, k, h, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    Potts = potts_model(q, k, h, Dcut)
    u, vh, _ = Potts.tensor_component()

    def impure_tensor(u, vh, Dcut:int):
        q = u.shape[0]

        T0 = oe.contract("ΦT,xΦ,ΦX,yΦ,ΦY,tΦ->ΦTxXyYt", u, vh, u, vh, u, vh)
        T0_shape = T0.shape
        T0 = cp.reshape(T0, newshape=(reduce(mul, T0_shape[:2]), reduce(mul, T0_shape[2:])))
        u0, s0, v0h = cp.linalg.svd(T0)
        u0 = u0[:,:q]
        s0 = s0[:q]
        v0h = v0h[:q,:]
        del T0

        T3 = oe.contract("tΦ,xΦ,ΦX,yΦ,ΦY,ΦT->ΦtxXyYT", vh, vh, u, vh, u, u)
        T3_shape = T3.shape
        T3 = cp.reshape(T3, newshape=(reduce(mul, T3_shape[:2]), reduce(mul, T3_shape[2:])))
        u3, s3, v3h = cp.linalg.svd(T3)
        u3 = u3[:,:q]
        s3 = s3[:q]
        v3h = v3h[:q,:]
        del T3

        M = oe.contract("Φα,α,Φβ,β->αβ", u0, s0, u3, s3)
        uM, sM, vMh = cp.linalg.svd(M)
        del M

        v0h = cp.reshape(v0h, newshape=(v0h.shape[0], T0_shape[2], T0_shape[3], T0_shape[4], T0_shape[5], T0_shape[6]))
        T0 = oe.contract("T,αT,αxXyYt->TxYtXy", cp.sqrt(sM), vMh, v0h)
        del u0, s0, v0h
        
        T0 = cp.reshape(T0, newshape=(reduce(mul, T0.shape[:3]), reduce(mul, T0.shape[3:])))
        k = min(T0.shape[0], Dcut)
        U0, S0, V0H = rsvd(T0, k, seed=1234)
        U0  = cp.reshape( U0, newshape=(T0_shape[6], T0_shape[2], T0_shape[5], S0.shape[0]))
        V0H = cp.reshape(V0H, newshape=(S0.shape[0], T0_shape[1], T0_shape[3], T0_shape[4]))
        T0imp = Tensor(U0, S0, V0H)
        del U0, S0, V0H, T0

        v3h = cp.reshape(v3h, newshape=(v3h.shape[0], T3_shape[2], T3_shape[3], T3_shape[4], T3_shape[5], T3_shape[6]))
        T3 = oe.contract("t,βt,βxXyYT->TxYtXy", cp.sqrt(sM), vMh, v3h)
        del u3, s3, v3h
        
        T3 = cp.reshape(T3, newshape=(reduce(mul, T3.shape[:3]), reduce(mul, T3.shape[3:])))
        k = min(T3.shape[0], Dcut)
        U3, S3, V3H = rsvd(T3, k, seed=1234)
        U3  = cp.reshape( U3, newshape=(T3_shape[6], T3_shape[2], T3_shape[5], S3.shape[0]))
        V3H = cp.reshape(V3H, newshape=(S3.shape[0], T3_shape[1], T3_shape[3], T3_shape[4]))
        T3imp = Tensor(U3, S3, V3H)
        del U3, S3, V3H, T3

        p = T0imp.s.shape[0]
        if p > Dcut:
            T0imp.U  = cp.pad(T0imp.U , ((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, 0)))
            T0imp.VH = cp.pad(T0imp.VH, ((0, 0), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
            T3imp.U  = cp.pad(T3imp.U , ((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, 0)))
            T3imp.VH = cp.pad(T3imp.VH, ((0, 0), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        else:
            T0imp.s  = cp.pad(T0imp.s, (0, Dcut-p))
            T0imp.U  = cp.pad(T0imp.U , ((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-p)))
            T0imp.VH = cp.pad(T0imp.VH, ((0, Dcut-p), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))

            T3imp.s  = cp.pad(T3imp.s, (0, Dcut-p))
            T3imp.U  = cp.pad(T3imp.U , ((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-p)))
            T3imp.VH = cp.pad(T3imp.VH, ((0, Dcut-p), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))

        return T0imp, T3imp

    def impure_tensor2(u, vh, Dcut:int):
        q = u.shape[0]
        M = oe.contract("aΦ,Φb,cΦ,Φd,eΦ,Φz,fΦ,Φg,hΦ,Φi,zΦ,Φj->abcdefghij",
                        vh, u, vh, u, vh, u, 
                        vh, u, vh, u, vh, u)
        M_shape = M.shape
        M = cp.reshape(M, newshape=(reduce(mul, M_shape[:5]), reduce(mul, M_shape[5:])))
        uM, sM, vMh = cp.linalg.svd(M)
        del M

        print("sM", sM[:Dcut])

        uM  = uM[:,:q]
        sM  = sM[:q]
        vMh = vMh[:q,:]
        uM  = cp.reshape(uM , newshape=(q,q,q,q,q,q))
        vMh = cp.reshape(vMh, newshape=(q,q,q,q,q,q))

        T0 = oe.contract("xXyYtT,T->TxYtXy", uM, cp.sqrt(sM))
        T0_shape = T0.shape
        T0 = cp.reshape(T0, newshape=(reduce(mul, T0_shape[:3]), reduce(mul, T0_shape[3:])))
        U0, s0, V0H = cp.linalg.svd(T0)
        print("s0", s0)
        del T0
        U0 = U0[:,:q]
        s0 = s0[:q]
        V0H = V0H[:q,:]
        U0 = cp.reshape(U0, newshape=(q,q,q,q))
        V0H = cp.reshape(V0H, newshape=(q,q,q,q))
        U0 = cp.pad(U0, pad_width=((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        s0 = cp.pad(s0, pad_width=(0, Dcut-q))
        V0H = cp.pad(V0H, pad_width=((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        T0 = Tensor(U0, s0, V0H)
        
        T3 = oe.contract("t,txXyYT->TxYtXy", cp.sqrt(sM), vMh)
        T3_shape = T3.shape
        T3 = cp.reshape(T3, newshape=(reduce(mul, T3_shape[:3]), reduce(mul, T3_shape[3:])))
        U3, s3, V3H = cp.linalg.svd(T3)
        print("s3", s3)
        del T3
        U3 = U3[:,:q]
        s3 = s3[:q]
        V3H = V3H[:q,:]
        U3 = cp.reshape(U3, newshape=(q,q,q,q))
        V3H = cp.reshape(V3H, newshape=(q,q,q,q))
        U3 = cp.pad(U3, pad_width=((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        s3 = cp.pad(s3, pad_width=(0, Dcut-q))
        V3H = cp.pad(V3H, pad_width=((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        T3 = Tensor(U3, s3, V3H)

        return T0, T3
        

    T0imp, T3imp = impure_tensor2(u, vh, Dcut)

    u  = cp.pad( u, ((0,0), (0,Dcut-q)))
    vh = cp.pad(vh, ((0,Dcut-q), (0,0)))
    
    w = cp.ones(Potts.q)
    I = cp.ones_like(w)
    U = [u, u]
    VH = [vh, vh]
    T = Potts_model_tensor(U, VH, I, w, Dcut)
    del U, VH, I, w

    print("U :",T.U )
    print("VH:",T.VH)

    T, Timp0, ln_normfact, ln_normfact_imp = atrg3d.ynearest_two_point_func_renorm(T, T0imp, T3imp, Dcut, XLOOPS, YLOOPS, ZLOOPS)

    trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
    ln_normfact = ln_normfact

    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    trace_imp = oe.contract("ijka,a,aijk", Timp0.U, Timp0.s, Timp0.VH)
    normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    two_point_z = normfact_imp * trace_imp / trace
    E = -3*two_point_z
    
    return lnZoV, E

def internal_energy2(q, k, h, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    Potts = potts_model(q, k, h, Dcut)
    u, vh, Z = Potts.tensor_component()

    def energy_impure_tensor(u, vh, Dcut:int):
        q = u.shape[0]
        M = oe.contract("aΦ,Φb,cΦ,Φd,eΦ,Φz,fΦ,Φg,hΦ,Φi,zΦ,Φj->abcdefghij",
                        vh, u, vh, u, vh, u, 
                        vh, u, vh, u, vh, u)
        M_shape = M.shape
        M = cp.reshape(M, newshape=(reduce(mul, M_shape[:5]), reduce(mul, M_shape[5:])))
        uM, sM, vMh = cp.linalg.svd(M)
        del M

        uM  = uM[:,:q]
        sM  = sM[:q]
        vMh = vMh[:q,:]
        uM  = cp.reshape(uM , newshape=(q,q,q,q,q,q))
        vMh = cp.reshape(vMh, newshape=(q,q,q,q,q,q))

        T0 = oe.contract("xXyYtT,T->TxYtXy", uM, cp.sqrt(sM))
        T0_shape = T0.shape
        T0 = cp.reshape(T0, newshape=(reduce(mul, T0_shape[:3]), reduce(mul, T0_shape[3:])))
        U0, s0, V0H = cp.linalg.svd(T0)
        del T0
        U0 = U0[:,:q]
        s0 = s0[:q]
        V0H = V0H[:q,:]
        U0 = cp.reshape(U0, newshape=(q,q,q,q))
        V0H = cp.reshape(V0H, newshape=(q,q,q,q))
        U0 = cp.pad(U0, pad_width=((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        s0 = cp.pad(s0, pad_width=(0, Dcut-q))
        V0H = cp.pad(V0H, pad_width=((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        T0 = Tensor(U0, s0, V0H)
        
        T3 = oe.contract("t,txXyYT->TxYtXy", cp.sqrt(sM), vMh)
        T3_shape = T3.shape
        T3 = cp.reshape(T3, newshape=(reduce(mul, T3_shape[:3]), reduce(mul, T3_shape[3:])))
        U3, s3, V3H = cp.linalg.svd(T3)
        U3 = U3[:,:q]
        s3 = s3[:q]
        V3H = V3H[:q,:]
        U3 = cp.reshape(U3, newshape=(q,q,q,q))
        V3H = cp.reshape(V3H, newshape=(q,q,q,q))
        U3 = cp.pad(U3, pad_width=((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        s3 = cp.pad(s3, pad_width=(0, Dcut-q))
        V3H = cp.pad(V3H, pad_width=((0, Dcut-q), (0, Dcut-q), (0, Dcut-q), (0, Dcut-q)))
        T3 = Tensor(U3, s3, V3H)

        return T0, T3
        

    T0imp_e, T3imp_e = energy_impure_tensor(u, vh, Dcut)

    u  = cp.pad( u, ((0,0), (0,Dcut-q)))
    vh = cp.pad(vh, ((0,Dcut-q), (0,0)))
    
    w = cp.ones(Potts.q)
    I = cp.ones_like(w)
    U = [u, u]
    VH = [vh, vh]
    T = Potts_model_tensor(U, VH, I, w, Dcut)
    T0imp_p = Potts_model_tensor(U, VH, Z, w, Dcut)
    del U, VH, I, w, Z

    print("U :",T.U )
    print("VH:",T.VH)


    T0imp = [T0imp_e, T0imp_p]
    T3imp = [T3imp_e, T]

    T, Timp0, ln_normfact, ln_normfact_imp = atrg3d.ynearest_two_point_func_renorm(T, T0imp, T3imp, Dcut, XLOOPS, YLOOPS, ZLOOPS)

    trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
    ln_normfact = ln_normfact

    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    trace_imp = oe.contract("ijka,a,aijk", Timp0[0].U, Timp0[0].s, Timp0[0].VH)
    ln_normfact_imp = ln_normfact_imp[0]
    normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    two_point_z = normfact_imp * trace_imp / trace
    E = -3*two_point_z
    
    return lnZoV, E


def oder_parameter(q, k, h, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    Potts = potts_model(q, k, h, Dcut)
    u, vh, Z = Potts.tensor_component()

    u  = cp.pad( u, ((0,0), (0,Dcut-q)))
    vh = cp.pad(vh, ((0,Dcut-q), (0,0)))

    w = cp.ones(Potts.q)
    I = cp.ones_like(w)
    U = [u, u]
    VH = [vh, vh]
    T = Potts_model_tensor(U, VH, I, w, Dcut)
    T0imp = Potts_model_tensor(U, VH, Z, w, Dcut)
    T1imp = T
    del U, VH, I, w, Z

    T, Timp0, ln_normfact, ln_normfact_imp = atrg3d.ynearest_two_point_func_renorm(T, T0imp, T1imp, Dcut, XLOOPS, YLOOPS, ZLOOPS)
    #T, Timp0, ln_normfact, ln_normfact_imp = hotrg3d.ynearest_two_point_func_renorm(T, T0imp, T1imp, Dcut, XLOOPS, YLOOPS, ZLOOPS)

    trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    trace_imp = oe.contract("ijka,a,aijk", Timp0.U, Timp0.s, Timp0.VH)
    normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    two_point_z = normfact_imp * trace_imp / trace
    phi = two_point_z
    
    return lnZoV, phi


def oder_parameter_hotrg(q, k, h, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    Potts = potts_model(q, k, h, Dcut)
    u, vh, Z = Potts.tensor_component()

    u  = cp.pad( u, ((0,0), (0,Dcut-q)))
    vh = cp.pad(vh, ((0,Dcut-q), (0,0)))

    w = cp.ones(Potts.q)
    I = cp.ones_like(w)
    T = oe.contract("a,a,ai,ja,ak,la,am,na->ijklmn", w, I, u, vh, u, vh, u, vh)
    T0imp = oe.contract("a,a,ai,ja,ak,la,am,na->ijklmn", w, Z, u, vh, u, vh, u, vh)
    T1imp = T

    T, Timp0, ln_normfact, ln_normfact_imp = hotrg3d.ynearest_two_point_func_renorm(T, T0imp, T1imp, Dcut, XLOOPS, YLOOPS, ZLOOPS)

    trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
    V = 2**(XLOOPS+YLOOPS+ZLOOPS)
    lnZoV = cp.sum(ln_normfact) + cp.log(trace) / V

    trace_imp = oe.contract("ijka,a,aijk", Timp0.U, Timp0.s, Timp0.VH)
    normfact_imp = cp.exp(cp.sum(ln_normfact_imp))
    two_point_z = normfact_imp * trace_imp / trace
    phi = two_point_z
    
    return lnZoV, phi