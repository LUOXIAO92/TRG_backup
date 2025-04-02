import sys
import numpy as np
import cupy as cp
import time
import opt_einsum as oe
from cuquantum import contract
from utility.randomized_svd import rsvd_for_3dATRG_tensor_inte
from tensor_class.tensor_class import Tensor

OUTPUT_DIR = sys.argv[8]

def tensor_rearrange(T:Tensor, direction:str, which:str):
    if direction == "X" or direction == "x":
        if which == "rearrange":
            U_temp  = T.U 
            T.U  = T.VH
            T.VH = U_temp
            del U_temp
            T.U  = cp.transpose(T.U , (2,1,3,0))
            T.VH = cp.transpose(T.VH, (3,1,0,2))
            return T

        elif which == "restore":
            U_temp  = T.U 
            T.U  = T.VH
            T.VH = U_temp
            del U_temp
            T.U  = cp.transpose(T.U , (2,1,3,0))
            T.VH = cp.transpose(T.VH, (3,1,0,2))
            return T

    elif direction == "Y" or direction == "y":
        if which == "rearrange":
            T.U  = cp.transpose(T.U , (2,0,1,3))
            T.VH = cp.transpose(T.VH, (0,3,1,2))
            return T
        elif which == "restore":
            T.U  = cp.transpose(T.U , (1,2,0,3))
            T.VH = cp.transpose(T.VH, (0,2,3,1))
            return T

    elif direction == "Z" or direction == "z":
        return T

def entanglement_filter(A, B, C, D, Dcut, epsilon):
    def filter(env_T_sqr, epsilon):
        _, s, vh = cp.linalg.svd(env_T_sqr, epsilon)
        u = cp.conj(vh.T)
        u = cp.reshape(u, newshape=(Dcut,Dcut,Dcut*Dcut))
        t = oe.contract("abi->i", u)
        t_filter = t * (s / (s + epsilon**2))
        #with open(OUTPUT_DIR+"/filter.dat", "w") as out:
        #    ss = s / (s + epsilon**2)
        #    for sss, tt, ttf in zip(ss, t, t_filter):
        #        out.write("{:.12e} {:.12e} {:.12e}\n".format(sss, tt.real, ttf.real))
        R = oe.contract("i,abi->ab", t_filter, cp.conj(u))
        u, s, vh = cp.linalg.svd(R)
        u  = oe.contract("ai,i->ai", u, cp.sqrt(s))
        vh = oe.contract("i,ib->ib", cp.sqrt(s), vh)
        return u, vh

    AdagA_i = oe.contract("cabA,cabα->Aα"  , cp.conj(A), A)
    CdagC_i = oe.contract("IefB,iefβ->IBiβ", cp.conj(C), C)
    DdagD_i = oe.contract("BhXg,βhxg->BXβx", cp.conj(D), D)
    AdagA_o = oe.contract("aXbA,axbα->AXαx", cp.conj(A), A)
    BdagB_o = oe.contract("AIab,αiab->AIαi", cp.conj(B), B)
    DdagD_o = oe.contract("Bhab,βhab->Bβ"  , cp.conj(D), D)

    #AB X direction
    inside  = oe.contract("Aα,AIwd,αiyd,IBiβ,xBzβ->wxyz", AdagA_i, cp.conj(B), B, CdagC_i, DdagD_i)
    outside = oe.contract("Awαy,AIαi,IxdB,izdβ,Bβ->wxyz", AdagA_o, BdagB_o, cp.conj(C), C, DdagD_o)
    env_T_sqr = oe.contract("iakb,jalb->ijkl", inside, outside)
    env_T_sqr = cp.reshape(env_T_sqr, newshape=(Dcut*Dcut, Dcut*Dcut))
    print("enviorment tensor hermit err", cp.linalg.norm(env_T_sqr-cp.conj(env_T_sqr.T)))
    print("enviorment tensor hermit relative err", cp.linalg.norm(env_T_sqr-cp.conj(env_T_sqr.T))/cp.linalg.norm(env_T_sqr))
    u, vh = filter(env_T_sqr, epsilon)
    #A = oe.contract("TbYα,xb->TxYα", A, vh)
    #B = oe.contract("αtay,aX->αtXy", B, u)

    #CD X direction
    env_T_sqr = oe.contract("aibk,ajbl->ijkl", inside, outside)
    env_T_sqr = cp.reshape(env_T_sqr, newshape=(Dcut*Dcut, Dcut*Dcut))
    print("enviorment tensor hermit err", cp.linalg.norm(env_T_sqr-cp.conj(env_T_sqr.T)))
    print("enviorment tensor hermit relative err", cp.linalg.norm(env_T_sqr-cp.conj(env_T_sqr.T))/cp.linalg.norm(env_T_sqr))
    u, vh = filter(env_T_sqr, epsilon)
    #C = oe.contract("TbYα,xb->TxYα", C, vh)
    #D = oe.contract("αtay,aX->αtXy", D, u)
    del inside, outside

    #BC T direction
    upside   = oe.contract("Aα,AIXd,αixd,BXbx,BJbj->IiJj", AdagA_i, cp.conj(B), B, AdagA_o, BdagB_o)
    downside = oe.contract("IBiβ,XBxβ,JXyA,jxya,Aa->IiJj", CdagC_i, DdagD_i, cp.conj(C), C, DdagD_o)
    env_T_sqr = oe.contract("AaJj,BbJj->ABab", upside, downside)
    env_T_sqr = cp.reshape(env_T_sqr, newshape=(Dcut*Dcut, Dcut*Dcut))
    print("enviorment tensor hermit err", cp.linalg.norm(env_T_sqr-cp.conj(env_T_sqr.T)))
    print("enviorment tensor hermit relative err", cp.linalg.norm(env_T_sqr-cp.conj(env_T_sqr.T))/cp.linalg.norm(env_T_sqr))
    u, vh = filter(env_T_sqr, epsilon)
    #B = oe.contract("αaXy,at->αtXy", B, u)
    #C = oe.contract("bxYα,Tb->TxYα", C, vh)
    del upside, downside

    return A, B, C, D


def leg_swarping(T0:Tensor, T1:Tensor, Dcut:int):
    #step (a)
    A = T1.U
    B = oe.contract("a,abcd->abcd", T1.s, T1.VH)
    C = oe.contract("abcd,d->abcd", T0.U, T0.s)
    D = T0.VH

    epsilon = 1e-4
    A, B, C, D = entanglement_filter(A, B, C, D, Dcut, epsilon)

    #step (b)~(c)
    UM, sM, VMH = rsvd_for_3dATRG_tensor_inte(B, C, Dcut, n_oversamples=2*Dcut, n_power_iter=Dcut)
    del B, C

    with open(OUTPUT_DIR+"/leg_exchange_sv_lx{:}_ly{:}_lt{:}.dat".format(count_xloop, count_yloop, count_zloop), "w") as out:
        for s in sM:
            out.write("{:.12e}\n".format(s))
    
    X = oe.contract("abcd,d->abcd", UM, cp.sqrt(sM))
    Y = oe.contract("a,abcd->abcd", cp.sqrt(sM), VMH)
    del UM, VMH, sM

    return A, X, Y, D

def squeezer(A, X, Y, D, Dcut:int):
    #step (d)
    MLdagML = oe.contract("abjc,bdie,aflc,fdke->ijkl", cp.conj(Y), cp.conj(D), Y, D)
    MRMRdag = oe.contract("ajbc,cide,albf,fkde->ijkl", A, X, cp.conj(A), cp.conj(X))
    MBdagMB = oe.contract("abcj,bdei,afcl,fdek->ijkl", cp.conj(Y), cp.conj(D), Y, D)
    MFMFdag = oe.contract("abjc,cdie,ablf,fdke->ijkl", A, X, cp.conj(A), cp.conj(X))
    MLdagML = cp.reshape(MLdagML, (Dcut*Dcut, Dcut*Dcut))
    MRMRdag = cp.reshape(MRMRdag, (Dcut*Dcut, Dcut*Dcut))
    MBdagMB = cp.reshape(MBdagMB, (Dcut*Dcut, Dcut*Dcut))
    MFMFdag = cp.reshape(MFMFdag, (Dcut*Dcut, Dcut*Dcut))

    vL, eL, _ = cp.linalg.svd(MLdagML)
    vR, eR, _ = cp.linalg.svd(MRMRdag)
    vB, eB, _ = cp.linalg.svd(MBdagMB)
    vF, eF, _ = cp.linalg.svd(MFMFdag)
    del MLdagML, MRMRdag, MBdagMB, MFMFdag


    RL = oe.contract("a,ab->ab", cp.sqrt(eL), cp.conj(vL.T))
    RR = oe.contract("ab,b->ab", vR, cp.sqrt(eR))
    RB = oe.contract("a,ab->ab", cp.sqrt(eB), cp.conj(vB.T))
    RF = oe.contract("ab,b->ab", vF, cp.sqrt(eF))

    U1, s1, V1H = cp.linalg.svd(RL@RR)
    U1H = cp.conj(U1.T)
    s1inv = (1 / s1)#.astype(cp.complex128)
    V1  = cp.conj(V1H.T)
    U1H = U1H[:Dcut,:]
    s1inv = s1inv[:Dcut]
    V1 = V1[:,:Dcut]
    del U1, s1, V1H, eL, vL, eR, vR

    U2, s2, V2H = cp.linalg.svd(RB@RF)
    U2H = cp.conj(U2.T)
    s2inv = (1 / s2)#.astype(cp.complex128)
    V2  = cp.conj(V2H.T)
    U2H = U2H[:Dcut,:]
    s2inv = s2inv[:Dcut]
    V2 = V2[:,:Dcut]
    del U2, s2, V2H, eB, vB, eF, vF

    P1 = oe.contract("a,ab,bc->ac", cp.sqrt(s1inv), U1H, RL)
    P2 = oe.contract("ab,bc,c->ac", RR, V1, cp.sqrt(s1inv))
    P3 = oe.contract("ab,bc,c->ac", RF, V2, cp.sqrt(s2inv))
    P4 = oe.contract("a,ab,bc->ac", cp.sqrt(s2inv), U2H, RB)
    del U1H, s1inv, V1, U2H, s2inv, V2
    
    P1 = cp.reshape(P1, (Dcut, Dcut, Dcut))
    P2 = cp.reshape(P2, (Dcut, Dcut, Dcut))
    P3 = cp.reshape(P3, (Dcut, Dcut, Dcut))
    P4 = cp.reshape(P4, (Dcut, Dcut, Dcut))

    return P1, P2, P3, P4

def new_tensor(A, X, Y, D, P1, P2, P3, P4, Dcut:int):
    #step (e)
    optimize={'slicing': {'min_slices': 4}}
    G = contract("xab,Ycd,Zbde,eaci->ZxYi", P1, P4, A, X, optimize=optimize)
    H = contract("iebd,ezac,abX,cdy->izXy", Y, D, P2, P3, optimize=optimize)

    G = cp.reshape(G, (Dcut*Dcut*Dcut, Dcut))
    H = cp.reshape(H, (Dcut, Dcut*Dcut*Dcut))
    
    UG, sG, VGH = cp.linalg.svd(G, full_matrices=False)
    UH, sH, VHH = cp.linalg.svd(H, full_matrices=False)
    del G, H

    UG  = cp.reshape(UG , (Dcut,Dcut,Dcut,Dcut))
    UH  = cp.reshape(UH , (Dcut,Dcut))
    VGH = cp.reshape(VGH, (Dcut,Dcut))
    VHH = cp.reshape(VHH, (Dcut,Dcut,Dcut,Dcut))

    K = oe.contract("a,aj,jb,b->ab", sG, VGH, UH, sH)
    UK, sK, VKH = cp.linalg.svd(K)

    U  = oe.contract("ZxYa,ai->ZxYi", UG, UK)
    VH = oe.contract("ib,bzXy->izXy", VKH, VHH)
    del UK, VKH

    T = Tensor(U, sK, VH)

    return T


def atrg_pure_tensor(T:Tensor, Dcut:int, direction:str):
    """
    T_{z'xyzx'y'} ≒ U_{z'xy,i} s_{i} VH_{i,zx'y'}\\
    Dcut: internal degrees of freedom of tensor\\
    i = 1~Dcut
    """

    #rearrange tensor legs
    T = tensor_rearrange(T, direction, "rearrange")

    #step (a)~(c)
    A, X, Y, D = leg_swarping(T, T, Dcut)
    
    #step (d)
    P1, P2, P3, P4 = squeezer(A, X, Y, D, Dcut)

    #step (e)
    T = new_tensor(A, X, Y, D, P1, P2, P3, P4, Dcut)

    #restore tensor legs to z'xyzx'y'
    T = tensor_rearrange(T, direction, "restore")

    return T

def atrg_impuer_tensor_2to1imp(T:Tensor, T0:Tensor, T1:Tensor, Dcut:int, direction:str):
    """
    T_{z'xyzx'y'} ≒ U_{z'xy,i} s_{i} VH_{i,zx'y'}\\
    Dcut: internal degrees of freedom of tensor\\
    i = 1~Dcut
    """

    #rearrange tensor legs
    t0 =time.time()
    T  = tensor_rearrange(T , direction, "rearrange")
    T0 = tensor_rearrange(T0, direction, "rearrange")
    T1 = tensor_rearrange(T1, direction, "rearrange")

    #step (a)~(c)
    t1 = time.time()
    A, X, Y, D = leg_swarping(T, T, Dcut)
    Aimp, Ximp, Yimp, Dimp = leg_swarping(T0, T1, Dcut)
    del T1
    t2 = time.time()

    #step (d)
    P1, P2, P3, P4 = squeezer(A, X, Y, D, Dcut)
    t3 = time.time()

    #step (e)
    T  = new_tensor(A, X, Y, D, P1, P2, P3, P4, Dcut)
    T0 = new_tensor(Aimp, Ximp, Yimp, Dimp, P1, P2, P3, P4, Dcut)
    t4 = time.time()

    #restore tensor legs to y'xyx'
    T  = tensor_rearrange(T , direction, "restore")
    T0 = tensor_rearrange(T0, direction, "restore")
    t5 = time.time()

    print("AXYD {:.2e} s, squeezer {:.2e} s, new tensor {:.2e} s".format(t2-t1,t3-t2,t4-t3))
    return T, T0

def atrg_impuer_tensor_1imp(T:Tensor, T0:Tensor, Dcut:int, direction:str):
    #rearrange tensor legs
    t0 =time.time()
    T  = tensor_rearrange(T , direction, "rearrange")
    T0 = tensor_rearrange(T0, direction, "rearrange")

    #step (a)~(c)
    t1 = time.time()
    A, X, Y, D = leg_swarping(T, T, Dcut)
    Aimp, Ximp, Yimp, Dimp = leg_swarping(T0, T, Dcut)
    t2 = time.time()

    #step (d)
    P1, P2, P3, P4 = squeezer(A, X, Y, D, Dcut)
    t3 = time.time()

    #step (e)
    T  = new_tensor(A, X, Y, D, P1, P2, P3, P4, Dcut)
    T0 = new_tensor(Aimp, Ximp, Yimp, Dimp, P1, P2, P3, P4, Dcut)
    t4 = time.time()

    #restore tensor legs to y'xyx'
    T  = tensor_rearrange(T , direction, "restore")
    T0 = tensor_rearrange(T0, direction, "restore")
    t5 = time.time()

    print("AXYD {:.2e} s, squeezer {:.2e} s, new tensor {:.2e} s".format(t2-t1,t3-t2,t4-t3))
    return T, T0



def __tensor_normalization__(T:Tensor):
    c = cp.max(T.s)
    T.s = T.s / c
    return T, c

def pure_tensor_renorm(T:Tensor, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    global count_xloop
    global count_yloop
    global count_zloop
    global count_totloop

    count_xloop = 0
    count_yloop = 0
    count_zloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+ZLOOPS+1, dtype=cp.float64)
    T, c = __tensor_normalization__(T)
    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)

    with open(OUTPUT_DIR+"/tensor_sv_lx{:}_ly{:}_lt{:}.dat".format(count_xloop, count_yloop, count_zloop), "w") as out:
        for s in T.s:
            out.write("{:.12e}\n".format(s))

    while (count_xloop < XLOOPS or count_yloop < YLOOPS or count_zloop < ZLOOPS):
        t0 = time.time()
        if count_zloop < ZLOOPS:
            count_zloop += 1
            count_totloop += 1
            T = atrg_pure_tensor(T, Dcut, "Z")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
        t1 = time.time()
        trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
        print("TrT",trace)
        I1 = oe.contract("ijka,ijkb->ab", cp.conj(T.U), T.U)
        I2 = oe.contract("aijk,bijk->ab", T.VH, cp.conj(T.VH))
        print("U^†_(aijk)U_(ijkb)", cp.linalg.norm(I1)**2, cp.trace(I1))
        print("V^†_(aijk)V_(ijkb)", cp.linalg.norm(I2)**2, cp.trace(I2))
        print("loop {:2d} finish. time: {:.6f} s\n".format(count_totloop, t1-t0))

        with open(OUTPUT_DIR+"/tensor_sv_lx{:}_ly{:}_lt{:}.dat".format(count_xloop, count_yloop, count_zloop), "w") as out:
            for s in T.s:
                out.write("{:.12e}\n".format(s))

        t0 = time.time()
        if count_xloop < XLOOPS:
            count_xloop += 1
            count_totloop += 1
            T = atrg_pure_tensor(T, Dcut, "X")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
        t1 = time.time()
        trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
        print("TrT",trace)
        I1 = oe.contract("ijka,ijkb->ab", cp.conj(T.U), T.U)
        I2 = oe.contract("aijk,bijk->ab", T.VH, cp.conj(T.VH))
        print("U^†_(aijk)U_(ijkb)", cp.linalg.norm(I1)**2, cp.trace(I1))
        print("V^†_(aijk)V_(ijkb)", cp.linalg.norm(I2)**2, cp.trace(I2))
        print("loop {:2d} finish. time: {:.6f} s\n".format(count_totloop, t1-t0))

        with open(OUTPUT_DIR+"/tensor_sv_lx{:}_ly{:}_lt{:}.dat".format(count_xloop, count_yloop, count_zloop), "w") as out:
            for s in T.s:
                out.write("{:.12e}\n".format(s))

        t0 = time.time()
        if count_yloop < YLOOPS:
            count_yloop += 1
            count_totloop += 1
            T = atrg_pure_tensor(T, Dcut, "Y")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
        t1 = time.time()
        trace = oe.contract("ijka,a,aijk", T.U, T.s, T.VH)
        print("TrT",trace)
        I1 = oe.contract("ijka,ijkb->ab", cp.conj(T.U), T.U)
        I2 = oe.contract("aijk,bijk->ab", T.VH, cp.conj(T.VH))
        print("U^†_(aijk)U_(ijkb)", cp.linalg.norm(I1)**2, cp.trace(I1))
        print("V^†_(aijk)V_(ijkb)", cp.linalg.norm(I2)**2, cp.trace(I2))
        print("loop {:2d} finish. time: {:.6f} s\n".format(count_totloop, t1-t0))

        with open(OUTPUT_DIR+"/tensor_sv_lx{:}_ly{:}_lt{:}.dat".format(count_xloop, count_yloop, count_zloop), "w") as out:
            for s in T.s:
                out.write("{:.12e}\n".format(s))

    return T, ln_normalized_factor

def ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    count_xloop = 0
    count_yloop = 0
    count_zloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+ZLOOPS+1, dtype=cp.float64)
    ln_normalized_factor_imp = cp.zeros(XLOOPS+YLOOPS+ZLOOPS+1, dtype=cp.float64)

    T    ,  c = __tensor_normalization__(T)
    Timp0, c0 = __tensor_normalization__(Timp0)
    Timp1, c1 = __tensor_normalization__(Timp1)
    trace = cp.einsum("ijka,a,aijk", T.U, T.s, T.VH)
    print("TrT",trace)
    
    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)
    ln_normalized_factor_imp[0] = cp.log(c0)+cp.log(c1) - 2*cp.log(c)

    while (count_xloop < XLOOPS or count_yloop < YLOOPS or count_zloop < ZLOOPS):

        print("loop {:2d} started".format(count_totloop), end=". ")
        t0 = time.time()
        if count_zloop < ZLOOPS:
            count_zloop += 1
            count_totloop += 1
            if count_totloop == 1:
                T, Timp0 = atrg_impuer_tensor_2to1imp(T, Timp0, Timp1, Dcut, "Z")
                T, c = __tensor_normalization__(T)
                Timp0, c0 = __tensor_normalization__(Timp0)
                ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
                ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)

            else:
                T, Timp0 = atrg_impuer_tensor_1imp(T, Timp0, Dcut, "Z")
                T, c = __tensor_normalization__(T)
                Timp0, c0 = __tensor_normalization__(Timp0)
                ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
                ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)
        t1 = time.time()
        trace = cp.einsum("ijka,a,aijk", T.U, T.s, T.VH)
        print("TrT",trace)
        print(" tot time: {:.2e} s\n".format(t1-t0))

        print("loop {:2d} started".format(count_totloop), end=". ")
        t0 = time.time()
        if count_xloop < XLOOPS:
            count_xloop += 1
            count_totloop += 1
            T, Timp0 = atrg_impuer_tensor_1imp(T, Timp0, Dcut, "X")
            T, c = __tensor_normalization__(T)
            Timp0, c0 = __tensor_normalization__(Timp0)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)
        t1 = time.time()
        trace = cp.einsum("ijka,a,aijk", T.U, T.s, T.VH)
        print("TrT",trace)
        print(" tot time: {:.2e} s".format(t1-t0))

        print("loop {:2d} started".format(count_totloop), end=". ")
        t0 = time.time()
        if count_yloop < YLOOPS:
            count_yloop += 1
            count_totloop += 1
            T, Timp0 = atrg_impuer_tensor_1imp(T, Timp0, Dcut, "Y")
            T, c = __tensor_normalization__(T)
            Timp0, c0 = __tensor_normalization__(Timp0)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)
        t1 = time.time()
        trace = cp.einsum("ijka,a,aijk", T.U, T.s, T.VH)
        print("TrT",trace)
        print(" tot time: {:.2e} s\n".format(t1-t0))

    return T, Timp0, ln_normalized_factor, ln_normalized_factor_imp