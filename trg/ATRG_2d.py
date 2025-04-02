import sys
import os
import numpy as np
import cupy as cp
import time
import opt_einsum as oe
from cuquantum import contract 

from utility.randomized_svd import rsvd
from tensor_class.tensor_class import Tensor

optimize={'slicing': {'min_slices': 4}}
OUTPUT_DIR = os.environ['OUTPUT_DIR']

def tensor_rearrange(T:Tensor, direction:str):
    if direction == "X" or direction == "x":
        U_temp  = T.U 
        T.U  = T.VH
        T.VH = U_temp
        del U_temp

        T.U  = cp.transpose(T.U , (2,1,0))
        T.VH = cp.transpose(T.VH, (2,1,0))

        return T

    elif direction == "Y" or direction == "y":
        return T

def intermedate_pure_tensor(T:Tensor, Dcut:int):
    #step (a)
    A = T.U
    D = T.VH

    #step (b)
    M = oe.contract("i,ial,ajk,k->ijkl", T.s, T.VH, T.U, T.s)

    #step (c)
    M = cp.reshape(M, (Dcut*Dcut, Dcut*Dcut))
    #UM, sM, VMH = rsvd(M, k=Dcut, n_oversamples=3*Dcut, n_power_iter=Dcut//2)
    UM, sM, VMH = cp.linalg.svd(M)
    __RG_flow_output__('swarping', sM)
    print("swarping error: {}".format(cp.sum(sM[Dcut:])/cp.sum(sM)))
    UM = UM[:,:Dcut]
    sM = sM[:Dcut]
    VMH= VMH[:Dcut,:]
    del M

    X = oe.contract("aj,j->aj", UM, cp.sqrt(sM))
    Y = oe.contract("j,ja->ja", cp.sqrt(sM), VMH)
    X = cp.reshape(X, (Dcut,Dcut,Dcut))
    Y = cp.reshape(Y, (Dcut,Dcut,Dcut))
    del UM, VMH

    return A, X, Y, D

def intermedate_tensor(T0:Tensor, T1:Tensor, Dcut:int):
    #step (a)
    A = T1.U
    D = T0.VH

    #step (b)
    #M_(α,x0',β,x1) = B_(α,i,x1) C_(i,x0',β)
    #M = oe.contract("i,ial,ajk,k->ijkl", T1.s, T1.VH, T0.U, T0.s)
    M = oe.contract("α,αiX,ixβ,β->αxβX", T1.s, T1.VH, T0.U, T0.s)

    #step (c)
    M = cp.reshape(M, (Dcut*Dcut, Dcut*Dcut))
    UM, sM, VMH = cp.linalg.svd(M)
    UM = UM[:,:Dcut]
    sM = sM[:Dcut]
    VMH= VMH[:Dcut,:]
    UM  = cp.reshape(UM, (Dcut,Dcut,Dcut))
    VMH = cp.reshape(VMH, (Dcut,Dcut,Dcut))
    del M

    X = oe.contract("abj,j->abj", UM, cp.sqrt(sM))
    Y = oe.contract("j,jab->jab", cp.sqrt(sM), VMH)
    
    del UM, VMH

    return A, X, Y, D

def leg_swapping(T:Tensor, T0:Tensor, T1:Tensor, Dcut:int):
    #step (a)
    A = T.U
    D = T.VH

    #step (b)
    M = oe.contract("i,ial,ajk,k->ijkl", T.s, T.VH, T.U, T.s)

    #step (c)
    M = cp.reshape(M, (Dcut*Dcut, Dcut*Dcut))
    UM, sM, VMH = cp.linalg.svd(M)
    __RG_flow_output__('swarping', sM)
    print("swarping error: {}".format(cp.sum(sM[Dcut:])/cp.sum(sM)))
    UM = UM[:,:Dcut]
    sM = sM[:Dcut]
    VMH= VMH[:Dcut,:]
    #del M

    X = oe.contract("aj,j->aj", UM, cp.sqrt(sM))
    Y = oe.contract("j,ja->ja", cp.sqrt(sM), VMH)
    X = cp.reshape(X, (Dcut,Dcut,Dcut))
    Y = cp.reshape(Y, (Dcut,Dcut,Dcut))

    Aimp = T1.U
    Dimp = T0.VH
    UM  = cp.reshape(UM,  (Dcut,Dcut,Dcut))
    VMH = cp.reshape(VMH, (Dcut,Dcut,Dcut))
    s0 = cp.diag(T0.s) if len(T0.s.shape) == 1 else T0.s
    s1 = cp.diag(T1.s) if len(T1.s.shape) == 1 else T1.s
    sMimp = oe.contract("ixk,iI,ItX,txJ,Jj,ljX->kl", cp.conj(UM), s1, T1.VH, T0.U, s0, cp.conj(VMH))
    us, ss, svh = cp.linalg.svd(sMimp)
    us = oe.contract("ki,i->ki", us, cp.sqrt(ss))
    svh = oe.contract("i,il->il", cp.sqrt(ss), svh)
    Ximp = oe.contract("ixk,kt->ixt", UM, us)
    Yimp = oe.contract("tl,ljX->tjX", svh, VMH)

    #err = cp.linalg.norm(cp.diag(sM) - sMimp) / cp.linalg.norm(cp.diag(sM))
    #print("err(sM,sMimp)=", err)
    #err = cp.linalg.norm(sM - ss) / cp.linalg.norm(cp.diag(sM))
    #print("err(sM,ss)=", err)
    #print("sM=",sM)
    #print("ss=",ss)

    #XY    = oe.contract("ixt,tjX->ixjX", X   , Y)
    #XYimp = oe.contract("ixt,tjX->ixjX", Ximp, Yimp)
    #XY    = cp.reshape(XY   , (Dcut*Dcut, Dcut*Dcut))
    #XYimp = cp.reshape(XYimp, (Dcut*Dcut, Dcut*Dcut))
    #err = cp.linalg.norm(XY-XYimp) / cp.linalg.norm(XY)
    #print("err(XY,XYimp)=",err)
    #err = cp.linalg.norm(XY-M) / cp.linalg.norm(M)
    #print("err(M,XY)=",err)
    #err = cp.linalg.norm(XYimp-M) / cp.linalg.norm(M)
    #print("err(M,XYimp)=",err)
    #err = cp.linalg.norm(Ximp-X) / cp.linalg.norm(X)
    #print("err(X,Ximp)=",err)
    #err = cp.linalg.norm(Yimp-Y) / cp.linalg.norm(Y)
    #print("err(Y,Yimp)=",err)

    return A, X, Y, D, Aimp, Ximp, Yimp, Dimp

def squeezer(A, X, Y, D, Dcut:int):
    #step (d)
    M1dagM1 = oe.contract("acj,cbi,adl,dbk->ijkl", cp.conj(Y), cp.conj(D), Y, D)
    M2M2dag = oe.contract("cia,bjc,dka,bld->ijkl", X, A, cp.conj(X), cp.conj(A))
    M1dagM1 = cp.reshape(M1dagM1, (Dcut*Dcut, Dcut*Dcut))
    M2M2dag = cp.reshape(M2M2dag, (Dcut*Dcut, Dcut*Dcut))

    #e1, v1 = cp.linalg.eigh(M1dagM1)
    #e2, v2 = cp.linalg.eigh(M2M2dag)
    v1, e1 , _ = cp.linalg.svd(M1dagM1)
    v2, e2 , _ = cp.linalg.svd(M2M2dag)
    e1 = e1.astype(cp.complex128)
    e2 = e2.astype(cp.complex128)
    del M1dagM1, M2M2dag

    R1 = oe.contract("a,ab->ab", cp.sqrt(e1), cp.conj(v1.T))
    R2 = oe.contract("ab,b->ab", v2, cp.sqrt(e2))
    U, s, VH = cp.linalg.svd(R1@R2)
    UH = cp.conj(U.T)
    sinv = (1 / s).astype(cp.complex128)
    V  = cp.conj(VH.T)
    del U, s, VH, e1, v1, e2, v2

    UH = UH[:Dcut,:]
    sinv = sinv[:Dcut]
    V = V[:,:Dcut]

    P1 = oe.contract("a,ab,bc->ac", cp.sqrt(sinv), UH, R1)
    P2 = oe.contract("ab,bc,c->ac", R2, V, cp.sqrt(sinv))
    del UH, sinv, V

    P1 = cp.reshape(P1, (Dcut, Dcut, Dcut))
    P2 = cp.reshape(P2, (Dcut, Dcut, Dcut))

    return P1, P2

def new_tensor(A, X, Y, D, P1, P2, Dcut:int):
    #step (e)
    G = oe.contract("jab,cak,ibc->ijk", P1, X, A)
    H = oe.contract("cja,icb,abk->ijk", D, Y, P2)

    G = cp.reshape(G, (Dcut*Dcut, Dcut))
    H = cp.reshape(H, (Dcut, Dcut*Dcut))
    
    UG, sG, VGH = cp.linalg.svd(G, full_matrices=False)
    UH, sH, VHH = cp.linalg.svd(H, full_matrices=False)
    del G, H

    UG  = cp.reshape(UG , (Dcut,Dcut,Dcut))
    UH  = cp.reshape(UH , (Dcut,Dcut))
    VGH = cp.reshape(VGH, (Dcut,Dcut))
    VHH = cp.reshape(VHH, (Dcut,Dcut,Dcut))

    K = oe.contract("a,aj,jb,b->ab", sG, VGH, UH, sH)
    UK, sK, VKH = cp.linalg.svd(K)

    U  = oe.contract("ija,ak->ijk", UG, UK)
    VH = oe.contract("ia,ajk->ijk", VKH, VHH)
    del UK, VKH

    T = Tensor(U, sK, VH)

    return T

def new_tensor_imp(A, X, Y, D,
                   Aimp, Ximp, Yimp, Dimp, 
                   P1, P2, Dcut:int):
    #step (e)
    G = oe.contract("jab,cak,ibc->ijk", P1, X, A)
    H = oe.contract("cja,icb,abk->ijk", D, Y, P2)
    G = cp.reshape(G, (Dcut*Dcut, Dcut))
    H = cp.reshape(H, (Dcut, Dcut*Dcut))
    
    UG, sG, VGH = cp.linalg.svd(G, full_matrices=False)
    UH, sH, VHH = cp.linalg.svd(H, full_matrices=False)
    #del G, H

    UG  = cp.reshape(UG , (Dcut,Dcut,Dcut))
    UH  = cp.reshape(UH , (Dcut,Dcut))
    VGH = cp.reshape(VGH, (Dcut,Dcut))
    VHH = cp.reshape(VHH, (Dcut,Dcut,Dcut))

    K = oe.contract("a,aj,jb,b->ab", sG, VGH, UH, sH)
    UK, sK, VKH = cp.linalg.svd(K)

    U  = oe.contract("ija,ak->ijk", UG, UK)
    VH = oe.contract("ia,ajk->ijk", VKH, VHH)
    #del UK, VKH

    T = Tensor(U, sK, VH)

    #impure tensor
    Gimp = oe.contract("jab,cak,ibc->ijk", P1, Ximp, Aimp)
    Himp = oe.contract("cja,icb,abk->ijk", Dimp, Yimp, P2)
    Kimp  = oe.contract("Yxα,Yxk,kyX,ByX->αB", cp.conj(UG), Gimp, Himp, cp.conj(VHH))
    sKimp = oe.contract("αi,αB,jB->ij", cp.conj(UK), Kimp, cp.conj(VKH))
    Uimp = oe.contract("Yxα,αi->Yxi", UG, UK)
    VHimp = oe.contract("jB,ByX->jyX", VKH, VHH)
    Timp = Tensor(Uimp, sKimp, VHimp)

    #G = cp.reshape(G, (Dcut*Dcut, Dcut))
    #H = cp.reshape(H, (Dcut, Dcut*Dcut))
    #Gimp = cp.reshape(Gimp, (Dcut*Dcut, Dcut))
    #Himp = cp.reshape(Himp, (Dcut, Dcut*Dcut))
    #GGHimp = Gimp #@ cp.conj(Gimp.T)
    #HHHimp = Himp #@ cp.conj(Himp.T)
    #GGH = G #@ cp.conj(G.T)
    #HHH = H #@ cp.conj(H.T)
    #normGGHimp = cp.linalg.norm(GGHimp)
    #normHHHimp = cp.linalg.norm(HHHimp)
    #normGGH = cp.linalg.norm(GGH)
    #normHHH = cp.linalg.norm(HHH)
    #normK = cp.linalg.norm(K)
    #normKimp = cp.linalg.norm(Kimp)
    #print("norm(Gimp)=", normGGHimp)
    #print("norm(G)=", normGGH)
    #print("norm(Himp)=", normHHHimp)
    #print("norm(H)=", normHHH)
    #print("norm(Kimp)=", normKimp)
    #print("norm(K)=", normK)
    #
    #
    #errGGimp = cp.linalg.norm(G-Gimp)/cp.linalg.norm(G)
    #errHHimp = cp.linalg.norm(H-Himp)/cp.linalg.norm(H)
    #print("err(G,Gimp)=", errGGimp)
    #print("err(H,Himp)=", errHHimp)
    #errKKimp = cp.linalg.norm(K-Kimp)/cp.linalg.norm(K)
    #print("err(K,Kimp)=", errKKimp)
    #errUUimp = cp.linalg.norm(U-Uimp)/cp.linalg.norm(U)
    #errVVimp = cp.linalg.norm(VH-VHimp)/cp.linalg.norm(VH)
    #print("err(U,Uimp)=", errUUimp)
    #print("err(VH,VHimp)=", errVVimp)

    return T, Timp

def atrg_pure_tensor(T:Tensor, Dcut:int, direction:str):
    """
    T_{y'xyx'} ≒ U_{y'x,i} s_{i} VH_{i,yx'}\\
    or T_{x'yxy'} ≒ U_{x'y,i} s_{i} VH_{i,xy'}
    """


    TrT = oe.contract("yxi,i,iyx", T.U, T.s, T.VH)
    print("TrT= {:}".format(TrT))    

    #rearrange tensor legs
    T = tensor_rearrange(T, direction)

    #step (a)~(c)
    A, X, Y, D = intermedate_tensor(T, T, Dcut)

    #step (d)
    P1, P2 = squeezer(A, X, Y, D, Dcut)

    #step (e)
    T = new_tensor(A, X, Y, D, P1, P2, Dcut)

    #restore tensor legs to y'xyx'
    T = tensor_rearrange(T, direction)

    return T

def atrg_impuer_tensor_2to1imp(T:Tensor, T0:Tensor, T1:Tensor, Dcut:int, direction:str):
    """
    T_{y'xyx'} ≒ U_{y'x,i} s_{i} VH_{i,yx'}\\
    or T_{x'yxy'} ≒ U_{x'y,i} s_{i} VH_{i,xy'}
    """

    #rearrange tensor legs
    T  = tensor_rearrange(T , direction)
    T0 = tensor_rearrange(T0, direction)
    T1 = tensor_rearrange(T1, direction)

    #TrT  = oe.contract("yxi,i,iyx", T.U, T.s, T.VH)
    #TrT0 = oe.contract("yxi,ij,jyx", T0.U, T0.s, T0.VH)
    #TrT1 = oe.contract("yxi,ij,jyx", T0.U, T0.s, T0.VH)
    #print("TrT ", TrT )
    #print("TrT0", TrT0)
    #print("TrT1", TrT1)

    #step (a)~(c)
    #A, X, Y, D = intermedate_pure_tensor(T, Dcut)
    #Aimp, Ximp, Yimp, Dimp = intermedate_tensor(T0, T1, Dcut)
    #del T1

    A, X, Y, D, Aimp, Ximp, Yimp, Dimp = leg_swapping(T, T0, T1, Dcut)
    del T1

    #errAAimp = cp.linalg.norm(A-Aimp) / cp.linalg.norm(A)
    #errXXimp = cp.linalg.norm(X-Ximp) / cp.linalg.norm(X)
    #errYYimp = cp.linalg.norm(Y-Yimp) / cp.linalg.norm(Y)
    #errDDimp = cp.linalg.norm(D-Dimp) / cp.linalg.norm(D)
    #print("err(A,Aimp)", errAAimp)
    #print("err(X,Ximp)", errXXimp)
    #print("err(Y,Yimp)", errYYimp)
    #print("err(D,Dimp)", errDDimp)

    #step (d)
    P1, P2 = squeezer(A, X, Y, D, Dcut)

    #step (e)
    #T  = new_tensor(A, X, Y, D, P1, P2, Dcut)
    #T0 = new_tensor(Aimp, Ximp, Yimp, Dimp, P1, P2, Dcut)

    T, T0 = new_tensor_imp(A, X, Y, D, Aimp, Ximp, Yimp, Dimp, P1, P2, Dcut)

    #TrT  = oe.contract("yxi,i,iyx", T.U, T.s, T.VH)
    #TrT0 = oe.contract("yxi,ij,jyx", T0.U, T0.s, T0.VH)
    #print("TrT ", TrT )
    #print("TrT0", TrT0)

    #restore tensor legs to y'xyx'
    T  = tensor_rearrange(T , direction)
    T0 = tensor_rearrange(T0, direction)

    return T, T0

def atrg_impuer_tensor_1imp(T:Tensor, T0:Tensor, Dcut:int, direction:str):
    #rearrange tensor legs
    T  = tensor_rearrange(T , direction)
    T0 = tensor_rearrange(T0, direction)

    #TrT  = oe.contract("yxi,i,iyx", T.U, T.s, T.VH)
    #TrT0 = oe.contract("yxi,ij,jyx", T0.U, T0.s, T0.VH)
    #print("TrT ", TrT )
    #print("TrT0", TrT0)

    #step (a)~(c)
    #A, X, Y, D = intermedate_pure_tensor(T, Dcut)
    #Aimp, Ximp, Yimp, Dimp = intermedate_tensor(T0, T, Dcut)

    A, X, Y, D, Aimp, Ximp, Yimp, Dimp = leg_swapping(T, T0, T, Dcut)

    #errAAimp = cp.linalg.norm(A-Aimp) / cp.linalg.norm(A)
    #errXXimp = cp.linalg.norm(X-Ximp) / cp.linalg.norm(X)
    #errYYimp = cp.linalg.norm(Y-Yimp) / cp.linalg.norm(Y)
    #errDDimp = cp.linalg.norm(D-Dimp) / cp.linalg.norm(D)
    #print("err(A,Aimp)", errAAimp)
    #print("err(X,Ximp)", errXXimp)
    #print("err(Y,Yimp)", errYYimp)
    #print("err(D,Dimp)", errDDimp)

    #step (d)
    P1, P2 = squeezer(A, X, Y, D, Dcut)

    #step (e)
    #T  = new_tensor(A, X, Y, D, P1, P2, Dcut)
    #T0 = new_tensor(Aimp, Ximp, Yimp, Dimp, P1, P2, Dcut)

    T, T0 = new_tensor_imp(A, X, Y, D, Aimp, Ximp, Yimp, Dimp, P1, P2, Dcut)

    #TrT  = oe.contract("yxi,i,iyx", T.U, T.s, T.VH)
    #TrT0 = oe.contract("yxi,ij,jyx", T0.U, T0.s, T0.VH)
    #print("TrT ", TrT )
    #print("TrT0", TrT0)

    #restore tensor legs to y'xyx'
    T  = tensor_rearrange(T , direction)
    T0 = tensor_rearrange(T0, direction)

    return T, T0


def __RG_flow_output__(type:str, data):
    if type == "swarping":
        with open("{:}/swarping_Lx2^{:}_Lt2^{:}.dat".format(OUTPUT_DIR, count_xloop, count_yloop), "w") as svout:
            for s in data:
                svout.write("{:.12e}\n".format(s))

    elif type == "tensor":
        with open("{:}/tensor_Lx2^{:}_Lt2^{:}.dat".format(OUTPUT_DIR, count_xloop, count_yloop), "w") as svout:
            for s in data:
               svout.write("{:.12e}\n".format(s))

def __tensor_normalization__(T:Tensor):
    if len(T.s.shape) == 1:
        Tr = oe.contract("yxi,i,iyx", T.U, T.s, T.VH)
        print("pure tensor TrT=",Tr)
    elif len(T.s.shape) == 2:
        Tr = oe.contract("yxi,ij,jyx", T.U, T.s, T.VH)
        print("impure tensor TrTimp=",Tr)
    #c = cp.abs(c)
    c = cp.max(cp.abs(T.s))
    print("c=",c)
    T.s = T.s / c
    return T, c

def pure_tensor_renorm(T:Tensor, Dcut:int, XLOOPS:int, YLOOPS:int):
    global count_xloop
    global count_yloop
    count_xloop = 0
    count_yloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+1, dtype=cp.float64)
    T, c = __tensor_normalization__(T)
    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)
    __RG_flow_output__('tensor', T.s)

    while (count_xloop < XLOOPS or count_yloop < YLOOPS):
        
        if count_yloop < YLOOPS:
            t0 = time.time()
            count_yloop += 1
            count_totloop += 1
            T = atrg_pure_tensor(T, Dcut, "Y")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            t1 = time.time()
            print("loop {:2d} finish. time: {:.6f} s\n".format(count_totloop, t1-t0))
            __RG_flow_output__('tensor', T.s)

        
        if count_xloop < XLOOPS:
            t0 = time.time()
            count_xloop += 1
            count_totloop += 1
            T = atrg_pure_tensor(T, Dcut, "X")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            t1 = time.time()
            print("loop {:2d} finish. time: {:.6f} s\n".format(count_totloop, t1-t0))
            __RG_flow_output__('tensor', T.s)

    return T, ln_normalized_factor

def ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut:int, XLOOPS:int, YLOOPS:int):
    global count_xloop
    global count_yloop
    count_xloop = 0
    count_yloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+1, dtype=cp.float64)
    ln_normalized_factor_imp = cp.zeros(XLOOPS+YLOOPS+1, dtype=cp.float64)

    T, c = __tensor_normalization__(T)
    Timp0, c0 = __tensor_normalization__(Timp0)
    Timp1, c1 = __tensor_normalization__(Timp1)

    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)
    ln_normalized_factor_imp[0] = cp.log(c0)+cp.log(c1) - 2*cp.log(c)

    while (count_xloop < XLOOPS or count_yloop < YLOOPS):

        t0 = time.time()
        if count_yloop < YLOOPS:
            count_yloop += 1
            count_totloop += 1
            if count_totloop == 1:
                T, Timp0 = atrg_impuer_tensor_2to1imp(T, Timp0, Timp1, Dcut, "Y")
                T, c = __tensor_normalization__(T)
                Timp0, c0 = __tensor_normalization__(Timp0)
                ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
                ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)

            else:
                T, Timp0 = atrg_impuer_tensor_1imp(T, Timp0, Dcut, "Y")
                T, c = __tensor_normalization__(T)
                Timp0, c0 = __tensor_normalization__(Timp0)
                ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
                ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)
        t1 = time.time()
        print("loop {:2d} finish. time: {:.6f} s".format(count_totloop, t1-t0))

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
        print("loop {:2d} finish. time: {:.6f} s".format(count_totloop, t1-t0))

    return T, Timp0, ln_normalized_factor, ln_normalized_factor_imp