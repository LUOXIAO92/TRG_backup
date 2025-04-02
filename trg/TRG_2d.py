import numpy as np
import cupy as cp
import sys
import time
from cuquantum import einsum
from mpi4py import MPI
from utility.randomized_svd import rsvd

import os
OUTPUT_DIR = os.environ['OUTPUT_DIR']

def SVD(T, Dcut, pattern:str):
    """
    pattern:
    A: T_{xy,x'y'}=U_{xy,i} s_{i} VH_{i,x'y'}
    B: T_{y'x,yx'}=U_{y'x,i} s_{i} VH_{i,yx'}

    returns:
    U=U_{abi}*sqrt(s_i)
    VH=sqrt(s_i)*VH_{icd}
    """
    if pattern == "A":
        T = cp.reshape(T, newshape=(Dcut*Dcut, Dcut*Dcut))
        U, s, VH = cp.linalg.svd(T)
        #U, s, VH = rsvd(T, k=Dcut, n_oversamples=int(Dcut*Dcut/2), n_power_iter=2)
        #T = cp.reshape(T, newshape=(Dcut, Dcut, Dcut, Dcut))

    elif pattern == "B":
        T = cp.transpose(T, axes=(3, 0, 1, 2))
        T = cp.reshape(T, newshape=(Dcut*Dcut, Dcut*Dcut))
        U, s, VH = cp.linalg.svd(T)
        #U, s, VH = rsvd(T, k=Dcut, n_oversamples=int(Dcut*Dcut/2), n_power_iter=2)
        #T = cp.reshape(T, newshape=(Dcut, Dcut, Dcut, Dcut))

    else:
        print("no such pattern")
        sys.exit(1)

    sv_sum = cp.sum(s)
    sv_1toDcut = cp.sum(s[:Dcut])
    sv_Dcuttoend = cp.sum(s[Dcut:])
    #print(s[:2*Dcut])
    #print("1~Dcut: {:6e} , Dcut~Dcut^2: {:.6e}".format(sv_1toDcut/sv_sum, sv_Dcuttoend/sv_sum))

    U  = cp.einsum("ai,i->ai", U , cp.sqrt(s))
    VH = cp.einsum("ia,i->ia", VH, cp.sqrt(s))

    U  = cp.reshape(U , newshape=(Dcut, Dcut, Dcut*Dcut))
    VH = cp.reshape(VH, newshape=(Dcut*Dcut, Dcut, Dcut))
    return U[:,:,:Dcut], VH[:Dcut,:,:], s

    #U  = cp.reshape(U , newshape=(Dcut, Dcut, Dcut))
    #VH = cp.reshape(VH, newshape=(Dcut, Dcut, Dcut))
    #return U, VH

def __new_tensor__(VH0, VH1, U0, U1, renormal_loop:int):

    if (renormal_loop % 2) == 1:
        T_new = einsum("iba,jcb,dck,adl->ijkl", VH0, VH1, U0, U1)
    elif (renormal_loop % 2) == 0: 
        T_new = einsum("iba,cbj,dck,lad->ijkl", VH1, U0, U1, VH0)
    
    return T_new

def TRG_pure_tensor(T, Dcut:int, renormal_loop:int):
    """
    initial configuration of the tensors is:\\
    ... T  T  T  T ... \\
    ... T  T  T  T ... \\
    ... T  T  T  T ... \\
    ... T  T  T  T ... 
    """
    #print(f"renormalization loops:{renormal_loop}")
    UB, VHB, _  = SVD(T , Dcut=Dcut, pattern="B")
    UA, VHA, _  = SVD(T , Dcut=Dcut, pattern="A")
    
    #if renormal_loop == 1:
    #    filename="TRG_sv/op2_{:d}_svd.dat".format(renormal_loop)
    #    out=open(filename, "w")
    #    out.write("#pure impure:{:}\n".format(renormal_loop))
    #    for i in range(Dcut*Dcut):
    #        out.write("{:d} {:.6e} {:.6e}\n".format(i+1, sa[i], s1[i]))
    #    out.close()

    if (renormal_loop % 2) == 1:
        T  = __new_tensor__(VHB, VHA, UB, UA, renormal_loop=renormal_loop)
        
    elif (renormal_loop % 2) == 0:
        T  = __new_tensor__(VHB, VHA, UB, UA, renormal_loop=renormal_loop)

    del UB, VHB, UA, VHA

    return T

def TRG_nearest_neighborhood(T, T0, T1, T2, T3, Dcut:int, renormal_loop:int):
    """
    initial configuration of the tensors is:\\
    ... T  T  T  T ... \\
    ... T  T1 T2 T ... \\
    ... T  T0 T3 T ... \\
    ... T  T  T  T ... 
    """
    #print(f"renormalization loops:{renormal_loop}")
    U0, VH0, _  = SVD(T0, Dcut=Dcut, pattern="B")
    U1, VH1, s1  = SVD(T1, Dcut=Dcut, pattern="A")
    U2, VH2, _  = SVD(T2, Dcut=Dcut, pattern="B")
    U3, VH3, _  = SVD(T3, Dcut=Dcut, pattern="A")
    UB, VHB, _  = SVD(T , Dcut=Dcut, pattern="B")
    UA, VHA, sa  = SVD(T , Dcut=Dcut, pattern="A")
    
    #if renormal_loop == 1:
    #    filename="TRG_sv/op2_{:d}_svd.dat".format(renormal_loop)
    #    out=open(filename, "w")
    #    out.write("#pure impure:{:}\n".format(renormal_loop))
    #    for i in range(Dcut*Dcut):
    #        out.write("{:d} {:.6e} {:.6e}\n".format(i+1, sa[i], s1[i]))
    #    out.close()

    if (renormal_loop % 2) == 1:
        T0 = __new_tensor__(VHB, VHA, U0, U1, renormal_loop=renormal_loop)
        T1 = __new_tensor__(VHB, VH1, U2, UA, renormal_loop=renormal_loop)
        T2 = __new_tensor__(VH2, VH3, UB, UA, renormal_loop=renormal_loop)
        T3 = __new_tensor__(VH0, VHA, UB, U3, renormal_loop=renormal_loop)
        T  = __new_tensor__(VHB, VHA, UB, UA, renormal_loop=renormal_loop)
        
    elif (renormal_loop % 2) == 0:
        T0 = __new_tensor__(VH0, VHA, UB, U3, renormal_loop=renormal_loop)
        T1 = __new_tensor__(VHB, VHA, U0, U1, renormal_loop=renormal_loop)
        T2 = __new_tensor__(VHB, VH1, U2, UA, renormal_loop=renormal_loop)
        T3 = __new_tensor__(VH2, VH3, UB, UA, renormal_loop=renormal_loop)
        T  = __new_tensor__(VHB, VHA, UB, UA, renormal_loop=renormal_loop)

    del U0, VH0, U1, VH1, U2, VH2, U3, VH3, UB, VHB, UA, VHA

    return T, T0, T1, T2, T3

def TRG_V_eq_4(T, T0, T1, T2, T3, Dcut:int, renormal_loop:int):
    """
    configuration of the tensors is:\\
    T1 T2 \\
    T0 T3 \\
    """
    #print(f"renormalization loops:{renormal_loop}")
    U0, VH0, _ = SVD(T0, Dcut=Dcut, pattern="B")
    U1, VH1, _ = SVD(T1, Dcut=Dcut, pattern="A")
    U2, VH2, _ = SVD(T2, Dcut=Dcut, pattern="B")
    U3, VH3, _ = SVD(T3, Dcut=Dcut, pattern="A")
    UB, VHB, _ = SVD(T , Dcut=Dcut, pattern="B")
    UA, VHA, _ = SVD(T , Dcut=Dcut, pattern="A")

    T0 = __new_tensor__(VH2, VH3, U0, U1, renormal_loop=renormal_loop)
    T1 = __new_tensor__(VH0, VH1, U2, U3, renormal_loop=renormal_loop)

    T  = __new_tensor__(VHB, VHA, UB, UA, renormal_loop=renormal_loop)

    del U0, VH0, U1, VH1, U2, VH2, U3, VH3, UB, VHB, UA, VHA, T2, T3

    return T, T0, T1

def TRG_V_eq_2(T, T0, T1, Dcut:int, renormal_loop:int):
    """
    configuration of the tensors is:
     T0    
         T1
    """
    #print(f"renormalization loops:{renormal_loop}")
    U0, VH0, _ = SVD(T0, Dcut=Dcut, pattern="B")
    U1, VH1, s1 = SVD(T1, Dcut=Dcut, pattern="A")
    UB, VHB, _ = SVD(T , Dcut=Dcut, pattern="B")
    UA, VHA, sa = SVD(T , Dcut=Dcut, pattern="A")

    T0 = __new_tensor__(VH0, VH1, U0, U1, renormal_loop=renormal_loop)
    T  = __new_tensor__(VHB, VHA, UB, UA, renormal_loop=renormal_loop)

    #filename="TRG_sv/op2_{:d}_svd.dat".format(renormal_loop)
    #out=open(filename, "w")
    #out.write("#pure impure:{:}\n".format(renormal_loop))
    #for i in range(Dcut*Dcut):
    #    out.write("{:d} {:.6e} {:.6e}\n".format(i+1, sa[i], s1[i]))
    #out.close()

    del U0, VH0, U1, VH1, UB, VHB, UA, VHA, T1

    return T, T0

def __RG_flow_output__(T):
    with open("{:}/tensor_l{:}.dat".format(OUTPUT_DIR, count_totloop), "w") as svout:
        T = cp.reshape(T, (T.shape[0]*T.shape[1], T.shape[2]*T.shape[3]))
        _, s, _ = cp.linalg.svd(T)
        for ss in s:
           #ss /= cp.max(s)
           svout.write("{:.12e}\n".format(ss))

def tensor_Trace(T):
    TrT = cp.einsum("xyxy", T)
    return TrT

def __tensor_normalization__(T):
    c = cp.max(cp.absolute(T))
    T = T / c
    return T, c

#def __tensor_normalization__(T):
#    #Tshape = T.shape
#    #T = cp.reshape(T, (T.shape[0]*T.shape[1], T.shape[2]*T.shape[3]))
#
#    Dens_Mat = cp.einsum("xxyY->yY", T)
#
#    _, s, _ = cp.linalg.svd(Dens_Mat)
#    c = cp.max(s)
#    T = T / c
#    
#    #s /= c
#    #T = cp.einsum("ij,j,jk->ik", u, s, vh)
#    #T = cp.reshape(T, (Tshape[0], Tshape[1], Tshape[2], Tshape[3]))
#    return T, c

def pure_tensor_renorm(T, Dcut:int, XLOOPS:int, YLOOPS:int):
    N = XLOOPS + YLOOPS
    global count_totloop
    count_totloop = 0

    ln_normalized_factor = cp.zeros(N+1, dtype=cp.float64)
    
    t0 = time.time()

    T , c  = __tensor_normalization__(T)
    ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)

    t1 = time.time()
    __RG_flow_output__(T)
    TrT = tensor_Trace(T)
    print("loop {:2d} finish, Re(TrT)= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s".format(count_totloop, TrT, c, t1-t0))
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    #print(count_totloop, ln_normalized_factor[count_totloop])
    while count_totloop < N:
        count_totloop += 1
        
        t0 = time.time()

        T = TRG_pure_tensor(T, Dcut, count_totloop)
        T , c  = __tensor_normalization__(T)
        ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)

        t1 = time.time()
        __RG_flow_output__(T)
        TrT = tensor_Trace(T)
        print("loop {:2d} finish, Re(TrT)= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s".format(count_totloop, TrT, c, t1-t0))

        #print(count_totloop, ln_normalized_factor[count_totloop])

    return T, ln_normalized_factor

def nearest_two_point_func_renorm(T, T0, T1, T2, T3, Dcut:int, XLOOPS:int, YLOOPS:int):
    N = XLOOPS + YLOOPS
    global count_totloop
    count_totloop = 0

    ln_normalized_factor = cp.zeros(N+1, dtype=cp.float64)
    ln_normalized_factor_imp = cp.zeros(N+1, dtype=cp.float64)
    
    t0 = time.time()

    T , c  = __tensor_normalization__(T)
    T0, c0 = __tensor_normalization__(T0)
    T1, c1 = __tensor_normalization__(T1)
    T2, c2 = __tensor_normalization__(T2)
    T3, c3 = __tensor_normalization__(T3)
    
    ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
    ln_normalized_factor_imp[count_totloop] = cp.log(c0)+cp.log(c1)+cp.log(c2)+cp.log(c3) - 4*cp.log(c)

    t1 = time.time()
    __RG_flow_output__(T)
    TrT = tensor_Trace(T)
    print("loop {:2d} finish, Re(TrT)= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s".format(count_totloop, TrT, c, t1-t0))
    
    #print(count_totloop, ln_normalized_factor[count_totloop])
    while count_totloop < N:
        count_totloop += 1
        if count_totloop <= N - 2:
            t0 = time.time()

            T, T0, T1, T2, T3 = TRG_nearest_neighborhood(T, T0, T1, T2, T3, Dcut, count_totloop)
            T , c  = __tensor_normalization__(T)
            T0, c0 = __tensor_normalization__(T0)
            T1, c1 = __tensor_normalization__(T1)
            T2, c2 = __tensor_normalization__(T2)
            T3, c3 = __tensor_normalization__(T3)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            ln_normalized_factor_imp[count_totloop] = cp.log(c0)+cp.log(c1)+cp.log(c2)+cp.log(c3) - 4*cp.log(c)

            t1 = time.time()
            __RG_flow_output__(T)
            TrT = tensor_Trace(T)
            print("loop {:2d} finish, Re(TrT)= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s".format(count_totloop, TrT, c, t1-t0))

        elif count_totloop == N - 1:
            t0 = time.time()

            T, T0, T1 = TRG_V_eq_4(T, T0, T1, T2, T3, Dcut, count_totloop)
            T , c  = __tensor_normalization__(T)
            T0, c0 = __tensor_normalization__(T0)
            T1, c1 = __tensor_normalization__(T1)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            ln_normalized_factor_imp[count_totloop] = cp.log(c0)+cp.log(c1) - 2*cp.log(c)

            t1 = time.time()
            __RG_flow_output__(T)
            TrT = tensor_Trace(T)
            print("loop {:2d} finish, Re(TrT)= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s".format(count_totloop, TrT, c, t1-t0))

        elif count_totloop == N:
            t0 = time.time()

            T, T0 = TRG_V_eq_2(T, T0, T1, Dcut, count_totloop)
            T , c  = __tensor_normalization__(T)
            T0, c0 = __tensor_normalization__(T0)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)

            t1 = time.time()
            __RG_flow_output__(T)
            TrT = tensor_Trace(T)
            print("loop {:2d} finish, Re(TrT)= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s".format(count_totloop, TrT, c, t1-t0))

        #print(count_totloop, ln_normalized_factor[count_totloop])

    return T, T0, ln_normalized_factor, ln_normalized_factor_imp