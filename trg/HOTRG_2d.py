import numpy as np
import cupy as cp
import sys
import time
from cuquantum import einsum
from mpi4py import MPI

def __tensor_legs_rearrange__(T, direction:str):
    if direction == "Y" or direction == "y":
        return T
    elif direction == "X" or direction == "x":
        T = einsum("abcd->cdab", T)
        return T

def __tensor_legs_restore__(T, direction:str):
    if direction == "Y" or direction == "y":
        return T
    elif direction == "X" or direction == "x":
        T = einsum("cdab->abcd", T)
        return T

def __contraction_projector__(T, Dcut:int):
    T_conj = cp.conj(T)

    t0 = time.time()
    MM_dag1 = einsum("aijk,blkm,cijn,dlnm->abcd", T, T, T_conj, T_conj)
    MM_dag2 = einsum("iajk,lbkm,icjn,ldnm->abcd", T, T, T_conj, T_conj)
    t1 = time.time()
    del T_conj

    MM_dag1 = cp.reshape(MM_dag1,  (cp.shape(MM_dag1)[0]*cp.shape(MM_dag1)[1], cp.shape(MM_dag1)[2]*cp.shape(MM_dag1)[3]))
    MM_dag2 = cp.reshape(MM_dag2,  (cp.shape(MM_dag2)[0]*cp.shape(MM_dag2)[1], cp.shape(MM_dag2)[2]*cp.shape(MM_dag2)[3]))
    Eigval1, Eigvect1 = cp.linalg.eigh(MM_dag1)
    Eigval2, Eigvect2 = cp.linalg.eigh(MM_dag2)
    #print(np.linalg.norm(MM_dag1-MM_dag2)/np.linalg.norm(MM_dag1))
    del MM_dag1, MM_dag2
   
    D = len(Eigval1)
    if D <= Dcut:
        Eigvect_cut = Eigvect1
    else:
        epsilon1 = cp.sum(Eigval1[:D - Dcut])
        epsilon2 = cp.sum(Eigval2[:D - Dcut])

        if epsilon1 < epsilon2:
            LDRU = "LD"
            Eigvect_cut = Eigvect1[:,D - Dcut:]
        else:
            LDRU = "RU"
            Eigvect_cut = Eigvect2[:,D - Dcut:]
    del Eigvect1, Eigval2

    Eigvect_cut = cp.reshape(Eigvect_cut, (Dcut, Dcut, Dcut))

    return Eigvect_cut, LDRU

def new_pure_tensor(T, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T = __tensor_legs_rearrange__(T, direction)
    U, LDRU = __contraction_projector__(T, Dcut)
    t0 = time.time()

    if LDRU == "LD":
        T = einsum("acke,bdel,abi,cdj->ijkl", T, T, cp.conj(U), U)
    elif LDRU == "RU":
        T = einsum("acke,bdel,abi,cdj->ijkl", T, T, U, cp.conj(U))
    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format(t1-t0))
    T = __tensor_legs_restore__(T, direction)

    return T


def new_impuer_tensor_2imp(T, Timp0, Timp1, nx, ny, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T     = __tensor_legs_rearrange__(T, direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    Timp1 = __tensor_legs_rearrange__(Timp1, direction)
    U, LDRU = __contraction_projector__(T, Dcut)

    t0 = time.time()
    if LDRU == "LD":
        Timp0 = einsum("acke,bdel,abi,cdj->ijkl", Timp0, T, cp.conj(U), U)
    elif LDRU == "RU":
        Timp0 = einsum("acke,bdel,abi,cdj->ijkl", Timp0, T, U, cp.conj(U))
    
    if LDRU == "LD":
        if (ny%2 == 1) and (direction == "Y" or direction == "y"):
            Timp1 = einsum("acke,bdel,abi,cdj->ijkl", T, Timp1, cp.conj(U), U)

        if (ny%2 == 0) and (direction == "Y" or direction == "y"):
            Timp1 = einsum("acke,bdel,abi,cdj->ijkl", Timp1, T, cp.conj(U), U)

        if (nx%2 == 1) and (direction == "X" or direction == "x"):
            Timp1 = einsum("acke,bdel,abi,cdj->ijkl", T, Timp1, cp.conj(U), U)

        if (nx%2 == 0) and (direction == "X" or direction == "x"):
            Timp1 = einsum("acke,bdel,abi,cdj->ijkl", Timp1, T, cp.conj(U), U)

    elif LDRU == "RU":
        if (ny%2 == 1) and (direction == "Y" or direction == "y"):
            Timp1 = einsum("acke,bdel,abi,cdj->ijkl", T, Timp1, U, cp.conj(U))

        if (ny%2 == 0) and (direction == "Y" or direction == "y"):
            Timp1 = einsum("acke,bdel,abi,cdj->ijkl", Timp1, T, U, cp.conj(U))

        if (nx%2 == 1) and (direction == "X" or direction == "x"):
            Timp1 = einsum("acke,bdel,abi,cdj->ijkl", T, Timp1, U, cp.conj(U))

        if (nx%2 == 0) and (direction == "X" or direction == "x"):
            Timp1 = einsum("acke,bdel,abi,cdj->ijkl", Timp1, T, U, cp.conj(U))

    if LDRU == "LD":
        T = einsum("acke,bdel,abi,cdj->ijkl", T, T, cp.conj(U), U)
    elif LDRU == "RU":
        T = einsum("acke,bdel,abi,cdj->ijkl", T, T, U, cp.conj(U))
    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format((t1-t0)/3))

    T     = __tensor_legs_rearrange__(T    , direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    Timp1 = __tensor_legs_rearrange__(Timp1, direction)

    del U

    return T, Timp0, Timp1

def new_impuer_tensor_2to1imp(T, Timp0, Timp1, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T     = __tensor_legs_rearrange__(T, direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    Timp1 = __tensor_legs_rearrange__(Timp1, direction)
    U, LDRU = __contraction_projector__(T, Dcut)

    t0 = time.time()
    if LDRU == "LD":
        Timp0 = einsum("acke,bdel,abi,cdj->ijkl", Timp0, Timp1, cp.conj(U), U)
        T = einsum("acke,bdel,abi,cdj->ijkl", T, T, cp.conj(U), U)
    elif LDRU == "RU":
        Timp0 = einsum("acke,bdel,abi,cdj->ijkl", Timp0, Timp1, U, cp.conj(U))
        T = einsum("acke,bdel,abi,cdj->ijkl", T, T, U, cp.conj(U))
    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format((t1-t0)/2))

    del Timp1, U

    T     = __tensor_legs_rearrange__(T    , direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)

    return T, Timp0

def new_impuer_tensor_1imp(T, Timp0, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T     = __tensor_legs_rearrange__(T    , direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    U, LDRU = __contraction_projector__(T, Dcut)

    if LDRU == "LD":
        Timp0 = einsum("acke,bdel,abi,cdj->ijkl", Timp0, T, cp.conj(U), U)
        T = einsum("acke,bdel,abi,cdj->ijkl", T, T, cp.conj(U), U)
    if LDRU == "RU":
        Timp0 = einsum("acke,bdel,abi,cdj->ijkl", Timp0, T, U, cp.conj(U))
        T = einsum("acke,bdel,abi,cdj->ijkl", T, T, U, cp.conj(U))

    t0 = time.time()
    T     = __tensor_legs_rearrange__(T    , direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format((t1-t0)/2))

    del U

    return T, Timp0


#Renormalize
def __tensor_normalization__(T):
    c = cp.max(cp.absolute(T))
    ##print(c)
    T = T / c
    return T, c

def pure_tensor_renorm(T, Dcut:int, XLOOPS:int, YLOOPS:int):
    count_xloop = 0
    count_yloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+1, dtype=cp.float64)
    T, c = __tensor_normalization__(T)
    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)

    while (count_xloop < XLOOPS or count_yloop < YLOOPS):
        t0 = time.time()
        if count_yloop < YLOOPS:
            count_yloop += 1
            count_totloop += 1
            T = new_pure_tensor(T, Dcut, "Y")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
        t1 = time.time()
        print("loop {:2d} finish. time: {:.6f} s".format(count_totloop, t1-t0))

        t0 = time.time()
        if count_xloop < XLOOPS:
            count_xloop += 1
            count_totloop += 1
            T = new_pure_tensor(T, Dcut, "X")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
        t1 = time.time()
        print("loop {:2d} finish. time: {:.6f} s".format(count_totloop, t1-t0))

    return T, ln_normalized_factor


def two_point_func_renorm(T, Timp0, Timp1, nx, ny, Dcut:int, XLOOPS:int, YLOOPS:int):
    count_xloop = 0
    count_yloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+1, dtype=cp.float64)
    ln_normalized_factor_imp = cp.zeros(XLOOPS+YLOOPS+1, dtype=cp.float64)

    T, c = __tensor_normalization__(T)
    Timp0, c0 = __tensor_normalization__(Timp0)
    Timp1, c1 = __tensor_normalization__(Timp1)

    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)
    ln_normalized_factor_imp[0] = cp.log(c0)+cp.log(c1) - 2*cp.log(c) #c0*c1 / (c**2)

    while (count_xloop < XLOOPS or count_yloop < YLOOPS):

        t0 = time.time()
        if count_yloop < YLOOPS:
            count_yloop += 1
            count_totloop += 1
            if int(nx**2 + ny**2) > 1 or (nx==1 and ny == 0):
                T, Timp0, Timp1 = new_impuer_tensor_2imp(T, Timp0, Timp1, nx, ny, Dcut, "Y")
                T, c = __tensor_normalization__(T)
                Timp0, c0 = __tensor_normalization__(Timp0)
                Timp1, c1 = __tensor_normalization__(Timp1)
                ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
                ln_normalized_factor_imp[count_totloop] = cp.log(c0)+cp.log(c1) - 2*cp.log(c)

            if nx == 0 and ny == 1:
                T, Timp0 = new_impuer_tensor_2to1imp(T, Timp0, Timp1, Dcut, "Y")
                T, c = __tensor_normalization__(T)
                Timp0, c0 = __tensor_normalization__(Timp0)
                ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
                ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c) 

            if nx == 0 and ny == 0:
                T, Timp0 = new_impuer_tensor_1imp(T, Timp0, Dcut, "Y")
                T, c = __tensor_normalization__(T)
                Timp0, c0 = __tensor_normalization__(Timp0)
                ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
                ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)

            ny = ny >> 1
        
        t1 = time.time()
        print("loop {:2d} finish. time: {:.6f} s".format(count_totloop, t1-t0))

        t0 = time.time()
        if count_xloop < XLOOPS:
            count_xloop += 1
            count_totloop += 1
            if int(nx**2 + ny**2) > 1 or (nx==0 and ny==1):
                T, Timp0, Timp1 = new_impuer_tensor_2imp(T, Timp0, Timp1, nx, ny, Dcut, "X")
                T, c = __tensor_normalization__(T)
                Timp0, c0 = __tensor_normalization__(Timp0)
                Timp1, c1 = __tensor_normalization__(Timp1)
                ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
                ln_normalized_factor_imp[count_totloop] = cp.log(c0)+cp.log(c1) - 2*cp.log(c)

            if nx == 1 and ny == 0:
                T, Timp0 = new_impuer_tensor_2to1imp(T, Timp0, Timp1, Dcut, "X")
                T, c = __tensor_normalization__(T)
                Timp0, c0 = __tensor_normalization__(Timp0)
                ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
                ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)

            if nx == 0 and ny == 0:
                T, Timp0 = new_impuer_tensor_1imp(T, Timp0, Dcut, "X")
                T, c = __tensor_normalization__(T)
                Timp0, c0 = __tensor_normalization__(Timp0)
                ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
                ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)

            nx = nx >> 1
        
        t1 = time.time()
        print("loop {:2d} finish. time: {:.6f} s".format(count_totloop, t1-t0))

    return T, Timp0, ln_normalized_factor, ln_normalized_factor_imp


def ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut:int, XLOOPS:int, YLOOPS:int):
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
                T, Timp0 = new_impuer_tensor_2to1imp(T, Timp0, Timp1, Dcut, "Y")
                T, c = __tensor_normalization__(T)
                Timp0, c0 = __tensor_normalization__(Timp0)
                ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
                ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)

            else:
                T, Timp0 = new_impuer_tensor_1imp(T, Timp0, Dcut, "Y")
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
            T, Timp0 = new_impuer_tensor_1imp(T, Timp0, Dcut, "X")
            T, c = __tensor_normalization__(T)
            Timp0, c0 = __tensor_normalization__(Timp0)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)
        t1 = time.time()
        print("loop {:2d} finish. time: {:.6f} s".format(count_totloop, t1-t0))

    return T, Timp0, ln_normalized_factor, ln_normalized_factor_imp