import numpy as np
import cupy as cp
import sys
import time
import opt_einsum as oe
from cuquantum import contract
from mpi4py import MPI

comm = MPI.COMM_WORLD 
myrank = comm.Get_rank() 
nproc = comm.Get_size() 
name = MPI.Get_processor_name() 
cuda = cp.cuda.Device(myrank)
cuda.use()

optimize={'slicing': {'min_slices': 4}}

def __tensor_legs_rearrange__(T, direction:str):
    if direction == "Z" or direction == "z":
        return T
    elif direction == "X" or direction == "x":
        T = contract("abcdef->cdefab", T)
        return T
    elif direction == "Y" or direction == "y":
        T = contract("abcdef->efabcd", T)
        return T

def __tensor_legs_restore__(T, direction:str):
    if direction == "Z" or direction == "z":
        return T
    elif direction == "X" or direction == "x":
        T = contract("cdefab->abcdef", T)
        return T
    elif direction == "Y" or direction == "y":
        T = contract("efabcd->abcdef", T)
        return T

def __contraction_projector__(T, Dcut:int):
    T_conj = cp.conj(T)

    t0 = time.time()
    MM_dag1m = contract("aefghm,bijkml,cefghn,dijknl->abcd", T, T, T_conj, T_conj, optimize=optimize)
    MM_dag1p = contract("eafghm,ibjkml,ecfghn,idjknl->abcd", T, T, T_conj, T_conj, optimize=optimize)
    MM_dag2m = contract("efaghm,ijbkml,efcghn,ijdknl->abcd", T, T, T_conj, T_conj, optimize=optimize)
    MM_dag2p = contract("efgahm,ijkbml,efgchn,ijkdnl->abcd", T, T, T_conj, T_conj, optimize=optimize)
    t1 = time.time()
    #print("Matrix for squeezer: {:.2e} s".format((t1-t0)/2), end=", ")
    del T_conj

    MM_dag1m = cp.reshape(MM_dag1m,  (cp.shape(MM_dag1m)[0]*cp.shape(MM_dag1m)[1], cp.shape(MM_dag1m)[2]*cp.shape(MM_dag1m)[3]))
    MM_dag1p = cp.reshape(MM_dag1p,  (cp.shape(MM_dag1p)[0]*cp.shape(MM_dag1p)[1], cp.shape(MM_dag1p)[2]*cp.shape(MM_dag1p)[3]))
    MM_dag2m = cp.reshape(MM_dag2m,  (cp.shape(MM_dag2m)[0]*cp.shape(MM_dag2m)[1], cp.shape(MM_dag2m)[2]*cp.shape(MM_dag2m)[3]))
    MM_dag2p = cp.reshape(MM_dag2p,  (cp.shape(MM_dag2p)[0]*cp.shape(MM_dag2p)[1], cp.shape(MM_dag2p)[2]*cp.shape(MM_dag2p)[3]))

    Eigval1m, Eigvect1m = cp.linalg.eigh(MM_dag1m)
    Eigval1p, Eigvect1p = cp.linalg.eigh(MM_dag1p)
    Eigval2m, Eigvect2m = cp.linalg.eigh(MM_dag2m)
    Eigval2p, Eigvect2p = cp.linalg.eigh(MM_dag2p)
    del MM_dag1m, MM_dag1p, MM_dag2m, MM_dag2p

    print("Eigval1m min",cp.min(Eigval1m), "Eigval1m max",cp.max(Eigval1m))
    print("Eigval1p min",cp.min(Eigval1p), "Eigval1p max",cp.max(Eigval1p))
    print("Eigval2m min",cp.min(Eigval2m), "Eigval2m max",cp.max(Eigval2m))
    print("Eigval2p min",cp.min(Eigval2p), "Eigval2p max",cp.max(Eigval2p))

    D = len(Eigval1m)
    if D <= Dcut:
        Eigvect_cut1 = Eigvect1m
        Eigvect_cut2 = Eigvect2m
    else:
        epsilon1m = cp.sum(Eigval1m[:D - Dcut])
        epsilon1p = cp.sum(Eigval1p[:D - Dcut])
        epsilon2m = cp.sum(Eigval2m[:D - Dcut])
        epsilon2p = cp.sum(Eigval2p[:D - Dcut])

        if epsilon1m < epsilon1p:
            PM1 = "minus"
            Eigvect_cut1 = Eigvect1m[:,D - Dcut:]
        else:
            PM1 = "plus"
            Eigvect_cut1 = Eigvect1p[:,D - Dcut:]

        if epsilon2m < epsilon2p:
            PM2 = "minus"
            Eigvect_cut2 = Eigvect2m[:,D - Dcut:]
        else:
            PM2 = "plus"
            Eigvect_cut2 = Eigvect2p[:,D - Dcut:]

    del Eigvect1m, Eigval1p

    if D <= Dcut:
        Eigvect_cut1 = cp.reshape(Eigvect_cut1, (D, D, D))
        Eigvect_cut2 = cp.reshape(Eigvect_cut2, (D, D, D))
    else:
        Eigvect_cut1 = cp.reshape(Eigvect_cut1, (Dcut, Dcut, Dcut))
        Eigvect_cut2 = cp.reshape(Eigvect_cut2, (Dcut, Dcut, Dcut))

    return Eigvect_cut1, Eigvect_cut2, PM1, PM2

def new_pure_tensor(T, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T = __tensor_legs_rearrange__(T, direction)
    U1, U2, PM1, PM2 = __contraction_projector__(T, Dcut)
    t0 = time.time()
    if PM1 == "minus":
        if PM2 == "minus":
            T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, cp.conj(U1), U1, cp.conj(U2), U2, optimize=optimize)
        elif PM2 == "plus":
            T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, cp.conj(U1), U1, U2, cp.conj(U2), optimize=optimize)
    elif PM1 == "plus":
        if PM2 == "minus":
            T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, U1, cp.conj(U1), cp.conj(U2), U2, optimize=optimize)
        elif PM2 == "plus":
            T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, U1, cp.conj(U1), U2, cp.conj(U2), optimize=optimize)
    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format(t1-t0))
    T = __tensor_legs_restore__(T, direction)

    return T


def new_impuer_tensor_2imp(T, Timp0, Timp1, nx, ny, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T     = __tensor_legs_rearrange__(T, direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    Timp1 = __tensor_legs_rearrange__(Timp1, direction)
    U = __contraction_projector__(T, Dcut)

    t0 = time.time()
    Timp0 = contract("acke,bdel,abi,cdj->ijkl", Timp0, T, U, cp.conj(U), optimize=optimize)

    if (ny%2 == 1) and (direction == "Y" or direction == "y"):
        Timp1 = contract("acke,bdel,abi,cdj->ijkl", T, Timp1, U, cp.conj(U), optimize=optimize)
    
    if (ny%2 == 0) and (direction == "Y" or direction == "y"):
        Timp1 = contract("acke,bdel,abi,cdj->ijkl", Timp1, T, U, cp.conj(U), optimize=optimize)

    if (nx%2 == 1) and (direction == "X" or direction == "x"):
        Timp1 = contract("acke,bdel,abi,cdj->ijkl", T, Timp1, U, cp.conj(U), optimize=optimize)

    if (nx%2 == 0) and (direction == "X" or direction == "x"):
        Timp1 = contract("acke,bdel,abi,cdj->ijkl", Timp1, T, U, cp.conj(U), optimize=optimize)

    T = contract("acke,bdel,abi,cdj->ijkl", T, T, U, cp.conj(U), optimize=optimize)
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
    U1, U2, PM1, PM2 = __contraction_projector__(T, Dcut)

    t0 = time.time()
    #Timp0 = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", Timp0, Timp1, U1, cp.conj(U1), U2, cp.conj(U2))
    #T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, U1, cp.conj(U1), U2, cp.conj(U2))

    if PM1 == "minus":
        if PM2 == "minus":
            Timp0 = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", Timp0, Timp1, cp.conj(U1), U1, cp.conj(U2), U2, optimize=optimize)
            T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, cp.conj(U1), U1, cp.conj(U2), U2, optimize=optimize)
        elif PM2 == "plus":
            Timp0 = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", Timp0, Timp1, cp.conj(U1), U1, U2, cp.conj(U2), optimize=optimize)
            T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, cp.conj(U1), U1, U2, cp.conj(U2), optimize=optimize)
    elif PM1 == "plus":
        if PM2 == "minus":
            Timp0 = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", Timp0, Timp1, U1, cp.conj(U1), cp.conj(U2), U2, optimize=optimize)
            T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, U1, cp.conj(U1), cp.conj(U2), U2, optimize=optimize)
        elif PM2 == "plus":
            Timp0 = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", Timp0, Timp1, U1, cp.conj(U1), U2, cp.conj(U2), optimize=optimize)
            T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, U1, cp.conj(U1), U2, cp.conj(U2), optimize=optimize)

    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format((t1-t0)/2))

    del Timp1, U1, U2

    T     = __tensor_legs_rearrange__(T    , direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)

    return T, Timp0

def new_impuer_tensor_1imp(T, Timp0, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T     = __tensor_legs_rearrange__(T    , direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    U1, U2, PM1, PM2 = __contraction_projector__(T, Dcut)

    #Timp0 = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", Timp0, T, U1, cp.conj(U1), U2, cp.conj(U2))
    #T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, U1, cp.conj(U1), U2, cp.conj(U2))
    if PM1 == "minus":
        if PM2 == "minus":
            Timp0 = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", Timp0, T, cp.conj(U1), U1, cp.conj(U2), U2, optimize=optimize)
            T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, cp.conj(U1), U1, cp.conj(U2), U2, optimize=optimize)
        elif PM2 == "plus":
            Timp0 = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", Timp0, T, cp.conj(U1), U1, U2, cp.conj(U2), optimize=optimize)
            T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, cp.conj(U1), U1, U2, cp.conj(U2), optimize=optimize)
    elif PM1 == "plus":
        if PM2 == "minus":
            Timp0 = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", Timp0, T, U1, cp.conj(U1), cp.conj(U2), U2, optimize=optimize)
            T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, U1, cp.conj(U1), cp.conj(U2), U2, optimize=optimize)
        elif PM2 == "plus":
            Timp0 = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", Timp0, T, U1, cp.conj(U1), U2, cp.conj(U2), optimize=optimize)
            T = contract("ijklez,mnopzf,ima,jnb,koc,lpd->abcdef", T, T, U1, cp.conj(U1), U2, cp.conj(U2), optimize=optimize)

    t0 = time.time()
    T     = __tensor_legs_restore__(T    , direction)
    Timp0 = __tensor_legs_restore__(Timp0, direction)
    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format((t1-t0)/2))

    del U1, U2

    return T, Timp0


#Renormalize
def __tensor_normalization__(T):
    #c = cp.max(cp.absolute(T))
    c = oe.contract("xxyytt", T)
    c = cp.abs(c)
    ##print(c)
    T = T / c
    return T, c

def __get_tensor_spectrum(T, Dcut):
    Ttmp = cp.transpose(T, (0,2,4,1,3,5))
    Ttmp = cp.reshape(Ttmp, (Dcut*Dcut*Dcut, Dcut*Dcut*Dcut))
    s = cp.linalg.svd(Ttmp, compute_uv=False)
    print("tensor spectrum", s[:2*Dcut]/cp.max(s))
    print(f"s1={cp.max(s):.6e}")

def pure_tensor_renorm(T, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    count_xloop = 0
    count_yloop = 0
    count_zloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+ZLOOPS+1, dtype=cp.float64)
    T, c = __tensor_normalization__(T)
    __get_tensor_spectrum(T, Dcut)
    print()
    #T[cp.abs(T)<1e-12] = 0.0
    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)

    while (count_xloop < XLOOPS or count_yloop < YLOOPS or count_zloop < ZLOOPS):

        if count_zloop < ZLOOPS:
            count_zloop += 1
            count_totloop += 1
            t0 = time.time()
            T = new_pure_tensor(T, Dcut, "Z")
            T, c = __tensor_normalization__(T)
            trace = oe.contract("aabbcc", T)
            print("TrT=",trace)
            #T[cp.abs(T)<1e-12] = 0.0
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            t1 = time.time()
            print("not-zero density (|T|>1e-12): {}".format(T[np.abs(T)>1e-12].size/T.size))
            print("not-zero density (|T|>1e-8): {}".format(T[np.abs(T)>1e-8].size/T.size))
            print("not-zero density (|T|>1e-5): {}".format(T[np.abs(T)>1e-5].size/T.size))
            #T[cp.abs(T)<1e-12] = 0.0
            __get_tensor_spectrum(T, Dcut)
            print("Z, loops:",count_totloop, "time= {:.2f} s\n".format(t1-t0))

        if count_yloop < YLOOPS:
            count_yloop += 1
            count_totloop += 1
            t0 = time.time()
            T = new_pure_tensor(T, Dcut, "Y")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            trace = oe.contract("aabbcc", T)
            print("TrT=",trace)
            #T[cp.abs(T)<1e-12] = 0.0
            t1 = time.time()
            print("not-zero density (|T|>1e-12): {}".format(T[np.abs(T)>1e-12].size/T.size))
            print("not-zero density (|T|>1e-8): {}".format(T[np.abs(T)>1e-8].size/T.size))
            print("not-zero density (|T|>1e-5): {}".format(T[np.abs(T)>1e-5].size/T.size))
            #T[cp.abs(T)<1e-12] = 0.0
            __get_tensor_spectrum(T, Dcut)
            print("Y, loops:",count_totloop, "time= {:.2f} s\n".format(t1-t0))

        #trace = oe.contract("aabbcc", T)
        #print(trace)

        if count_xloop < XLOOPS:
            count_xloop += 1
            count_totloop += 1
            t0 = time.time()
            T = new_pure_tensor(T, Dcut, "X")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            trace = oe.contract("aabbcc", T)
            print("TrT=",trace)
            #T[cp.abs(T)<1e-12] = 0.0
            t1 = time.time()
            print("not-zero density (|T|>1e-12): {}".format(T[np.abs(T)>1e-12].size/T.size))
            print("not-zero density (|T|>1e-8): {}".format(T[np.abs(T)>1e-8].size/T.size))
            print("not-zero density (|T|>1e-5): {}".format(T[np.abs(T)>1e-5].size/T.size))
            #T[cp.abs(T)<1e-12] = 0.0
            __get_tensor_spectrum(T, Dcut)
            print("X, loops:",count_totloop, "time= {:.2f} s\n".format(t1-t0))

        #trace = oe.contract("aabbcc", T)
        #print(trace)
    #__get_tensor_spectrum(T, Dcut)

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

    return T, Timp0, ln_normalized_factor, ln_normalized_factor_imp


def ynearest_two_point_func_renorm(T, Timp0, Timp1, Dcut:int, XLOOPS:int, YLOOPS:int, ZLOOPS:int):
    count_xloop = 0
    count_yloop = 0
    count_zloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+ZLOOPS+1, dtype=cp.float64)
    ln_normalized_factor_imp = cp.zeros(XLOOPS+YLOOPS+ZLOOPS+1, dtype=cp.float64)

    #trace = oe.contract("aabbic,ddeeci",Timp1,Timp0)
    #print("TrT1T0  =",trace)

    T, c = __tensor_normalization__(T)
    Timp0, c0 = __tensor_normalization__(Timp0)
    Timp1, c1 = __tensor_normalization__(Timp1)

    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)
    ln_normalized_factor_imp[0] = cp.log(c0)+cp.log(c1) - 2*cp.log(c) #c0*c1 / (c**2)

    while (count_xloop < XLOOPS or count_yloop < YLOOPS or count_zloop < ZLOOPS):

        if count_zloop < ZLOOPS:
            count_zloop += 1
            count_totloop += 1
            if count_totloop == 1:
                T, Timp0 = new_impuer_tensor_2to1imp(T, Timp0, Timp1, Dcut, "Z")
                T, c = __tensor_normalization__(T)
                Timp0, c0 = __tensor_normalization__(Timp0)
                ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
                ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c) #c0 / c

            else:
                T, Timp0 = new_impuer_tensor_1imp(T, Timp0, Dcut, "Z")
                T, c = __tensor_normalization__(T)
                Timp0, c0 = __tensor_normalization__(Timp0)
                ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
                ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)

        if count_yloop < YLOOPS:
            count_yloop += 1
            count_totloop += 1
            T, Timp0 = new_impuer_tensor_1imp(T, Timp0, Dcut, "Y")
            T, c = __tensor_normalization__(T)
            Timp0, c0 = __tensor_normalization__(Timp0)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)

        if count_xloop < XLOOPS:
            count_xloop += 1
            count_totloop += 1
            T, Timp0 = new_impuer_tensor_1imp(T, Timp0, Dcut, "X")
            T, c = __tensor_normalization__(T)
            Timp0, c0 = __tensor_normalization__(Timp0)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)

    return T, Timp0, ln_normalized_factor, ln_normalized_factor_imp