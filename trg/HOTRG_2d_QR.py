import os
import numpy as np
import cupy as cp
import sys
import time
from cuquantum import contract 
import opt_einsum as oe

from utility.randomized_svd import rsvd

optimize={'slicing': {'min_slices': 4}}

OUTPUT_DIR = os.environ["OUTPUT_DIR"]

def __tensor_legs_rearrange__(T, direction:str):
    if direction == "Y" or direction == "y":
        return T
    elif direction == "X" or direction == "x":
        T = contract("abcd->cdab", T)
        return T

def __tensor_legs_restore__(T, direction:str):
    if direction == "Y" or direction == "y":
        return T
    elif direction == "X" or direction == "x":
        T = contract("cdab->abcd", T)
        return T

def __squeezer__(T, Dcut:int):
    """
    returns left/down projector PLD and right/up projector PRU \\
    PLD_{x,x0,x1} or PLD_{y,y0,y1} \\
    PRU_{x'0,x'1,x'} or PRU_{y'0,y'1,y'}
    """
    T_conj = cp.conj(T)

    t0 = time.time()
    A1dagA1 = oe.contract("aibe,cjed,akbf,clfd->ijkl", T_conj, T_conj, T, T)
    A2A2dag = oe.contract("iabe,jced,kabf,lcfd->ijkl", T, T, T_conj, T_conj)
    t1 = time.time()
    del T_conj

    A1dagA1 = cp.reshape(A1dagA1,  (cp.shape(A1dagA1)[0]*cp.shape(A1dagA1)[1], cp.shape(A1dagA1)[2]*cp.shape(A1dagA1)[3]))
    A2A2dag = cp.reshape(A2A2dag,  (cp.shape(A2A2dag)[0]*cp.shape(A2A2dag)[1], cp.shape(A2A2dag)[2]*cp.shape(A2A2dag)[3]))
    Eigval1, Eigvect1 = cp.linalg.eigh(A1dagA1)
    Eigval2, Eigvect2 = cp.linalg.eigh(A2A2dag)
    #Eigval1 = Eigval1[::-1]
    #Eigval2 = Eigval2[::-1]
    #Eigvect1 = Eigvect1[:,::-1]
    #Eigvect2 = Eigvect2[:,::-1]
    #Eigvect11, Eigval11, _ = cp.linalg.svd(A1dagA1)
    #Eigvect21, Eigval21, _ = cp.linalg.svd(A2A2dag)
    del A1dagA1, A2A2dag
    #print("eigh:",Eigval1[::-1])
    #print("svd :",Eigval11)

    e1, e2 = Eigval1/cp.max(Eigval1), Eigval2/cp.max(Eigval2)
    e1 = e1[::-1]
    e2 = e2[::-1]
    __RG_flow_output__(type="squeezer", data=zip(e1, e2))
    del e1, e2
   
    Eigval1 = Eigval1.astype(cp.complex128)
    Eigval2 = Eigval2.astype(cp.complex128)
    R1 = contract("ia,a->ai", cp.conj(Eigvect1), cp.sqrt(Eigval1))
    R2 = contract("ia,a->ia", Eigvect2, cp.sqrt(Eigval2))

    U, S, VH = cp.linalg.svd(R1@R2)
    #U, S, VH = rsvd(R1@R2, k=Dcut, n_power_iter=5)
    UH = cp.conj(U.T)
    Sinv = 1 / S
    Sinv = Sinv.astype(cp.complex128)
    V = cp.conj(VH.T)
    del U, S, VH

    UH = UH[:Dcut,:]
    Sinv = Sinv[:Dcut]
    V = V[:,:Dcut]

    P1 = contract("ia,aj,j->ij", R2, V , cp.sqrt(Sinv))
    P2 = contract("i,ia,aj->ij", cp.sqrt(Sinv), UH, R1)
    del R1, R2, Eigval1, Eigval2, Eigvect1, Eigvect2

    #print(cp.linalg.norm(P2@P1)**2)
    
    P1 = cp.reshape(P1, (Dcut,Dcut,Dcut))
    P2 = cp.reshape(P2, (Dcut,Dcut,Dcut))

    return P2, P1


def new_pure_tensor(T, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    #gilt_eps = 0
    #if gilt_eps != 0:
    #    from itertools import cycle
    #    from trg.Gilt import gilt_for_2dHOTRG
    #    legs = 'ijkl'
    #    done_legs = {leg:False for leg in legs}
    #    for leg in cycle(legs):
    #        T, _, _, _, done = gilt_for_2dHOTRG(T, T, T, T, gilt_eps, leg)
    #        done_legs[leg] = done
    #
    #        if all(done_legs.values()):
    #            break
    #print("Tshape", T.shape)
    #T = cp.pad(T, pad_width=(Dcut-T.shape[0], Dcut-T.shape[1], Dcut-T.shape[2], Dcut-T.shape[3]), 
    #           mode='constant', constant_values =1e-16)

    T = __tensor_legs_rearrange__(T, direction)
    PLD, PRU = __squeezer__(T, Dcut)

    t0 = time.time()
    T = contract("acke,bdel,iab,cdj->ijkl", T, T, PLD, PRU)
    t1 = time.time()

    del PLD, PRU

    #print("average contraction time per tensor: {:.2e} s".format(t1-t0))
    T = __tensor_legs_restore__(T, direction)
    
    return T


def new_impuer_tensor_2imp(T, Timp0, Timp1, nx, ny, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T     = __tensor_legs_rearrange__(T, direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    Timp1 = __tensor_legs_rearrange__(Timp1, direction)
    PLD, PRU = __squeezer__(T, Dcut)

    t0 = time.time()
    Timp0 = contract("acke,bdel,iab,cdj->ijkl", Timp0, T, PLD, PRU)
    
    if (ny%2 == 1) and (direction == "Y" or direction == "y"):
        Timp1 = contract("acke,bdel,iab,cdj->ijkl", T, Timp1, PLD, PRU)

    if (ny%2 == 0) and (direction == "Y" or direction == "y"):
        Timp1 = contract("acke,bdel,iab,cdj->ijkl", Timp1, T, PLD, PRU)

    if (nx%2 == 1) and (direction == "X" or direction == "x"):
        Timp1 = contract("acke,bdel,iab,cdj->ijkl", T, Timp1, PLD, PRU)

    if (nx%2 == 0) and (direction == "X" or direction == "x"):
        Timp1 = contract("acke,bdel,iab,cdj->ijkl", Timp1, T, PLD, PRU)

    T = contract("acke,bdel,iab,cdj->ijkl", T, T, PLD, PRU)
    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format((t1-t0)/3))

    T     = __tensor_legs_rearrange__(T    , direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    Timp1 = __tensor_legs_rearrange__(Timp1, direction)

    del PLD, PRU

    return T, Timp0, Timp1

def new_impuer_tensor_2to1imp(T, Timp0, Timp1, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T     = __tensor_legs_rearrange__(T, direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    Timp1 = __tensor_legs_rearrange__(Timp1, direction)
    PLD, PRU = __squeezer__(T, Dcut)

    t0 = time.time()
    Timp0 = contract("acke,bdel,iab,cdj->ijkl", Timp0, Timp1, PLD, PRU, optimize=optimize)
    T = contract("acke,bdel,iab,cdj->ijkl", T, T, PLD, PRU, optimize=optimize)
    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format((t1-t0)/2))

    del Timp1, PLD, PRU

    T     = __tensor_legs_rearrange__(T    , direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)

    return T, Timp0

def new_impuer_tensor_1imp(T, Timp0, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T     = __tensor_legs_rearrange__(T    , direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    PLD, PRU = __squeezer__(T, Dcut)

    Timp0 = contract("acke,bdel,iab,cdj->ijkl", Timp0, T, PLD, PRU, optimize=optimize)
    T = contract("acke,bdel,iab,cdj->ijkl", T, T, PLD, PRU, optimize=optimize)

    t0 = time.time()
    T     = __tensor_legs_rearrange__(T    , direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format((t1-t0)/2))

    del PLD, PRU

    return T, Timp0


    
#Renormalize
def __tensor_normalization__(T):
    c = cp.max(cp.absolute(T))
    T = T / c
    return T, c

def __RG_flow_output__(type:str, data):
    if type == "squeezer":
        with open("{:}/squeezer_Lx2^{:}_Lt2^{:}.dat".format(OUTPUT_DIR, count_xloop, count_yloop), "w") as svout:
            for sL, sR in data:
                svout.write("{:.12e} {:.12e}\n".format(sL, sR))

    elif type == "tensor":
        with open("{:}/tensor_Lx2^{:}_Lt2^{:}.dat".format(OUTPUT_DIR, count_xloop, count_yloop), "w") as svout:
            T = cp.transpose(data, axes=(0,3,1,2))
            T = cp.reshape(T, (T.shape[0]*T.shape[1], T.shape[2]*T.shape[3]))
            _, s, _ = cp.linalg.svd(T)
            for ss in s:
               ss /= cp.max(s)
               svout.write("{:.12e}\n".format(ss))

def pure_tensor_renorm(T, Dcut:int, XLOOPS:int, YLOOPS:int):
    global count_xloop
    global count_yloop
    count_xloop = 0
    count_yloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+1, dtype=cp.float64)
    T, c = __tensor_normalization__(T)
    __RG_flow_output__(type="tensor", data=T)
    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)

    TrT = contract("iijj", T)
    print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}.".format(count_totloop, TrT, c))

    while (count_xloop < XLOOPS or count_yloop < YLOOPS):
        
        if count_yloop < YLOOPS:
            t0 = time.time()
            count_yloop += 1
            count_totloop += 1
            T = new_pure_tensor(T, Dcut, "Y")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            t1 = time.time()
            TrT = contract("iijj", T)
            print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s".format(count_totloop, TrT, c, t1-t0))
            __RG_flow_output__(type="tensor", data=T)
        
        if count_xloop < XLOOPS:
            t0 = time.time()
            count_xloop += 1
            count_totloop += 1
            T = new_pure_tensor(T, Dcut, "X")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            t1 = time.time()
            TrT = contract("iijj", T)
            print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s".format(count_totloop, TrT, c, t1-t0))
            __RG_flow_output__(type="tensor", data=T)


    return T, ln_normalized_factor


def renyi_entropy_renorm(T, Dcut:int, XLOOPS:int, YLOOPS:int):
    global count_xloop
    global count_yloop
    count_xloop = 0
    count_yloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+1, dtype=cp.float64)
    T, c = __tensor_normalization__(T)
    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)
    __RG_flow_output__(type="tensor", data=T)

    while (count_xloop < XLOOPS or count_yloop < YLOOPS):
        
        if count_yloop < YLOOPS:
            t0 = time.time()
            count_yloop += 1
            count_totloop += 1
            T = new_pure_tensor(T, Dcut, "Y")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            t1 = time.time()
            print("loop {:2d} finish. time: {:.6f} s".format(count_totloop, t1-t0))
            __RG_flow_output__(type="tensor", data=T)
        
        if count_xloop < XLOOPS:
            t0 = time.time()
            count_xloop += 1
            count_totloop += 1
            T = new_pure_tensor(T, Dcut, "X")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            t1 = time.time()
            print("loop {:2d} finish. time: {:.6f} s".format(count_totloop, t1-t0))
            __RG_flow_output__(type="tensor", data=T)


    return T, ln_normalized_factor


def entanglement_entropy_renorm(T, Dcut:int, XLOOPS:int, YLOOPS:int):
    count_xloop = 0
    count_yloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+1, dtype=cp.float64)
    T, c = __tensor_normalization__(T)
    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)

    ln_emax = []
    ln_ZoV  = []
    SEE = []
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

        #lnZ/V
        V = 2**count_totloop
        trace = cp.einsum("aabb", T)
        lnZoV = cp.sum(ln_normalized_factor) + cp.log(trace) / V
        ln_ZoV.append(lnZoV)


        TT = cp.transpose(T, axes=(0,3,1,2))
        TT = cp.reshape(TT, (Dcut*Dcut, Dcut*Dcut))
        t0 = time.time()
        #u, s, vh = cp.linalg.svd(TT)
        #e = cp.einsum("ia,ai,i->i", vh, u, s)
        _, e, _ = cp.linalg.svd(TT)
        
        t1 = time.time()
        
        #lnλ_0
        lnemax = cp.sum(ln_normalized_factor) + cp.log(e[0]) / V
        ln_emax.append(lnemax)

        #S_EE
        lam = e/cp.sum(e[:Dcut])
        see = - cp.sum(lam * cp.log(lam))
        SEE.append(see)

        #TM = cp.einsum("aayY->yY",T)
        #print(cp.linalg.norm(TM-cp.conj(TM.T))/cp.linalg.norm(TM))
        #print(e)
        #for ee,lamm,ee1 in zip(e,lam,e1):
        #    print("{:11.4e} {:11.4e} {:11.4e}".format(ee.real,lamm.real, ee1.real))
        #with open("O3_2d_D48_b3.0_eigvs.txt", "w") as out:
        #    for ee,lamm,ee1 in zip(e,lam,e1):
        #        print("{:11.4e} {:11.4e} {:11.4e}".format(ee.real,lamm.real, ee1.real))
        #        out.write("{:.12e} {:.12e} {:.12e}\n".format(ee.real,lamm.real, ee1.real))
    
    return ln_ZoV, SEE, ln_emax


def two_point_func_renorm(T, Timp0, Timp1, nx, ny, Dcut:int, XLOOPS:int, YLOOPS:int):
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
    __RG_flow_output__(type="tensor", data=T)

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

            __RG_flow_output__(type="tensor", data=T)
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

            __RG_flow_output__(type="tensor", data=T)
        t1 = time.time()
        print("loop {:2d} finish. time: {:.6f} s".format(count_totloop, t1-t0))

    return T, Timp0, ln_normalized_factor, ln_normalized_factor_imp

