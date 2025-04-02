import os
import numpy as np
import cupy as cp
import sys
import time
import math
from opt_einsum import contract 
import opt_einsum as oe
from itertools import product

from trg.gauge_fixing import gauge_fixing_2d
from utility.randomized_svd import rsvd
from utility.truncated_svd import svd

optimize={'slicing': {'min_slices': 4}}
truncate_eps = 1e-10

#OUTPUT_DIR = sys.argv[7]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]

def __tensor_legs_rearrange__(T, direction:str):
    if direction == "Y" or direction == "y":
        return T
    elif direction == "X" or direction == "x":
        #T = contract("abcd->cdab", T)
        T = cp.transpose(T, axes=(2,3,0,1))
        return T

def __tensor_legs_restore__(T, direction:str):
    if direction == "Y" or direction == "y":
        return T
    elif direction == "X" or direction == "x":
        #T = contract("cdab->abcd", T)
        T = cp.transpose(T, axes=(2,3,0,1))
        return T

def __squeezer__(T0, T1, T2, T3, Dcut:int):
    """
    >>>     d        f                d                f
    >>>     |    j   |                |                |
    >>> c---T1-------T2---e       c---T1---\       /---T2---e
    >>>     |        |                |     \     /    |
    >>>    i|        |k              i|    P1\---/P2   |k
    >>>     |        |                |     /     \    |
    >>> a---T0-------T3---g       a---T0---/       \---T3---g
    >>>     |    l   |                |                |
    >>>     b        h                b                h 
    returns left/down projector PLD and right/up projector PRU \\
    PLD_{x,x0,x1} or PLD_{y,y0,y1} \\
    PRU_{x'0,x'1,x'} or PRU_{y'0,y'1,y'}
    """

    t0 = time.time()
    LdagL = oe.contract("aibe,cjed,akbf,clfd->ijkl", cp.conj(T0), cp.conj(T1), T0, T1)
    RRdag = oe.contract("iabe,jced,kabf,lcfd->ijkl", T3, T2, cp.conj(T3), cp.conj(T2))
    t1 = time.time()

    #__sparse_check__(LdagL, "hotrg Env tensor")

    LdagL = cp.reshape(LdagL,  (LdagL.shape[0]*LdagL.shape[1], LdagL.shape[2]*LdagL.shape[3]))
    RRdag = cp.reshape(RRdag,  (RRdag.shape[0]*RRdag.shape[1], RRdag.shape[2]*RRdag.shape[3]))
    #Eigval1, Eigvect1 = cp.linalg.eigh(LdagL)
    #Eigval2, Eigvect2 = cp.linalg.eigh(RRdag)
    Eigvect1, Eigval1, _ = svd(LdagL, shape=[[0], [1]], k=min(*LdagL.shape), truncate_eps=truncate_eps)
    Eigvect2, Eigval2, _ = svd(RRdag, shape=[[0], [1]], k=min(*RRdag.shape), truncate_eps=truncate_eps)

    
    #Eigvect1, Eigval1, _ = svd(LdagL, shape=[[0], [1]], truncate_eps=truncate_eps)
    #Eigvect2, Eigval2, _ = svd(RRdag, shape=[[0], [1]], truncate_eps=truncate_eps)
    print(f"Tr(LdagL)={cp.trace(LdagL)}, Tr(RRdag)={cp.trace(RRdag)}")
    del LdagL, RRdag

    save_RG_flow(type="squeezer", data=zip(Eigval1, Eigval2), Dcut=Dcut)
   
    #Eigval1 = Eigval1.astype(cp.complex128)
    #Eigval2 = Eigval2.astype(cp.complex128)
    R1 = oe.contract("ia,a->ai", cp.conj(Eigvect1), cp.sqrt(Eigval1))
    R2 = oe.contract("ia,a->ia", Eigvect2, cp.sqrt(Eigval2))

    #U, S, VH = cp.linalg.svd(R1@R2)
    R1R2 = R1@R2
    k = min(*R1R2.shape, Dcut)
    U, S, VH = svd(R1R2, shape=[[0], [1]], k=k, truncate_eps=truncate_eps)
    UH = cp.conj(U.T)
    Sinv = 1 / S
    #Sinv = Sinv.astype(cp.complex128)
    V = cp.conj(VH.T)

    print("eL",Eigval1[:k])
    print("eR",Eigval2[:k])
    print("S", S[:k])

    del U, S, VH

    #if len(Sinv) > Dcut:
    #    UH = UH[:Dcut,:]
    #    Sinv = Sinv[:Dcut]
    #    V = V[:,:Dcut]

    P1 = oe.contract("ia,aj,j->ij", R2, V , cp.sqrt(Sinv))
    P2 = oe.contract("i,ia,aj->ij", cp.sqrt(Sinv), UH, R1)
    del R1, R2, Eigval1, Eigval2, Eigvect1, Eigvect2

    print("Tr(P1@P2)=", cp.trace(P1@P2), "|P1@P2|^2=", cp.linalg.norm(P1@P2)**2)
    
    P1_shape = (T0.shape[1], T1.shape[1], P1.shape[1])
    P2_shape = (P2.shape[0], T3.shape[0], T2.shape[0])
    P1 = cp.reshape(P1, P1_shape)
    P2 = cp.reshape(P2, P2_shape)

    return P2, P1

def slice_legs(leg_size, slicing):
    """
    leg_size, slicing: can be a exponential notation if the number is large
    slicing leg_size to slicing parts \\
    if leg_size < slicing, return a list of slice(i, i+1) \\
    if slicing==0, return one element list of slice(0, leg_size) \\
    return a list of slice class
    """
    leg_size = int(leg_size)
    slicing = int(slicing)

    if leg_size == 0:
        print(leg_size, slicing)
        import sys
        print("leg_size or slicing is zero!")
        sys.exit(0)

    if leg_size < slicing:
        return [slice(i, i+1) for i in range(leg_size)]
    
    if slicing == 0:
        return [slice(0, leg_size)]

    bs1 = leg_size // slicing
    bs2 = (leg_size // slicing) + 1
    n1 = (leg_size - bs2*slicing) // (bs1 - bs2)
    n2 = slicing - n1
    
    slice_list = [slice(i, i+bs1) for i in range(0, n1*bs1, bs1)]
    slice_list += [slice(i, i+bs2) for i in range(n1*bs1, leg_size, bs2)]

    return slice_list

def coarse_graining(T0:cp.ndarray, T1:cp.ndarray, PLD:cp.ndarray, PRU:cp.ndarray, slicing, Dcut):
    """
    >>> T0_{acke}, T1_{bedl}, PLD_{iab}, PRU_{cdj}
    >>>            l
    >>>            |
    >>>       /b---T1---d\\
    >>> i--PLD     |      d
    >>>      \\     e--e    \\
    >>>       a       |     PRU--j
    >>>         \\a---T0---c/  
    >>>               |   
    >>>               k
    """
    slicing = int(slicing)
    chi_a = PLD.shape[1]
    chi_e = T0.shape[2]
    chi_d = T1.shape[0]
    chi_i = PLD.shape[0]
    chi_j = PRU.shape[2]
    chi_k = T0.shape[2]
    chi_l = T1.shape[3]

    if slicing <= Dcut:
        slice_list_e = slice_legs(leg_size=chi_e, slicing=slicing)
        select = 1
    elif (Dcut < slicing) and (slicing <= Dcut*Dcut):
        slicing_e = slicing
        slicing_d = math.ceil(slicing_e / chi_d)
        slice_list_e = slice_legs(leg_size=chi_e, slicing=slicing_e)
        slice_list_d = slice_legs(leg_size=chi_d, slicing=slicing_d)
        select = 2
    elif (slicing < Dcut*Dcut) and (slicing < Dcut*Dcut*Dcut):
        slicing_e = slicing
        slicing_d = math.ceil(slicing_e / chi_d)
        slicing_a = math.ceil(slicing_d / chi_a)
        slice_list_e = slice_legs(leg_size=chi_e, slicing=slicing_e)
        slice_list_d = slice_legs(leg_size=chi_d, slicing=slicing_d)
        slice_list_a = slice_legs(leg_size=chi_a, slicing=slicing_a)
        select = 3

    path = [(0, 2), (0, 1), (0, 1)]
    subscripts = "acke,bdel,iab,cdj->ijkl"
    T = cp.zeros(shape=(chi_i, chi_j, chi_k, chi_l), dtype=T0.dtype)
    if select == 1:
        for e in slice_list_e:
            T += oe.contract(subscripts, T0[:,:,:,e], T1[:,:,e,:], PLD, PRU, optimize=path)

    elif select == 2:
        iteration = product(slice_list_e, slice_list_d)
        for e,d in iteration:
            T += oe.contract(subscripts, T0[:,:,:,e], T1[:,d,e,:], PLD, PRU[:,d,:], optimize=path)

    elif select == 3:
        iteration = product(slice_list_e, slice_list_d, slice_list_a)
        for e,d,a in iteration:
            T += oe.contract(subscripts, T0[a,:,:,e], T1[:,d,e,:], PLD[:,a,:], PRU[:,d,:], optimize=path)
    
    return T


def new_pure_tensor(T, Dcut:int, gilt_eps, direction:str):

    from trg.Gilt import gilt_plaq_2dHOTRG

    Ngilt = int(os.environ['NGILT'])
    gilt_legs = int(os.environ['NCUTLEGS'])

    if Ngilt == 1:
        if direction == 'y' or direction == 'Y':
            T0, T1 = gilt_plaq_2dHOTRG(T, T, gilt_eps, direction, gilt_legs)
        elif direction == 'x' or direction == 'X':
            T0, T1 = T, T
    elif Ngilt == 2:
        T0, T1 = gilt_plaq_2dHOTRG(T, T, gilt_eps, direction, gilt_legs)


    #from trg.Gilt import gilt_plaq_2dHOTRG_22222
    #if Ngilt == 1:
    #    if direction == 'y' or direction == 'Y':
    #        T0, T1 = gilt_plaq_2dHOTRG_22222(T, T, gilt_eps, direction, gilt_legs)
    #    elif direction == 'x' or direction == 'X':
    #        T0, T1 = T, T
    #elif Ngilt == 2:
    #    T0, T1 = gilt_plaq_2dHOTRG_22222(T, T, gilt_eps, direction, gilt_legs)


    print(f"normT0={cp.linalg.norm(T0)}, normT1={cp.linalg.norm(T1)}")
    T0 = __tensor_legs_rearrange__(T0, direction)
    T1 = __tensor_legs_rearrange__(T1, direction)

    PLD, PRU = __squeezer__(T0, T1, T1, T0, Dcut)

    #print("mem_info:", cp.cuda.Device(0).mem_info)
    
    t0 = time.time()
    slicing=Dcut
    T = coarse_graining(T0, T1, PLD, PRU, slicing=slicing, Dcut=Dcut)
    t1 = time.time()

    #if direction == 'y' or direction == 'Y':
    #    T = cp.transpose(T, axes=(0,3,1,2))
    #    T = gauge_fixing_2d(T)
    #    T = cp.transpose(T, axes=(0,2,3,1))
    
    del PLD, PRU, T0, T1

    T = __tensor_legs_restore__(T, direction)
    
    return T

#def new_pure_tensor_chain(T, Dcut:int, gilt_eps, direction):
#    from trg.Gilt import gilt_chain_2dHOTRG
#
#    T0, T1 = gilt_chain_2dHOTRG(T, T, gilt_eps, direction)
#
#    T0 = __tensor_legs_rearrange__(T0, direction)
#    T1 = __tensor_legs_rearrange__(T1, direction)
#    PLD, PRU = __squeezer__(T0, T1, T1, T0, Dcut)
#    
#    T = coarse_graining(T0, T1, PLD, PRU, slicing=100, Dcut=Dcut)
#
#    del PLD, PRU, T0, T1
#
#    T = __tensor_legs_restore__(T, direction)
#    
#    return T

def new_impuer_tensor_2imp(T, Timp0, Timp1, nx, ny, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T     = __tensor_legs_rearrange__(T, direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    Timp1 = __tensor_legs_rearrange__(Timp1, direction)
    PLD, PRU = __squeezer__(T, T, T, T, Dcut)

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
    PLD, PRU = __squeezer__(T, T, T, T, Dcut)

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
    PLD, PRU = __squeezer__(T, T, T, T, Dcut)

    Timp0 = contract("acke,bdel,iab,cdj->ijkl", Timp0, T, PLD, PRU, optimize=optimize)
    T = contract("acke,bdel,iab,cdj->ijkl", T, T, PLD, PRU, optimize=optimize)

    t0 = time.time()
    T     = __tensor_legs_rearrange__(T    , direction)
    Timp0 = __tensor_legs_rearrange__(Timp0, direction)
    t1 = time.time()
    #print("average contraction time per tensor: {:.2e} s".format((t1-t0)/2))

    del PLD, PRU

    return T, Timp0

def __sparse_check__(T, which):
    Tabs = cp.abs(T)
    maxabs = cp.max(Tabs)
    count1 = Tabs[Tabs/maxabs < 1e-12].size / Tabs.size
    count2 = Tabs[Tabs/maxabs < 1e-4].size  / Tabs.size
    count3 = Tabs[Tabs/maxabs < 1e-3].size  / Tabs.size
    count4 = Tabs[Tabs/maxabs < 1e-2].size  / Tabs.size
    count5 = Tabs[Tabs/maxabs < 1e-5].size  / Tabs.size
    count6 = Tabs[Tabs/maxabs < 1e-6].size  / Tabs.size
    print(f"{which}: |T|/|max(|T|)|<1e-12: {count1}")
    print(f"{which}: |T|/|max(|T|)|<1e-6 : {count6}")
    print(f"{which}: |T|/|max(|T|)|<1e-5 : {count5}")
    print(f"{which}: |T|/|max(|T|)|<1e-4 : {count2}")
    print(f"{which}: |T|/|max(|T|)|<1e-3 : {count3}")
    print(f"{which}: |T|/|max(|T|)|<1e-2 : {count4}")
    del Tabs

    
#Renormalize
def __tensor_normalization__(T):
    #if (count_xloop+count_yloop) % 2 == 0:
    #    c = oe.contract("iijj", T)
    #    c = cp.abs(c)
    #else:
    #    c = 1
    c = oe.contract("iijj", T)
    c = cp.abs(c)
    T = T / c
    return T, c

def save_tensor_data(T, c):
    path = "{:}/data".format(OUTPUT_DIR)
    if not os.path.isdir(path):
        os.mkdir(path)
    file = "{:}/tensor_lx{:}_ly{:}.npz".format(path, count_xloop, count_yloop)
    if not os.path.isfile(file):
        cp.savez(file, tensor=T, factor=c)

def load_tensor_data(T:cp.ndarray):
    path = "{:}/data".format(OUTPUT_DIR)
    file = "{:}/tensor_lx{:}_ly{:}.npz".format(path, count_xloop, count_yloop)
    if os.path.isfile(file):
        data = cp.load(file, mmap_mode='r')
        T = data['tensor']
        #c = data['factor']
    return T

def get_new_pure_tensor(T:cp.ndarray, Dcut:int, gilt_eps:float, direction:str, contract_chain=False):
    path = "{:}/data".format(OUTPUT_DIR)
    file = "{:}/tensor_lx{:}_ly{:}.npz".format(path, count_xloop, count_yloop)
    if os.path.isfile(file):
        print("load /data/tensor_lx{:}_ly{:}.npz".format(count_xloop, count_yloop))
        data = cp.load(file, mmap_mode='r')
        T = data['tensor']
        c = data['factor']
    else:
        if not contract_chain:
            T = new_pure_tensor(T, Dcut, gilt_eps, direction)
        else:
            T = new_pure_tensor(T, Dcut, 0.0, direction)
        T, c = __tensor_normalization__(T)
    return T, c

def save_entropy(T, Dcut):
    TrTBTA = contract("ijaa,jibb", T, T)
    print("TrTBTA=",TrTBTA)
    Dens_Mat = contract("ijab,jicd->acbd", T, T) / TrTBTA
    chi_1, chi_2, chi_3, chi_4 = Dens_Mat.shape[0], Dens_Mat.shape[1], Dens_Mat.shape[2], Dens_Mat.shape[3]
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(chi_1*chi_2, chi_3*chi_4))
    _, e_Dens_Mat, _ = cp.linalg.svd(Dens_Mat)
    print("dens_mat hermit err",cp.linalg.norm(Dens_Mat-cp.conj(Dens_Mat.T))/cp.linalg.norm(Dens_Mat))

    STE = - cp.sum( e_Dens_Mat * cp.log(e_Dens_Mat))
    if count_xloop+count_yloop == 0:
        with open("{:}/STE.dat".format(OUTPUT_DIR), "w") as ste_out:
            ste_out.write("{:.15e}\n".format(STE))
    else:
        with open("{:}/STE.dat".format(OUTPUT_DIR), "a") as ste_out:
            ste_out.write("{:.15e}\n".format(STE))

    if not os.path.exists("{:}/densitymatrix".format(OUTPUT_DIR)):
        os.mkdir("{:}/densitymatrix".format(OUTPUT_DIR))
    with open("{:}/densitymatrix/densitymatrix_lx{:}_lt{:}.dat".format(OUTPUT_DIR, count_xloop, count_yloop), "w") as svout:
        emax = cp.max(e_Dens_Mat)
        e_dens_mat = e_Dens_Mat / emax
        if len(e_dens_mat) < Dcut*Dcut:
            e_dens_mat = cp.pad(e_dens_mat, pad_width=(0,Dcut*Dcut-len(e_dens_mat)), mode='constant', constant_values=0.0)
        svout.write("#ρmax={:.12e}\n".format(emax))
        for ee in e_dens_mat:
           svout.write("{:.12e}\n".format(ee))

    Dens_Mat = cp.reshape(Dens_Mat, newshape=(chi_1, chi_2, chi_3, chi_4))
    rho_A = contract("aiaj->ij", Dens_Mat)
    _, e_A, _ = cp.linalg.svd(rho_A)
    print("rho_A hermit err",cp.linalg.norm(rho_A-cp.conj(rho_A.T))/cp.linalg.norm(rho_A))

    SEE = -cp.sum(e_A * cp.log(e_A))
    if count_xloop+count_yloop == 0:
        with open("{:}/SEE.dat".format(OUTPUT_DIR), "w") as see_out:
            see_out.write("{:.15e}\n".format(SEE))
    else:
        with open("{:}/SEE.dat".format(OUTPUT_DIR), "a") as see_out:
            see_out.write("{:.15e}\n".format(SEE))

    if not os.path.exists("{:}/densitymatrix_A".format(OUTPUT_DIR)):
        os.mkdir("{:}/densitymatrix_A".format(OUTPUT_DIR))
    with open("{:}/densitymatrix_A/densitymatrix_A_lx{:}_lt{:}.dat".format(OUTPUT_DIR, count_xloop, count_yloop), "w") as svout:
        emax = cp.max(e_A)
        e_a = e_A / emax
        if len(e_a) < Dcut:
            e_a = cp.pad(e_a, pad_width=(0,Dcut-len(e_a)), mode='constant', constant_values=0.0)
        svout.write("#ρAmax={:.12e}\n".format(emax))
        for ee in e_a:
           svout.write("{:.12e}\n".format(ee))

def save_free_energy(T, ln_normfact, xloops, yloops):
    trace = contract("aabb", T)
    V = 2**(xloops+yloops)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V
    if count_xloop+count_yloop == 0:
        with open("{:}/free_energy.dat".format(OUTPUT_DIR), "w") as fe:
            fe.write("{:.15e}\n".format(ln_ZoverV))
    else:
        with open("{:}/free_energy.dat".format(OUTPUT_DIR), "a") as fe:
            fe.write("{:.15e}\n".format(ln_ZoverV))

def save_RG_flow(type:str, data, Dcut):
    if type == "squeezer":
        if not os.path.exists("{:}/squeezer".format(OUTPUT_DIR)):
            os.mkdir("{:}/squeezer".format(OUTPUT_DIR))
        with open("{:}/squeezer/squeezer_lx{:}_lt{:}.dat".format(OUTPUT_DIR, count_xloop, count_yloop), "w") as svout:
            data_list = list(zip(*data))
            sL, sR = data_list[0], data_list[1]
            del data_list
            sL = cp.asarray(sL)
            sR = cp.asarray(sR)
            if len(sL) < Dcut*Dcut:
                sL = cp.pad(sL, pad_width=(0,Dcut*Dcut-len(sL)), mode='constant', constant_values=0.0)
                sR = cp.pad(sR, pad_width=(0,Dcut*Dcut-len(sR)), mode='constant', constant_values=0.0)
            sLmax = cp.max(sL)
            sRmax = cp.max(sR)
            svout.write("#max(sL)={:.12e} max(sR)={:.12e}\n".format(sLmax, sRmax))
            sL /= sLmax
            sR /= sRmax
            for sl, sr in zip(sL, sR):
                svout.write("{:.12e} {:.12e}\n".format(sl, sr))

        return sLmax, sRmax

    elif type == "tensor":
        if not os.path.exists("{:}/tensor".format(OUTPUT_DIR)):
            os.mkdir("{:}/tensor".format(OUTPUT_DIR))
        with open("{:}/tensor/tensor_lx{:}_lt{:}.dat".format(OUTPUT_DIR, count_xloop, count_yloop), "w") as svout:
            T = cp.transpose(data, axes=(0,3,1,2))
            T = cp.reshape(T, (T.shape[0]*T.shape[1], T.shape[2]*T.shape[3]))
            _, s, _ = cp.linalg.svd(T)
            if len(s) < Dcut*Dcut:
                s = cp.pad(s, pad_width=(0,Dcut*Dcut-len(s)), mode='constant', constant_values=0.0)
            smax = cp.max(s)
            svout.write("#max(s)={:.12e}\n".format(smax))
            s /= smax
            for ss in s:
               svout.write("{:.12e}\n".format(ss))

        return smax

    elif type == "transfer_matrix":
        if not os.path.exists("{:}/transfer_matrix".format(OUTPUT_DIR)):
            os.mkdir("{:}/transfer_matrix".format(OUTPUT_DIR))
        with open("{:}/transfer_matrix/transfer_matrix_lx{:}_lt{:}.dat".format(OUTPUT_DIR, count_xloop, count_yloop), "w") as svout:
            M = cp.einsum("xxyY->yY", data)
            print("transfer matrix hermit err:", cp.linalg.norm(M-cp.conj(M.T))/cp.linalg.norm(M))
            _, e, _ = cp.linalg.svd(M)
            if len(e) < Dcut*Dcut:
                e = cp.pad(e, pad_width=(0,Dcut*Dcut-len(e)), mode='constant', constant_values=0.0)
            emax = cp.max(e)
            svout.write("#max(e)={:.12e}\n".format(emax))
            e /= emax
            for ee in e:
               svout.write("{:.12e}\n".format(ee))

        if count_xloop+count_yloop == 0:
            with open(f"{OUTPUT_DIR}/max_eigvs.dat", "w") as output:
                output.write(f"{emax:.12e}\n")
        elif count_xloop+count_yloop > 0:
            with open(f"{OUTPUT_DIR}/max_eigvs.dat", "a") as output:
                output.write(f"{emax:.12e}\n")

        return emax
    
def save_correlation_length(T:cp.ndarray, Dcut:int):

    outdir = "{:}/correlation_length".format(OUTPUT_DIR)
    if not os.path.exists(outdir):
            os.mkdir(outdir)
    
    #calculate ξt
    #Ls=L
    M = oe.contract("xxyY->yY", T)
    _, e, _ = cp.linalg.svd(M)
    if len(e) < Dcut*Dcut:
        e = cp.pad(e, pad_width=(0,Dcut*Dcut-len(e)), mode='constant', constant_values=0.0)
    emax = cp.max(e)
    e /= emax
    for i in range(1,7):
        eL = e**i
        if count_xloop+count_yloop == 0:
            with open(f"{outdir}/ξt_Lt{i}L_Ls1L.dat", "w") as output:
                e1 = eL[1] if eL[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + count_yloop * cp.log(2) - cp.log(-1 / cp.log(e1))
                output.write(f"#λ2^{i}={e1}\n")
                output.write(f"{lnxi_t:.12e}\n")
        elif count_xloop+count_yloop > 0:
            with open(f"{outdir}/ξt_Lt{i}L_Ls1L.dat", "a") as output:
                e1 = eL[1] if eL[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + count_yloop * cp.log(2) - cp.log(-1 / cp.log(e1))
                output.write(f"{lnxi_t:.12e}\n")

    #Ls=2L
    M2L = oe.contract("ijab,jicd->acbd", T, T)
    chi_1, chi_2, chi_3, chi_4 = M2L.shape[0], M2L.shape[1], M2L.shape[2], M2L.shape[3]
    M2L = cp.reshape(M2L, newshape=(chi_1*chi_2, chi_3*chi_4))
    _, e2, _ = cp.linalg.svd(M2L)
    if len(e2) < Dcut*Dcut:
        e2 = cp.pad(e2, pad_width=(0,Dcut*Dcut-len(e2)), mode='constant', constant_values=0.0)
    e2max = cp.max(e2)
    e2 /= e2max
    for i in range(1,7):
        e2L = e2**i
        if count_xloop+count_yloop == 0:
            with open(f"{outdir}/ξt_Lt{i}L_Ls2L.dat", "w") as output:
                e1 = e2L[1] if e2L[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + count_yloop * cp.log(2) - cp.log(-1 / cp.log(e1))
                output.write(f"#λ2^{i}={e1}\n")
                output.write(f"{lnxi_t:.12e}\n")
        elif count_xloop+count_yloop > 0:
            with open(f"{outdir}/ξt_Lt{i}L_Ls2L.dat", "a") as output:
                e1 = e2L[1] if e2L[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + count_yloop * cp.log(2) - cp.log(-1 / cp.log(e1))
                output.write(f"{lnxi_t:.12e}\n")


    #calculate ξs
    #Lt=L
    M = oe.contract("xXyy->xX", T)
    _, e, _ = cp.linalg.svd(M)
    if len(e) < Dcut*Dcut:
        e = cp.pad(e, pad_width=(0,Dcut*Dcut-len(e)), mode='constant', constant_values=0.0)
    emax = cp.max(e)
    e /= emax
    for i in range(1,7):
        eL = e**i
        if count_xloop+count_yloop == 0:
            with open(f"{outdir}/ξs_Lt1L_Ls{i}L.dat", "w") as output:
                e1 = eL[1] if eL[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + count_xloop * cp.log(2) - cp.log(-1 / cp.log(e1))
                output.write(f"#λ2^{i}={e1}\n")
                output.write(f"{lnxi_t:.12e}\n")
        elif count_xloop+count_yloop > 0:
            with open(f"{outdir}/ξs_Lt1L_Ls{i}L.dat", "a") as output:
                e1 = eL[1] if eL[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + count_xloop * cp.log(2) - cp.log(-1 / cp.log(e1))
                output.write(f"{lnxi_t:.12e}\n")

    #Lt=2L
    M2L = oe.contract("abij,cdji->acbd", T, T)
    chi_1, chi_2, chi_3, chi_4 = M2L.shape[0], M2L.shape[1], M2L.shape[2], M2L.shape[3]
    M2L = cp.reshape(M2L, newshape=(chi_1*chi_2, chi_3*chi_4))
    _, e2, _ = cp.linalg.svd(M2L)
    if len(e2) < Dcut*Dcut:
        e2 = cp.pad(e2, pad_width=(0,Dcut*Dcut-len(e2)), mode='constant', constant_values=0.0)
    e2max = cp.max(e2)
    e2 /= e2max
    for i in range(1,7):
        e2L = e2**i
        if count_xloop+count_yloop == 0:
            with open(f"{outdir}/ξs_Lt2L_Ls{i}L.dat", "w") as output:
                e1 = e2L[1] if e2L[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + count_xloop * cp.log(2) - cp.log(-1 / cp.log(e1))
                output.write(f"#λ2^{i}={e1}\n")
                output.write(f"{lnxi_t:.12e}\n")
        elif count_xloop+count_yloop > 0:
            with open(f"{outdir}/ξs_Lt2L_Ls{i}L.dat", "a") as output:
                e1 = e2L[1] if e2L[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + count_xloop * cp.log(2) - cp.log(-1 / cp.log(e1))
                output.write(f"{lnxi_t:.12e}\n")


def cal_X(T:cp.ndarray, save=False):
    TrT = oe.contract("xxyy", T)
    TTx = oe.contract("xXaa,Xxbb", T, T)
    TTy = oe.contract("aayY,bbYy", T, T)

    Xx = TrT**2 / TTx
    Xy = TrT**2 / TTy
    
    if save:
        fname = OUTPUT_DIR + "/X.dat"
        if count_xloop + count_yloop == 0:
            mode = "w"
        else:
            mode = "a"
        with open(fname, mode) as out:
            out.write(f"{Xy.real:.12e} {Xx.real:.12e}\n")
    
    print(f"Xy={Xy:.12e}")
    print(f"Xx={Xx:.12e}")

    return Xy, Xx

def find_1stto3rdmax(T:cp.ndarray, save=False):
    #chiX, chiY, chiT = T.shape[::2]
    #Tdiag = [T[x,x,y,y,t,t] for x in range(chiX) for y in range(chiY) for t in range(chiT)]
    Tdiag = T.copy()
    #Tdiag = oe.contract("xxyytt->xyt", T)
    Tdiag = Tdiag.flatten()
    Tabs  = cp.abs(Tdiag)
    indices = cp.argsort(Tabs)[::-1]
    print("diagnal of T:", (Tdiag[indices])[:32])
    Tmax = cp.zeros(3, complex)
    for n,i in enumerate(indices):
        if n >= 3:
            break
        Tmax[n] = Tdiag[i]

    if save:
        if count_xloop + count_yloop == 0:
            mode = "w"
        else:
            mode = "a"
        fname = OUTPUT_DIR + "/max_1to3-th_values.dat"
        with open(fname, mode) as output:
            write = f"{Tmax[0]:40.12e} {Tmax[1]:40.12e} {Tmax[2]:40.12e}\n"
            output.write(write)

from measurement.measurement import save_ground_state_energy
def pure_tensor_renorm(T, Dcut:int, gilt_eps, XLOOPS:int, YLOOPS:int):
    global count_xloop
    global count_yloop
    count_xloop = 0
    count_yloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+1, dtype=cp.float64)

    t0 = time.time()
    T, c = __tensor_normalization__(T)
    #save_tensor_data(T, c)
    save_RG_flow(type="tensor", data=T, Dcut=Dcut)
    max_eigv = save_RG_flow(type="transfer_matrix", data=T, Dcut=Dcut)
    save_entropy(T, Dcut)
    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)
    save_ground_state_energy(ln_normalized_factor, max_eigv, count_totloop, OUTPUT_DIR)

    with open(f"{OUTPUT_DIR}/sum_normfact.dat", "w") as output:
        sum_normfact = cp.sum(ln_normalized_factor)
        output.write(f"{sum_normfact:.12e}\n")

    t1 = time.time()

    save_free_energy(T, ln_normalized_factor, count_xloop, count_yloop)
    save_correlation_length(T, Dcut)
    #__sparse_check__(T, "tensor")
    #print("loop {:2d} finish.\n".format(count_totloop))
    cal_X(T, save=True)
    find_1stto3rdmax(T, save=False)

    TrT = contract("iijj", T)
    print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_totloop, TrT, c, t1-t0))

    while (count_xloop < XLOOPS or count_yloop < YLOOPS):

        if count_yloop < YLOOPS:
            t0 = time.time()
            T = load_tensor_data(T)
            count_yloop += 1
            count_totloop += 1
            print(f"loop {count_totloop} start.")
            T, c = get_new_pure_tensor(T, Dcut, gilt_eps, "Y")
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            #save_tensor_data(T, c)
            save_RG_flow(type="tensor", data=T, Dcut=Dcut)
            max_eigv = save_RG_flow(type="transfer_matrix", data=T, Dcut=Dcut)
            save_entropy(T, Dcut)
            save_free_energy(T, ln_normalized_factor, count_xloop, count_yloop)
            save_ground_state_energy(ln_normalized_factor, max_eigv, count_totloop, OUTPUT_DIR)
            save_correlation_length(T, Dcut)
            cal_X(T, save=True)
            find_1stto3rdmax(T, save=False)
            TrT = contract("iijj", T)
            t1 = time.time()

            with open(f"{OUTPUT_DIR}/sum_normfact.dat", "a") as output:
                sum_normfact = cp.sum(ln_normalized_factor)
                output.write(f"{sum_normfact:.12e}\n")
            print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_totloop, TrT, c, t1-t0))
            
        if count_xloop < XLOOPS:
            t0 = time.time()
            T = load_tensor_data(T)
            count_xloop += 1
            count_totloop += 1
            print(f"loop {count_totloop} start.")
            T, c = get_new_pure_tensor(T, Dcut, gilt_eps, "X")
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            #save_tensor_data(T, c)
            max_eigv = save_RG_flow(type="tensor", data=T, Dcut=Dcut)
            save_RG_flow(type="transfer_matrix", data=T, Dcut=Dcut)
            save_entropy(T, Dcut)
            save_free_energy(T, ln_normalized_factor, count_xloop, count_yloop)
            save_ground_state_energy(ln_normalized_factor, max_eigv, count_totloop, OUTPUT_DIR)
            save_correlation_length(T, Dcut)
            cal_X(T, save=True)
            find_1stto3rdmax(T, save=False)
            TrT = contract("iijj", T)
            t1 = time.time()

            with open(f"{OUTPUT_DIR}/sum_normfact.dat", "a") as output:
                sum_normfact = cp.sum(ln_normalized_factor)
                output.write(f"{sum_normfact:.12e}\n")
            print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_totloop, TrT, c, t1-t0))
            
    return T, ln_normalized_factor

def pure_tensor_renorm_Xfist_thenY(T, Dcut:int, gilt_eps, XLOOPS:int, YLOOPS:int):
    global count_xloop
    global count_yloop
    count_xloop = 0
    count_yloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+1, dtype=cp.float64)

    t0 = time.time()
    T, c = __tensor_normalization__(T)
    save_tensor_data(T, c)
    save_RG_flow(type="tensor", data=T, Dcut=Dcut)
    max_eigv = save_RG_flow(type="transfer_matrix", data=T, Dcut=Dcut)
    save_entropy(T, Dcut)
    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)
    save_ground_state_energy(ln_normalized_factor, max_eigv, count_totloop, OUTPUT_DIR)

    with open(f"{OUTPUT_DIR}/sum_normfact.dat", "w") as output:
        sum_normfact = cp.sum(ln_normalized_factor)
        output.write(f"{sum_normfact:.12e}\n")

    t1 = time.time()

    save_free_energy(T, ln_normalized_factor, count_xloop, count_yloop)
    #__sparse_check__(T, "tensor")
    #print("loop {:2d} finish.\n".format(count_totloop))

    TrT = contract("iijj", T)
    print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_totloop, TrT, c, t1-t0))

    while (count_xloop < XLOOPS):
        t0 = time.time()
        T = load_tensor_data(T)
        count_xloop += 1
        count_totloop += 1
        print(f"loop {count_totloop} start.")
        T, c = get_new_pure_tensor(T, Dcut, gilt_eps, "X")
        ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
        save_tensor_data(T, c)
        max_eigv = save_RG_flow(type="tensor", data=T, Dcut=Dcut)
        save_RG_flow(type="transfer_matrix", data=T, Dcut=Dcut)
        save_entropy(T, Dcut)
        save_free_energy(T, ln_normalized_factor, count_xloop, count_yloop)
        save_ground_state_energy(ln_normalized_factor, max_eigv, count_totloop, OUTPUT_DIR)
        TrT = contract("iijj", T)
        t1 = time.time()

        with open(f"{OUTPUT_DIR}/sum_normfact.dat", "a") as output:
            sum_normfact = cp.sum(ln_normalized_factor)
            output.write(f"{sum_normfact:.12e}\n")
        print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_totloop, TrT, c, t1-t0))

    while (count_yloop < YLOOPS):
        t0 = time.time()
        T = load_tensor_data(T)
        count_yloop += 1
        count_totloop += 1
        print(f"loop {count_totloop} start.")
        T, c = get_new_pure_tensor(T, Dcut, gilt_eps, "Y")
        ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
        save_tensor_data(T, c)
        save_RG_flow(type="tensor", data=T, Dcut=Dcut)
        max_eigv = save_RG_flow(type="transfer_matrix", data=T, Dcut=Dcut)
        save_entropy(T, Dcut)
        save_free_energy(T, ln_normalized_factor, count_xloop, count_yloop)
        save_ground_state_energy(ln_normalized_factor, max_eigv, count_totloop, OUTPUT_DIR)
        TrT = contract("iijj", T)
        t1 = time.time()

        with open(f"{OUTPUT_DIR}/sum_normfact.dat", "a") as output:
            sum_normfact = cp.sum(ln_normalized_factor)
            output.write(f"{sum_normfact:.12e}\n")
        print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_totloop, TrT, c, t1-t0))
            
    return T, ln_normalized_factor

def pure_tensor_renorm_Y2X1(T, Dcut:int, gilt_eps, XLOOPS:int, YLOOPS:int):
    global count_xloop
    global count_yloop
    count_xloop = 0
    count_yloop = 0
    count_totloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+1, dtype=cp.float64)

    t0 = time.time()
    T, c = __tensor_normalization__(T)
    save_tensor_data(T, c)
    save_RG_flow(type="tensor", data=T, Dcut=Dcut)
    max_eigv = save_RG_flow(type="transfer_matrix", data=T, Dcut=Dcut)
    save_entropy(T, Dcut)
    ln_normalized_factor[0] = cp.log(c) / 2**(count_totloop)
    save_ground_state_energy(ln_normalized_factor, max_eigv, count_totloop, OUTPUT_DIR)

    with open(f"{OUTPUT_DIR}/sum_normfact.dat", "w") as output:
        sum_normfact = cp.sum(ln_normalized_factor)
        output.write(f"{sum_normfact:.12e}\n")

    t1 = time.time()

    save_free_energy(T, ln_normalized_factor, count_xloop, count_yloop)

    TrT = contract("iijj", T)
    print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_totloop, TrT, c, t1-t0))

    while (count_xloop < XLOOPS or count_yloop < YLOOPS):

        if count_yloop < YLOOPS:
            t0 = time.time()
            T = load_tensor_data(T)
            count_yloop += 1
            count_totloop += 1
            print(f"loop {count_totloop} start.")
            T, c = get_new_pure_tensor(T, Dcut, gilt_eps, "Y")
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            save_tensor_data(T, c)
            save_RG_flow(type="tensor", data=T, Dcut=Dcut)
            max_eigv = save_RG_flow(type="transfer_matrix", data=T, Dcut=Dcut)
            save_entropy(T, Dcut)
            save_free_energy(T, ln_normalized_factor, count_xloop, count_yloop)
            save_ground_state_energy(ln_normalized_factor, max_eigv, count_totloop, OUTPUT_DIR)
            TrT = contract("iijj", T)
            t1 = time.time()

            with open(f"{OUTPUT_DIR}/sum_normfact.dat", "a") as output:
                sum_normfact = cp.sum(ln_normalized_factor)
                output.write(f"{sum_normfact:.12e}\n")
            print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_totloop, TrT, c, t1-t0))
            
        if count_yloop < YLOOPS:
            t0 = time.time()
            T = load_tensor_data(T)
            count_yloop += 1
            count_totloop += 1
            print(f"loop {count_totloop} start.")
            T, c = get_new_pure_tensor(T, Dcut, gilt_eps, "Y")
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            save_tensor_data(T, c)
            save_RG_flow(type="tensor", data=T, Dcut=Dcut)
            max_eigv = save_RG_flow(type="transfer_matrix", data=T, Dcut=Dcut)
            save_entropy(T, Dcut)
            save_free_energy(T, ln_normalized_factor, count_xloop, count_yloop)
            save_ground_state_energy(ln_normalized_factor, max_eigv, count_totloop, OUTPUT_DIR)
            TrT = contract("iijj", T)
            t1 = time.time()

            with open(f"{OUTPUT_DIR}/sum_normfact.dat", "a") as output:
                sum_normfact = cp.sum(ln_normalized_factor)
                output.write(f"{sum_normfact:.12e}\n")
            print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_totloop, TrT, c, t1-t0))

        if count_xloop < XLOOPS:
            t0 = time.time()
            T = load_tensor_data(T)
            count_xloop += 1
            count_totloop += 1
            print(f"loop {count_totloop} start.")
            T, c = get_new_pure_tensor(T, Dcut, gilt_eps, "X")
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            save_tensor_data(T, c)
            max_eigv = save_RG_flow(type="tensor", data=T, Dcut=Dcut)
            save_RG_flow(type="transfer_matrix", data=T, Dcut=Dcut)
            save_entropy(T, Dcut)
            save_free_energy(T, ln_normalized_factor, count_xloop, count_yloop)
            save_ground_state_energy(ln_normalized_factor, max_eigv, count_totloop, OUTPUT_DIR)
            TrT = contract("iijj", T)
            t1 = time.time()

            with open(f"{OUTPUT_DIR}/sum_normfact.dat", "a") as output:
                sum_normfact = cp.sum(ln_normalized_factor)
                output.write(f"{sum_normfact:.12e}\n")
            print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_totloop, TrT, c, t1-t0))
            
    return T, ln_normalized_factor

def transfermatrix_renorm(T, Dcut:int, gilt_eps, XLOOPS:int, YLOOPS:int):
    from measurement.measurement import save_entropy_transfermatrix_method
    global count_xloop
    global count_yloop
    count_xloop = 0
    count_yloop = 0
    ln_normalized_factor = cp.zeros(XLOOPS+YLOOPS+1, dtype=cp.float64)

    t0 = time.time()
    T, c = __tensor_normalization__(T)
    ln_normalized_factor[count_xloop] = cp.log(c) / 2**(count_xloop)

    save_tensor_data(T, c)
    save_RG_flow(type="tensor", data=T, Dcut=Dcut)
    save_RG_flow(type="transfer_matrix", data=T, Dcut=Dcut)
    save_entropy_transfermatrix_method(T, ln_normalized_factor, Dcut, "hotrg", count_xloop, YLOOPS, OUTPUT_DIR)
    t1 = time.time()

    TrT = contract("iijj", T)
    print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_xloop, TrT, c, t1-t0))

    while count_xloop < XLOOPS:
        t0 = time.time()
        T = load_tensor_data(T)
        count_xloop += 1
        T, c = get_new_pure_tensor(T, Dcut, gilt_eps, "X", contract_chain=True)
        ln_normalized_factor[count_xloop] = cp.log(c) / 2**(count_xloop)
        
        save_tensor_data(T, c)
        save_RG_flow(type="tensor", data=T, Dcut=Dcut)
        save_RG_flow(type="transfer_matrix", data=T, Dcut=Dcut)
        save_entropy_transfermatrix_method(T, ln_normalized_factor, Dcut, "hotrg", count_xloop, YLOOPS, OUTPUT_DIR)
        
        TrT = contract("iijj", T)
        t1 = time.time()
        print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_xloop, TrT, c, t1-t0))
            
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
    save_RG_flow(type="tensor", data=T, Dcut=Dcut)

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
            save_RG_flow(type="tensor", data=T, Dcut=Dcut)
        
        if count_xloop < XLOOPS:
            t0 = time.time()
            count_xloop += 1
            count_totloop += 1
            T = new_pure_tensor(T, Dcut, "X")
            T, c = __tensor_normalization__(T)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            t1 = time.time()
            print("loop {:2d} finish. time: {:.6f} s".format(count_totloop, t1-t0))
            save_RG_flow(type="tensor", data=T, Dcut=Dcut)


    return T, ln_normalized_factor


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

