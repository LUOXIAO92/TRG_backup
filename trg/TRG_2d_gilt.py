import numpy as np
import cupy as cp
import sys
import time
import opt_einsum as oe
from cuquantum import einsum
from mpi4py import MPI
from utility.truncated_svd import svd
from trg.Gilt import gilt_plaq_TRG
from trg.entanglement_filtering import legoptimize_2dTRG, legoptimize_2dTRG_FET

#OUTPUT_DIR = sys.argv[7]
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
        Us, s, sVH = svd(T, shape=[[0,1], [2,3]], truncate_err=1e-12, split=True)
    elif pattern == "B":
        Us, s, sVH = svd(T, shape=[[3,0], [1,2]], truncate_err=1e-12, split=True)
    else:
        print("no such pattern")
        sys.exit(1)

    if len(s) > Dcut:
        Us, sVH, s = Us[:,:,:Dcut], sVH[:Dcut,:,:], s[:Dcut]
    
    return Us, sVH, s
    

def __new_tensor__(VHB, VHA, UB, UA, renormal_loop:int):
    if (renormal_loop % 2) == 1:
        T_new = oe.contract("iba,jcb,dck,adl->ijkl", VHB, VHA, UB, UA)
    elif (renormal_loop % 2) == 0: 
        T_new = oe.contract("iba,cbj,dck,lad->ijkl", VHA, UB, UA, VHB)
    
    return T_new

def TRG_pure_tensor(T, Dcut:int, renormal_loop:int, gilt_eps):
    """
    initial configuration of the tensors is:\\
    ... T  T  T  T ... \\
    ... T  T  T  T ... \\
    ... T  T  T  T ... \\
    ... T  T  T  T ... 
    """

    if gilt_eps > 1e-12:
        if renormal_loop % 2 == 1:
            B, A = gilt_plaq_TRG(T, T, gilt_eps)
            UB, VHB, _  = SVD(B , Dcut=Dcut, pattern="B")
            UA, VHA, _  = SVD(A , Dcut=Dcut, pattern="A")
        elif renormal_loop % 2 == 0:
            UB, VHB, _  = SVD(T , Dcut=Dcut, pattern="B")
            UA, VHA, _  = SVD(T , Dcut=Dcut, pattern="A")
    else:
        if renormal_loop % 2 == 1:
            UB, VHB, _  = SVD(T , Dcut=Dcut, pattern="B")
            UA, VHA, _  = SVD(T , Dcut=Dcut, pattern="A")
        elif renormal_loop % 2 == 0:
            UB, VHB, _  = SVD(T , Dcut=Dcut, pattern="B")
            UA, VHA, _  = SVD(T , Dcut=Dcut, pattern="A")
    
    
    T  = __new_tensor__(VHB, VHA, UB, UA, renormal_loop=renormal_loop)
   
    del UB, VHB, UA, VHA

    return T

def TRG_pure_tensor_FET(T, Dcut:int, renormal_loop:int, gilt_eps):
    """
    initial configuration of the tensors is:\\
    ... T  T  T  T ... \\
    ... T  T  T  T ... \\
    ... T  T  T  T ... \\
    ... T  T  T  T ... 
    """

    if renormal_loop % 2 == 1:
        #B, A = legoptimize_2dTRG(T, T, gilt_eps, cut_scheme="gilt")
        #B, A = legoptimize_2dTRG(T, T, gilt_eps, cut_scheme="FET", maxiter=20)
        B, A = legoptimize_2dTRG_FET(T, T, 20)
        UB, VHB, _  = SVD(B , Dcut=Dcut, pattern="B")
        UA, VHA, _  = SVD(A , Dcut=Dcut, pattern="A")
    elif renormal_loop % 2 == 0:
        UB, VHB, _  = SVD(T , Dcut=Dcut, pattern="B")
        UA, VHA, _  = SVD(T , Dcut=Dcut, pattern="A")
    
    
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

    #if (renormal_loop % 2) == 1:
    T0 = __new_tensor__(VHB, VHA, U0, U1, renormal_loop=renormal_loop)
    T1 = __new_tensor__(VHB, VH1, U2, UA, renormal_loop=renormal_loop)
    T2 = __new_tensor__(VH2, VH3, UB, UA, renormal_loop=renormal_loop)
    T3 = __new_tensor__(VH0, VHA, UB, U3, renormal_loop=renormal_loop)
    T  = __new_tensor__(VHB, VHA, UB, UA, renormal_loop=renormal_loop)
        
    #elif (renormal_loop % 2) == 0:
    #    T0 = __new_tensor__(VH0, VHA, UB, U3, renormal_loop=renormal_loop)
    #    T1 = __new_tensor__(VHB, VHA, U0, U1, renormal_loop=renormal_loop)
    #    T2 = __new_tensor__(VHB, VH1, U2, UA, renormal_loop=renormal_loop)
    #    T3 = __new_tensor__(VH2, VH3, UB, UA, renormal_loop=renormal_loop)
    #    T  = __new_tensor__(VHB, VHA, UB, UA, renormal_loop=renormal_loop)

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

def save_correlation_length(T:cp.ndarray, Dcut:int):

    outdir = "{:}/correlation_length".format(OUTPUT_DIR)
    if not os.path.exists(outdir):
            os.mkdir(outdir)
    
    #calculate ξt
    #Ls=L
    M = oe.contract("xyxY->yY", T)
    _, e, _ = cp.linalg.svd(M)
    if len(e) < Dcut*Dcut:
        e = cp.pad(e, pad_width=(0,Dcut*Dcut-len(e)), mode='constant', constant_values=0.0)
    emax = cp.max(e)
    e /= emax
    for i in range(1,7):
        eL = e**i
        if count_totloop == 0:
            with open(f"{outdir}/ξt_Lt{i}L_Ls1L.dat", "w") as output:
                e1 = eL[1] if eL[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + 0.5 * count_totloop * cp.log(2) + cp.log(-1 / cp.log(e1))
                #output.write(f"#λ2^{i}={e1}\n")
                output.write(f"{lnxi_t:.12e}\n")
        elif count_totloop > 0:
            with open(f"{outdir}/ξt_Lt{i}L_Ls1L.dat", "a") as output:
                e1 = eL[1] if eL[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + 0.5 * count_totloop * cp.log(2) + cp.log(-1 / cp.log(e1))
                output.write(f"{lnxi_t:.12e}\n")

    #Ls=2L
    # {x0,y0,X0,Y0}, {x1,y1,X1,Y1} -> {Y0,Y1,y0,y1}
    #  a  k  b  i  ,  b  l  a  j   ->  i  j  k  l
    M2L = oe.contract("akbi,blaj->ijkl", T, T)
    chi_1, chi_2, chi_3, chi_4 = M2L.shape[0], M2L.shape[1], M2L.shape[2], M2L.shape[3]
    M2L = cp.reshape(M2L, newshape=(chi_1*chi_2, chi_3*chi_4))
    _, e2, _ = cp.linalg.svd(M2L)
    if len(e2) < Dcut*Dcut:
        e2 = cp.pad(e2, pad_width=(0,Dcut*Dcut-len(e2)), mode='constant', constant_values=0.0)
    e2max = cp.max(e2)
    e2 /= e2max
    for i in range(1,7):
        e2L = e2**i
        if count_totloop == 0:
            with open(f"{outdir}/ξt_Lt{i}L_Ls2L.dat", "w") as output:
                e1 = e2L[1] if e2L[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + 0.5 * count_totloop * cp.log(2) + cp.log(-1 / cp.log(e1))
                #output.write(f"#λ2^{i}={e1}\n")
                output.write(f"{lnxi_t:.12e}\n")
        elif count_totloop > 0:
            with open(f"{outdir}/ξt_Lt{i}L_Ls2L.dat", "a") as output:
                e1 = e2L[1] if e2L[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + 0.5 * count_totloop * cp.log(2) + cp.log(-1 / cp.log(e1))
                output.write(f"{lnxi_t:.12e}\n")

    #calculate ξs
    #Lt=L
    M = oe.contract("xyXy->xX", T)
    _, e, _ = cp.linalg.svd(M)
    if len(e) < Dcut*Dcut:
        e = cp.pad(e, pad_width=(0,Dcut*Dcut-len(e)), mode='constant', constant_values=0.0)
    emax = cp.max(e)
    e /= emax
    for i in range(1,7):
        eL = e**i
        if count_totloop == 0:
            with open(f"{outdir}/ξs_Lt1L_Ls{i}L.dat", "w") as output:
                e1 = eL[1] if eL[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + 0.5 * count_totloop * cp.log(2) + cp.log(-1 / cp.log(e1))
                #output.write(f"#λ2^{i}={e1}\n")
                output.write(f"{lnxi_t:.12e}\n")
        elif count_totloop > 0:
            with open(f"{outdir}/ξs_Lt1L_Ls{i}L.dat", "a") as output:
                e1 = eL[1] if eL[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + 0.5 * count_totloop * cp.log(2) + cp.log(-1 / cp.log(e1))
                output.write(f"{lnxi_t:.12e}\n")

    #Lt=2L
    # {x0,y0,X0,Y0}, {x1,y1,X1,Y1} -> {X0,X1,x0,x1}
    #  k  b  i  a  ,  l  a  j  b   ->  i  j  k  l
    M2L = oe.contract("kbia,lajb->ijkl", T, T)
    chi_1, chi_2, chi_3, chi_4 = M2L.shape[0], M2L.shape[1], M2L.shape[2], M2L.shape[3]
    M2L = cp.reshape(M2L, newshape=(chi_1*chi_2, chi_3*chi_4))
    _, e2, _ = cp.linalg.svd(M2L)
    if len(e2) < Dcut*Dcut:
        e2 = cp.pad(e2, pad_width=(0,Dcut*Dcut-len(e2)), mode='constant', constant_values=0.0)
    e2max = cp.max(e2)
    e2 /= e2max
    for i in range(1,7):
        e2L = e2**i
        if count_totloop == 0:
            with open(f"{outdir}/ξs_Lt2L_Ls{i}L.dat", "w") as output:
                e1 = e2L[1] if e2L[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + 0.5 * count_totloop * cp.log(2) + cp.log(-1 / cp.log(e1))
                #output.write(f"#λ2^{i}={e1}\n")
                output.write(f"{lnxi_t:.12e}\n")
        elif count_totloop > 0:
            with open(f"{outdir}/ξs_Lt2L_Ls{i}L.dat", "a") as output:
                e1 = e2L[1] if e2L[1] > 1e-200 else 1e-200
                lnxi_t = cp.log(i) + 0.5 * count_totloop * cp.log(2) + cp.log(-1 / cp.log(e1))
                output.write(f"{lnxi_t:.12e}\n")

def save_RG_flow(T, Dcut):
    if not os.path.exists("{:}/tensor".format(OUTPUT_DIR)):
        os.mkdir("{:}/tensor".format(OUTPUT_DIR))
    with open("{:}/tensor/tensor_l{:}.dat".format(OUTPUT_DIR, count_totloop), "w") as svout:
        M = cp.reshape(T, (T.shape[0]*T.shape[1], T.shape[2]*T.shape[3]))
        _, s, _ = cp.linalg.svd(M)
        if len(s) < Dcut*Dcut:
            s = cp.pad(s, pad_width=(0,Dcut*Dcut-len(s)), mode='constant', constant_values=0.0)
        s0 = s[0]
        svout.write("#max(s)={:.12e}\n".format(s0))
        s /= s0
        for ss in s:
            ss /= cp.max(s)
            svout.write("{:.12e}\n".format(ss))

    if not os.path.exists("{:}/transfer_matrix".format(OUTPUT_DIR)):
        os.mkdir("{:}/transfer_matrix".format(OUTPUT_DIR))
    with open("{:}/transfer_matrix/transfer_matrix_l{:}.dat".format(OUTPUT_DIR, count_totloop), "w") as svout:
        M = oe.contract("xyxY->yY", T)
        print("transfer matrix hermit err:", cp.linalg.norm(M-cp.conj(M.T))/cp.linalg.norm(M))
        _, e, _ = cp.linalg.svd(M)
        if len(e) < Dcut*Dcut:
            e = cp.pad(e, pad_width=(0,Dcut*Dcut-len(e)), mode='constant', constant_values=0.0)
        emax = cp.max(e)
        svout.write("#max(e)={:.12e}\n".format(emax))
        e /= emax
        for ee in e:
           svout.write("{:.12e}\n".format(ee))

    if count_totloop == 0:
        with open(f"{OUTPUT_DIR}/correlation_length_t.dat", "w") as output:
            e1 = e[1] if e[1] > 1e-100 else 1e-100
            lnxi_t = ((count_totloop) / 2) * cp.log(2) + cp.log(-1 / cp.log(e1))
            output.write(f"{lnxi_t:.12e}\n")
    elif count_totloop > 0:
        with open(f"{OUTPUT_DIR}/correlation_length_t.dat", "a") as output:
            e1 = e[1] if e[1] > 1e-100 else 1e-100
            lnxi_t = ((count_totloop) / 2) * cp.log(2) + cp.log(-1 / cp.log(e1))
            output.write(f"{lnxi_t:.12e}\n")

    if count_totloop == 0:
        with open(f"{OUTPUT_DIR}/max_eigvs.dat", "w") as output:
            output.write(f"{emax:.12e}\n")
    elif count_totloop > 0:
        with open(f"{OUTPUT_DIR}/max_eigvs.dat", "a") as output:
            output.write(f"{emax:.12e}\n")

    return s0, emax

def save_entropy(T, Dcut):

    TA = oe.contract("xyXY->xXyY", T.copy())
    TB = TA
    TrTBTA = oe.contract("ijaa,jibb", TB, TA)
    print("TrTBTA=",TrTBTA)
    Dens_Mat = oe.contract("ijab,jicd->acbd", TB, TA) / TrTBTA
    chi_1, chi_2, chi_3, chi_4 = Dens_Mat.shape[0], Dens_Mat.shape[1], Dens_Mat.shape[2], Dens_Mat.shape[3]
    Dens_Mat = cp.reshape(Dens_Mat, newshape=(chi_1*chi_2, chi_3*chi_4))
    _, e_Dens_Mat, _ = cp.linalg.svd(Dens_Mat)
    print("dens_mat hermit err",cp.linalg.norm(Dens_Mat-cp.conj(Dens_Mat.T))/cp.linalg.norm(Dens_Mat))

    if not os.path.exists("{:}/densitymatrix".format(OUTPUT_DIR)):
        os.mkdir("{:}/densitymatrix".format(OUTPUT_DIR))
    with open("{:}/densitymatrix/densitymatrix_l{:}.dat".format(OUTPUT_DIR, count_totloop), "w") as svout:
        emax = cp.max(e_Dens_Mat)
        e_dens_mat = e_Dens_Mat / emax
        if len(e_dens_mat) < Dcut*Dcut:
            e_dens_mat = cp.pad(e_dens_mat, pad_width=(0,Dcut*Dcut-len(e_dens_mat)), mode='constant', constant_values=0.0)
        svout.write("#ρmax={:.12e}\n".format(emax))
        for ee in e_dens_mat:
           svout.write("{:.12e}\n".format(ee))

    STE = - cp.sum( e_Dens_Mat * cp.log(e_Dens_Mat))
    if count_totloop == 0:
        with open("{:}/STE.dat".format(OUTPUT_DIR), "w") as ste_out:
            ste_out.write("{:.15e}\n".format(STE))
    else:
        with open("{:}/STE.dat".format(OUTPUT_DIR), "a") as ste_out:
            ste_out.write("{:.15e}\n".format(STE))


    Dens_Mat = cp.reshape(Dens_Mat, newshape=(chi_1, chi_2, chi_3, chi_4))
    rho_A = oe.contract("aiaj->ij", Dens_Mat)
    _, e_A, _ = cp.linalg.svd(rho_A)
    SEE = -cp.sum(e_A * cp.log(e_A))
    print("rho_A hermit err",cp.linalg.norm(rho_A-cp.conj(rho_A.T))/cp.linalg.norm(rho_A))

    if not os.path.exists("{:}/densitymatrix_A".format(OUTPUT_DIR)):
        os.mkdir("{:}/densitymatrix_A".format(OUTPUT_DIR))
    with open("{:}/densitymatrix_A/densitymatrix_A_l{:}.dat".format(OUTPUT_DIR, count_totloop), "w") as svout:
        emax = cp.max(e_A)
        e_a = e_A / emax
        if len(e_a) < Dcut:
            e_a = cp.pad(e_a, pad_width=(0,Dcut-len(e_a)), mode='constant', constant_values=0.0)
        svout.write("#ρAmax={:.12e}\n".format(emax))
        for ee in e_a:
           svout.write("{:.12e}\n".format(ee))

    if count_totloop == 0:
        with open("{:}/SEE.dat".format(OUTPUT_DIR), "w") as see_out:
            see_out.write("{:.15e}\n".format(SEE))
    else:
        with open("{:}/SEE.dat".format(OUTPUT_DIR), "a") as see_out:
           see_out.write("{:.15e}\n".format(SEE))

def save_tensor_data(T, c):
    path = "{:}/data".format(OUTPUT_DIR)
    if not os.path.isdir(path):
        os.mkdir(path)
    file = "{:}/tensor_l{:}.npz".format(path, count_totloop)
    if not os.path.isfile(file):
        cp.savez(file, tensor=T, factor=c)

def load_tensor_data(T:cp.ndarray):
    path = "{:}/data".format(OUTPUT_DIR)
    file = "{:}/tensor_l{:}.npz".format(path, count_totloop)
    if os.path.isfile(file):
        data = cp.load(file, mmap_mode='r')
        T = data['tensor']
        #c = data['factor']
    return T

def save_free_energy(T, ln_normfact, rgstep):
    trace = oe.contract("abab", T)
    V = 2**(rgstep)
    ln_ZoverV = cp.sum(ln_normfact) + cp.log(trace) / V

    if count_totloop == 0:
        with open("{:}/free_energy.dat".format(OUTPUT_DIR), "w") as fe:
            fe.write("{:.15e}\n".format(ln_ZoverV))
    else:
        with open("{:}/free_energy.dat".format(OUTPUT_DIR), "a") as fe:
            fe.write("{:.15e}\n".format(ln_ZoverV))

def get_new_pure_tensor(T:cp.ndarray, Dcut:int, renormal_loop:int, gilt_eps:float):
    path = "{:}/data".format(OUTPUT_DIR)
    file = "{:}/tensor_l{:}.npz".format(path, renormal_loop)
    if os.path.isfile(file):
        data = cp.load(file, mmap_mode='r')
        T = data['tensor']
        c = data['factor']
    else:
        T = TRG_pure_tensor(T, Dcut, renormal_loop, gilt_eps)
        #T = TRG_pure_tensor_FET(T, Dcut, renormal_loop, gilt_eps)
        T, c = __tensor_normalization__(T)
    return T, c

def tensor_Trace(T):
    TrT = oe.contract("xyxy", T)
    return TrT

def __tensor_normalization__(T):
    #c = cp.max(cp.absolute(T))
    c = cp.abs(tensor_Trace(T))
    T = T / c
    return T, c

from measurement.measurement import save_ground_state_energy
def pure_tensor_renorm(T, Dcut:int, gilt_eps, XLOOPS:int, YLOOPS:int):
    N = XLOOPS + YLOOPS
    global count_totloop
    count_totloop = 0

    ln_normalized_factor = cp.zeros(N+1, dtype=cp.float64)
    
    t0 = time.time()
    T , c  = __tensor_normalization__(T)
    ln_normalized_factor[count_totloop] = cp.log(c) / 2.0**(count_totloop)
    save_tensor_data(T, c)
    _, max_eigv = save_RG_flow(T, Dcut)
    #save_entropy(T, Dcut)
    save_ground_state_energy(ln_normalized_factor, max_eigv, count_totloop, OUTPUT_DIR)
    save_free_energy(T, ln_normalized_factor, count_totloop)
    save_correlation_length(T, Dcut)
    t1 = time.time()

    TrT = tensor_Trace(T)
    print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_totloop, TrT, c, t1-t0))
    
    while count_totloop < N:
        
        t0 = time.time()
        T = load_tensor_data(T)
        
        count_totloop += 1
        #T = TRG_pure_tensor(T, Dcut, count_totloop, gilt_eps)
        T, c = get_new_pure_tensor(T, Dcut, count_totloop, gilt_eps)
        #T , c  = __tensor_normalization__(T)
        ln_normalized_factor[count_totloop] = cp.log(c) / 2.0**(count_totloop)

        save_tensor_data(T, c)
        _, max_eigv = save_RG_flow(T, Dcut)
        #save_entropy(T, Dcut)
        save_ground_state_energy(ln_normalized_factor, max_eigv, count_totloop, OUTPUT_DIR)
        save_free_energy(T, ln_normalized_factor, count_totloop)
        save_correlation_length(T, Dcut)
        t1 = time.time()

        TrT = tensor_Trace(T)
        print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s\n".format(count_totloop, TrT, c, t1-t0))

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
    save_RG_flow(T, Dcut)
    TrT = tensor_Trace(T)
    print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s".format(count_totloop, TrT, c, t1-t0))
    
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
            save_RG_flow(T, Dcut)
            TrT = tensor_Trace(T)
            print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s".format(count_totloop, TrT, c, t1-t0))

        elif count_totloop == N - 1:
            t0 = time.time()

            T, T0, T1 = TRG_V_eq_4(T, T0, T1, T2, T3, Dcut, count_totloop)
            T , c  = __tensor_normalization__(T)
            T0, c0 = __tensor_normalization__(T0)
            T1, c1 = __tensor_normalization__(T1)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            ln_normalized_factor_imp[count_totloop] = cp.log(c0)+cp.log(c1) - 2*cp.log(c)

            t1 = time.time()
            save_RG_flow(T, Dcut)
            TrT = tensor_Trace(T)
            print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s".format(count_totloop, TrT, c, t1-t0))

        elif count_totloop == N:
            t0 = time.time()

            T, T0 = TRG_V_eq_2(T, T0, T1, Dcut, count_totloop)
            T , c  = __tensor_normalization__(T)
            T0, c0 = __tensor_normalization__(T0)
            ln_normalized_factor[count_totloop] = cp.log(c) / 2**(count_totloop)
            ln_normalized_factor_imp[count_totloop] = cp.log(c0) - cp.log(c)

            t1 = time.time()
            save_RG_flow(T, Dcut)
            TrT = tensor_Trace(T)
            print("loop {:2d} finish, TrT= {:.15e} , norm_fact c= {:.15e}. time: {:.6f} s".format(count_totloop, TrT, c, t1-t0))

        #print(count_totloop, ln_normalized_factor[count_totloop])

    return T, T0, ln_normalized_factor, ln_normalized_factor_imp

