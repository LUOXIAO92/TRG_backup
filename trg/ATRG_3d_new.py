import os
import numpy as np
import cupy as cp
import time
import opt_einsum as oe
import math

import itertools as iter
from utility.randomized_svd import rsvd_for_3dATRG_tensor_inte, rsvd
from utility.truncated_svd import svd, eigh, truncate
from tensor_class.tensor_class import ATRG_Tensor as Tensor

OUTPUT_DIR = os.environ['OUTPUT_DIR']

degeneracy_eps = float(os.environ['DGENERACY_EPS'])
truncate_eps = 1e-10

def leg_transposition(A:Tensor, do_what="transpose", direction="t"):
    """
    A: tensor A_{TXYtxy} = U_{TXYi} s_i VH_{itxy} 
    >>>        T  Y
    >>>        | /
    >>>        U -- X
    >>>        s
    >>>   x -- VH
    >>>      / |
    >>>     y  t
    
    do_what: "transpose" or "restore"
    direction: "t" or "T", temporal direction; 
               "x" or "X", x direction; 
               "y" or "Y", y direction. 
    -------------------------------------------------------------
    >>> "transpose":
    >>>     "t": U_{TXYi} s_i VH_{itxy} -> U_{TXYi} s_i VH_{itxy}
    >>>     "x": U_{TXYi} s_i VH_{itxy} -> U_{XYTi} s_i VH_{ixyt}
    >>>     "y": U_{TXYi} s_i VH_{itxy} -> U_{YTXi} s_i VH_{iytx}
    >>> "restore":
    >>>     "t": U_{TXYi} s_i VH_{itxy} -> U_{TXYi} s_i VH_{itxy}
    >>>     "x": U_{XYTi} s_i VH_{ixyt} -> U_{TXYi} s_i VH_{itxy}
    >>>     "y": U_{YTXi} s_i VH_{iytx} -> U_{TXYi} s_i VH_{itxy}
    -------------------------------------------------------------
    """

    if do_what == "transpose":
        if   direction == "x" or direction == "X":
            A.U  = cp.transpose(A.U , axes=(1,2,0,3))
            A.VH = cp.transpose(A.VH, axes=(0,2,3,1))
        elif direction == "y" or direction == "Y":
            A.U  = cp.transpose(A.U , axes=(2,0,1,3))
            A.VH = cp.transpose(A.VH, axes=(0,3,1,2))
        elif direction == "t" or direction == "T":
            return A

    elif do_what == "restore":
        if   direction == "x" or direction == "X":
            A.U  = cp.transpose(A.U , axes=(2,0,1,3))
            A.VH = cp.transpose(A.VH, axes=(0,3,1,2))
        elif direction == "y" or direction == "Y":
            A.U  = cp.transpose(A.U , axes=(1,2,0,3))
            A.VH = cp.transpose(A.VH, axes=(0,2,3,1))
        elif direction == "t" or direction == "T":
            return A

    return A

def rsvd_for_ATRG_leg_swapping(B:cp.ndarray, C:cp.ndarray, chi_k:int, p=32, q=64, eps=1e-10, seed=12345):
    """
    do svd of B_{βit} C_{tjγ} = M_{βijγ} = M_{iγjβ} = UM_{iγk} sM_{k} VMH_{kjβ}
    B: tensor B_{βit}
    C: tensor C_{tjγ}
    >>>    i     γ         γ \   / j       γ \             / j
    >>>    |     |     ->      M      ->      UM-k-sM-k-VMH       
    >>> β--B--t--C--j      i /   \ β       i /             \ β

    chi_k: bond dimension of leg k
    p: number of oversample
    q: times of iteration

    returns: UM_{iγk}, sM_{k}, VMH_{kjβ}
    """

    chiB_b, chiB_i, chiB_t = B.shape
    chiC_t, chiC_j, chiC_g = C.shape

    #M = oe.contract("βit,tjγ->iγjβ", B, C)

    #generate standard normal tensor Y_{jβl}
    l = chi_k + p
    rs = cp.random.RandomState(seed)
    Y = rs.standard_normal(size=(chiC_j, chiB_b, l))
    Y = Y.astype(complex)

    #contract Y with B and C
    # γ\ /j\          γ\
    #   M   Y--l -->    Y'--l
    # i/ \β/          i/
    # M_{iγjβ}Y_{jβl} => Y'_{iγl}
    Y = oe.contract("βit,tjγ,jβl->iγl", B, C, Y)

    #start QR iteration
    #compute Q_{iγl} with qr(Y_{iγl}), Y_{iγl} = M_{iγjβ} Y'_{jβl}
    Y = cp.reshape(Y, newshape=(chiB_i*chiC_g, l))
    Q, _ = cp.linalg.qr(Y)
    l_size = min(chiB_i*chiC_g, l)
    Q = cp.reshape(Q, newshape=(chiB_i, chiC_g, l_size))
    for _ in range(q):
        #compute Q_{jβl} with qr(Y'_{jβl}) , Y'_{jβl} = M^†_{jβiγ} Y_{iγl}
        Y = oe.contract("βit,tjγ,iγl->jβl", cp.conj(B), cp.conj(C), Q)
        Y = cp.reshape(Y, newshape=(chiC_j*chiB_b, l_size))
        Q, _ = cp.linalg.qr(Y)
        Q = cp.reshape(Q, newshape=(chiC_j, chiB_b, l_size))

        #compute Q_{iγl} with qr(Y_{iγl}), Y_{iγl} = M_{iγjβ} Y'_{jβl}
        Y = oe.contract("βit,tjγ,jβl->iγl", B, C, Q)
        Y = cp.reshape(Y, newshape=(chiB_i*chiC_g, l_size))
        Q, _ = cp.linalg.qr(Y)
        Q = cp.reshape(Q, newshape=(chiB_i, chiC_g, l_size))
    #QR iteration finish

    #compute X = Q^† @ M, X_{ljβ} = Q^*_{iγl} B_{βit} C_{tjγ}
    X = oe.contract("iγl,βit,tjγ->ljβ", cp.conj(Q), B, C)
    X = cp.reshape(X, (l_size, chiC_j*chiB_b))
    UM, sM, VMH = cp.linalg.svd(X, full_matrices=False)
    del X

    VMH = cp.reshape(VMH, newshape=(l_size, chiC_j, chiB_b))
    UM = oe.contract("iγl,lL->iγL", Q, UM)

    s0 = sM
    smax = cp.max(s0)
    s0 = s0 / smax
    k = s0[s0>eps].shape[0]
    k = min(k, chi_k)

    UM = UM[:,:,:k]
    sM = sM[:k]
    VMH = VMH[:k,:,:]

    return UM, sM, VMH

def leg_swapping(B:cp.ndarray, C:cp.ndarray, Dcut:int, ispure:bool):
    """
    Swap the legs as flowing. 
    >>>     i                   i  Y0   
    >>>     |                   | /     
    >>> x1--B                   UM--X0
    >>>   / |                  k|     
    >>> y1  t  Y0    --->       sM    
    >>>     | /                 |     
    >>>     C--X0          x1--VM    
    >>>     |                 / |     
    >>>     j               y1  j     
    return: UM, sM, VMH or UM, sM, VMH, sM_imp when T0 and T1 is not None
    >>> UM: UM_{i,X0,Y0,k}
    >>> sM: sM_{k}
    >>> VMH: VMH_{k,j,x1,y1}
    """
    #chiA_T1, chiA_X1, chiA_Y1, chiA_i  = A.shape
    chiB_i , chiB_t1, chiB_x1, chiB_y1 = B.shape
    chiC_T0, chiC_X0, chiC_Y0, chiC_j  = C.shape
    #chiD_j , chiD_t0, chiD_x0, chiD_y0 = D.shape

    #            i                               i        Y0               i  Y0  
    #            |                               |       /                 | /    
    # x1--UB--β--B                               UM--γ--VCH--X0            UM--X0
    #    /       |                               |                         |     
    #  y1        t       Y0     --->             sM              --->      sM    
    #            |      /                        |                         |     
    #            C--γ--VCH--X0        x1--UB--β--VM                    x1--VM    
    #            |                       /       |                       / |     
    #            j                     y1        j                     y1  j     

    #B_{itxy} = B_{xyit} = UB_{xyβ} sB_{β} VBH_{βit}, B' = sB_{β} VBH_{βit}
    B = cp.transpose(B, axes=(2,3,0,1)).reshape((chiB_x1*chiB_y1, chiB_i*chiB_t1))
    UB, sB, VBH = cp.linalg.svd(B, full_matrices=False)
    beta_size = min(chiB_x1*chiB_y1, chiB_i*chiB_t1)
    UB  = cp.reshape(UB,  newshape=(chiB_x1, chiB_y1, beta_size))
    VBH = cp.reshape(VBH, newshape=(beta_size, chiB_i, chiB_t1))
    B_tilde = oe.contract("β,βit->βit", sB, VBH)
    
    #C_{TXYj} = C_{TjXY} = UC_{Tjγ} sC_{γ} VCH_{γXY}, C' = UC_{Tjγ} sC_{γ}
    C = cp.transpose(C, axes=(0,3,1,2)).reshape((chiC_T0*chiC_j, chiC_X0*chiC_Y0))
    UC, sC, VCH = cp.linalg.svd(C, full_matrices=False)
    gamma_size = min(chiC_T0*chiC_j, chiC_X0*chiC_Y0)
    UC  = cp.reshape(UC,  newshape=(chiC_T0, chiC_j, gamma_size))
    VCH = cp.reshape(VCH, newshape=(gamma_size, chiC_X0, chiC_Y0))
    C_tilde  = oe.contract("tjγ,γ->tjγ", UC, sC)

    #UM_{iγk}, sM_{k}, VMH_{kjβ}
    chi_k = min(Dcut, chiC_j*beta_size, chiB_i*gamma_size)
    t0 = time.time()
    UM, sM, VMH = rsvd_for_ATRG_leg_swapping(B=B_tilde, C=C_tilde, chi_k=3*chi_k, p=0, q=Dcut, eps=truncate_eps)
    t1 = time.time()
    print(f"Rsvd in leg swapping finished. Time= {t1-t0:.2e} s")

    #Out put sM----------------------------------------------------------------------------------
    if ispure:
        dname = OUTPUT_DIR+"/leg_exchange_pure/"
    else:
        dname = OUTPUT_DIR+"/leg_exchange_impure/"
    fname = "leg_exchange_sv_lx{:}_ly{:}_lt{:}.dat".format(count_xloop, count_yloop, count_zloop)
    if not os.path.exists(dname):
        os.mkdir(dname)
    with open(dname+fname, "w") as out:
        sMmax = 1 #cp.max(sM)
        sMout = sM / sMmax
        out.write("#s1={:.12e}\n".format(sMmax.real))
        k = max(Dcut, chi_k)
        sMout = cp.pad(sMout, pad_width=(0,3*k-len(sMout)), mode='constant', constant_values=0.0)
        for s in sMout:
            out.write("{:.12e}\n".format(s.real))
    #--------------------------------------------------------------------------------------------

    #chi = min(Dcut, chi_k, chiB_i, len(sM))
    #print("sM:", sM)
    #print("chi_k", chi_k)

    #k = min(Dcut, chi_k, chiB_i, len(sM))
    #k = min(Dcut, chiB_i)
    k = chi_k
    chi, leg_swapping_err = truncate(sM, k=k, degeneracy_eps=degeneracy_eps, truncate_eps=truncate_eps)
    print(f"len(sM)={len(sM)}, k={k}, chi={chi}")
    #print(f"k={k}, chi={chi}, err={leg_swapping_err}")
    #print(f"sM_{{Dcut}}/s1={sM[chi-1]/sM[0]}")
    #print(f"sM_{{4*Dcut}}/s1={sM[-1]/sM[0]}")
    print(f"leg swapping error= {leg_swapping_err:.6e}")

    UM  = UM[:,:,:chi]
    sM  = sM[:chi]
    VMH = VMH[:chi,:,:]
    UM  = oe.contract("iγk,γXY->iXYk", UM, VCH)
    VMH = oe.contract("kjβ,xyβ->kjxy", VMH, UB)

    return UM, sM, VMH


#def truncated_svd(A, eps=1e-10):
#        u, s, vh = cp.linalg.svd(A)
#        k = s[s>eps].shape[0]
#        u = u[:,:k]
#        s = s[:k]
#        vh = vh[:k,:]
#        return u, s, vh

def squeezer(A, UM, sM, VMH, D, Dcut):
    """
    return: Px, PX, Py, PY
    >>> Px: Px_{x,x1,x0}
    >>> PX: PX_{X1,X0,X}
    >>> Py: Py_{y,y1,y0}
    >>> PY: PY_{Y1,Y0,Y}
    """
    #UM_{i,X0,Y0,k}, VMH_{k,j,x1,y1}
    chi_T1, chi_X1, chi_Y1, chi_i  = A.shape
    chi_i , chi_X0, chi_Y0, chi_k  = UM.shape
    chi_k , chi_j , chi_x1, chi_y1 = VMH.shape
    chi_j , chi_t0, chi_x0, chi_y0 = D.shape    
    #chi_k = len(sM)

    path = ['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)]
    XdagX = oe.contract("aibc,akbf,cjde,e,flde->ijkl", cp.conj(A), A, cp.conj(UM), sM, UM  , optimize=path[1:])
    YdagY = oe.contract("abic,abkf,cdje,e,fdle->ijkl", cp.conj(A), A, cp.conj(UM), sM, UM  , optimize=path[1:])
    xxdag = oe.contract("bejf,delf,abic,a,adkc->ijkl", D, cp.conj(D), VMH, sM, cp.conj(VMH), optimize=path[1:])
    yydag = oe.contract("befj,defl,abci,a,adck->ijkl", D, cp.conj(D), VMH, sM, cp.conj(VMH), optimize=path[1:])
    XdagX = cp.reshape(XdagX, newshape=(chi_X1*chi_X0, chi_X1*chi_X0))
    YdagY = cp.reshape(YdagY, newshape=(chi_Y1*chi_Y0, chi_Y1*chi_Y0))
    xxdag = cp.reshape(xxdag, newshape=(chi_x1*chi_x0, chi_x1*chi_x0))
    yydag = cp.reshape(yydag, newshape=(chi_y1*chi_y0, chi_y1*chi_y0))

    _, sX, vXh = cp.linalg.svd(XdagX, full_matrices=False)
    _, sY, vYh = cp.linalg.svd(YdagY, full_matrices=False)
    ux, sx, _  = cp.linalg.svd(xxdag, full_matrices=False)
    uy, sy, _  = cp.linalg.svd(yydag, full_matrices=False)
    del XdagX, YdagY, xxdag, yydag

    #Squeezer of x direction---------------------------------------
    RX = oe.contract("a,ab->ab", cp.sqrt(sX), vXh)
    Rx = oe.contract("ab,b->ab", ux, cp.sqrt(sx))
    k1 = min(RX.shape[0], Rx.shape[1], Dcut)
    u, s, vh = svd(RX@Rx, shape=[[0], [1]], k=k1, degeneracy_eps=degeneracy_eps, truncate_eps=truncate_eps)
    uh = cp.conj(u.T)
    v  = cp.conj(vh.T)
    s_invsqrt = cp.sqrt(1/s)
    PX = oe.contract("ab,bx,x->ax", Rx, v, s_invsqrt)
    Px = oe.contract("x,xc,cb->xb", s_invsqrt, uh, RX)

    chix = min(len(s), Dcut)
    PX = PX[:,:chix]
    Px = Px[:chix,:]
    PX = cp.reshape(PX, newshape=(chi_X1, chi_X0, chix))
    Px = cp.reshape(Px, newshape=(chix, chi_x1, chi_x0))
    del RX, Rx
    #---------------------------------------------------------------

    #Squeezer of y direction---------------------------------------
    RY = oe.contract("a,ab->ab", cp.sqrt(sY), vYh)
    Ry = oe.contract("ab,b->ab", uy, cp.sqrt(sy))
    k2 = min(RY.shape[0], Ry.shape[1], Dcut)
    u, s, vh = svd(RY@Ry, shape=[[0], [1]], k=k2, degeneracy_eps=degeneracy_eps, truncate_eps=truncate_eps)
    uh = cp.conj(u.T)
    v  = cp.conj(vh.T)
    s_invsqrt = cp.sqrt(1/s)
    PY = oe.contract("ab,by,y->ay", Ry, v, s_invsqrt)
    Py = oe.contract("y,yc,cb->yb", s_invsqrt, uh, RY)

    chiy = min(len(s), Dcut)
    PY = PY[:,:chiy]
    Py = Py[:chiy,:]
    PY = cp.reshape(PY, newshape=(chi_Y1, chi_Y0, chiy))
    Py = cp.reshape(Py, newshape=(chiy, chi_y1, chi_y0))
    del RY, Ry
    #---------------------------------------------------------------

    del sX, vXh, sY, vYh, ux, sx, uy, sy

    return Px, PX, Py, PY

def leg_slicing(leg_size, slicing):
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
        slice_list = [slice(i, i+1) for i in range(leg_size)]
        return slice_list, len(slice_list)
    
    if slicing <= 1:
        slice_list = [slice(0, leg_size)]
        return slice_list, len(slice_list)

    bs1 = leg_size // slicing + 1
    bs2 = leg_size // slicing

    #n1*bs1 + n2*bs2 = leg_size
    #n1 + n2 = slicing
    n1 = (leg_size - bs2*slicing) // (bs1 - bs2)
    n2 = slicing - n1
    
    slice_list = [slice(i, i+bs1) for i in range(0, n1*bs1, bs1)]
    slice_list += [slice(i, i+bs2) for i in range(n1*bs1, leg_size, bs2)]

    return slice_list, len(slice_list)

def coarse_graining(A, UM, sM, VMH, D, Px, PX, Py, PY, slicing=100):
    chi_T1, chi_X1, chi_Y1, chi_i  = A.shape
    chi_i , chi_X0, chi_Y0, chi_k  = UM.shape
    chi_k , chi_j , chi_x1, chi_y1 = VMH.shape
    chi_j , chi_t0, chi_x0, chi_y0 = D.shape
    chi_x, chi_y = Px.shape[0], Py.shape[0]
    chi_X, chi_Y = PX.shape[2], PY.shape[2]
    chi_T, chi_t = chi_T1, chi_t0

    #Contract G_{T1,X,Y,k} = A_{T1,X1,Y1,i} PX_{X1,X0,X} UM_{i,X0,Y0,k} PY_{Y1,Y0,Y}----
    slicing_i = slicing
    slic_list_i , nslic_i  = leg_slicing(leg_size=chi_i , slicing=slicing_i)

    slicing_X0 = math.ceil(slicing/nslic_i)
    slic_list_X0, nslic_X0 = leg_slicing(leg_size=chi_X0, slicing=slicing_X0)

    slicing_Y1 = math.ceil(slicing/nslic_i/nslic_X0)
    slic_list_Y1, nslic_Y1 = leg_slicing(leg_size=chi_Y1, slicing=slicing_Y1)

    iteration = iter.product(slic_list_Y1, slic_list_X0, slic_list_i)
    path = ['einsum_path', (0, 1), (0, 1), (0, 1)]
    G = 0
    for Y1, X0, i in iteration:
        a  = A[:,:,Y1,i]
        pX = PX[:,X0,:]
        uM = UM[i,X0,:,:]
        pY = PY[Y1,:,:]
        G += oe.contract("Tabi,acX,icdk,bdY->TXYk", a, pX, uM, pY, optimize=path[1:])
    #------------------------------------------------------------------------------------

    #Contract H_{k,x,y,t0} = VMH_{k,j,x1,y1} Px_{x,x1,x0} D_{j,t0,x0,y0} Py_{y,y1,y0}----
    slicing_j = slicing
    slic_list_j , nslic_j  = leg_slicing(leg_size=chi_j , slicing=slicing_j)

    slicing_x0 = math.ceil(slicing/nslic_j)
    slic_list_x0, nslic_x0 = leg_slicing(leg_size=chi_x0, slicing=slicing_x0)

    slicing_y1 = math.ceil(slicing/nslic_i/nslic_x0)
    slic_list_y1, nslic_y1 = leg_slicing(leg_size=chi_y1, slicing=slicing_y1)

    iteration = iter.product(slic_list_y1, slic_list_x0, slic_list_j)
    path = ['einsum_path', (0, 1), (0, 1), (0, 1)]
    H = 0
    for y1, x0, j in iteration:
        vMh = VMH[:,j,:,y1]
        px  = Px[:,:,x0]
        d   = D[j,:,x0,:]
        py  = Py[:,y1,:]
        H += oe.contract("kjab,xac,jtcd,ybd->ktxy", vMh, px, d, py, optimize=path[1:])
    #------------------------------------------------------------------------------------

    #G_{T1,X,Y,k}, H_{k,t0,x,y}
    #G = cp.reshape(G, newshape=(chi_T*chi_X*chi_Y, chi_k))
    #H = cp.reshape(H, newshape=(chi_k, chi_t*chi_x*chi_y))
    #UG, sG, VGH = cp.linalg.svd(G, full_matrices=False)
    #UH, sH, VHH = cp.linalg.svd(H, full_matrices=False)

    kG = min(chi_T*chi_X*chi_Y, chi_k)
    kH = min(chi_k, chi_t*chi_x*chi_y)
    UG, sG, VGH = svd(G, shape=[[0,1,2], [3]], k=kG, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps)
    UH, sH, VHH = svd(H, shape=[[0], [1,2,3]], k=kH, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps)
    del G, H

    chi_kG, chi_kH = len(sG), len(sH)
    UG  = cp.reshape(UG , newshape=(chi_T, chi_X, chi_Y, chi_kG))
    VHH = cp.reshape(VHH, newshape=(chi_kH, chi_t, chi_x, chi_y))

    K = oe.contract("a,ak,k,kb,b->ab", sG, VGH, sM, UH, sH)
    #UK, sK, VKH = cp.linalg.svd(K)
    kK = min(*K.shape)
    UK, sK, VKH = svd(K, shape=[[0], [1]], k=kK, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps)

    #K = oe.contract("a,aj,j,jb,b->ab", sG, VGH, sM, UH, sH)
    #UK, sK, VKH = cp.linalg.svd(K)

    U  = oe.contract("TXYa,ai->TXYi", UG, UK)
    VH = oe.contract("ib,btxy->itxy", VKH, VHH)
    del UK, VKH

    #T = Tensor(U, sK, VH)

    return U, sK, VH


def atrg_pure_tensor(T:Tensor, Dcut:int, direction:str):

    #TdagT = oe.contract("i,izxy,iZxy,zXYj,ZXYj,j", 
    #                    T.s**2, T.VH, cp.conj(T.VH), T.U, cp.conj(T.U), T.s**2)

    T = leg_transposition(T, "transpose", direction)

    #step (a)~(c)
    t0 = time.time()
    A, D = T.U, T.VH
    B = oe.contract("i,itxy->itxy", T.s, T.VH)
    C = oe.contract("TXYj,j->TXYj", T.U, T.s)
    UM, sM, VMH = leg_swapping(B, C, Dcut, ispure=True)
    t1 = time.time()
    print(f"Leg swapping finished. Time= {t1-t0:.2e} s")

    print("Singular values of sM is:")
    with cp.printoptions(formatter={'float_kind':'{:.6e}'.format}):
        print(sM)
        print(f"sM/s1 = {sM[-1]/sM[0]:.6e}")

    #step (d)
    t0 = time.time()
    Px, PX, Py, PY = squeezer(A, UM, sM, VMH, D, Dcut)
    t1 = time.time()
    print(f"Squeezer finished. Time= {t1-t0:.2e} s")

    #step (e)
    t0 = time.time()
    U, s, VH = coarse_graining(A, UM, sM, VMH, D, Px, PX, Py, PY, slicing=Dcut)
    T = T.update(U, s, VH)
    t1 = time.time()
    print(f"Coarse graining finished. Time= {t1-t0:.2e} s")

    T = leg_transposition(T, "restore", direction)

    #err
    #tdagt = cp.sum(T.s**2)
    #err = cp.sqrt((TdagT - tdagt)/TdagT)
    #print(f"total coarse graining error: {err:.6e}")

    return T


def new_impuer_tensor_2points(T:Tensor, T0:Tensor, Tn:Tensor, Dcut:int, direction:str):

    T  = leg_transposition(T , "transpose", direction)
    T0 = leg_transposition(T0, "transpose", direction)
    Tn = leg_transposition(Tn, "transpose", direction)

    #step (a)~(c)
    t0 = time.time()

    A, D = T.U, T.VH
    B = oe.contract("i,itxy->itxy", T.s, T.VH)
    C = oe.contract("TXYj,j->TXYj", T.U, T.s)
    UM, sM, VMH = leg_swapping(B, C, Dcut, ispure=True)

    print("Singular values of sM is:")
    with cp.printoptions(formatter={'float_kind':'{:.6e}'.format}):
        print(sM)
        print(f"sM/s1 = {sM[-1]/sM[0]:.6e}")

    tensors = [T0, Tn]
    Aimp = []
    Dimp = []
    UMimp = []
    sMimp = []
    VMHimp = []
    for tensor in tensors:
        if tensor.loc[direction] % 2 == 1:
            Aimp.append(tensor.U)
            Bimp = oe.contract("i,itxy->itxy", tensor.s, tensor.VH)
            Cimp = oe.contract("TXYj,j->TXYj", T.U, T.s)
            Dimp.append(T.VH)
            umimp, smimp, vmhimp = leg_swapping(Bimp, Cimp, Dcut, ispure=False)
        elif tensor.loc[direction] % 2 == 0:
            Aimp.append(T.U)
            Bimp = oe.contract("i,itxy->itxy", T.U, T.s)
            Cimp = oe.contract("TXYj,j->TXYj", tensor.s, tensor.VH)
            Dimp.append(tensor.VH)
            umimp, smimp, vmhimp = leg_swapping(Bimp, Cimp, Dcut, ispure=False)
        UMimp.append(umimp)
        sMimp.append(smimp)
        VMHimp.append(vmhimp)

    t1 = time.time()
    print(f"Leg swapping finished. Time= {t1-t0:.2e} s")

    #step (d)
    t0 = time.time()
    Px, PX, Py, PY = squeezer(A, UM, sM, VMH, D, Dcut)
    t1 = time.time()
    print(f"Squeezer finished. Time= {t1-t0:.2e} s")

    #step (e)
    t0 = time.time()

    for i in range(len(tensors)):
        U, s, VH = coarse_graining(Aimp[i], UMimp[i], sMimp[i], VMHimp[i], Dimp[i], 
                                     Px, PX, Py, PY, slicing=Dcut)
        tensors[i] = tensors[i].update(U, s, VH)
    T0, Tn = tensors

    U, s, VH = coarse_graining(T, A, UM, sM, VMH, D, Px, PX, Py, PY, slicing=Dcut)
    T = T.update(U, s, VH)

    t1 = time.time()
    print(f"Coarse graining finished. Time= {t1-t0:.2e} s")

    T  = leg_transposition(T , "restore", direction)
    T0 = leg_transposition(T0, "restore", direction)
    Tn = leg_transposition(Tn, "restore", direction)
    
    return T, T0, Tn

def new_impuer_tensor_absorb(T:Tensor, T0:Tensor, Tn:Tensor, Dcut:int, direction:str):
    T  = leg_transposition(T , "transpose", direction)
    T0 = leg_transposition(T0, "transpose", direction)
    Tn = leg_transposition(Tn, "transpose", direction)

    #step (a)~(c)
    t0 = time.time()

    A, D = T.U, T.VH
    B = oe.contract("i,itxy->itxy", T.s, T.VH)
    C = oe.contract("TXYj,j->TXYj", T.U, T.s)
    UM, sM, VMH = leg_swapping(B, C, Dcut, ispure=True)

    print("Singular values of sM is:")
    with cp.printoptions(formatter={'float_kind':'{:.6e}'.format}):
        print(sM)
        print(f"sM/s1 = {sM[-1]/sM[0]:.6e}")
    
    if T0.loc[direction] % 2 == 0 and Tn.loc[direction] % 2 == 1:
        Aimp = Tn.U
        Bimp = oe.contract("i,itxy->itxy", Tn.s, Tn.VH)
        Cimp = oe.contract("TXYj,j->TXYj", T0.U, T0.s)
        Dimp = T0.VH
    elif T0.loc[direction] % 2 == 1 and Tn.loc[direction] % 2 == 0:
        Aimp = T0.U
        Bimp = oe.contract("i,itxy->itxy", T0.s, T0.VH)
        Cimp = oe.contract("TXYj,j->TXYj", Tn.U, Tn.s)
        Dimp = Tn.VH
    UMimp, sMimp, VMHimp = leg_swapping(Bimp, Cimp, Dcut, ispure=False)

    t1 = time.time()
    print(f"Leg swapping finished. Time= {t1-t0:.2e} s")

    
    #step (d)
    t0 = time.time()
    Px, PX, Py, PY = squeezer(A, UM, sM, VMH, D, Dcut)
    t1 = time.time()
    print(f"Squeezer finished. Time= {t1-t0:.2e} s")

    U, s, VH = coarse_graining(Aimp, UMimp, sMimp, VMHimp, Dimp, 
                         Px, PX, Py, PY, slicing=Dcut)
    T0 = T0.update(U, s, VH)
    
    U, s, VH  = coarse_graining(A, UM, sM, VMH, D, Px, PX, Py, PY, slicing=Dcut)
    T = T.update(U, s, VH)

    t1 = time.time()
    print(f"Coarse graining finished. Time= {t1-t0:.2e} s")

    T  = leg_transposition(T , "restore", direction)
    T0 = leg_transposition(T0, "restore", direction)

    del Tn
    
    return T, T0

def new_impuer_tensor_1point(T:Tensor, T0:Tensor, Dcut:int, direction:str):
    T  = leg_transposition(T , "transpose", direction)
    T0 = leg_transposition(T0, "transpose", direction)

    #step (a)~(c)
    t0 = time.time()

    A, D = T.U, T.VH
    B = oe.contract("i,itxy->itxy", T.s, T.VH)
    C = oe.contract("TXYj,j->TXYj", T.U, T.s)
    UM, sM, VMH = leg_swapping(B, C, Dcut, ispure=True)

    print("Singular values of sM is:")
    with cp.printoptions(formatter={'float_kind':'{:.6e}'.format}):
        print(sM)
        print(f"sM/s1 = {sM[-1]/sM[0]:.6e}")

    if T0.loc[direction] % 2 == 0:
        Aimp = T.U
        Bimp = oe.contract("i,itxy->itxy", T.s, T.VH)
        Cimp = oe.contract("TXYj,j->TXYj", T0.U, T0.s)
        Dimp = T0.VH
    elif T0.loc[direction] % 2 == 1:
        Aimp = T0.U
        Bimp = oe.contract("i,itxy->itxy", T0.s, T0.VH)
        Cimp = oe.contract("TXYj,j->TXYj", T.U, T.s)
        Dimp = T.VH
    UMimp, sMimp, VMHimp = leg_swapping(Bimp, Cimp, Dcut, ispure=False)

    t1 = time.time()
    print(f"Leg swapping finished. Time= {t1-t0:.2e} s")

    
    #step (d)
    t0 = time.time()
    Px, PX, Py, PY = squeezer(A, UM, sM, VMH, D, Dcut)
    t1 = time.time()
    print(f"Squeezer finished. Time= {t1-t0:.2e} s")

    U, s, VH = coarse_graining(Aimp, UMimp, sMimp, VMHimp, Dimp, 
                         Px, PX, Py, PY, slicing=Dcut)
    T0 = T0.update(U, s, VH)
    
    U, s, VH = coarse_graining(A, UM, sM, VMH, D, 
                         Px, PX, Py, PY, slicing=Dcut)
    T = T.update(U, s, VH)

    t1 = time.time()
    print(f"Coarse graining finished. Time= {t1-t0:.2e} s")

    T  = leg_transposition(T , "restore", direction)
    T0 = leg_transposition(T0, "restore", direction)

    return T, T0

def new_impuer_tensor_1point_MultipleImpureTensors(T:Tensor, T0:list, Dcut:int, direction:str):
    n_T0 = len(T0)
    for i in range(n_T0):
        assert type(T0[i]) == Tensor, f"T0[{i}] must be ATRG_Tensor type"

    T  = leg_transposition(T , "transpose", direction)
    for i in range(n_T0):
        T0[i] = leg_transposition(T0[i], "transpose", direction)

    #step (a)~(c)
    t0 = time.time()
    A, D = T.U, T.VH
    B = oe.contract("i,itxy->itxy", T.s, T.VH)
    C = oe.contract("TXYj,j->TXYj", T.U, T.s)
    UM, sM, VMH = leg_swapping(B, C, Dcut, ispure=True)
    t1 = time.time()
    print(f"Leg swapping of pure tensor finished. Time= {t1-t0:.2e} s")

    print("Singular values of sM is:")
    with cp.printoptions(formatter={'float_kind':'{:.6e}'.format}):
        print(sM)
        print(f"sM/s1 = {sM[-1]/sM[0]:.6e}")

    #step (d)
    t0 = time.time()
    Px, PX, Py, PY = squeezer(A, UM, sM, VMH, D, Dcut)
    t1 = time.time()
    print(f"Squeezer finished. Time= {t1-t0:.2e} s")

    for i in range(n_T0):
        t0 = time.time()
        if T0[i].loc[direction] % 2 == 0:
            Aimp = T.U
            Bimp = oe.contract("i,itxy->itxy", T.s, T.VH)
            Cimp = oe.contract("TXYj,j->TXYj", T0[i].U, T0[i].s)
            Dimp = T0[i].VH
        elif T0[i].loc[direction] % 2 == 1:
            Aimp = T0[i].U
            Bimp = oe.contract("i,itxy->itxy", T0[i].s, T0[i].VH)
            Cimp = oe.contract("TXYj,j->TXYj", T.U, T.s)
            Dimp = T.VH
        UMimp, sMimp, VMHimp = leg_swapping(Bimp, Cimp, Dcut, ispure=False)
        t1 = time.time()
        print(f"Leg swapping of T0[{i}] finished. Time= {t1-t0:.2e} s")

        t0 = time.time()
        U, s, VH = coarse_graining(Aimp, UMimp, sMimp, VMHimp, Dimp, 
                             Px, PX, Py, PY, slicing=Dcut)
        T0[i] = T0[i].update(U, s, VH)
        t1 = time.time()
        print(f"Coarse graining of T0[{i}] finished finished. Time= {t1-t0:.2e} s")

    t0 = time.time()

    U, s, VH = coarse_graining(A, UM, sM, VMH, D, 
                         Px, PX, Py, PY, slicing=Dcut)
    T = T.update(U, s, VH)

    t1 = time.time()
    print(f"Coarse graining of pure tensor finished. Time= {t1-t0:.2e} s")

    T  = leg_transposition(T , "restore", direction)
    for i in range(n_T0):
        T0[i] = leg_transposition(T0[i], "restore", direction)

    return T, T0

def save_singularvalues(T:Tensor, k:int):
    dname = OUTPUT_DIR+"/tensor_sv/"
    fname = dname + "tensor_sv_lx{:}_ly{:}_lt{:}.dat".format(rgstep['X'], rgstep['Y'], rgstep['T'])
    if not os.path.exists(dname):
        os.mkdir(dname)
    with open(fname, "w") as out:
        smax = cp.max(T.s)
        s = T.s / smax
        out.write("#s1={:.12e}\n".format(smax.real))
        K = max(k, len(s))
        s = cp.pad(s, pad_width=(0,K-len(s)), mode='constant', constant_values=0.0)
        for si in s:
            out.write("{:.12e}\n".format(si))

def cal_X(T:Tensor, save=False):
    Xt, Xx, Xy = T.cal_X()
    
    if save:
        fname = OUTPUT_DIR + "/X.dat"
        if rgstep['T'] + rgstep['X'] + rgstep['Y'] == 0:
            mode = "w"
        else:
            mode = "a"
        with open(fname, mode) as out:
            out.write(f"{Xt.real:.12e} {Xx.real:.12e} {Xy.real:.12e}\n")
    
    print(f"Xt={Xt:.12e}")
    print(f"Xx={Xx:.12e}")
    print(f"Xy={Xy:.12e}")

    return Xt, Xx, Xy

def find_max(T:Tensor, max_nth):
    chiT, chiX, chiY, _ = T.U.shape
    xy = iter.product(range(chiX), range(chiY))
    for ix,iy in xy:
        t0 = time.time()
        Tsub = oe.contract("TXYi,i,itx->TXYt", T.U, T.s, T.VH[:,:,ix,iy])
        Tsub = Tsub.flatten()
        idx_Tsub = np.arange(len(Tsub))

        idx_Tsub = np.unravel_index(indices=idx_Tsub, shape=(chiT,chiX,chiY,chiT))
        idx_Tsub += ((np.repeat(ix, len(Tsub)), ) + (np.repeat(iy, len(Tsub)), ))

        #idx_Tsub = np.unravel_index(indices=idx_Tsub, shape=(chiT,chiX,chiY,chiT,chiX))
        #idx_Tsub += (np.repeat(iy, len(Tsub)), )

        idx_Tsub = np.ravel_multi_index(idx_Tsub, dims=(chiT,chiX,chiY,chiT,chiX,chiY))


        max_size = len(Tsub)

        if 0 + iy == 0:
            Tmax = Tsub
            indices = idx_Tsub
        elif 0 + iy > 0:
            Tmax = cp.concatenate((Tmax, Tsub), axis=0)
            indices = np.concatenate((indices, idx_Tsub), axis=0)

            Tmax_idx_sorted = cp.argsort(cp.abs(Tmax))[::-1]
            Tmax_idx_sorted = Tmax_idx_sorted.get()
            Tmax = Tmax[Tmax_idx_sorted]
            Tmax = Tmax[:max_size]
            indices = indices[Tmax_idx_sorted]
            indices = indices[:max_size]
        t1 = time.time()
        print(f"{ix},{iy} finish. Time= {t1-t0:.2e} s")

    Tmax = Tmax[:max_nth]
    indices = indices[:max_nth]
    print("indices", indices)

    print("Tmax=",Tmax)
    indices = np.unravel_index(indices=indices, shape=(chiT,chiX,chiY,chiT,chiX,chiY))
    indices = np.asarray(list(zip(*indices)))
    print("indices", indices)

    import sys
    sys.exit(0)

    return Tmax, indices

def find_1stto3rdmax(T:Tensor, save=False):
    #Tmax, indices = find_max(T, 3)
    #for i in range(3):
    #    print(f"max {i}-th: T_{tuple(indices[i])}= {T[indices[i]]}, |T|= {cp.abs(T[indices[i]]):.12e}, Tmax={Tmax[i]}")

    chiT, chiX, chiY, _ = T.U.shape
    
    def get_Tdiag(i,j,k):
        diag = oe.contract("i,i,i", T.U[i,j,k,:], T.s, T.VH[:,i,j,k])
        return diag

    #get_Tdiavg_vectorize = cp.vectorize(get_Tdiag)

    Tdiag = [get_Tdiag(i,j,k) for i in range(chiT) for j in range(chiX) for k in range(chiY)]
    Tdiag = cp.asarray(Tdiag)

    Tdiag = Tdiag.flatten()
    Tabs  = cp.abs(Tdiag)
    print("diagnal of |T|:", (cp.sort(Tabs)[::-1])[:10])
    indices = cp.argsort(Tabs)[::-1]
    Tmax = cp.zeros(3, complex)
    for n,i in enumerate(indices):
        if n >= 3:
            break
        Tmax[n] = Tdiag[i]

    if save:
        if rgstep["T"] + rgstep["X"] + rgstep["Y"] == 0:
            mode = "w"
        else:
            mode = "a"
        fname = OUTPUT_DIR + "/max_1to3-th_values.dat"
        with open(fname, mode) as output:
            write = f"{Tmax[0]:40.12e} {Tmax[1]:40.12e} {Tmax[2]:40.12e}\n"
            output.write(write)

def cal_scaling_dimensions(T:Tensor, save=False):
    DensMat = oe.contract("Txyi,i,itxy->Tt", T.U, T.s, T.VH)
    e, _ = cp.linalg.eigh(DensMat)
    e = e[::-1]
    e[e < 1e-16] = 1e-16
    e = cp.pad(e, pad_width=(0,T.Dcut-len(e)), mode='constant', constant_values=1e-16)
    e0 = cp.max(e)

    scaldims = cp.log(e0 / e) / (2*cp.pi)

    if save:
        if rgstep["T"] + rgstep["X"] + rgstep["Y"] == 0:
            mode = "w"
        else:
            mode = "a"
        fname = OUTPUT_DIR + "/scaling_dimensions.dat"
        with open(fname, mode) as output:
            write = ""
            for sd in scaldims:
                write += f"{sd:.12e} "
            write = write.rstrip(" ")
            write += "\n"
            output.write(write)


def cal_coorelation(T:Tensor, save=False):
    outdir = "{:}".format(OUTPUT_DIR)
    if not os.path.exists(outdir):
            os.mkdir(outdir)

    #calculate ξt
    Denst_t = oe.contract("TXYi,i,itXY->Tt", T.U, T.s, T.VH)
    _, et, _ = cp.linalg.svd(Denst_t)
    if len(et) < T.Dcut:
        et = cp.pad(et, pad_width=(0,T.Dcut - len(et)), mode='constant', constant_values=0.0)
    etmax = cp.max(et)
    et /= etmax

    if rgstep["T"] + rgstep["X"] + rgstep["Y"] == 0:
        mode = "w"
    elif rgstep["T"] + rgstep["X"] + rgstep["Y"] > 0:
        mode = "a"

    if save:
        with open(f"{outdir}/lnξt.dat", mode) as output:
            et1 = et[1] if et[1] > 1e-200 else 1e-200
            lnxi_t = rgstep["T"] * cp.log(2) - cp.log(-1 / cp.log(et1))
            output.write(f"{lnxi_t:.12e}\n")

        densmat_dir = f"{outdir}/density_matrix_t"
        if not os.path.exists(densmat_dir):
            os.mkdir(densmat_dir)
        with open(f'{densmat_dir}/DentMatT_lx{rgstep["X"]}_ly{rgstep["Y"]}_lt{rgstep["T"]}.dat', mode) as output:
            output.write(f"λ1={etmax:.12e}\n")
            for et_ in et:
                output.write(f"{et_:.12e}\n")

    
    #calculate ξx
    Denst_x = oe.contract("TXYi,i,iTxY->Xx", T.U, T.s, T.VH)
    _, ex, _ = cp.linalg.svd(Denst_x)
    if len(ex) < T.Dcut:
        ex = cp.pad(ex, pad_width=(0,T.Dcut - len(ex)), mode='constant', constant_values=0.0)
    exmax = cp.max(ex)
    ex /= exmax

    if rgstep["T"] + rgstep["X"] + rgstep["Y"] == 0:
        mode = "w"
    elif rgstep["T"] + rgstep["X"] + rgstep["Y"] > 0:
        mode = "a"

    if save:
        with open(f"{outdir}/lnξx.dat", mode) as output:
            ex1 = ex[1] if ex[1] > 1e-200 else 1e-200
            lnxi_x = rgstep["X"] * cp.log(2) - cp.log(-1 / cp.log(ex1))
            output.write(f"{lnxi_x:.12e}\n")

    
    #calculate ξy
    Denst_y = oe.contract("TXYi,i,iTXy->Yy", T.U, T.s, T.VH)
    _, ey, _ = cp.linalg.svd(Denst_y)
    if len(ey) < T.Dcut:
        ey = cp.pad(ey, pad_width=(0,T.Dcut - len(ey)), mode='constant', constant_values=0.0)
    eymax = cp.max(ey)
    ey /= eymax

    if rgstep["T"] + rgstep["X"] + rgstep["Y"] == 0:
        mode = "w"
    elif rgstep["T"] + rgstep["X"] + rgstep["Y"] > 0:
        mode = "a"

    if save:
        with open(f"{outdir}/lnξy.dat", mode) as output:
            ey1 = ey[1] if ex[1] > 1e-200 else 1e-200
            lnxi_y = rgstep["Y"] * cp.log(2) - cp.log(-1 / cp.log(ey1))
            output.write(f"{lnxi_y:.12e}\n")
        
    print(f"ln(ξt)={lnxi_t:.12e}, ln(ξx)={lnxi_x:.12e}, ln(ξy)={lnxi_y:.12e}")


def normalization(T:Tensor):
    #c = cp.max(T.s)
    c = cp.abs(T.trace()).item()
    T.s = T.s / c
    return T, c

def sqr_distance(loc1:dict, loc2:dict):
    r1 = np.array(list(loc1.values()))
    r2 = np.array(list(loc2.values()))
    dist2 = np.sum((r1 - r2)*(r1 - r2))
    return dist2


def exec_pure_tensor_renorm(T:Tensor, Dcut:int, direction:str, rgstep:dict):
    T = atrg_pure_tensor(T, Dcut, direction)
    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['T']})"
    T , c  = normalization(T)
    T.normal_const[factor_key] = math.log(c) / 2**(sum(rgstep.values()))
    return T

def pure_tensor(T:Tensor, Dcut:int, TOT_RGSTEPS:dict):
    global rgstep
    global count_xloop
    global count_yloop
    global count_zloop
    count_xloop = 0
    count_yloop = 0
    count_zloop = 0
    rgstep = {'X':0, 'Y':0, 'T':0}


    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['T']})"
    T , c  = normalization(T)
    
    T .normal_const[factor_key] = math.log(c) / 2**(sum(rgstep.values()))

    #calculate some physical quantities---
    save_singularvalues(T, Dcut)
    
    cal_X(T, save=True)
    #cal_scaling_dimensions(T, save=True)
    #cal_coorelation(T, save=True)
    #find_1stto3rdmax(T, save=True)
    #-------------------------------------

    cycle = "TXY"
    for direction in iter.cycle(cycle):
        if sum(rgstep.values()) >= sum(TOT_RGSTEPS.values()):
            break

        #do renormalization steps
        if rgstep[direction] < TOT_RGSTEPS[direction]:
            if direction == 'T':
                count_zloop += 1
            elif direction == 'X':
                count_xloop += 1
            elif direction == 'Y':
                count_yloop += 1
            rgstep[direction] += 1
            print()
            print(f"rgstep(nt,nx,ny)=({rgstep['T']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization start")
            
            t0 = time.time()
            Ushape_before = T.U.shape
            VHshape_before = T.VH.shape
            T = exec_pure_tensor_renorm(T, Dcut, direction, rgstep)

            #calculate some physical quantities---
            save_singularvalues(T, Dcut)

            cal_X(T, save=True)
            #cal_scaling_dimensions(T, save=True)
            #cal_coorelation(T, save=True)
            #find_1stto3rdmax(T, save=True)
            #-------------------------------------

            Ushape_after = T.U.shape
            VHshape_after = T.VH.shape
            t1 = time.time()
            print(f"bond dimensions of U,VH: {Ushape_before},{VHshape_before} -> {Ushape_after},{VHshape_after}")
            print(f"rgstep(nt,nx,ny)=({rgstep['T']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization finished")
            print(f"time= {t1-t0:.2e} s")
            
    return T


#one point function ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
def cal_field_expected_value(T:Tensor, T0:list, rgstep:dict, save:bool):
    """
    T0[0] : Field
    T0[1] : Complex conjugate of field
    """
    TrT = T.trace()
    Tfactor  = T.get_normalization_const()

    n_T0 = len(T0)
    P = [0.0 for i in range(n_T0)]
    for i in range(n_T0):
        TrT0 = T0[i].trace()
        T0factor = T0[i].get_normalization_const()
        fact0 = cp.exp(cp.sum(T0factor))
        P[i] = fact0*TrT0/TrT

    current_sum_rgsteps = sum(rgstep.values())
    V = 2**(current_sum_rgsteps)
    ln_ZoverV = cp.sum(Tfactor) + cp.log(TrT) / V

    print(f"rgstep={current_sum_rgsteps}, lnZ/V={ln_ZoverV:.12e}, <P>={P[0]:.12e}, <P*>={P[1]:.12e}")

    if save:
        if current_sum_rgsteps == 0:
            mode = "w"
            #write = "#nrgsteps lnZ/V <P> <P†>\n"
            write = ""
        else:
            mode = "a"
            write = ""
        fname = OUTPUT_DIR + "/field_expected_value_rgsteps_dependence.dat"
        with open(fname, mode) as output:
            write += f"{current_sum_rgsteps} {ln_ZoverV.real:.12e} {P[0].real:.12e} {P[1].real:.12e}\n"
            output.write(write)


def exec_rgstep_1point_func(T:Tensor, T0:list, Dcut:int, direction:str, rgstep:dict):
    n_T0 = len(T0)
    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['T']})"
    #loc0_after = T0.loc.copy()
    #loc0_after[direction] = loc0_after[direction] // 2

    T, T0 = new_impuer_tensor_1point_MultipleImpureTensors(T, T0, Dcut, direction)

    t0 = time.time()
    T , c  = normalization(T)
    T.normal_const[factor_key]  = math.log(c) / 2**(sum(rgstep.values()))
    data_print = f"c={c}"

    for i in range(n_T0):
        T0[i], c0 = normalization(T0[i])
        T0[i].normal_const[factor_key] = math.log(c0) - math.log(c)
        T0[i].loc[direction] //= 2
        data_print += f", c0[{i}]={c0}"
    print(data_print)
    t1 = time.time()
    print(f"normalization time= {t1-t0:.2e} s")

    return T, T0

def one_point_function(T:Tensor, T0:list, Dcut:int, TOT_RGSTEPS:dict):
    """
    T : Pure tensor
    T0 : A list of impure tensor
    Dcut : Bond dimension
    TOT_RGSTEPS : dict type, {'X':xsteps, 'Y':ysteps, 'T':tsteps}
    """
    global rgstep
    global count_xloop
    global count_yloop
    global count_zloop
    count_xloop = 0
    count_yloop = 0
    count_zloop = 0
    rgstep = {'X':0, 'Y':0, 'T':0}

    n_T0 = len(T0)
    for i in range(n_T0):
        assert type(T0[i]) == Tensor, f"T0[{i}] must be ATRG_Tensor type"

    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['T']})"
    T, c  = normalization(T)
    T.normal_const[factor_key] = math.log(c) / 2**(sum(rgstep.values()))
    for i in range(n_T0):
        T0[i], c0 = normalization(T0[i])
        T0[i].normal_const[factor_key] = math.log(c0) - math.log(c)
        print(f"c={c}, c0={c0}")
    

    #calculate some physical quantities---
    save_singularvalues(T, Dcut)
    
    #cal_X(T, save=True)
    #cal_scaling_dimensions(T, save=True)
    #cal_coorelation(T, save=True)
    #find_1stto3rdmax(T, save=True)
    cal_field_expected_value(T, T0, rgstep, True)
    #-------------------------------------

    cycle = "TXY"
    for direction in iter.cycle(cycle):
        if sum(rgstep.values()) >= sum(TOT_RGSTEPS.values()):
            break

        #do renormalization steps
        if rgstep[direction] < TOT_RGSTEPS[direction]:
            if direction == 'T':
                count_zloop += 1
            elif direction == 'X':
                count_xloop += 1
            elif direction == 'Y':
                count_yloop += 1
            rgstep[direction] += 1
            print()
            print(f"rgstep(nt,nx,ny)=({rgstep['T']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization start")
            
            t0 = time.time()
            Ushape_before = T.U.shape
            VHshape_before = T.VH.shape
            T, T0 = exec_rgstep_1point_func(T, T0, Dcut, direction, rgstep)

            #calculate some physical quantities---
            save_singularvalues(T, Dcut)

            #cal_X(T, save=True)
            #cal_scaling_dimensions(T, save=True)
            #cal_coorelation(T, save=True)
            #find_1stto3rdmax(T, save=True)
            cal_field_expected_value(T, T0, rgstep, True)
            #-------------------------------------

            Ushape_after = T.U.shape
            VHshape_after = T.VH.shape
            t1 = time.time()
            print(f"bond dimensions of U,VH: {Ushape_before},{VHshape_before} -> {Ushape_after},{VHshape_after}")
            print(f"rgstep(nt,nx,ny)=({rgstep['T']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization finished")
            print(f"time= {t1-t0:.2e} s")
            
    return T, T0
#one point function ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑





#one point function ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
def cal_field_expected_value1(T:Tensor, T0:Tensor, rgstep:dict, save:bool):
    """
    T0[0] : Field
    T0[1] : Complex conjugate of field
    """
    TrT = T.trace()
    Tfactor  = T.get_normalization_const()

    TrT0 = T0.trace()
    T0factor = T0.get_normalization_const()
    fact0 = cp.exp(cp.sum(T0factor))
    P = fact0*TrT0/TrT

    current_sum_rgsteps = sum(rgstep.values())
    V = 2**(current_sum_rgsteps)
    ln_ZoverV = cp.sum(Tfactor) + cp.log(TrT) / V

    print(f"rgstep={current_sum_rgsteps}, lnZ/V={ln_ZoverV:.12e}, <P>={P:.12e}")

    if save:
        if current_sum_rgsteps == 0:
            mode = "w"
            #write = "#nrgsteps lnZ/V <P> <P†>\n"
            write = ""
        else:
            mode = "a"
            write = ""
        fname = OUTPUT_DIR + "/field_expected_value_rgsteps_dependence.dat"
        with open(fname, mode) as output:
            write += f"{current_sum_rgsteps} {ln_ZoverV.real:.12e} {P.real:.12e}\n"
            output.write(write)


def exec_rgstep_1point_func1(T:Tensor, T0:Tensor, Dcut:int, direction:str, rgstep:dict):
    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['T']})"
    #loc0_after = T0.loc.copy()
    #loc0_after[direction] = loc0_after[direction] // 2

    T, T0 = new_impuer_tensor_1point(T, T0, Dcut, direction)

    t0 = time.time()
    T , c  = normalization(T)
    T.normal_const[factor_key]  = math.log(c) / 2**(sum(rgstep.values()))

    data_print = f"c={c}"
    T0, c0 = normalization(T0)
    T0.normal_const[factor_key] = math.log(c0) - math.log(c)
    T0.loc[direction] //= 2
    data_print += f", c0={c0}"
    print(data_print)
    t1 = time.time()
    print(f"normalization time= {t1-t0:.2e} s")

    return T, T0

def one_point_function1(T:Tensor, T0:Tensor, Dcut:int, TOT_RGSTEPS:dict):
    """
    T : Pure tensor
    T0 : A list of impure tensor
    Dcut : Bond dimension
    TOT_RGSTEPS : dict type, {'X':xsteps, 'Y':ysteps, 'T':tsteps}
    """
    global rgstep
    global count_xloop
    global count_yloop
    global count_zloop
    count_xloop = 0
    count_yloop = 0
    count_zloop = 0
    rgstep = {'X':0, 'Y':0, 'T':0}


    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['T']})"
    T, c  = normalization(T)
    T.normal_const[factor_key] = math.log(c) / 2**(sum(rgstep.values()))
    T0, c0 = normalization(T0)
    T0.normal_const[factor_key] = math.log(c0) - math.log(c)
    print(f"c={c}, c0={c0}")
    

    #calculate some physical quantities---
    save_singularvalues(T, Dcut)
    
    #cal_X(T, save=True)
    #cal_scaling_dimensions(T, save=True)
    #cal_coorelation(T, save=True)
    #find_1stto3rdmax(T, save=True)
    cal_field_expected_value1(T, T0, rgstep, True)
    #-------------------------------------

    cycle = "TXY"
    for direction in iter.cycle(cycle):
        if sum(rgstep.values()) >= sum(TOT_RGSTEPS.values()):
            break

        #do renormalization steps
        if rgstep[direction] < TOT_RGSTEPS[direction]:
            if direction == 'T':
                count_zloop += 1
            elif direction == 'X':
                count_xloop += 1
            elif direction == 'Y':
                count_yloop += 1
            rgstep[direction] += 1
            print()
            print(f"rgstep(nt,nx,ny)=({rgstep['T']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization start")
            
            t0 = time.time()
            Ushape_before = T.U.shape
            VHshape_before = T.VH.shape
            T, T0 = exec_rgstep_1point_func1(T, T0, Dcut, direction, rgstep)

            #calculate some physical quantities---
            save_singularvalues(T, Dcut)

            #cal_X(T, save=True)
            #cal_scaling_dimensions(T, save=True)
            #cal_coorelation(T, save=True)
            #find_1stto3rdmax(T, save=True)
            cal_field_expected_value1(T, T0, rgstep, True)
            #-------------------------------------

            Ushape_after = T.U.shape
            VHshape_after = T.VH.shape
            t1 = time.time()
            print(f"bond dimensions of U,VH: {Ushape_before},{VHshape_before} -> {Ushape_after},{VHshape_after}")
            print(f"rgstep(nt,nx,ny)=({rgstep['T']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization finished")
            print(f"time= {t1-t0:.2e} s")
            
    return T, T0
#one point function ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑








#two point function ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
def exec_rgstep_2point_func(T:Tensor, T0:Tensor, Tn:Tensor|None, Dcut:int, direction:str, rgstep:dict):
    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['T']})"
    loc0_after = T0.loc.copy()
    loc0_after[direction] = loc0_after[direction] // 2

    if Tn is not None:
        locn_after = Tn.loc.copy()
        locn_after[direction] = locn_after[direction] // 2
        sqr_distance_after = sqr_distance(loc0_after, locn_after)
        sqr_distance_current = sqr_distance(T0.loc, Tn.loc)

        if sqr_distance_after > 0:
            T, T0, Tn = new_impuer_tensor_2points(T, T0, Tn, Dcut, direction)

            t0 = time.time()
            T , c  = normalization(T)
            T0, c0 = normalization(T0)
            Tn, c1 = normalization(Tn)

            T.normal_const[factor_key]  = math.log(c) / 2**(sum(rgstep.values()))
            T0.normal_const[factor_key] = math.log(c0) + math.log(c1) - 2*math.log(c)

            #print coordinate of impure tensor T0, Tn
            coor0 = f"({T0.loc['X']},{T0.loc['Y']},{T0.loc['T']})" \
                  + f" -> ({loc0_after['X']},{loc0_after['Y']},{loc0_after['T']})"
            coorn = f"({Tn.loc['X']},{Tn.loc['Y']},{Tn.loc['T']})" \
                  + f" -> ({locn_after['X']},{locn_after['Y']},{locn_after['T']})"
            print(f"T0: {coor0}")
            print(f"Tn: {coorn}")

            T0.loc[direction] //= 2
            Tn.loc[direction] //= 2

            print(f"c={c}, c0={c0}, cn={c1}")
            t1 = time.time()
            print(f"normalization time= {t1-t0:.2e} s")

        if sqr_distance_after == 0:
            T, T0 = new_impuer_tensor_absorb(T, T0, Tn, Dcut, direction)

            t0 = time.time()
            T , c  = normalization(T)
            T0, c0 = normalization(T0)
            T.normal_const[factor_key]  = math.log(c) / 2**(sum(rgstep.values()))
            T0.normal_const[factor_key] = math.log(c0) - math.log(c)

            #print coordinate of impure tensor T0, Tn
            print("impure tensor Tn have been absorbed!")
            coor0 = f"({T0.loc['X']},{T0.loc['Y']},{T0.loc['T']})" \
                  + f" -> ({loc0_after['X']},{loc0_after['Y']},{loc0_after['T']})"
            coorn = f"({Tn.loc['X']},{Tn.loc['Y']},{Tn.loc['T']})" \
                  + f" -> ({locn_after['X']},{locn_after['Y']},{locn_after['T']})"
            print(f"T0: {coor0}")
            print(f"Tn: {coorn}")

            T0.loc[direction] //= 2
            Tn = None

            print(f"c={c}, c0={c0}")
            t1 = time.time()
            print(f"normalization time= {t1-t0:.2e} s")

    else:
        T, T0, new_impuer_tensor_1point(T, T0, Dcut, direction)

        t0 = time.time()
        T , c  = normalization(T)
        T0, c0 = normalization(T0)
        T.normal_const[factor_key]  = math.log(c) / 2**(sum(rgstep.values()))
        T0.normal_const[factor_key] = math.log(c0) - math.log(c)

        #print coordinate of impure tensor T0, Tn
        coor0 = f"({T0.loc['X']},{T0.loc['Y']},{T0.loc['T']})" \
              + f" -> ({loc0_after['X']},{loc0_after['Y']},{loc0_after['T']})"
        print(f"T0: {coor0}")

        T0.loc[direction] //= 2
        Tn = None

        print(f"c={c}, c0={c0}")
        t1 = time.time()
        print(f"normalization time= {t1-t0:.2e} s")

    return T, T0, Tn


def two_point_function(T:Tensor, T0:Tensor, Tn:Tensor, Dcut:int, TOT_RGSTEPS:dict):
    global rgstep
    global count_xloop
    global count_yloop
    global count_zloop
    count_xloop = 0
    count_yloop = 0
    count_zloop = 0
    rgstep = {'X':0, 'Y':0, 'T':0}


    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['T']})"
    T , c  = normalization(T)
    T0, c0 = normalization(T0)
    Tn, c1 = normalization(Tn)

    T .normal_const[factor_key] = math.log(c) / 2**(sum(rgstep.values()))
    T0.normal_const[factor_key] = math.log(c0) + math.log(c1) - 2*math.log(c)

    #calculate some physical quantities---
    save_singularvalues(T, Dcut)
    
    #cal_X(T, save=True)
    #cal_scaling_dimensions(T, save=True)
    #cal_coorelation(T, save=True)
    #find_1stto3rdmax(T, save=True)
    #-------------------------------------

    print(f"c={c}, c0={c0}, cn={c1}")

    cycle = "TXY"
    for direction in iter.cycle(cycle):
        if sum(rgstep.values()) >= sum(TOT_RGSTEPS.values()):
            break

        #do renormalization steps
        if rgstep[direction] < TOT_RGSTEPS[direction]:
            if direction == 'T':
                count_zloop += 1
            elif direction == 'X':
                count_xloop += 1
            elif direction == 'Y':
                count_yloop += 1
            rgstep[direction] += 1
            print()
            print(f"rgstep(nt,nx,ny)=({rgstep['T']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization start")
            
            t0 = time.time()
            Ushape_before = T.U.shape
            VHshape_before = T.VH.shape
            T, T0, Tn = exec_rgstep_2point_func(T, T0, Tn, Dcut, direction, rgstep)

            #calculate some physical quantities---
            save_singularvalues(T, Dcut)

            #cal_X(T, save=True)
            #cal_scaling_dimensions(T, save=True)
            #cal_coorelation(T, save=True)
            #find_1stto3rdmax(T, save=True)
            #-------------------------------------

            Ushape_after = T.U.shape
            VHshape_after = T.VH.shape
            t1 = time.time()
            print(f"bond dimensions of U,VH: {Ushape_before},{VHshape_before} -> {Ushape_after},{VHshape_after}")
            print(f"rgstep(nt,nx,ny)=({rgstep['T']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization finished")
            print(f"time= {t1-t0:.2e} s")
            
    return T, T0
#two point function ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
