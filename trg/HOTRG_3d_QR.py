import gc
import math
import time
import numpy as np
import cupy as cp
import opt_einsum as oe
import itertools as iter

from itertools import product

from tensor_class.tensor_class import HOTRG_Tensor as Tensor
from trg.gauge_fixing import gauge_fixing_2d
#from utility.randomized_svd import rsvd
from utility.truncated_svd import svd, eigh

optimize={'slicing': {'min_slices': 4}}

import sys
dcut = int(sys.argv[4])
slicing = dcut*5

import os
select_squeezer = os.environ["HOTRG_SQUEEZER"]

OUTPUT_DIR = os.environ['OUTPUT_DIR']
degeneracy_eps = float(os.environ['DGENERACY_EPS'])
truncate_eps = 1e-10

def leg_transposition(T:cp.ndarray, do_what="transpose", direction='Z'):
    """
    A: tensor A_{xXyYzZ}
    >>>        Z  Y
    >>>        | /
    >>>   x -- T -- X
    >>>      / |
    >>>     y  z
    
    do_what: "transpose" or "restore"\\
    direction:
    >>> 'Z' or 'Z', temporal or z direction;
    >>> 'X' or 'X', x direction;
    >>> 'Y' or 'Y', y direction. 
    -------------------------------------------------------------
    >>> "transpose":
    >>>     "Z": T_{xXyYzZ} -> T_{xXyYzZ}
    >>>     'X': T_{xXyYzZ} -> T_{yYzZxX}
    >>>     'Y': T_{xXyYzZ} -> T_{zZxXyY}
    >>> "restore":
    >>>     "Z": T_{xXyYzZ} -> T_{xXyYzZ}
    >>>     'X': T_{yYzZxX} -> T_{xXyYzZ}
    >>>     'Y': T_{zZxXyY} -> T_{xXyYzZ}
    -------------------------------------------------------------
    """
    if do_what == "transpose":
        if direction == 'Z' or direction == 'Z':
            return T
        elif direction == 'X' or direction == 'X':
            T = cp.transpose(T, axes=(2,3,4,5,0,1))
            return T
        elif direction == 'Y' or direction == 'Y':
            T = cp.transpose(T, axes=(4,5,0,1,2,3))
            return T

    elif do_what == "restore":
        if direction == 'Z' or direction == 'Z':
            return T
        elif direction == 'X' or direction == 'X':
            T = cp.transpose(T, axes=(4,5,0,1,2,3))
            return T
        elif direction == 'Y' or direction == 'Y':
            T = cp.transpose(T, axes=(2,3,4,5,0,1))
            return T
        
def squeezer_fix(T0, T1, T2, T3, Dcut:int):
    chix0, chiX0, chiy0, chiY0, chiz0, chiZ0 = T0.shape
    chix1, chiX1, chiy1, chiY1, chiz1, chiZ1 = T1.shape
    chix2, chiX2, chiy2, chiY2, chiz2, chiZ2 = T2.shape
    chix3, chiX3, chiy3, chiY3, chiz3, chiZ3 = T3.shape

    t0 = time.time()
    path = [(0,2), (0,1), (0,1)]
    LXdagLX = oe.contract("aibcdm,ejfgmh,akbcdn,elfgnh->ijkl", cp.conj(T0), cp.conj(T1), T0, T1, optimize=path)
    LydagLy = oe.contract("abicdm,efjgmh,abkcdn,eflgnh->ijkl", cp.conj(T0), cp.conj(T1), T0, T1, optimize=path)
    t1 = time.time()
    TdagT = oe.contract("xXxX", LXdagLX)

    LXdagLX = cp.reshape(LXdagLX,  (chiX0*chiX1, chiX0*chiX1))
    LydagLy = cp.reshape(LydagLy,  (chiy0*chiy1, chiy0*chiy1))

    #eLX, ULX = cp.linalg.eigh(LXdagLX)
    #eLy, ULy = cp.linalg.eigh(LydagLy)
    #eLX, ULX = eLX[::-1], ULX[:,::-1]
    #eLy, ULy = eLy[::-1], ULy[:,::-1]

    k1 = min(*LXdagLX.shape, Dcut)
    k2 = min(*LydagLy.shape, Dcut)
    eLX, ULX = eigh(LXdagLX, shape=[[0], [1]], k=k1, degeneracy_eps=degeneracy_eps, truncate_eps=truncate_eps)
    eLy, ULy = eigh(LydagLy, shape=[[0], [1]], k=k2, degeneracy_eps=degeneracy_eps, truncate_eps=truncate_eps)

    #squeezer of x direction:
    k = min(len(eLX), Dcut)
    PX = ULX
    Px = cp.conj(ULX.T)
    PX = PX[:,:k]
    Px = Px[:k,:]
    PX = cp.reshape(PX, (chiX0, chiX1, k))
    Px = cp.reshape(Px, (k, chix0, chix1))

    #squeezer of y direction:
    k = min(len(eLy), Dcut)
    PY = cp.conj(ULy.T)
    Py = ULy
    PY = PY[:k,:]
    Py = Py[:,:k]
    PY = cp.reshape(PY, (k, chiY3, chiY2))
    Py = cp.reshape(Py, (chiy0, chiy1, k))

    return Px, PX, Py, PY, TdagT
        
def squeezer_original(T0, T1, T2, T3, Dcut:int):
    chix0, chiX0, chiy0, chiY0, chiz0, chiZ0 = T0.shape
    chix1, chiX1, chiy1, chiY1, chiz1, chiZ1 = T1.shape
    chix2, chiX2, chiy2, chiY2, chiz2, chiZ2 = T2.shape
    chix3, chiX3, chiy3, chiY3, chiz3, chiZ3 = T3.shape

    t0 = time.time()
    path = [(0,2), (0,1), (0,1)]
    LXdagLX = oe.contract("aibcdm,ejfgmh,akbcdn,elfgnh->ijkl", cp.conj(T0), cp.conj(T1), T0, T1, optimize=path)
    RxRxdag = oe.contract("iabcdm,jefgmh,kabcdn,lefgnh->ijkl", T3, T2, cp.conj(T3), cp.conj(T2), optimize=path)
    LydagLy = oe.contract("abicdm,efjgmh,abkcdn,eflgnh->ijkl", cp.conj(T0), cp.conj(T1), T0, T1, optimize=path)
    RYRYdag = oe.contract("abcidm,efgjmh,abckdn,efglnh->ijkl", T3, T2, cp.conj(T3), cp.conj(T2), optimize=path)
    t1 = time.time()
    TdagT = oe.contract("xXxX", LXdagLX)

    LXdagLX = cp.reshape(LXdagLX,  (chiX0*chiX1, chiX0*chiX1))
    RxRxdag = cp.reshape(RxRxdag,  (chix3*chix2, chix3*chix2))
    LydagLy = cp.reshape(LydagLy,  (chiy0*chiy1, chiy0*chiy1))
    RYRYdag = cp.reshape(RYRYdag,  (chiY3*chiY2, chiY3*chiY2))

    #eLX, ULX = cp.linalg.eigh(LXdagLX)
    #eRx, URx = cp.linalg.eigh(RxRxdag)
    #eLy, ULy = cp.linalg.eigh(LydagLy)
    #eRY, URY = cp.linalg.eigh(RYRYdag)

    k1 = min(*LXdagLX.shape, Dcut)
    k2 = min(*RxRxdag.shape, Dcut)
    k3 = min(*LydagLy.shape, Dcut)
    k4 = min(*RYRYdag.shape, Dcut)
    eLX, ULX = eigh(LXdagLX, shape=[[0], [1]], k=k1, degeneracy_eps=degeneracy_eps, truncate_eps=truncate_eps)
    eRx, URx = eigh(RxRxdag, shape=[[0], [1]], k=k2, degeneracy_eps=degeneracy_eps, truncate_eps=truncate_eps)
    eLy, ULy = eigh(LydagLy, shape=[[0], [1]], k=k3, degeneracy_eps=degeneracy_eps, truncate_eps=truncate_eps)
    eRY, URY = eigh(RYRYdag, shape=[[0], [1]], k=k4, degeneracy_eps=degeneracy_eps, truncate_eps=truncate_eps)

    #eLX, ULX = eLX[::-1], ULX[:,::-1]
    #eRx, URx = eRx[::-1], URx[:,::-1]
    #eLy, ULy = eLy[::-1], ULy[:,::-1]
    #eRY, URY = eRY[::-1], URY[:,::-1]
    print(f"eLX: {eLX[:Dcut]}")
    #print(f"eRx: {eRx}")
    print(f"eLy: {eLy[:Dcut]}")
    #print(f"eRY: {eRY}")

    #squeezer of x direction:
    k = min(len(eLX), Dcut)
    errLX = cp.sum(eLX[k: Dcut*Dcut])
    errRx = cp.sum(eRx[k: Dcut*Dcut])
    if errLX < errRx:
        PX = ULX
        Px = cp.conj(ULX.T)
    else:
        PX = URx
        Px = cp.conj(URx.T)
    PX = PX[:,:k]
    Px = Px[:k,:]
    PX = cp.reshape(PX, (chiX0, chiX1, k))
    Px = cp.reshape(Px, (k, chix0, chix1))

    #squeezer of y direction:
    k = min(len(eLy), Dcut)
    errLy = cp.sum(eLy[k: Dcut*Dcut])
    errRY = cp.sum(eRY[k: Dcut*Dcut])
    if errLy < errRY:
        PY = cp.conj(URY.T)
        Py = URY
    else:
        PY = cp.conj(ULy.T)
        Py = ULy
    PY = PY[:k,:]
    Py = Py[:,:k]
    PY = cp.reshape(PY, (k, chiY3, chiY2))
    Py = cp.reshape(Py, (chiy0, chiy1, k))

    return Px, PX, Py, PY, TdagT

def squeezer_QR(T0, T1, T2, T3, Dcut:int):
    """
    >>>     z'1           z'2               z'1                    z'2
    >>> y'1  |        y'2  |            y'1  |                 y'2  |
    >>>    \ |           \ |               \ |                    \ |
    >>> x1---T1------------T2---x'2     x1---T1---\             /---T2---x'2
    >>>      | \    j      | \               | \   \           /    | \      
    >>>      |  y1         |  y2             |  y1  \         /     |  y2
    >>>     i|             |k               i|       \-------/      |k      
    >>> y'0  |        y'2  |            y'0  |       /       \ y'2  |
    >>>    \ |           \ |               \ |      /         \   \ |       
    >>> x0---T0------------T3---x'3     x0---T0----/           \----T3---x'3
    >>>      | \    l      | \               | \                    | \  
    >>>      |  y0         |  y3             |  y0                  |  y3
    >>>      z0            z3                z0                    z3
    returns projector Px, PX, Py, PY \\
    >>> Px_{x, x0, x1}
    >>> PX_{x'0, x'1, x'}
    >>> Py_{y0, y1, y}
    >>> PY_{y', y'0, y'1}
    """
    
    chix0, chiX0, chiy0, chiY0, chiz0, chiZ0 = T0.shape
    chix1, chiX1, chiy1, chiY1, chiz1, chiZ1 = T1.shape
    chix2, chiX2, chiy2, chiY2, chiz2, chiZ2 = T2.shape
    chix3, chiX3, chiy3, chiY3, chiz3, chiZ3 = T3.shape

    t0 = time.time()
    path = [(0,2), (0,1), (0,1)]
    LXdagLX = oe.contract("aibcdm,ejfgmh,akbcdn,elfgnh->ijkl", cp.conj(T0), cp.conj(T1), T0, T1, optimize=path)
    RxRxdag = oe.contract("iabcdm,jefgmh,kabcdn,lefgnh->ijkl", T3, T2, cp.conj(T3), cp.conj(T2), optimize=path)
    LydagLy = oe.contract("abicdm,efjgmh,abkcdn,eflgnh->ijkl", cp.conj(T0), cp.conj(T1), T0, T1, optimize=path)
    RYRYdag = oe.contract("abcidm,efgjmh,abckdn,efglnh->ijkl", T3, T2, cp.conj(T3), cp.conj(T2), optimize=path)
    t1 = time.time()
    #print("Matrix for squeezer: {:.2e} s".format((t1-t0)/2), end=", ")

    TdagT = oe.contract("xXxX", LXdagLX)

    LXdagLX = cp.reshape(LXdagLX,  (chiX0*chiX1, chiX0*chiX1))
    RxRxdag = cp.reshape(RxRxdag,  (chix3*chix2, chix3*chix2))
    LydagLy = cp.reshape(LydagLy,  (chiy0*chiy1, chiy0*chiy1))
    RYRYdag = cp.reshape(RYRYdag,  (chiY3*chiY2, chiY3*chiY2))

    ULx, SLx, VLxH = cp.linalg.svd(LXdagLX)
    URx, SRx, VRxH = cp.linalg.svd(RxRxdag)
    ULy, SLy, VLyH = cp.linalg.svd(LydagLy)
    URy, SRy, VRyH = cp.linalg.svd(RYRYdag)
    del LXdagLX, RxRxdag, LydagLy, RYRYdag

    print(f"SLx: {SLx[:Dcut]}")
    #print(f"SRx: {SRx}")
    print(f"SLy: {SLy[:Dcut]}")
    #print(f"SRy: {SRy}")
    
    def generalized_inverse(A:cp.ndarray, degeneracy_eps, truncate_eps):
        k = min(*A.shape, Dcut)
        U, S, VH = svd(A, shape=[[0], [1]], k=k, degeneracy_eps=degeneracy_eps, truncate_eps=truncate_eps)
        UH = cp.conj(U.T)
        Sinv = 1 / S
        V = cp.conj(VH.T)

        return UH, Sinv, V

    RLx = oe.contract("a,ax->ax", cp.sqrt(SLx), VLxH)
    RRx = oe.contract("xb,b->xb", URx, cp.sqrt(SRx))
    UxH, Sxinv, Vx = generalized_inverse(RLx@RRx, degeneracy_eps, truncate_eps)
    PX = oe.contract("ib,bX,X->iX", RRx, Vx, cp.sqrt(Sxinv))
    Px = oe.contract("x,xa,ai->xi", cp.sqrt(Sxinv), UxH, RLx)
    #print("Tr(PX@Px)=", cp.trace(PX@Px), "|Px@PX|^2=", cp.linalg.norm(Px@PX)**2)
    PX = cp.reshape(PX, (chiX0, chiX1, PX.shape[1]))
    Px = cp.reshape(Px, (Px.shape[0], chix0, chix1))


    RLy = oe.contract("a,ay->ay", cp.sqrt(SLy), VLyH)
    RRy = oe.contract("yb,b->yb", URy, cp.sqrt(SRy))
    UyH, Syinv, Vy = generalized_inverse(RLy@RRy, degeneracy_eps, truncate_eps)
    Py = oe.contract("ib,by,y->iy", RRy, Vy, cp.sqrt(Syinv))
    PY = oe.contract("Y,Ya,ai->Yi", cp.sqrt(Syinv), UyH, RLy)
    #print("Tr(Py@PY)=", cp.trace(Py@PY), "|PY@Py|^2=", cp.linalg.norm(PY@Py)**2)
    Py = cp.reshape(Py, (chiy0, chiy1, Py.shape[1]))
    PY = cp.reshape(PY, (PY.shape[0], chiY0, chiY1))

    #Py = cp.transpose(Py, (2,0,1))
    #PY = cp.transpose(PY, (1,2,0))
    #Py, PY = PY, Py

    return Px, PX, Py, PY, TdagT
    

def slice_legs(leg_size, slicing):
    """
    leg_size, slicing: can be a exponential notation if the number is large
    slicing leg_size to slicing parts \\
    if leg_size < slicing, return a list of slice(i, i+1) \\
    if slicing<=1, return one element list of slice(0, leg_size) \\
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
        return [slice(0, leg_size)], 1

    bs1 = leg_size // slicing
    bs2 = (leg_size // slicing) + 1
    n1 = (leg_size - bs2*slicing) // (bs1 - bs2)
    n2 = slicing - n1
    
    slice_list = [slice(i, i+bs1) for i in range(0, n1*bs1, bs1)]
    slice_list += [slice(i, i+bs2) for i in range(n1*bs1, leg_size, bs2)]

    return slice_list, len(slice_list)

def coarse_graining(T0:cp.ndarray, T1:cp.ndarray, 
                    Px:cp.ndarray, PX:cp.ndarray, 
                    Py:cp.ndarray, PY:cp.ndarray, 
                    slicing:int, Dcut:int):
    """
    T: T_{x,X,y,Y,z,Z}\\
    Px: Px_{x,x0,x1}\\
    PX: PX_{X0,X1,X}\\
    Py: Py_{y0,y1,y}\\
    PY: PY_{Y,Y0,Y1}
    >>>                     f   Z                               
    >>>                  Y\ |\  |                               
    >>>                    \| \ |                               
    >>>               /b----|---T1--------c\                    
    >>>              /      |   | \         \                   
    >>>      x      /       |   |  \g        \      X           
    >>>      ------/       e|  i|   |         \------           
    >>>            \         \  |   |         /                 
    >>>             \         \ |   |        /                  
    >>>              \-a--------T0--|\-----d/                   
    >>>                         | \ | \                         
    >>>                         |  \|  y                        
    >>>                         z   h                           
    """
    slicing = int(slicing)
    chiZ = T1.shape[5]
    chiz = T0.shape[4]
    chix = Px.shape[0]
    chiX = PX.shape[2]
    chiy = Py.shape[2]
    chiY = PY.shape[0]
    chia, chid, chih, chie = T0.shape[0:4]
    chib, chic, chig, chif = T1.shape[0:4]
    chii = T0.shape[5]

    slicing_i = min(chii, slicing)
    remain = math.ceil(slicing / slicing_i)
    slicing_b = min(remain, chib)
    remain = math.ceil(remain / slicing_b)
    slicing_f = min(remain, chif)
    remain = math.ceil(remain / slicing_f)
    slicing_d = min(remain, chid)
    remain = math.ceil(remain / slicing_d)
    slicing_h = remain

    slice_list_i, nslice_i = slice_legs(chii, slicing_i)
    slice_list_b, nslice_b = slice_legs(chib, slicing_b)
    slice_list_f, nslice_f = slice_legs(chif, slicing_f)
    slice_list_d, nslice_d = slice_legs(chid, slicing_d)
    slice_list_h, nslice_h = slice_legs(chih, slicing_h)

    
    T0 = T0.get()
    T1 = T1.get()
    Px = Px.get()
    PX = PX.get()
    Py = Py.get()
    PY = PY.get()

    iteration = product(slice_list_i, slice_list_b, slice_list_f, slice_list_d, slice_list_h)
    T = cp.zeros(shape=(chix,chiX,chiy,chiY,chiz,chiZ), dtype=complex)
    for i,b,f,d,h in iteration:
        t0 = cp.asarray(T0[:,d,h,:,:,i])
        t1 = cp.asarray(T1[b,:,:,f,i,:])
        px = cp.asarray(Px[:,:,b])
        pX = cp.asarray(PX[d,:,:])
        py = cp.asarray(Py[h,:,:])
        pY = cp.asarray(PY[:,:,f])
        T += oe.contract("adhezi,bcgfiZ,xab,dcX,hgy,Yef->xXyYzZ", t0, t1, px, pX, py, pY)

    del T0, T1
    gc.collect()

    return T


def squeezer(T0, T1, T2, T3, Dcut, squeezer_type:str):
    """
    squeezer_type: "original", "QR", "fix"
    """
    if squeezer_type == "original":
        Px, PX, Py, PY, TdagT = squeezer_original(T0, T1, T2, T3, Dcut) 
        return Px, PX, Py, PY, TdagT
    elif squeezer_type == "QR":
        Px, PX, Py, PY, TdagT = squeezer_QR(T0, T1, T2, T3, Dcut) 
        return Px, PX, Py, PY, TdagT
    elif squeezer_type == "fix":
        Px, PX, Py, PY, TdagT = squeezer_fix(T0, T1, T2, T3, Dcut) 
        return Px, PX, Py, PY, TdagT

def new_pure_tensor(T:dict, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T['Tensor'] = leg_transposition(T['Tensor'], "transpose", direction)

    t0 = time.time()
    Px, PX, Py, PY, TdagT = squeezer(T['Tensor'], T['Tensor'], T['Tensor'], T['Tensor'], Dcut, select_squeezer)
    t1 = time.time()
    print("squeezer time: {:.2e} s".format(t1-t0))

    t0 = time.time()
    T['Tensor'] = coarse_graining(T['Tensor'], T['Tensor'], Px, PX, Py, PY, slicing=slicing, Dcut=Dcut)
    t1 = time.time()
    print("coarse graining time: {:.2e} s".format(t1-t0))

    T['Tensor'] = leg_transposition(T['Tensor'], "restore", direction)

    return T


def new_impuer_tensor_2points(T:dict, T0:dict, Tn:dict, Dcut:int, direction:str):
    
    T['Tensor']  = leg_transposition( T['Tensor'], "transpose", direction)
    T0['Tensor'] = leg_transposition(T0['Tensor'], "transpose", direction)
    Tn['Tensor'] = leg_transposition(Tn['Tensor'], "transpose", direction)
    
    t0 = time.time()
    Px, PX, Py, PY, TdagT = squeezer(T['Tensor'], T['Tensor'], T['Tensor'], T['Tensor'], Dcut, select_squeezer)
    t1 = time.time()
    print("squeezer time: {:.2e} s".format(t1-t0))

    t0 = time.time()
    
    if T0["loc"][direction] % 2 == 1:
        T0['Tensor'] = coarse_graining(T['Tensor'], T0['Tensor'], 
                                       Px, PX, Py, PY, 
                                       slicing=slicing, Dcut=Dcut)
    elif T0["loc"][direction] % 2 == 0:
        T0['Tensor'] = coarse_graining(T0['Tensor'], T['Tensor'], 
                                       Px, PX, Py, PY, 
                                       slicing=slicing, Dcut=Dcut)
            
    if Tn["loc"][direction] % 2 == 1:
        Tn['Tensor'] = coarse_graining(T['Tensor'], Tn['Tensor'], 
                                       Px, PX, Py, PY, 
                                       slicing=slicing, Dcut=Dcut)
    elif Tn["loc"][direction] % 2 == 0:
        Tn['Tensor'] = coarse_graining(Tn['Tensor'], T['Tensor'], 
                                       Px, PX, Py, PY, 
                                       slicing=slicing, Dcut=Dcut)

    T['Tensor'] = coarse_graining(T['Tensor'], T['Tensor'], 
                                  Px, PX, Py, PY, 
                                  slicing=slicing, Dcut=Dcut)

    t1 = time.time()
    print("coarse graining time: {:.2e} s".format(t1-t0))

    T['Tensor']  = leg_transposition( T['Tensor'], "restore", direction)
    T0['Tensor'] = leg_transposition(T0['Tensor'], "restore", direction)
    Tn['Tensor'] = leg_transposition(Tn['Tensor'], "restore", direction)

    #err
    tdagt = oe.contract("xXyYzZ,xXyYzZ", cp.conj(T['Tensor']), T['Tensor'])
    err = cp.sqrt((TdagT - tdagt)/TdagT)
    print(f"total coarse graining error: {err:.6e}")

    return T, T0, Tn

def new_impuer_tensor_absorb(T:dict, T0:dict, Tn:dict, Dcut:int, direction:str):
    #print("Contracting direction:",direction, end=", ")

    T['Tensor']  = leg_transposition( T['Tensor'], "transpose", direction)
    T0['Tensor'] = leg_transposition(T0['Tensor'], "transpose", direction)
    Tn['Tensor'] = leg_transposition(Tn['Tensor'], "transpose", direction)

    t0 = time.time()
    Px, PX, Py, PY, TdagT = squeezer(T['Tensor'], T['Tensor'], T['Tensor'], T['Tensor'], Dcut, select_squeezer)
    t1 = time.time()
    print("squeezer time: {:.2e} s".format(t1-t0))

    t0 = time.time()

    if T0["loc"][direction] % 2 == 0 and Tn["loc"][direction] % 2 == 1:
        T0['Tensor'] = coarse_graining(T0['Tensor'], Tn['Tensor'], 
                                       Px, PX, Py, PY, 
                                       slicing=slicing, Dcut=Dcut)
        
    elif T0["loc"][direction] % 2 == 1 and Tn["loc"][direction] % 2 == 0:
        T0['Tensor'] = coarse_graining(Tn['Tensor'], T0['Tensor'], 
                                       Px, PX, Py, PY, 
                                       slicing=slicing, Dcut=Dcut)
        
    T['Tensor'] = coarse_graining(T['Tensor'], T['Tensor'], 
                                  Px, PX, Py, PY, 
                                  slicing=slicing, Dcut=Dcut)
    
    t1 = time.time()
    print("coarse graining time: {:.2e} s".format(t1-t0))
    
    T['Tensor']  = leg_transposition( T['Tensor'], "restore", direction)
    T0['Tensor'] = leg_transposition(T0['Tensor'], "restore", direction)

    #err
    tdagt = oe.contract("xXyYzZ,xXyYzZ", cp.conj(T['Tensor']), T['Tensor'])
    err = cp.sqrt((TdagT - tdagt)/TdagT)
    print(f"total coarse graining error: {err:.6e}")
    
    del Tn

    return T, T0

def new_impuer_tensor_1point(T:dict, T0:dict, Dcut:int, direction:str):
    
    T['Tensor']  = leg_transposition( T['Tensor'], "transpose", direction)
    T0['Tensor'] = leg_transposition(T0['Tensor'], "transpose", direction)

    t0 = time.time()
    Px, PX, Py, PY, TdagT = squeezer(T['Tensor'], T['Tensor'], T['Tensor'], T['Tensor'], Dcut, select_squeezer)
    t1 = time.time()
    print("squeezer time: {:.2e} s".format(t1-t0))

    t0 = time.time()

    if T0["loc"][direction] % 2 == 1:
        T0['Tensor'] = coarse_graining(T['Tensor'], T0['Tensor'], 
                                       Px, PX, Py, PY, 
                                       slicing=slicing, Dcut=Dcut)
    elif T0["loc"][direction] % 2 == 0:
        T0['Tensor'] = coarse_graining(T0['Tensor'], T['Tensor'], 
                                       Px, PX, Py, PY, 
                                       slicing=slicing, Dcut=Dcut)
        
    T['Tensor'] = coarse_graining(T['Tensor'], T['Tensor'], 
                                  Px, PX, Py, PY, 
                                  slicing=slicing, Dcut=Dcut)
        
    t1 = time.time()
    print("coarse graining time: {:.2e} s".format(t1-t0))

    T['Tensor']  = leg_transposition( T['Tensor'], "restore", direction)
    T0['Tensor'] = leg_transposition(T0['Tensor'], "restore", direction)

    #err
    tdagt = oe.contract("xXyYzZ,xXyYzZ", cp.conj(T['Tensor']), T['Tensor'])
    err = cp.sqrt((TdagT - tdagt)/TdagT)
    print(f"total coarse graining error: {err:.6e}")

    return T, T0

#
def cal_X(Tensor:dict, save=False):
    T = Tensor['Tensor']
    TTt = oe.contract("xxyytT,aabbTt", T, T)
    TTx = oe.contract("xXyytt,Xxaabb", T, T)
    TTy = oe.contract("xxyYtt,aaYybb", T, T)
    TrT = oe.contract("xxyytt", T)

    Xt = TrT**2 / TTt
    Xx = TrT**2 / TTx
    Xy = TrT**2 / TTy
    
    if save:
        fname = OUTPUT_DIR + "/X.dat"
        if rgstep['Z'] + rgstep['X'] + rgstep['Y'] == 0:
            mode = "w"
        else:
            mode = "a"
        with open(fname, mode) as out:
            out.write(f"{Xt.real:.12e} {Xx.real:.12e} {Xy.real:.12e}\n")
    
    print(f"Xt={Xt:.12e}")
    print(f"Xx={Xx:.12e}")
    print(f"Xy={Xy:.12e}")

    return Xt, Xx, Xy

def find_1stto3rdmax(Tensor:dict, save=False):
    T = Tensor['Tensor'].get()
    chiX, chiY, chiT = T.shape[::2]
    Tdiag = [T[x,x,y,y,t,t] for x in range(chiX) for y in range(chiY) for t in range(chiT)]
    Tdiag = cp.asarray(Tdiag)
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
        if rgstep["Z"] + rgstep["X"] + rgstep["Y"] == 0:
            mode = "w"
        else:
            mode = "a"
        fname = OUTPUT_DIR + "/max_1to3-th_values.dat"
        with open(fname, mode) as output:
            #Tmax = cp.abs(Tmax)
            write = f"{Tmax[0]:40.12e} {Tmax[1]:40.12e} {Tmax[2]:40.12e}\n"
            output.write(write)


def cal_scaling_dimensions(Tensor:dict, Dcut:int, save=False):
    T = Tensor['Tensor']

    DensMat = oe.contract("xxyytT->Tt", T)
    e, _ = cp.linalg.eigh(DensMat)
    e = e[::-1]
    e[e < 1e-16] = 1e-16
    e = cp.pad(e, pad_width=(0,Dcut-len(e)), mode='constant', constant_values=1e-16)
    e0 = cp.max(e)

    scaldims = cp.log(e0 / e) / (2*cp.pi)

    if save:
        if rgstep["Z"] + rgstep["X"] + rgstep["Y"] == 0:
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

def cal_coorelation(Tensor:dict, Dcut:int, save=False):
    outdir = "{:}".format(OUTPUT_DIR)
    

    T = Tensor['Tensor']

    #calculate ξt
    Denst_t = oe.contract("xxyytT->Tt", T)
    _, et, _ = cp.linalg.svd(Denst_t)
    if len(et) < Dcut * Dcut:
        et = cp.pad(et, pad_width=(0,Dcut * Dcut - len(et)), mode='constant', constant_values=0.0)
    etmax = cp.max(et)
    et /= etmax

    if rgstep["Z"] + rgstep["X"] + rgstep["Y"] == 0:
        mode = "w"
    elif rgstep["Z"] + rgstep["X"] + rgstep["Y"] > 0:
        mode = "a"

    if save:
        with open(f"{outdir}/lnξt.dat", mode) as output:
            et1 = et[1] if et[1] > 1e-200 else 1e-200
            lnxi_t = rgstep["Z"] * cp.log(2) - cp.log(-1 / cp.log(et1))
            output.write(f"{lnxi_t:.12e}\n")

        densmat_dir = f"{outdir}/density_matrix_t"
        if not os.path.exists(densmat_dir):
            os.mkdir(densmat_dir)
        with open(f'{densmat_dir}/DentMatT_lx{rgstep["X"]}_ly{rgstep["Y"]}_lt{rgstep["Z"]}.dat', mode) as output:
            output.write(f"λ1={etmax:.12e}\n")
            for et_ in et:
                output.write(f"{et_:.12e}\n")

    
    #calculate ξx
    Denst_x = oe.contract("xXyytt->Xx", T)
    _, ex, _ = cp.linalg.svd(Denst_x)
    if len(ex) < Dcut * Dcut:
        ex = cp.pad(ex, pad_width=(0,Dcut * Dcut - len(ex)), mode='constant', constant_values=0.0)
    exmax = cp.max(ex)
    ex /= exmax

    if rgstep["Z"] + rgstep["X"] + rgstep["Y"] == 0:
        mode = "w"
    elif rgstep["Z"] + rgstep["X"] + rgstep["Y"] > 0:
        mode = "a"

    if save:
        with open(f"{outdir}/lnξx.dat", mode) as output:
            ex1 = ex[1] if ex[1] > 1e-200 else 1e-200
            lnxi_x = rgstep["X"] * cp.log(2) - cp.log(-1 / cp.log(ex1))
            output.write(f"{lnxi_x:.12e}\n")

    
    #calculate ξy
    Denst_y = oe.contract("xxyYtt->Yy", T)
    _, ey, _ = cp.linalg.svd(Denst_y)
    if len(ey) < Dcut * Dcut:
        ey = cp.pad(ey, pad_width=(0,Dcut * Dcut - len(ey)), mode='constant', constant_values=0.0)
    eymax = cp.max(ey)
    ey /= eymax

    if rgstep["Z"] + rgstep["X"] + rgstep["Y"] == 0:
        mode = "w"
    elif rgstep["Z"] + rgstep["X"] + rgstep["Y"] > 0:
        mode = "a"

    if save:
        with open(f"{outdir}/lnξy.dat", mode) as output:
            ey1 = ey[1] if ex[1] > 1e-200 else 1e-200
            lnxi_y = rgstep["Y"] * cp.log(2) - cp.log(-1 / cp.log(ey1))
            output.write(f"{lnxi_y:.12e}\n")
        
    print(f"ln(ξt)={lnxi_t:.12e}, ln(ξx)={lnxi_x:.12e}, ln(ξy)={lnxi_y:.12e}")

#Renormalization
def normalization(T):
    c = oe.contract("xxyyzz", T)
    c = cp.abs(c).item()
    T = T / c
    return T, c

def sqr_distance(loc1:dict, loc2:dict):
    r1 = np.array(list(loc1.values()))
    r2 = np.array(list(loc2.values()))
    dist2 = np.sum((r1 - r2)*(r1 - r2))
    return dist2

def __get_tensor_spectrum(T, Dcut):
    Ttmp = cp.transpose(T, (0,2,4,1,3,5))
    Ttmp = cp.reshape(Ttmp, (Dcut*Dcut*Dcut, Dcut*Dcut*Dcut))
    s = cp.linalg.svd(Ttmp, compute_uv=False)
    print("tensor spectrum", s[:2*Dcut]/cp.max(s))


def exec_rgstep_pure_tensor(T:dict, Dcut:int, direction:str, rgstep:dict):
    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['Z']})"
    T = new_pure_tensor(T, Dcut, direction)

    t0 = time.time()
    T['Tensor'] , c  = normalization(T['Tensor'])
    T['factor'][factor_key]  = math.log(c) / 2**(sum(rgstep.values()))
    print(f"c={c}")
    t1 = time.time()
    print(f"normalization time= {t1-t0:.2e} s")

    return T

def pure_tensor(T:dict, Dcut:int, TOT_RGSTEPS:dict):
    global rgstep
    rgstep = {'X':0, 'Y':0, 'Z':0}

    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['Z']})"
    T['Tensor'] , c  = normalization(T['Tensor'])
    T['factor'][factor_key]  = math.log(c) / 2**(sum(rgstep.values()))
    print(f"c={c}")

    #calculate some physical quantities---
    cal_X(T, save=True)
    cal_coorelation(T, Dcut, save=True)
    cal_scaling_dimensions(T, Dcut, save=True)

    #find_1stto3rdmax(T, save=True)
    #-------------------------------------

    cycle = "ZXY"
    for direction in iter.cycle(cycle):
        if sum(rgstep.values()) >= sum(TOT_RGSTEPS.values()):
            break

        #execute renormalization steps
        if rgstep[direction] < TOT_RGSTEPS[direction]:
            rgstep[direction] += 1
            print()
            print(f"rgstep(nz,nx,ny)=({rgstep['Z']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization start")
            
            t0 = time.time()
            Tshape_before = T['Tensor'].shape
            T = exec_rgstep_pure_tensor(T, Dcut, direction, rgstep)

            #calculate some physical quantities---
            cal_X(T, save=True)
            cal_coorelation(T, Dcut, save=True)
            cal_scaling_dimensions(T, Dcut, save=True)

            #find_1stto3rdmax(T, save=True)
            #-------------------------------------

            Tshape_after = T['Tensor'].shape
            t1 = time.time()
            print(f"bond dimensions of T: {Tshape_before} -> {Tshape_after}")
            print(f"rgstep(nz,nx,ny)=({rgstep['Z']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization finished")
            print(f"time= {t1-t0:.2e} s")
    
    return T

#one point function ↓------------------------------------------------------------------------------
def cal_field_expected_value(T:dict, T0:list, rgstep:dict, save:bool):
    """
    T0[0] : Field
    T0[1] : Complex conjugate of field
    """
    TrT = oe.contract("xxyytt", T['Tensor'])
    Tfactor  = cp.asarray(list(T['factor'].values()))

    n_T0 = len(T0)
    P = [0.0 for i in range(n_T0)]
    for i in range(n_T0):
        TrT0 = oe.contract("xxyytt", T0[i]['Tensor'])
        T0factor = cp.asarray(list(T0[i]['factor'].values()))
        fact0 = cp.exp(cp.sum(T0factor))
        P[i] = fact0*TrT0/TrT

    current_sum_rgsteps = sum(rgstep.values())
    V = 2**(current_sum_rgsteps)
    ln_ZoverV = cp.sum(Tfactor) + cp.log(TrT) / V

    print(f"rgstep={current_sum_rgsteps}, lnZ/V={ln_ZoverV:.12e}, <P>={P[0]:.12e}, <P*>={0.0}")

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
            write += f"{current_sum_rgsteps} {ln_ZoverV.real:.12e} {P[0].real:.12e} {0.0}\n"
            output.write(write)

def exec_rgstep_1point_func(T:dict, T0:dict, Dcut:int, direction:str, rgstep:dict):
    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['Z']})"
    loc0_after = T0["loc"].copy()
    loc0_after[direction] = loc0_after[direction] // 2

    T, T0 = new_impuer_tensor_1point(T, T0, Dcut, direction)

    t0 = time.time()
    T['Tensor'] , c  = normalization(T['Tensor'])
    T0['Tensor'], c0 = normalization(T0['Tensor'])
    T['factor'][factor_key]  = math.log(c) / 2**(sum(rgstep.values()))
    T0['factor'][factor_key] = math.log(c0) - math.log(c)

    #print coordinate of impure tensor T0, Tn
    coor0 = f"({T0['loc']['X']},{T0['loc']['Y']},{T0['loc']['Z']})" \
          + f" -> ({loc0_after['X']},{loc0_after['Y']},{loc0_after['Z']})"
    print(f"T0: {coor0}")

    T0["loc"][direction] //= 2

    print(f"c={c}, c0={c0}")
    t1 = time.time()
    print(f"normalization time= {t1-t0:.2e} s")

    return T, T0

def one_point_function(T:dict, T0:dict, Dcut:int, TOT_RGSTEPS:dict):
    rgstep = {'X':0, 'Y':0, 'Z':0}

    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['Z']})"
    T['Tensor'] , c  = normalization(T['Tensor'])
    T0['Tensor'], c0 = normalization(T0['Tensor'])

    T['factor'][factor_key]  = math.log(c) / 2**(sum(rgstep.values()))
    T0['factor'][factor_key] = math.log(c0) - math.log(c) #c0 / c

    print(f"c={c}, c0={c0}")


    cal_field_expected_value(T, [T0], rgstep, True)


    cycle = "ZXY"
    for direction in iter.cycle(cycle):
        if sum(rgstep.values()) >= sum(TOT_RGSTEPS.values()):
            break

        #do renormalization steps
        if rgstep[direction] < TOT_RGSTEPS[direction]:
            rgstep[direction] += 1
            print()
            print(f"rgstep(nz,nx,ny)=({rgstep['Z']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization start")
            
            t0 = time.time()
            Tshape_before = T['Tensor'].shape
            T, T0 = exec_rgstep_1point_func(T, T0, Dcut, direction, rgstep)
            Tshape_after = T['Tensor'].shape
            t1 = time.time()
            print(f"bond dimensions of T: {Tshape_before} -> {Tshape_after}")
            print(f"rgstep(nz,nx,ny)=({rgstep['Z']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization finished")
            

            cal_field_expected_value(T, [T0], rgstep, True)



            print(f"time= {t1-t0:.2e} s")
            
    return T, T0
#one point function ↑------------------------------------------------------------------------------




#two point function ↓------------------------------------------------------------------------------
def exec_rgstep_2point_func(T:dict, T0:dict, Tn:dict, Dcut:int, direction:str, rgstep:dict):
    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['Z']})"
    loc0_after = T0["loc"].copy()
    loc0_after[direction] = loc0_after[direction] // 2

    if len(Tn) > 0:
        locn_after = Tn["loc"].copy()
        locn_after[direction] = locn_after[direction] // 2
        sqr_distance_after = sqr_distance(loc0_after, locn_after)
        sqr_distance_current = sqr_distance(T0["loc"], Tn["loc"])

        if sqr_distance_after > 0:
            T, T0, Tn = new_impuer_tensor_2points(T, T0, Tn, Dcut, direction)

            t0 = time.time()
            T['Tensor'] , c  = normalization(T['Tensor'])
            T0['Tensor'], c0 = normalization(T0['Tensor'])
            Tn['Tensor'], c1 = normalization(Tn['Tensor'])

            #factorT = math.log(c) / 2**(sum(rgstep.values()))
            #factorT0 = math.log(c0) + math.log(c1) - 2*math.log(c)
            T['factor'][factor_key]  = math.log(c) / 2**(sum(rgstep.values()))
            T0['factor'][factor_key] = math.log(c0) + math.log(c1) - 2*math.log(c)

            #print coordinate of impure tensor T0, Tn
            coor0 = f"({T0['loc']['X']},{T0['loc']['Y']},{T0['loc']['Z']})" \
                  + f" -> ({loc0_after['X']},{loc0_after['Y']},{loc0_after['Z']})"
            coorn = f"({Tn['loc']['X']},{Tn['loc']['Y']},{Tn['loc']['Z']})" \
                  + f" -> ({locn_after['X']},{locn_after['Y']},{locn_after['Z']})"
            print(f"T0: {coor0}")
            print(f"Tn: {coorn}")

            T0["loc"][direction] //= 2
            Tn["loc"][direction] //= 2

            print(f"c={c}, c0={c0}, cn={c1}")
            t1 = time.time()
            print(f"normalization time= {t1-t0:.2e} s")

        if sqr_distance_after == 0:
            T, T0 = new_impuer_tensor_absorb(T, T0, Tn, Dcut, direction)

            t0 = time.time()
            T['Tensor'] , c  = normalization(T['Tensor'])
            T0['Tensor'], c0 = normalization(T0['Tensor'])
            T['factor'][factor_key]  = math.log(c) / 2**(sum(rgstep.values()))
            T0['factor'][factor_key] = math.log(c0) - math.log(c)

            #print coordinate of impure tensor T0, Tn
            print("impure tensor Tn have been absorbed!")
            coor0 = f"({T0['loc']['X']},{T0['loc']['Y']},{T0['loc']['Z']})" \
                  + f" -> ({loc0_after['X']},{loc0_after['Y']},{loc0_after['Z']})"
            coorn = f"({Tn['loc']['X']},{Tn['loc']['Y']},{Tn['loc']['Z']})" \
                  + f" -> ({locn_after['X']},{locn_after['Y']},{locn_after['Z']})"
            print(f"T0: {coor0}")
            print(f"Tn: {coorn}")

            T0["loc"][direction] //= 2
            Tn = {}

            print(f"c={c}, c0={c0}")
            t1 = time.time()
            print(f"normalization time= {t1-t0:.2e} s")

    else:
        T, T0, new_impuer_tensor_1point(T, T0, Dcut, direction)

        t0 = time.time()
        T['Tensor'] , c  = normalization(T['Tensor'])
        T0['Tensor'], c0 = normalization(T0['Tensor'])
        T['factor'][factor_key]  = math.log(c) / 2**(sum(rgstep.values()))
        T0['factor'][factor_key] = math.log(c0) - math.log(c)

        #print coordinate of impure tensor T0, Tn
        coor0 = f"({T0['loc']['X']},{T0['loc']['Y']},{T0['loc']['Z']})" \
              + f" -> ({loc0_after['X']},{loc0_after['Y']},{loc0_after['Z']})"
        print(f"T0: {coor0}")

        T0["loc"][direction] //= 2
        Tn = {}

        print(f"c={c}, c0={c0}")
        t1 = time.time()
        print(f"normalization time= {t1-t0:.2e} s")

    return T, T0, Tn

def two_point_function(T:dict, T0:dict, Tn:dict, Dcut:int, TOT_RGSTEPS:dict):
    rgstep = {'X':0, 'Y':0, 'Z':0}

    factor_key = f"({rgstep['X']},{rgstep['Y']},{rgstep['Z']})"
    T['Tensor'] , c  = normalization(T['Tensor'])
    T0['Tensor'], c0 = normalization(T0['Tensor'])
    Tn['Tensor'], c1 = normalization(Tn['Tensor'])

    T['factor'][factor_key]  = math.log(c) / 2**(sum(rgstep.values()))
    T0['factor'][factor_key] = math.log(c0) + math.log(c1) - 2*math.log(c) #c0*c1 / (c**2)

    print(f"c={c}, c0={c0}, cn={c1}")

    cycle = "ZXY"
    for direction in iter.cycle(cycle):
        if sum(rgstep.values()) >= sum(TOT_RGSTEPS.values()):
            break

        #do renormalization steps
        if rgstep[direction] < TOT_RGSTEPS[direction]:
            rgstep[direction] += 1
            print()
            print(f"rgstep(nz,nx,ny)=({rgstep['Z']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization start")
            
            t0 = time.time()
            Tshape_before = T['Tensor'].shape
            T, T0, Tn = exec_rgstep_2point_func(T, T0, Tn, Dcut, direction, rgstep)
            Tshape_after = T['Tensor'].shape
            t1 = time.time()
            print(f"bond dimensions of T: {Tshape_before} -> {Tshape_after}")
            print(f"rgstep(nz,nx,ny)=({rgstep['Z']},{rgstep['X']},{rgstep['Y']}), " \
                  + f"{sum(rgstep.values())}-th renormalization finished")
            print(f"time= {t1-t0:.2e} s")
            
    return T, T0
    #two point function ↑------------------------------------------------------------------------------
