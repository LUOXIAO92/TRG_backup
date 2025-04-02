import time
import numpy as np
import cupy as cp
try:
    import cupy as xp
    import cupyx.scipy.linalg as slag
    _USE_GPU_ = True
except:
    import numpy as xp
    import scipy.linalg as slag
    _USE_GPU_ = False

import opt_einsum as oe
from utility.truncated_svd import svd, eigh

convergence_err = 1e-2
max_iteration = 1000

def __timming__(_USE_GPU_ = True):
    if _USE_GPU_:
        xp.cuda.Stream.null.synchronize()
        t = time.time()
    else:
        t = time.time()
    return t

def optimize_Rp(U, S, gilt_eps):
    """
    U: U_{αβ,i}
    """
    #compute initial Rp
    Rp, Rp_ai, s, Rp_ib = compute_Rp(U, S, gilt_eps, need_svd=True, split=True)
    U_inner = U
    S_inner = S
    us_inner = Rp_ai
    svh_inner = Rp_ib

    time_diff = 0
    count = 0
    global convergence_err
    #compute and truncate bond matrix untill all singular values converge to 1
    while (xp.abs(s-1).max() >= convergence_err) and (count < max_iteration):
        t0 = __timming__(_USE_GPU_ == True)
        #compute inner enviroment tensor and then svd it
        E_inner = oe.contract("ABi,Aa,bB,i->abi", U_inner, us_inner, svh_inner, S_inner)
        U_inner, S_inner, _ = svd(E_inner, shape=[[0,1], [2]])
        S_inner = S_inner / xp.sum(S_inner)
        

        #compute truncated bond matrix and split it into two parts
        _, us_inner, s, svh_inner = compute_Rp(U_inner, S_inner, gilt_eps, need_svd=True, split=True)
        Rp_ai = oe.contract("aA,Ai->ai", Rp_ai, us_inner)
        Rp_ib = oe.contract("iB,Bb->ib", svh_inner, Rp_ib)
        count += 1
        t1 = __timming__(_USE_GPU_ == True)
        time_diff += t1-t0
        if count % 20 == 0:
            #__sparse_check__(E_inner, "inner Env tensor")
            print("iteration:{}, s[:20]:{}, time:{:.2e}s".format(count, s[:20], time_diff))
            time_diff = 0

        del E_inner

    Rp = oe.contract("ai,ib->ab", Rp_ai, Rp_ib)
    del U_inner, S_inner, us_inner, svh_inner, s

    return Rp, count

def compute_Rp(U, S, gilt_eps, need_svd:bool, split=False):
    #compute trace t_i=TrU_i
    t = oe.contract("aai->i", U)

    #compute t'_i = t_i * S_i^2 / (ε^2 + S_i^2)
    if gilt_eps != 0:
        ratio = S / gilt_eps
        weight = ratio**2 / (1 + ratio**2)
        tp = t * weight
    else:
        tp = t

    #compute R'
    Rp = oe.contract("i,abi->ab", tp, xp.conj(U))

    del tp

    #svd R' 
    if need_svd or split:
        u, s, vh = svd(Rp, shape=[[0], [1]], truncate_err=gilt_eps*1e-3, split=split)
        return Rp, u, s, vh
    else:
        return Rp


def Gilt_plaq_2dHOTRG(T0:xp.ndarray, T1:xp.ndarray, gilt_eps, leg:str, direction:str):
    """
    >>>       direction:Y            direction:X      
    >>>       |        |             |        |     
    >>>    ---T0---j---T0---      ---T0---i---T1--- 
    >>>       |        |             |        |     
    >>>       i        k             l        j     
    >>>       |        |             |        |     
    >>>    ---T1---l---T1---      ---T0---k---T1--- 
    >>>       |        |             |        |     
    gilting leg = i,j,k,l
    with tensors T_{x,x',y,y'}
    """

    #compute eigenvalues of environment tensor, S:=S^2
    #     -------
    #     |     |    
    #  |--LU----RU--|
    #  |  |     |   |
    #  |  |     |   |
    #  |--LD----RD--|
    #     |     |    
    #     -------
    if direction == "y" or direction == "Y":
        LD = oe.contract("ixjy,iXjY->xyXY", T1, xp.conj(T1))
        LU = oe.contract("ixyj,iXYj->xyXY", T0, xp.conj(T0))
        RU = oe.contract("xiyj,XiYj->xyXY", T0, xp.conj(T0))
        RD = oe.contract("xijy,XijY->xyXY", T1, xp.conj(T1))
    elif direction == "x" or direction == "X":
        LD = oe.contract("ixjy,iXjY->xyXY", T0, xp.conj(T0))
        LU = oe.contract("ixyj,iXYj->xyXY", T0, xp.conj(T0))
        RU = oe.contract("xiyj,XiYj->xyXY", T1, xp.conj(T1))
        RD = oe.contract("xijy,XijY->xyXY", T1, xp.conj(T1))

    if direction == "y" or direction == "Y": 
        if   leg == "i":
            subscripts = "aαbA,cβdB,cedf,aebf->αβAB"
        elif leg == "j":
            subscripts = "acbd,αcAd,βeBf,aebf->αβAB"
        elif leg == "k":
            subscripts = "acbd,ecfd,eαfA,aβbB->αβAB"
        elif leg == "l":
            subscripts = "αaAb,cadb,cedf,βeBf->αβAB"
    elif direction == "x" or direction == "X": 
        if   leg == "l":
            subscripts = "aαbA,cβdB,cedf,aebf->αβAB"
        elif leg == "i":
            subscripts = "acbd,αcAd,βeBf,aebf->αβAB"
        elif leg == "j":
            subscripts = "acbd,ecfd,eαfA,aβbB->αβAB"
        elif leg == "k":
            subscripts = "αaAb,cadb,cedf,βeBf->αβAB"
    
    Env = oe.contract(subscripts, LD, LU, RU, RD)
    #__sparse_check__(Env, "Env tensor")
    #U, S, _ = svd(Env, shape=[[0,1], [2,3]])
    S, U = eigh(Env, shape=[[0,1], [2,3]])
    S[S < 0] = 0
    S = xp.sqrt(S)
    #print("Env", cp.abs(Env))
    #print("min Env singularval", S[max(0,len(S)-10):len(S)])
    S = S / xp.sum(S)
    del LD, LU, RU, RD, Env

    Rp, count = optimize_Rp(U, S, gilt_eps)
    #print("Rp is nan", Rp)
    uRp, sRp, vRph = svd(Rp, shape=[[0], [1]], truncate_err=gilt_eps*1e-3)
    global convergence_err
    done = xp.abs(sRp-1).max() < convergence_err
    Rp_ai = oe.contract("ai,i->ai", uRp,  xp.sqrt(sRp))
    Rp_ib = oe.contract("ib,i->ib", vRph, xp.sqrt(sRp))
    print("sRp:", sRp)
    err = gilt_error(U, S, Rp_ai, Rp_ib)

    if gilt_eps != 0:
        if direction == "y" or direction == "Y": 
            if leg == "i":
                T1 = oe.contract("xXya,ai->xXyi", T1, Rp_ai)
                T0 = oe.contract("xXbY,ib->xXiY", T0, Rp_ib)
            elif leg == "j":
                T0 = oe.contract("xayY,ai->xiyY", T0, Rp_ai) 
                T0 = oe.contract("bXyY,ib->iXyY", T0, Rp_ib)
            elif leg == "k":
                T0 = oe.contract("xXaY,ai->xXiY", T0, Rp_ai)
                T1 = oe.contract("xXyb,ib->xXyi", T1, Rp_ib)
            elif leg == "l":
                T1 = oe.contract("xayY,ai->xiyY", T1, Rp_ai) 
                T1 = oe.contract("bXyY,ib->iXyY", T1, Rp_ib)
        elif direction == "x" or direction == "X": 
            if leg == "i":
                T0 = oe.contract("xayY,ai->xiyY", T0, Rp_ai)
                T1 = oe.contract("bXyY,ib->iXyY", T1, Rp_ib)
            elif leg == "j":
                T1 = oe.contract("xXaY,ai->xXiY", T1, Rp_ai)
                T1 = oe.contract("xXyb,ib->xXyi", T1, Rp_ib)
            elif leg == "k":
                T0 = oe.contract("xayY,ai->xiyY", T0, Rp_ai)
                T1 = oe.contract("bXyY,ib->iXyY", T1, Rp_ib)
            elif leg == "l":
                T0 = oe.contract("xXya,ai->xXyi", T0, Rp_ai) 
                T0 = oe.contract("xXbY,ib->xXiY", T0, Rp_ib)
    
    return T0, T1, err, done, count
    
def Gilt_plaq_2dTRG(T1:xp.ndarray, T2:xp.ndarray, gilt_eps, leg:str):
    """
    >>>     d        f        
    >>>     |    j   |        
    >>> c---T2-------T1---e   
    >>>     |        |        
    >>>    i|        |k       
    >>>     |        |        
    >>> a---T1-------T2---g   
    >>>     |    l   |        
    >>>     b        h        
    gilting leg = i,j,k,l
    with tensors T_{x,y,x',y'}
    """

    #compute eigenvalues of environment tensor, S:=S^2
    LD = oe.contract("ijxy,ijXY->xyXY", T1, xp.conj(T1))
    LU = oe.contract("iyxj,iYXj->yxYX", T2, xp.conj(T2))
    RU = oe.contract("xyij,XYij->xyXY", T1, xp.conj(T1))
    RD = oe.contract("xijy,XijY->xyXY", T2, xp.conj(T2))

    if   leg == "i":
        subscripts = "aibk,jcld,cedf,aebf->ijkl"
    elif leg == "j":
        subscripts = "abcd,bidk,jelf,aecf->ijkl"
    elif leg == "k":
        subscripts = "abcd,bedf,eifk,ajcl->ijkl"
    elif leg == "l":
        subscripts = "iakb,acbd,cedf,jelf->ijkl"

    Env = oe.contract(subscripts, LD, LU, RU, RD)
    U, S, _ = svd(Env, shape=[[0,1], [2,3]])
    S = xp.sqrt(S)
    S = S / xp.sum(S)
    del LD, LU, RU, RD

    #compute optimized R' and split it to 2 parts
    Rp, count = optimize_Rp(U, S, gilt_eps)
    uRp, sRp, vRph = svd(Rp, shape=[[0], [1]], truncate_err=gilt_eps*1e-3)
    global convergence_err
    done = xp.abs(sRp-1).max() < convergence_err
    Rp_ai = oe.contract("ai,i->ai", uRp,  xp.sqrt(sRp))
    Rp_ib = oe.contract("ib,i->ib", vRph, xp.sqrt(sRp))

    print("sRp:", sRp)

    err = gilt_error(U, S, Rp_ai, Rp_ib)

    #apply gilt
    if   leg == "i":
        T1 = oe.contract("xyXa,ai->xyXi", T1, Rp_ai)
        T2 = oe.contract("xbXY,ib->xiXY", T2, Rp_ib)
    elif leg == "j":
        T2 = oe.contract("xyaY,ai->xyiY", T2, Rp_ai)
        T1 = oe.contract("byXY,ib->iyXY", T1, Rp_ib)
    elif leg == "k":
        T1 = oe.contract("xaXY,ai->xiXY", T1, Rp_ai)
        T2 = oe.contract("xyXb,ib->xyXi", T2, Rp_ib)
    elif leg == "l":
        T1 = oe.contract("xyaY,ai->xyiY", T1, Rp_ai)
        T2 = oe.contract("byXY,ib->iyXY", T2, Rp_ib)

    return T1, T2, err, done, count

def Gilt_chain_2dHOTRG(T0:xp.ndarray, T1:xp.ndarray, gilt_eps, leg:str, direction:str):
    """
    >>>     <-----
    >>>     |    |i   |j   |
    >>>  ---T0---T1---T0---T1---
    >>>     |    |i   |j   |
    """
    if direction == "y" or direction == "Y":
        if leg == "i":
            subscripts = "αβab,cdbe,ABaf,cdfe->αβAB"
        elif leg == "j":
            subscripts = "abcd,αβde,abcf,ABfe->αβAB"
        else:
            raise ValueError("No such leg!")
    elif direction == "x" or direction == "X":
        if leg == "i":
            subscripts = "abαβ,bcde,afAB,fcde->αβAB"
        elif leg == "j":
            subscripts = "abcd,beαβ,afcd,feAB->αβAB"
    else:
        raise ValueError("No such direction!")
    Env = oe.contract(subscripts, T0, T1, xp.conj(T0), xp.conj(T1))
    #Env = oe.contract(subscripts, T1, T0, xp.conj(T1), xp.conj(T0))
    #U, S, _ = svd(Env, shape=[[0,1], [2,3]])
    S, U = eigh(Env, shape=[[0,1], [2,3]])
    S = xp.sqrt(S)
    S = S / xp.sum(S)
    del Env

    Rp, count = optimize_Rp(U, S, gilt_eps)
    uRp, sRp, vRph = svd(Rp, shape=[[0], [1]], truncate_err=gilt_eps*1e-3)
    global convergence_err
    done = xp.abs(sRp-1).max() < convergence_err
    Rp_ai = oe.contract("ai,i->ai", uRp,  xp.sqrt(sRp))
    Rp_ib = oe.contract("ib,i->ib", vRph, xp.sqrt(sRp))
    print("sRp:", sRp)
    err = gilt_error(U, S, Rp_ai, Rp_ib)

    #if gilt_eps != 0:
    if direction == "y" or direction == "Y":
        if leg == "i":
            #T1 = oe.contract("aXyY,ai->iXyY", T1, Rp_ai)
            #T1 = oe.contract("xbyY,ib->xiyY", T1, Rp_ib)

            T0 = oe.contract("aXyY,ai->iXyY", T0, Rp_ai)
            T0 = oe.contract("xbyY,ib->xiyY", T0, Rp_ib)
        elif leg == "j":
            #T0 = oe.contract("aXyY,ai->iXyY", T0, Rp_ai)
            #T0 = oe.contract("xbyY,ib->xiyY", T0, Rp_ib)

            T1 = oe.contract("aXyY,ai->iXyY", T1, Rp_ai)
            T1 = oe.contract("xbyY,ib->xiyY", T1, Rp_ib)

    elif direction == "x" or direction == "X":
        if leg == "i":
            #T1 = oe.contract("xXaY,ai->xXiY", T1, Rp_ai)
            #T1 = oe.contract("xXyb,ib->xXyi", T1, Rp_ib)

            T0 = oe.contract("xXaY,ai->xXiY", T0, Rp_ai)
            T0 = oe.contract("xXyb,ib->xXyi", T0, Rp_ib)
        elif leg == "j":
            #T0 = oe.contract("xXaY,ai->xXiY", T0, Rp_ai)
            #T0 = oe.contract("xXyb,ib->xXyi", T0, Rp_ib)

            T1 = oe.contract("xXaY,ai->xXiY", T1, Rp_ai)
            T1 = oe.contract("xXyb,ib->xXyi", T1, Rp_ib)

    return T0, T1, err, done, count

def gilt_error(U, S, Rp_ai, Rp_ib):
    t = xp.einsum("iij->j", U)
    tp = xp.einsum("abt,ai,ib->t", U, Rp_ai, Rp_ib)
    diff = t-tp
    diff = diff*S
    err = xp.linalg.norm(diff) / xp.linalg.norm(t*S)
    return err

def hermit_err(A, shape):
    if len(shape) != 2:
        print(f"len of {shape} should be 2")
        import sys
        sys.exit()
    
    a = xp.transpose(A.copy(), axes=tuple(shape[0]+shape[1]))

    from functools import reduce
    from operator import mul
    llen = len(shape[0])
    rlen = len(shape[1])
    m = reduce(mul, a.shape[:llen])
    n = reduce(mul, a.shape[rlen:])
    a = xp.reshape(a, newshape=(m,n))
    a_dag = xp.conj(a.T)

    err = xp.linalg.norm(a - a_dag) / xp.linalg.norm(a)

    return err


def gilt_plaq_2dHOTRG(T0, T1, gilt_eps=1e-7, direction="y", gilt_legs=2):

    if gilt_eps < 1e-12:
        return T0, T1

    gilt_err = 0
    from itertools import cycle
    if gilt_legs == 2:
        legs = 'jl'
    elif gilt_legs == 4:
        legs = 'ijkl'
    done_legs = {leg:False for leg in legs}
    for leg in cycle(legs):
        T0_shape0, T1_shape0 = T0.shape, T1.shape
        t0 = __timming__(_USE_GPU_ == True)
        T0, T1, err, done, count = Gilt_plaq_2dHOTRG(T0, T1, gilt_eps=gilt_eps, leg=leg, direction=direction)
        t1 = __timming__(_USE_GPU_ == True)
        T0_shape1, T1_shape1 = T0.shape, T1.shape
        gilt_err += err
        print("T0:{}->{}\nT1:{}->{}\nleg:{}, gilt err= {:.6e}, iteration count:{:}, done:{}, time:{:.2e}s"\
              .format(T0_shape0, T0_shape1, T1_shape0, T1_shape1, leg, gilt_err, count, done, t1-t0))
        done_legs[leg] = True
    
        if all(done_legs.values()):
            break
    
    return T0, T1

def gilt_chain_2dHOTRG(T0, T1, gilt_eps=1e-7, direction="y"):

    gilt_err = 0
    from itertools import cycle
    legs = 'ij'
    done_legs = {leg:False for leg in legs}
    for leg in cycle(legs):
        T0_shape0, T1_shape0 = T0.shape, T1.shape
        t0 = __timming__(_USE_GPU_ == True)
        T0, T1, err, done, count = Gilt_chain_2dHOTRG(T0, T1, gilt_eps=gilt_eps, leg=leg, direction=direction)
        t1 = __timming__(_USE_GPU_ == True)
        T0_shape1, T1_shape1 = T0.shape, T1.shape
        gilt_err += err
        print("T0:{}->{}\nT1:{}->{}\nleg:{}, gilt err= {:.6e}, iteration count:{:}, done:{}, time:{:.2e}s"\
              .format(T0_shape0, T0_shape1, T1_shape0, T1_shape1, leg, gilt_err, count, done, t1-t0))
        done_legs[leg] = True
    
        if all(done_legs.values()):
            break
    
    return T0, T1


def gilt_plaq_TRG(B, A, gilt_eps):

    gilt_err = 0
    if True:
        from itertools import cycle
        legs = 'ijkl'
        done_legs = {leg:False for leg in legs}
        for leg in cycle(legs):
            A_shape0, B_shape0 = A.shape, B.shape
            t0 = __timming__(_USE_GPU_ == True)
            B, A, err, done, count = Gilt_plaq_2dTRG(B, A, gilt_eps=gilt_eps, leg=leg)
            t1 = __timming__(_USE_GPU_ == True)
            done_legs[leg] = done
            A_shape1, B_shape1 = A.shape, B.shape
            gilt_err += err
            print("A:{}->{}\nB:{}->{}\ngilt err= {:.6e}, iteration count:{}, done:{}, time:{:.2e}s"\
                  .format(A_shape0,A_shape1,B_shape0,B_shape1, gilt_err, count, done, t1-t0))
    
            if all(done_legs.values()):
                break

    return B, A



def gilt_plaq_2dHOTRG_22222(T0, T1, gilt_eps, direction, legcut):
    from trg.__gilts__ import cutLeg, applyRp, gilt_err, gilt_hotrgplaq
    from abeliantensors import Tensor

    T0 = oe.contract("xXyY->xYXy", T0).get()
    T1 = oe.contract("xXyY->xYXy", T1).get()
    T0 = Tensor.from_ndarray(T0)
    T1 = Tensor.from_ndarray(T1)
    print(T0.shape)
    direction = 'v' if direction=='y' or direction=='Y' else 'h'
    T0, T1, RABs = gilt_hotrgplaq(T0, T1, epsilon=gilt_eps, convergence_eps=0.01, 
                                  direction=direction, legcut=legcut)
    
    T0 = T0.to_ndarray()
    T1 = T1.to_ndarray()
    T0 = xp.asarray(T0)
    T1 = xp.asarray(T1)
    T0 = oe.contract("xYXy->xXyY", T0)
    T1 = oe.contract("xYXy->xXyY", T1)

    return T0, T1