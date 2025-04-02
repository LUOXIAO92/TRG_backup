import time
import numpy as np
import cupy as cp
from itertools import cycle
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

def __timming__(_USE_GPU_ = True):
    if _USE_GPU_:
        xp.cuda.Stream.null.synchronize()
        t = time.time()
    else:
        t = time.time()
    return t

def Environment_plaq_2d(T0:xp.ndarray, T1:xp.ndarray, leg:str, scheme:str, direction="y", env_only=False):
    """
    ------------------------HOTRG-------------------------
    >>>                                 <-- x         
    >>>       |    j   |           i |        |  i  
    >>>  y ---T1-------T1---      ---T0-------T1--- 
    >>>  |    |        |             |        |     
    >>>  |  i |        | k         l |        | j   
    >>>  V    |        |             |        |     
    >>>    ---T0-------T0---      ---T0-------T1--- 
    >>>       |    l   |           k |        |  k  
    >>>    T_{x,x',y,y'}
    ------------------------------------------------------

    --------------------------TRG-------------------------
    >>>     d        f        
    >>>     |    j   |        
    >>> c---T1-------T0---e   
    >>>     |        |        
    >>>    i|        |k       
    >>>     |        |        
    >>> a---T0-------T1---g   
    >>>     |    l   |        
    >>>     b        h        
    >>>    T_{x,y,x',y'}
    ------------------------------------------------------
    
    return Env_{αβAB}, U_{αβi}, S_i
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

    if scheme == "hotrg":
        if direction == "y" or direction == "Y":
            LD = oe.contract("ixjy,iXjY->xyXY", T0, xp.conj(T0))
            LU = oe.contract("ixyj,iXYj->xyXY", T1, xp.conj(T1))
            RU = oe.contract("xiyj,XiYj->xyXY", T1, xp.conj(T1))
            RD = oe.contract("xijy,XijY->xyXY", T0, xp.conj(T0))
        elif direction == "x" or direction == "X":
            LD = oe.contract("ixjy,iXjY->xyXY", T1, xp.conj(T1))
            LU = oe.contract("ixyj,iXYj->xyXY", T1, xp.conj(T1))
            RU = oe.contract("xiyj,XiYj->xyXY", T0, xp.conj(T0))
            RD = oe.contract("xijy,XijY->xyXY", T0, xp.conj(T0))

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
    
    elif scheme == "trg":
        LD = oe.contract("ijxy,ijXY->xyXY", T0, xp.conj(T0))
        LU = oe.contract("iyxj,iYXj->yxYX", T1, xp.conj(T1))
        RU = oe.contract("xyij,XYij->xyXY", T0, xp.conj(T0))
        RD = oe.contract("xijy,XijY->xyXY", T1, xp.conj(T1))

        if   leg == "i":
            subscripts = "aibk,jcld,cedf,aebf->ijkl"
        elif leg == "j":
            subscripts = "abcd,bidk,jelf,aecf->ijkl"
        elif leg == "k":
            subscripts = "abcd,bedf,eifk,ajcl->ijkl"
        elif leg == "l":
            subscripts = "iakb,acbd,cedf,jelf->ijkl"

    else:
        raise ValueError(f"scheme {scheme} is not support yet")

    Env = oe.contract(subscripts, LD, LU, RU, RD)

    if not env_only:
        U, S, _ = svd(Env, shape=[[0,1], [2,3]])
        S = xp.sqrt(S)
        return Env, U, S
    else:
        return Env

def apply_Rp_2d(T0, T1, Rp_ai, Rp_ib, leg, scheme:str, direction="y"):
    if scheme == "hotrg":
        if direction == "y" or direction == "Y": 
            if leg == "i":
                T0 = oe.contract("xXya,ai->xXyi", T0, Rp_ai)
                T1 = oe.contract("xXbY,ib->xXiY", T1, Rp_ib)
            elif leg == "j":
                T1 = oe.contract("xayY,ai->xiyY", T1, Rp_ai)
                T1 = oe.contract("bXyY,ib->iXyY", T1, Rp_ib)
            elif leg == "k":
                T1 = oe.contract("xXaY,ai->xXiY", T1, Rp_ai)
                T0 = oe.contract("xXyb,ib->xXyi", T0, Rp_ib)
            elif leg == "l":
                T0 = oe.contract("xayY,ai->xiyY", T0, Rp_ai) 
                T0 = oe.contract("bXyY,ib->iXyY", T0, Rp_ib)
        elif direction == "x" or direction == "X": 
            if leg == "i":
                T1 = oe.contract("xayY,ai->xiyY", T1, Rp_ai)
                T0 = oe.contract("bXyY,ib->iXyY", T0, Rp_ib)
            elif leg == "j":
                T0 = oe.contract("xXaY,ai->xXiY", T0, Rp_ai)
                T0 = oe.contract("xXyb,ib->xXyi", T0, Rp_ib)
            elif leg == "k":
                T1 = oe.contract("xayY,ai->xiyY", T1, Rp_ai)
                T0 = oe.contract("bXyY,ib->iXyY", T0, Rp_ib)
            elif leg == "l":
                T1 = oe.contract("xXya,ai->xXyi", T1, Rp_ai) 
                T1 = oe.contract("xXbY,ib->xXiY", T1, Rp_ib)

    elif scheme == "trg":
        if   leg == "i":
            T0 = oe.contract("xyXa,ai->xyXi", T0, Rp_ai)
            T1 = oe.contract("xbXY,ib->xiXY", T1, Rp_ib)
        elif leg == "j":
            T1 = oe.contract("xyaY,ai->xyiY", T1, Rp_ai)
            T0 = oe.contract("byXY,ib->iyXY", T0, Rp_ib)
        elif leg == "k":
            T0 = oe.contract("xaXY,ai->xiXY", T0, Rp_ai)
            T1 = oe.contract("xyXb,ib->xyXi", T1, Rp_ib)
        elif leg == "l":
            T0 = oe.contract("xyaY,ai->xyiY", T0, Rp_ai)
            T1 = oe.contract("byXY,ib->iyXY", T1, Rp_ib)

    else:
        raise ValueError(f"scheme {scheme} is not support yet")

    return T0, T1

def optimize_Rp(U, S, gilt_eps, max_iteration=1000):
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
            print("iteration:{}, s[-10:-1]:{}, time:{:.2e}s".format(count, s[-10:-1], time_diff))
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
    

#---------------------GILT-------------------------
def gilt_error(U, S, Rp_ai, Rp_ib):
    t = xp.einsum("iij->j", U)
    tp = xp.einsum("abt,ai,ib->t", U, Rp_ai, Rp_ib)
    diff = t-tp
    diff = diff*S
    err = xp.linalg.norm(diff) / xp.linalg.norm(t*S)
    return err
    
def cutlegs_2dGILT(T0:xp.ndarray, T1:xp.ndarray, gilt_eps:float, leg:str, scheme:str, direction="y"):
    _, U, S = Environment_plaq_2d(T0, T1, leg, scheme, direction)
    S = S / xp.sum(S)

    Rp, count = optimize_Rp(U, S, gilt_eps)
    uRp, sRp, vRph = svd(Rp, shape=[[0], [1]], truncate_err=gilt_eps*1e-3)
    global convergence_err
    done = xp.abs(sRp-1).max() < convergence_err
    Rp_ai = oe.contract("ai,i->ai", uRp,  xp.sqrt(sRp))
    Rp_ib = oe.contract("ib,i->ib", vRph, xp.sqrt(sRp))

    print("sRp:", sRp)
    err = gilt_error(U, S, Rp_ai, Rp_ib)
    T0, T1 = apply_Rp_2d(T0, T1, Rp_ai, Rp_ib, leg, scheme, direction)
    
    return T0, T1, err, done, count
#---------------------GILT END----------------------


#----------------------FET--------------------------
def normEnvironment(Env):
    norm_fact = oe.contract("iijj", Env)
    Env_norm = Env / norm_fact
    return Env_norm, norm_fact

def fidelity(Env, u, s, vh):
    """
    Env should be normalized \\
    return f, err=1-f, phipsi, psiphi, phiphi
    """
    R = oe.contract("αi,i,iβ->αβ", u, s, vh)
    phipsi = oe.contract("αβAA,αβ", Env, R)
    psiphi = oe.contract("ααAB,AB", Env, xp.conj(R))
    phiphi = oe.contract("αβAB,αβ,AB", Env, R, xp.conj(R))
    
    f = phipsi*psiphi / phiphi
    err = 1 - f
    return f, err, phipsi, psiphi, phiphi

def inv(B):
    B_shape = B.shape
    B = xp.reshape(B, newshape=(B_shape[0]*B_shape[1], B_shape[2]*B_shape[3]))
    e, u = xp.linalg.eigh(B)
    einv = 1 / e
    Binv = oe.contract("ij,j,kj->ik", u, einv, xp.conj(u))

    BBinv = oe.contract("ij,ji", B, Binv) / len(e)
    print(f"BB^-1={BBinv:.6e}")

    Binv = xp.reshape(Binv, newshape=(B_shape[2], B_shape[3], B_shape[0], B_shape[1]))
    B = xp.reshape(B, newshape=B_shape)

    #e, u, err = eigh(B, shape=[[0,1],[2,3]], truncate_err=1e-16, return_err=True)
    #e_inv = 1 / e
    #Binv = oe.contract("IBa,a,iβa->IBiβ", u, e_inv, xp.conj(u))

    #UUinv = oe.contract("iβa,iβa", u, xp.conj(u)) / len(e)
    #UinvU = oe.contract("iβa,iβa", xp.conj(u), u) / len(e)
    #BBinv = oe.contract("iβIB,IBiβ", B, Binv) / len(e)
    #print(f"BB^-1={BBinv:.6e}, UU^-1={UUinv:.6e}, eigh_err={err:.6e}")
    return Binv

def optimizeFET(Env, u, s, vh, whichfix):
    #Env_norm, c = normEnvironment(Env)
    #print(f"Env normalization factor{c}")
    if whichfix == "u":
        P = oe.contract("jjAB,AI->IB", Env, xp.conj(u))
        B = oe.contract("αβAB,αi,AI->iβIB", Env, u, xp.conj(u))

        Binv = inv(B)
        R = oe.contract("IB,IBiβ->iβ", P, Binv)

        uR = oe.contract("αi,iβ->αβ", u, R)
        up, sp, vhp = svd(uR, shape=[[0],[1]], truncate_err=1e-10)

    elif whichfix == "v":
        P = oe.contract("jjAB,IB->AI", Env, xp.conj(vh))
        B = oe.contract("αβAB,iβ,IB->αiAI", Env, vh, xp.conj(vh))

        Binv = inv(B)
        R = oe.contract("AI,AIαi->αi", P, Binv)

        Rvh = oe.contract("αi,iβ->αβ", R, vh)
        up, sp, vhp = svd(Rvh, shape=[[0],[1]], truncate_err=1e-10)

    return up, sp, vhp

def cutlegs_2dFET(T0:xp.ndarray, T1:xp.ndarray, leg:str, scheme:str, maxiter:int, direction='y'):
    Env, U, S = Environment_plaq_2d(T0, T1, leg, scheme, direction, env_only=False)

    #normalize environment matrix
    Env, norm_fact = normEnvironment(Env)

    #initialize bond matrix
    S /= xp.max(S)
    initgliteps = S[len(S)//2]
    initgliteps = min(initgliteps, 1e-4)
    print(f"init gilteps={initgliteps:.6e}")
    #U, S, _ = svd(Env, shape=[[0,1],[2,3]])
    S = S / xp.sum(S)
    R, count = optimize_Rp(U, S, initgliteps, 5)
    u0, s0, vh0 = svd(R, shape=[[0],[1]], truncate_err=1e-10)
    print("init gilt count", count)

    #calculate initial error
    f0, err0, phipsi, psiphi, phiphi = fidelity(Env, u0, s0, vh0)

    uvcycle = cycle("uv")
    whichfix = "u"
    for k in range(maxiter):

        #u1, s1, vh1 = optimizeFET(Env, u0, s0, vh0, whichfix)
        #f1, err1, phipsi, psiphi, phiphi = fidelity(Env, u1, s1, vh1)

        whichfix = next(uvcycle)
        u0, s0, vh0 = optimizeFET(Env, u0, s0, vh0, whichfix)
        f1, err1, phipsi, psiphi, phiphi = fidelity(Env, u0, s0, vh0)
        print(f"count:{k+1}, fix{whichfix}, err={err1}, fidelity={f1}, \ns={s0}")

        if (np.abs(err0) < np.abs(err1)) and (k == 0):
            print(f"initgilt is enougth, err={err0:.2e}")
            break

        if np.abs(err1) < 1e-12:
            print("error has converged, exit the loop")
            k += 1
            break

        if np.abs(err0-err1)/np.abs(err0) < convergence_err:
            print("change of error is small, exit the loop")
            k += 1
            break

        whichfix = next(uvcycle)
        #u0, s0, vh0 = u1, s1, vh1
        err0 = err1

    #s0 = s0 * (((phipsi + psiphi) / (2*phiphi)).real)
    f, err, phipsi, psiphi, phiphi = fidelity(Env, u0, s0, vh0)
    print(f"<Φ|Ψ>-<Ψ|Φ>= {phipsi-psiphi:.12e} , fidelity={f:.4e}")
    leastsq_err = 1 - phipsi - psiphi + phiphi

    Rp_ai = oe.contract("αi,i->αi", u0,  xp.sqrt(s0))
    Rp_ib = oe.contract("iβ,i->iβ", vh0, xp.sqrt(s0))
    T0, T1 = apply_Rp_2d(T0, T1, Rp_ai, Rp_ib, leg, scheme, direction)

    print(f"sp:{s0}")

    return T0, T1, err, leastsq_err, k

def legoptimize_2dHOTRG(T0, T1, gilt_eps=1e-7, direction="y", gilt_legs=2, cut_scheme="gilt", maxiter=20):
    loop_err = 0
    from itertools import cycle
    if gilt_legs == 2:
        legs = 'jl'
    elif gilt_legs == 4:
        legs = 'ijkl'

    done_legs = {leg:False for leg in legs}
    for leg in cycle(legs):
        T0_shape0, T1_shape0 = T0.shape, T1.shape
        t0 = __timming__(_USE_GPU_ == True)

        if cut_scheme == "gilt":
            T0, T1, err, done, count = cutlegs_2dGILT(T0, T1, gilt_eps=gilt_eps, leg=leg, 
                                                      scheme="hotrg", direction=direction)
        elif cut_scheme == "FET":
            T0, T1, ferr, err, count = cutlegs_2dFET(T0, T1, leg=leg, scheme="hotrg", 
                                                     maxiter=maxiter, direction=direction)
            done = (count < maxiter)

        t1 = __timming__(_USE_GPU_ == True)
        T0_shape1, T1_shape1 = T0.shape, T1.shape
        loop_err += err
        print("T0:{}->{}\nT1:{}->{}\nleg:{}, gilt err= {:.6e}, iteration count:{:}, done:{}, time:{:.2e}s"\
              .format(T0_shape0, T0_shape1, T1_shape0, T1_shape1, leg, loop_err, count, done, t1-t0))
        done_legs[leg] = True

        if all(done_legs.values()):
            break
    
    return T0, T1

def legoptimize_2dTRG(B, A, gilt_eps=1e-7, cut_scheme="gilt", maxiter=20):

    gilt_err = 0
    from itertools import cycle
    legs = 'ijkl'
    done_legs = {leg:False for leg in legs}
    for leg in cycle(legs):
        A_shape0, B_shape0 = A.shape, B.shape
        t0 = __timming__(_USE_GPU_ == True)

        if cut_scheme == "gilt":
            B, A, err, done, count = cutlegs_2dGILT(B, A, gilt_eps=gilt_eps, leg=leg, 
                                                    scheme="trg")
        elif cut_scheme == "FET":
            B, A, ferr, err, count = cutlegs_2dFET(B, A, leg=leg, scheme="trg", 
                                                   maxiter=maxiter)
            done = (err < 1e-3)
            done_legs[leg] = done
        t1 = __timming__(_USE_GPU_ == True)
        A_shape1, B_shape1 = A.shape, B.shape
        gilt_err += err
        print("A:{}->{}\nB:{}->{}\ngilt err= {:.6e}, iteration count:{}, done:{}, time:{:.2e}s"\
              .format(A_shape0,A_shape1,B_shape0,B_shape1, gilt_err, count, done, t1-t0))
    
        if all(done_legs.values()):
            break

    return B, A

from abeliantensors import Tensor
from trg.__gilts__ import fetCut, fidelity, applyRp
def legoptimize_2dTRG_FET(T0:xp.ndarray, T1:xp.ndarray, maxiter:int):
    
    chitid = 64
    maxiter = maxiter
    initscheme = "Gilt"
    giltdeg = 0.5

    T0 = oe.contract("xyXY->xYXy", T0)
    T1 = oe.contract("xyXY->xYXy", T1)
    T0 = T0.get()
    T1 = T1.get()
    T0 = Tensor.from_ndarray(T0)
    T1 = Tensor.from_ndarray(T1)

    loop_red_err = 0.0
    RABs = [1] * 6
    ord_leg = cycle(['II','IV','I','III'])
    done_legs = {leg:False for leg in ['II','IV','I','III']}
    for leg in ord_leg:
        A_shape0, B_shape0 = T1.shape, T0.shape
        t0 = __timming__(_USE_GPU_ == True)


        up, sp, vp, err, loop_red_err_cur, N_iter = fetCut(T1, T0, leg, chitid, maxiter, forwhat = "trg",
                            initscheme = initscheme, giltdeg = giltdeg, verbose=True)
        
        spSqrt = sp.abs().sqrt()
        Ruprime = up.multiply_diag(spSqrt, axis = 1, direction = 'r')
        Rvprime = vp.multiply_diag(spSqrt, axis = 0, direction = 'l')
        # Absorb Rup and Rvp into on our tensors A and B
        T1,T0 = applyRp(T1,T0,Ruprime, Rvprime, leg, forwhat = 'trg',
                      RABs = RABs)
        loop_red_err += loop_red_err_cur

        done = (sp/sp.max() - 1).abs().max() < 1e-2
        done_legs[leg] = done

        t1 = __timming__(_USE_GPU_ == True)
        A_shape1, B_shape1 = T1.shape, T0.shape

        print("A:{}->{}\nB:{}->{}\niteration count:{}, loop_red_err= {:.6e}, err:{}, time:{:.2e}s"\
              .format(A_shape0,A_shape1,B_shape0,B_shape1, N_iter, loop_red_err, err, t1-t0))

        if all(done_legs.values()):
            break

    T0 = T0.to_ndarray()
    T1 = T1.to_ndarray()
    T0 = xp.asarray(T0)
    T1 = xp.asarray(T1)
    T0 = oe.contract("xYXy->xyXY", T0)
    T1 = oe.contract("xYXy->xyXY", T1)

    return T0, T1