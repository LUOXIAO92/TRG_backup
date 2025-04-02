import numpy as np
import cupy as cp
import opt_einsum as oe

from math import ceil

def initial_tensor_for_ATRG(dim:int,
                            J:cp.ndarray, w:cp.ndarray, 
                            As:cp.ndarray, Bs:cp.ndarray, 
                            At:cp.ndarray, Bt:cp.ndarray, 
                            k:int, p:int, q:int,
                            seed=1234):
    """
    dim: Dimension of the system
    J : Jacobian determinant of integration. 
    w : weights of quadrature rules.
    As: Spacial legs of positive direction As_{iX}, As_{iY}, As_{iZ}.
    Bs: Temporal legs of negative direction Bs_{xi}, As_{yi}, As_{zi}.
    At: Spacial leg of positive direction As_{iT}.
    Bt: Temporal leg of negative direction Bs_{ti}.
    k : Svd rank or bond dimensions of internal degrees of freedom.
    p : Over samples.
    q : Times of iteration.
    seed: Seed for random standard normal compression matrix.

    About As, Bs of At, Bt: for a boltzmann weight M_{ij} between i and j site, we do svd as following
    >>> M_{ij} = U_{ik} s_{k} VH_{kj} = U_{ik} sqrt(s_{k}) sqrt(s_{k}) VH_{kj} = A_{ik} B_{kj}
    >>> A_{ik} = U_{ik} sqrt(s_{k}), k = X, Y, Z, T
    >>> B_{kj} = sqrt(s_{k}) VH_{kj}, k = x, y, z, t

    Example: we do svd as following, where i is the internal degrees of freedom of a initial tensor.
    For a 2-d system or a 4-leg tensor
    >>>        X                 
    >>>        |                 
    >>>        As                    X          
    >>>        |           rsvd      |           
    >>> T--At--i--Bt--t    --->   T--U--k..s..k--VH--t
    >>>        |                                 |   
    >>>       Bs                                 x   
    >>>        |                 
    >>>        x                 

    For a 3-d system or a 6-leg tensor
    >>>  Y     X                 
    >>>   \    |                 
    >>>    As  As                  Y X          
    >>>      \ |           rsvd     \|           
    >>> T--At--i--Bt--t    --->   T--U--k..s..k--VH--t
    >>>        | \                               | \ 
    >>>       Bs  Bs                             x  y
    >>>        |    \            
    >>>        x     y           

    For a 4-d system or a 8-leg tensor
    >>>  Z     Y     X            
    >>>   \    |    /            
    >>>    As  As As                Z Y X         
    >>>      \ | /          rsvd     \|/          
    >>> T--At--i--At--t    --->    T--U--k..s..k--VH--t
    >>>      / | \                               /|\ 
    >>>    Bs  Bs Bs                            x y z
    >>>   /    |    \            
    >>>  x     y     z           
    """

    l = k + p
    chi, chi_i = Bs.shape #bond dimensions of external legs and internal leg
    rs = cp.random.RandomState(seed=seed)

    if dim == 2:
        #Compression tensor
        Ytensor_shape  = (chi, chi, l)
        Ymatrix_shape  = (int(chi**2), l)

        #QR decomposition
        Qtensor_shape  = (chi, chi, l)

        #RSVD
        Xtensor_shape  = (l, chi, chi)
        Xmatrix_shape  = (l, int(chi**2))

        #Contraction path
        TY_path = ["einsum_path", (0, 1), (0, 5), (0, 1), (0, 2), (1, 2), (0, 1)]
        TY_subscripts    = "a,a,aT,aX,ta,xa,txl->TXl"
        TdagQ_subscripts = "a,a,ta,xa,aT,aX,TXl->txl"
        TQ_subscripts    = "a,a,aT,aX,ta,xa,txl->TXl"
        QdagT_subscripts = "a,a,ta,xa,aT,aX,TXl->ltx"

        #ATRG tensor
        u_shape  = (chi, chi, k)
        vh_shape = (k, chi, chi)

    elif dim == 3:
        #Compression tensor
        Ytensor_shape  = (chi, chi, chi, l)
        Ymatrix_shape  = (int(chi**3), l)

        #QR decomposition
        Qtensor_shape  = (chi, chi, chi, l)

        #RSVD
        Xtensor_shape  = (l, chi, chi, chi)
        Xmatrix_shape  = (l, int(chi**3))

        #Contraction path
        TY_path = ["einsum_path", (0, 1), (0, 7), (0, 1), (0, 1), (0, 2), (0, 2), (1, 2), (0, 1)]
        TY_subscripts    = "a,a,aT,aX,aY,ta,xa,ya,txyl->TXYl"
        TdagQ_subscripts = "a,a,ta,xa,ya,aT,aX,aY,TXYl->txyl"
        TQ_subscripts    = "a,a,aT,aX,aY,ta,xa,ya,txyl->TXYl"
        QdagT_subscripts = "a,a,ta,xa,ya,aT,aX,aY,TXYl->ltxy"

        #ATRG tensor
        u_shape  = (chi, chi, chi, k)
        vh_shape = (k, chi, chi, chi)

    elif dim == 4:
        #Compression tensor
        Ytensor_shape  = (chi, chi, chi, chi, l)
        Ymatrix_shape  = (int(chi**4), l)

        #QR decomposition
        Qtensor_shape  = (chi, chi, chi, chi, l)

        #RSVD
        Xtensor_shape  = (l, chi, chi, chi, chi)
        Xmatrix_shape  = (l, int(chi**4))

        #Contraction path
        TY_path = ["einsum_path", (0,1), (0,9), (0,8), (0,1), (0,1), (0,1), (0,3,4), (0,1,2)]
        TY_subscripts    = "a,a,aT,aX,aY,aZ,ta,xa,ya,za,txyzl->TXYZl"
        TdagQ_subscripts = "a,a,ta,xa,ya,za,aT,aX,aY,aZ,TXYZl->txyzl"
        TQ_subscripts    = "a,a,aT,aX,aY,aZ,ta,xa,ya,za,txyzl->TXYZl"
        QdagT_subscripts = "a,a,ta,xa,ya,za,aT,aX,aY,aZ,TXYZl->ltxyz"

        #ATRG tensor
        u_shape  = (chi, chi, chi, chi, k)
        vh_shape = (k, chi, chi, chi, chi)

    Y = rs.standard_normal(size=Ytensor_shape)

    #Calculate compress matrix/tensor Y'_{TXYZl} = T_{TXYZtxyz} Y_{txyzl}
    #Then do qr decomposition Q_{TXYZ,a} @ R_{a,l} = Y'_{TXYZl}
    if   dim == 2:
        Y = oe.contract(TY_subscripts, w, J, At, As, Bt, Bs, Y, optimize=TY_path[1:])
    elif dim == 3:
        Y = oe.contract(TY_subscripts, w, J, At, As, As, Bt, Bs, Bs, Y, optimize=TY_path[1:])
    elif dim == 4:
        Y = oe.contract(TY_subscripts, w, J, At, As, As, As, Bt, Bs, Bs, Bs, Y, optimize=TY_path[1:])
        
    Y = cp.reshape(Y, newshape=Ymatrix_shape)
    Q, _ = cp.linalg.qr(Y)
    Q = cp.reshape(Q, newshape=Qtensor_shape)
    del Y

    #calculate QR iteration
    def qr_iteration(q:int, dim:int, 
                     w:cp.ndarray , J:cp.ndarray, 
                     At:cp.ndarray, Bt:cp.ndarray, 
                     As:cp.ndarray, Bs:cp.ndarray, 
                     Q:cp.ndarray , 
                     Ymatrix_shape:tuple, 
                     Qtensor_shape:tuple, 
                     TdagQ_subscripts:str, TQ_subscripts:str, path:str):
        
        for q_ in range(q):
        
            #Calculate Y'_{txyzl} = T^†_{txyzTXYZ} @ Q_{TXYZl} = T^*_{TXYZtxyz} Q_{TXYZl}
            #Then do qr decomposition Q_{txyz,a} @ R_{a,l} = Y'_{txyzl}
            if   dim == 2:
                Y = oe.contract(TdagQ_subscripts, cp.conj(w) , cp.conj(J), 
                                                  cp.conj(Bt), cp.conj(Bs),
                                                  cp.conj(At), cp.conj(As), 
                                                  Q, optimize=path[1:])
            elif dim == 3:
                Y = oe.contract(TdagQ_subscripts, cp.conj(w) , cp.conj(J), 
                                                  cp.conj(Bt), cp.conj(Bs), cp.conj(Bs), 
                                                  cp.conj(At), cp.conj(As), cp.conj(As), 
                                                  Q, optimize=path[1:])
            elif dim == 4:
                Y = oe.contract(TdagQ_subscripts, cp.conj(w) , cp.conj(J), 
                                                  cp.conj(Bt), cp.conj(Bs), cp.conj(Bs), cp.conj(Bs), 
                                                  cp.conj(At), cp.conj(As), cp.conj(As), cp.conj(As), 
                                                  Q, optimize=path[1:])
            Y = cp.reshape(Y, newshape=Ymatrix_shape)
            Q, _ = cp.linalg.qr(Y)
            Q = cp.reshape(Q, newshape=Qtensor_shape)
            del Y

            #Calculate Y'_{TXYZl} = T_{TXYZtxyz} @ Q_{txyzl} T_{TXYZtxyz} Q_{txyzl} 
            #Then do qr decomposition Q_{TXYZ,a} @ R_{a,l} = Y'_{TXYZl}
            if dim == 2:
                Y = oe.contract(TQ_subscripts, w, J, 
                                               At, As, 
                                               Bt, Bs, 
                                               Q, optimize=path[1:])
            elif dim == 3:
                Y = oe.contract(TQ_subscripts, w, J, 
                                               At, As, As, 
                                               Bt, Bs, Bs, 
                                               Q, optimize=path[1:])
            elif dim == 4:
                Y = oe.contract(TQ_subscripts, w, J, 
                                               At, As, As, As, 
                                               Bt, Bs, Bs, Bs, 
                                               Q, optimize=path[1:])
            Y = cp.reshape(Y, newshape=Ymatrix_shape)
            Q, _ = cp.linalg.qr(Y)
            Q = cp.reshape(Q, newshape=Qtensor_shape)
            del Y

        return Q

    #Do qr iteration. 
    #Q: Q_{TXYZl}
    Q = qr_iteration(q, dim, w, J, 
                     At, Bt, As, Bs, Q, 
                     Ymatrix_shape, Qtensor_shape, 
                     TdagQ_subscripts, TQ_subscripts, TY_path)
    
    #Calculate X_{ltxyz} = Q^†_{l,TXYZ} @ T_{TXYZ,txyz} = Q^*_{TXYZl} @ T_{TXYZtxyz}
    if dim == 2:
        X = oe.contract(QdagT_subscripts, w, J, 
                                          Bt, Bs, 
                                          At, As, 
                                          cp.conj(Q), optimize=TY_path[1:])
    elif dim == 3:
        X = oe.contract(QdagT_subscripts, w, J, 
                                          Bt, Bs, Bs, 
                                          At, As, As, 
                                          cp.conj(Q), optimize=TY_path[1:])
    elif dim == 4:
        X = oe.contract(QdagT_subscripts, w, J, 
                                          Bt, Bs, Bs, Bs, 
                                          At, As, As, As, 
                                          cp.conj(Q), optimize=TY_path[1:])
    X = cp.reshape(X, newshape=Xmatrix_shape)
    u, s, vh = cp.linalg.svd(X, full_matrices=False)
    u, s, vh = u[:,:k], s[:k], vh[:k,:]
    del X

    #u: u_{l,i}
    #s: s_{i}
    #vh: vh_{i,txyz}
    #Calculate u_{TXYZi} = Q_{TXYZl} u_{li}
    if   dim == 2:
        u = oe.contract("TXl,li->TXi", Q, u)
    elif dim == 3:
        u = oe.contract("TXYl,li->TXYi", Q, u)
    elif dim == 4:
        u = oe.contract("TXYZl,li->TXYZi", Q, u)
    
    vh = cp.reshape(vh, newshape=vh_shape)
    del Q

    return u, s, vh