import cupy as cp
import opt_einsum as oe
from opt_einsum import contract

def rsvd(A, k:int, n_oversamples=0, n_power_iter=0, iterator="QR", seed=1234):
    """
    A: MxN array\\
    k:rank\\
    n_oversamples+k=l << min{M,N} must be satisfied\\
    n_power_iter: iteration times\\
    iterator: "QR" and "power", "QR" is recommend
    seed:default is numpy.random.RandomState
    """
    rs = cp.random.RandomState(seed)

    M, N = (cp.shape(A)[0], cp.shape(A)[1])
    l = k + n_oversamples

    Y = rs.standard_normal(size=(N, l))
    Y = A @ Y
    
    if iterator == "power":
        Q = __power_iteration__(A, Y, n_power_iter)
    elif iterator == "QR":
        Q = __QR_iteration__(A, Y, n_power_iter)

    if A.dtype == cp.complex64 or A.dtype == cp.complex128:
        B = cp.conj(cp.transpose(Q)) @ A
    else:
        B = cp.transpose(Q) @ A

    u_tilde, s, vh = cp.linalg.svd(B, full_matrices=False)
    del B
    u = Q @ u_tilde
    del u_tilde

    return u[:,:k], s[:k], vh[:k,:]

def rsvd2(A:cp.ndarray, k:int, n_oversamples=0, seed=1, del_A=False):
    """
    A: MxN array\\
    k: rank\\
    n_oversamples+k=l << min{M,N} must be satisfied\\
    n_power_iter: iteration times\\
    iterator: "QR" and "power", "QR" is recommend
    seed:default is numpy.random.RandomState
    """
    #if seed == NULL:
    #    seed = cp.random.RandomState(np.random.RandomState)

    rs = cp.random.RandomState(seed)

    M, N = A.shape
    l = k + n_oversamples

    Y = rs.standard_normal(size=(N, l))
    Y = A @ Y
    
    Q, _ = cp.linalg.qr(Y)
    B = cp.conj(cp.transpose(Q)) @ A

    if del_A:
        del A

    u_tilde, s, vh = cp.linalg.svd(B, full_matrices=False)
    del B
    u = Q @ u_tilde
    del u_tilde

    return u[:,:k], s[:k], vh[:k,:]

def __power_iteration__(A, Y, n_power_iter):
    if A.dtype == cp.complex64 or A.dtype == cp.complex128:
        for q in range(n_power_iter):
            Y = A @ (cp.conj(cp.transpose(A)) @ Y)
    else:
        for q in range(n_power_iter):
            Y = A @ (cp.transpose(A) @ Y)
    Q, _ = cp.linalg.qr(Y)

    return Q

def __QR_iteration__(A, Y, n_power_iter):
    Q, _ = cp.linalg.qr(Y)

    for _ in range(n_power_iter):
        Y = cp.conj(A.T) @ Q
        Q, _ = cp.linalg.qr(Y)

        Y = A @ Q
        Q, _ = cp.linalg.qr(Y)

    del Y

    return Q



#RSVD for atrg tensor initialized
def rsvd_for_3dATRG_tensor_init(A, B, Dcut:int, n_oversamples=10, n_power_iter=0, iterator="QR", seed=cp.random.RandomState(seed=1234)):

    def __QR_iteration_4atrg__(A, B, Y, Dcut, l, n_power_iter):
        Y = cp.reshape(Y, (Dcut*Dcut*Dcut,l))
        Q, _ = cp.linalg.qr(Y)
        #print("norm(R)= {:.12e}".format(cp.linalg.norm(R)))

        for _ in range(n_power_iter):
            Y = __TdagY__(A, B, Q, Dcut, l)
            Q, _ = cp.linalg.qr(Y)

            Y = __TY__(A, B, Q, Dcut, l)
            Q, _ = cp.linalg.qr(Y)

        return Q

    def __power_iteration_4atrg__(A, B, Y, Dcut, l, n_power_iter):
        for q in range(n_power_iter):
            Y = __TdagY__(A, B, Y, Dcut, l)
            Y = __TY__(A, B, Y, Dcut, l)
        Q, _ = cp.linalg.qr(Y)

        return Q 

    def __TdagY__(A, B, Y, Dcut, l):
        Y = cp.reshape(Y, (Dcut,Dcut,Dcut,l))
        Y = contract("iabc,defi,defl->abcl", cp.conj(B), cp.conj(A), Y)
        #Y = contract("iabc,defi,defl->abcl", (B), (A), Y)
        Y = cp.reshape(Y, (Dcut*Dcut*Dcut,l))
        return Y

    def __TY__(A, B, Y, Dcut, l):
        Y = cp.reshape(Y, (Dcut,Dcut,Dcut,l))
        Y = contract("abci,idef,defl->abcl", A, B, Y)
        Y = cp.reshape(Y, (Dcut*Dcut*Dcut,l))
        return Y

    l = Dcut + n_oversamples
    Y = seed.standard_normal(size=(Dcut,Dcut,Dcut,l))
    Y = Y.astype(cp.complex128)
    Y = contract("abci,idef,defl->abcl", A, B, Y)
    #print("norm(A)= {:.12e}".format(cp.linalg.norm(A)))
    #print("norm(B)= {:.12e}".format(cp.linalg.norm(B)))
    #print("norm(Y)= {:.12e}".format(cp.linalg.norm(Y)))

    if iterator == "QR":
        Q = __QR_iteration_4atrg__(A, B, Y, Dcut, l, n_power_iter)
    elif iterator == "power":
        Q = __power_iteration_4atrg__(A, B, Y, Dcut, l, n_power_iter)
    del Y

    Q = cp.reshape(Q, (Dcut,Dcut,Dcut,l))
    X = contract("abcl,abci,idef->ldef", cp.conj(Q), A, B)
    X = cp.reshape(X, (l,Dcut*Dcut*Dcut))
    u, s, vh = cp.linalg.svd(X, full_matrices=False)
    del X

    u = contract("abci,ij->abcj", Q, u)
    vh = cp.reshape(vh, (l,Dcut,Dcut,Dcut))

    del Q

    return u[:,:,:,:Dcut], s[:Dcut], vh[:Dcut,:,:,:]
    

def rsvd_for_3dATRG_tensor_inte(B, C, Dcut:int, n_oversamples=10, n_power_iter=0, iterator="QR", seed=cp.random.RandomState(seed=1234)):
    """
    Calculate svd of M_{ax0y'0,bx'1y1} = UM_{ax0y'0,j} sM_{j} VMH_{j,bx'1y1}
    """

    def __QR_iteration_4atrg__(B, C, O, Dcut, l, n_power_iter):
        O = cp.reshape(O, (Dcut*Dcut*Dcut,l))
        Q, _ = cp.linalg.qr(O)

        for _ in range(n_power_iter):
            O = __MdagQ__(B, C, Q, Dcut, l)
            Q, _ = cp.linalg.qr(O)

            O = __MQ__(B, C, Q, Dcut, l)
            Q, _ = cp.linalg.qr(O)
        del O

        return Q

    def __power_iteration_4atrg__(B, C, Q, Dcut, l, n_power_iter):
        for q in range(n_power_iter):
            O = __MdagQ__(B, C, Q, Dcut, l)
            O = __MQ__(B, C, Q, Dcut, l)
        Q, _ = cp.linalg.qr(O)
        del O

        return Q 

    def __MdagQ__(B, C, Q, Dcut, l):
        Q = cp.reshape(Q, (Dcut,Dcut,Dcut,l))
        O = contract("adjk,dbci,abcl->ijkl", cp.conj(B), cp.conj(C), Q)
        #O = contract("adjk,dbci,abcl->ijkl", (B), (C), Q)
        O = cp.reshape(O, (Dcut*Dcut*Dcut,l))
        return O

    def __MQ__(B, C, Q, Dcut, l):
        Q = cp.reshape(Q, (Dcut,Dcut,Dcut,l))
        O = contract("idbc,djka,abcl->ijkl", B, C, Q)
        O = cp.reshape(O, (Dcut*Dcut*Dcut,l))
        return O

    l = Dcut + n_oversamples
    O = seed.normal(size=(Dcut,Dcut,Dcut,l))
    O = O.astype(cp.complex128)
    O = contract("iabc,ajkd,dbcl->ijkl", B, C, O)

    if iterator == "QR":
        Q = __QR_iteration_4atrg__(B, C, O, Dcut, l, n_power_iter)
    elif iterator == "power":
        Q = __power_iteration_4atrg__(B, C, O, Dcut, l, n_power_iter)
    del O
    #print(Q.shape)
    #print("tr Q†Q=",cp.linalg.norm(cp.conj(Q.T)@Q)**2)
    Q = cp.reshape(Q, (Dcut,Dcut,Dcut,l))
    X = oe.contract("abcl,adjk,dbci->lijk", cp.conj(Q), B, C)
    X = cp.reshape(X, (l,Dcut*Dcut*Dcut))
    u, s, vh = cp.linalg.svd(X, full_matrices=False)
    del X

    u = oe.contract("ijka,al->ijkl", Q, u)
    vh = cp.reshape(vh, (l,Dcut,Dcut,Dcut))

    return u[:,:,:,:Dcut], s[:Dcut], vh[:Dcut,:,:,:]