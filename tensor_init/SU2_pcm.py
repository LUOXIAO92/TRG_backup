import gc
import numpy as np
import cupy as cp
import opt_einsum as oe

from utility.randomized_svd import rsvd
from time import time

from tensor_class.tensor_class import ATRG_Tensor as Tensor

class SU2_pcm_initialize():
    def __init__(self, K:int, dim:int, Dcut:int, beta:float, h=0.0, mu1=0.0, mu2=0.0):
        self.K = K
        self.Dcut = Dcut
        self.dim  = dim
        self.beta = beta
        self.h    = h
        self.mu1  = mu1
        self.mu2  = mu2

        #self.U, self.w, self.J, self.I = self.SU2_matrix()

    def SU2_matrix(self):
        Kt = self.K
        Ka = self.K
        Kb = self.K

        from scipy.special import roots_legendre
        t, wt = roots_legendre(Kt)
        a, wa = roots_legendre(Ka)
        b, wb = roots_legendre(Kb)

        t = np.asarray(np.pi * (t + 1) / 4)
        a = np.asarray(np.pi * (a + 1))
        b = np.asarray(np.pi * (b + 1))

        epia = np.exp( 1j*a)
        epib = np.exp( 1j*b)
        emia = np.exp(-1j*a)
        emib = np.exp(-1j*b)
        st = np.sin(t)
        ct = np.cos(t)

        It = np.ones(shape=Kt, dtype=complex)
        Ia = np.ones(shape=Ka, dtype=complex)
        Ib = np.ones(shape=Kb, dtype=complex)

        #Uij = Uij(θ, α, β)
        U = np.zeros(shape=(2, 2, Kt, Ka, Kb), dtype=complex)
        subscript = "t,a,b->tab"
        U[0,0] =  oe.contract("t,a,b->tab", ct, epia, Ib)
        U[0,1] =  oe.contract("t,a,b->tab", st, Ia, epib)
        U[1,0] = -oe.contract("t,a,b->tab", st, Ia, emib)
        U[1,1] =  oe.contract("t,a,b->tab", ct, emia, Ib)
        
        I = np.zeros_like(U)
        I[0,0] = oe.contract(subscript, It, Ia, Ib)
        I[1,1] = oe.contract(subscript, It, Ia, Ib)

        #w[0] = contract("a,a,a->a", cp.sin(theta), cp.cos(theta), w[0])
        #Jacobian = (π/8) * sin(θ)cos(θ)
        Jt = st*ct
        Ja = Ia
        Jb = Ib
        J = oe.contract(subscript, Jt, Ja, Jb) * (cp.pi / 8)

        #weight
        w = oe.contract(subscript, wt, wa, wb)

        U = cp.reshape(U, newshape=(2, 2, int(self.K**3)))
        I = cp.reshape(I, newshape=(2, 2, int(self.K**3)))
        w = cp.reshape(w, newshape=(int(self.K**3)))
        J = cp.reshape(J, newshape=(int(self.K**3)))

        U = cp.asarray(U)
        I = cp.asarray(I)
        w = cp.asarray(w)
        J = cp.asarray(J)

        return U, w, J, I
    
    def chemical_potential_term(self):
        def chemical_potential_matrix(a):
            D = cp.zeros(shape=(2,2), dtype=float)
            D[0,0] = np.exp( a)
            D[1,1] = np.exp(-a)
            return D
        mu1 = self.mu1
        mu2 = self.mu2
        D1 = chemical_potential_matrix( (mu1+mu2)/2)
        D2 = chemical_potential_matrix( (mu1-mu2)/2)
        D3 = chemical_potential_matrix(-(mu1-mu2)/2)
        D4 = chemical_potential_matrix(-(mu1+mu2)/2)
        return D1, D2, D3, D4
    
    def cal_Boltzmann_weight(self, U, I, direction="spatial"):
        """
        direction: spatial or temporal 
        return: Boltzmann weight matrix M
        """

        if direction == "spatial":
            M =   self.beta * 2 * oe.contract("ija,ijb->ab", U, cp.conj(U)) \
                + self.beta * 2 * oe.contract("ija,ijb->ab", cp.conj(U), U) \
                + (self.h / (2*self.dim)) * oe.contract("ija,ijb->ab", U, I) \
                + (self.h / (2*self.dim)) * oe.contract("ija,ijb->ab", I, U)
            M = cp.exp(M)
        elif direction == "temporal":
            #M = oe.contract("ija,ijb->ab", U, cp.conj(U))
            #M = cp.exp(self.beta * 2 * 2 * M.real)
            D1, D2, D3, D4 = self.chemical_potential_term()
            M =   self.beta * 2 * oe.contract("ij,jka,kl,ilb->ab", D1, U, D2, cp.conj(U)) \
                + self.beta * 2 * oe.contract("ij,kja,kl,lib->ab", D3, cp.conj(U), D4, U) \
                + (self.h / (2*self.dim)) * oe.contract("ija,ijb->ab", U, I) \
                + (self.h / (2*self.dim)) * oe.contract("ija,ijb->ab", I, U)
            M = cp.exp(M)

        return M 
    
    def svd_Boltzmann_weight(self, M, k, del_M=True, split=False):

        #from utility.randomized_svd import rsvd2
        #u, s, vh = rsvd2(A=M, k=k, seed=1234, del_A=del_M)
        u, s, vh = rsvd(A=M, k=k, n_oversamples=k, n_power_iter=k, seed=1234)
        del M

        print("Singular values of Boltzmann weight is:")
        with cp.printoptions(formatter={'float_kind':'{:.6e}'.format}):
            print(s)
            print(f"s_Dcut/s_1 = {s[-1]/s[0]:.6e}")
        print()

        if split:
            A = oe.contract("ia,a->ia", u, cp.sqrt(s))
            B = oe.contract("a,aj->aj", cp.sqrt(s), vh)
            return A, B
        else:
            return u, s, vh
        
    def form_hotrg_tensor(self, J, w, s, Bs, As, Bt, At):

        if self.dim==2:
            subscript = "U,U,U,xU,UX,yU,UY->xXyY"
            T = oe.contract(subscript, J, w, s, Bs, As, Bt, At)

        elif self.dim==3:
            UxX = oe.contract("U,U,U,xU,UX->UxX", J, w, s, Bs, As)
            UyY = oe.contract("yU,UY->UyY", Bs, As)
            UtT = oe.contract("tU,UT->UtT", Bt, At)

            path = [(0, 1, 2)]
            slicing = 100
            n = len(J) // slicing
            T = cp.zeros(shape=(Bs.shape[0], As.shape[1], Bs.shape[0], As.shape[1], Bt.shape[0], At.shape[1]), dtype=complex)
            for i in range(0, len(J), n):
                T += oe.contract("UxX,UyY,UzZ->xXyYzZ", UxX[i:i+n,:,:], UyY[i:i+n,:,:], UtT[i:i+n,:,:], optimize=path)

        elif self.dim==4:
            subscript = "U,U,U,xU,UX,yU,UY,zU,UZ,tU,UT->xXyYzZtT"
            T = oe.contract(subscript, J, w, s, Bs, As, Bs, As, Bs, As, Bt, At)

        return T
    
    def cal_hotrg_init_tensor(self, Dinit, impure=False):
        U, w, J, I = self.SU2_matrix()

        if (self.mu1 + self.mu2) <= 1e-12:
            Ms = self.cal_Boltzmann_weight(U, I, direction="spatial")
            As, Bs = self.svd_Boltzmann_weight(Ms, k=Dinit, del_M=True, split=True)
            At, Bt = As, Bs
        else:
            Ms = self.cal_Boltzmann_weight(U, I, direction="spatial")
            As, Bs = self.svd_Boltzmann_weight(Ms, k=Dinit, del_M=True, split=True)
            del Ms
            Mt = self.cal_Boltzmann_weight(U, I, direction="temporal")
            At, Bt = self.svd_Boltzmann_weight(Mt, k=Dinit, del_M=True, split=True)
            del Mt

        I = cp.ones_like(w)
        T = self.form_hotrg_tensor(J, w, I, Bs, As, Bt, At)
        if impure:
            return T, J, w, I, Bs, As, Bt, At, U
        else:
            return T
        
    
    def form_ATRG_tensor(self, J, w, O, Bs, As, Bt, At, k:int, is_impure=False, p=0, q=0, seed=1234)->Tensor:
        """
        J: Jacobian determinant of integration. \\
        w: weights of quadrature rules.\\
        O: Operator of impure tensor. All one if making a pure tensor.\\
        Bs, As, Bt, At: Components of spatial and temporal direction.\\
        k: Svd rank or bond dimensions of internal degrees of freedom.\\
        p: Over samples.\\
        q: Times of iteration.\\
        seed: Seed for random standard normal compression matrix.\\
        """

        #ichunks = self.para_chunks if ichunks is None else ichunks
        from tensor_init.ATRG_init import initial_tensor_for_ATRG as init
        U, s, VH = init(dim=self.dim, J=J, w=w*O, 
                        As=As, Bs=Bs, 
                        At=At, Bt=Bt,
                        k=k, 
                        p=p, 
                        q=q, 
                        seed=seed)
        T = Tensor(U, s, VH, Dcut=self.Dcut, dim=self.dim, is_impure=is_impure, loc={})
        return T

    def cal_ATRG_init_tensor(self, Dinit:int, k:int, impure=False, p=0, q=0, seed=1234):
        """
        J : Jacobian determinant of integration. 
        w : weights of quadrature rules.
        k : Svd rank or bond dimensions of internal degrees of freedom.
        p : Over samples.
        q : Times of iteration.
        seed: Seed for random standard normal compression matrix.
        """

        U, w, J, I = self.SU2_matrix()

        if (self.mu1 + self.mu2) <= 1e-12:
            Ms = self.cal_Boltzmann_weight(U, I, direction="spatial")
            As, Bs = self.svd_Boltzmann_weight(Ms, k=Dinit, del_M=True, split=True)
            del Ms
            At, Bt = As, Bs
        else:
            Ms = self.cal_Boltzmann_weight(U, I, direction="spatial")
            As, Bs = self.svd_Boltzmann_weight(Ms, k=Dinit, del_M=True, split=True)
            del Ms

            Ms = self.cal_Boltzmann_weight(U, I, direction="temporal")
            At, Bt = self.svd_Boltzmann_weight(Ms, k=Dinit, del_M=True, split=True)
            
        #from tensor_init.ATRG_init import initial_tensor_for_ATRG as init
        #U, s, VH = init(dim=self.dim, J=J, w=w, 
        #                As=As, Bs=Bs, 
        #                At=At, Bt=Bt,
        #                k=k, 
        #                p=p, 
        #                q=q, 
        #                seed=seed)
        #
        #T = Tensor(U, s, VH, Dcut=self.Dcut, dim=self.dim, is_impure=False, loc={})

        I = cp.ones_like(w)
        T = self.form_ATRG_tensor(J, w, I, Bs, As, Bt, At, k, impure, p, q, seed)
        
        if impure:
            return T, J, w, U, I, Bs, As, Bt, At
        else:
            return T