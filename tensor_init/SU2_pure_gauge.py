import time
import math
import copy

try:
    import cupy as cp
except:
    pass
import numpy as np

import opt_einsum as oe
from itertools import product

from mpi4py import MPI

from tools.mpi_tools import contract_slicer, pure_gauge_slice, gpu_syn, flatten_2dim_job_results
from utility.truncated_svd import eigh, svd

class SU2_pure_gauge:
    def __init__(self, 
                 dim:int, 
                 Dcut:int, 
                 Ks:tuple|list,
                 β:float, 
                 ε:float|None, 
                 comm:MPI.Intracomm, 
                 use_gpu=False,
                 quadrature = "GL",
                 mesh_size  = None):
        
        self.dim  = dim
        self.Dcut = Dcut
        self.Ks   = Ks
        self.β = β
        self.ε = ε
        self.quadrature = quadrature
        self.mesh_size  = mesh_size

        self.comm    = comm
        self.use_gpu = use_gpu

    def plaquette_tensor(self, chi, chunk:tuple, legs_to_hosvd:list):
        """
        >>>         U1†
        >>>     ----------
        >>>     |        |
        >>>  U0†|        |U2
        >>>     |        |
        >>>     ----------
        >>>         U3
        """

        len_legs_to_hosvd = len(legs_to_hosvd)
        assert (len_legs_to_hosvd == 1) or (len_legs_to_hosvd == 2) or (len_legs_to_hosvd == 4), f"number of legs must be 1,2 or 4"

        for leg in legs_to_hosvd:
             assert legs_to_hosvd[leg] == leg, "legs_to_hosvd must be [0], [0,1] or [0,1,2,3]"

        MPI_COMM = self.comm
        MPI_RANK = MPI_COMM.Get_rank()
        MPI_SIZE = MPI_COMM.Get_size()        

        
        if self.quadrature == "DE":
            print(f"Using double exponential quadrature, mesh size= {self.mesh_size}")
            assert self.mesh_size is not None
            from tools.trg_tools import SU2_matrix_double_exponential_formula
            U, w, J, I = SU2_matrix_double_exponential_formula(self.Ks[0], self.mesh_size, self.Ks[1], self.Ks[2], 
                                                          self.comm, to_cupy = self.use_gpu)
        else:
            print(f"Using Gauss Legendre quadrature")
            from tools.trg_tools import SU2_matrix_Gauss_Legendre_quadrature
            U, w, J, I = SU2_matrix_Gauss_Legendre_quadrature(self.Ks[0], self.Ks[1], self.Ks[2], 
                                                          self.comm, to_cupy = self.use_gpu)

        N = self.Ks[0] * self.Ks[1] * self.Ks[2]

        assert len(chunk) == 3, "length of `chunk` must be 3"
        
        MPI_COMM.barrier()
        iteration = [
                     pure_gauge_slice(shape=(N, N, N, N), chunk=(N, chunk[0], chunk[1], chunk[2])),
                     pure_gauge_slice(shape=(N, N, N, N), chunk=(chunk[0], N, chunk[1], chunk[2])),
                     pure_gauge_slice(shape=(N, N, N, N), chunk=(chunk[0], chunk[1], N, chunk[2])),
                     pure_gauge_slice(shape=(N, N, N, N), chunk=(chunk[0], chunk[1], chunk[2], N)),
                     ]
        MPI_COMM.barrier()


        #compute hosvd P_{U0, U1, U2, U3} = s_{a, b, c, d} V_{U0, a} V_{U1, b} V_{U2, c} V_{U3, d}
        #compute M_{U0, U0'} = P_{U0, U1, U2, U3} P*_{U'0, U1, U2, U3}, ..., M_{U3, U3'}
        M_list = []
        for i in range(4):
            if i <= legs_to_hosvd[-1]:
                MPI_COMM.barrier()
                gpu_syn(self.use_gpu)
                leg_hosvd = legs_to_hosvd[i]
                M_tmp = plaquette_contraction_for_hosvd(self.β, self.ε, U, w, J, leg_hosvd, iteration[leg_hosvd], 
                                                        self.comm, use_gpu=self.use_gpu, verbose=True)
                M_list.append(M_tmp)
                MPI_COMM.barrier()
                if MPI_RANK == 0:
                    print(f"Calculation of leg U{i} finished.")
                del M_tmp
            else:
                M_list.append(M_list[i % len_legs_to_hosvd])
                

        MPI_COMM.barrier()
        if MPI_RANK == 0:
            M =[]
            for rank in range(MPI_SIZE):
                M_send = []
                for i in range(rank, 4, MPI_SIZE):
                    if rank == 0:
                        M.append(M_list[i])
                    else:
                        M_send.append(M_list[i])
                if rank != 0:
                    MPI_COMM.send(obj=M_send, dest=rank, tag=rank)
        else:
            gpu_syn(self.use_gpu)
            M = MPI_COMM.recv(source=0, tag=MPI_RANK)
        MPI_COMM.barrier()


        MPI_COMM.barrier()
        gpu_syn(self.use_gpu)
        eigvals, vs = [], []
        for m in M:
            if m is not None:
                e, v = eigh(m, shape=[[0], [1]], k=chi, truncate_eps=1e-10, degeneracy_eps=1e-5)
                eigvals.append(e)
                vs.append(v)
        MPI_COMM.barrier()
        

        MPI_COMM.barrier()
        for i, e in enumerate(eigvals):
            leg = i*MPI_SIZE + MPI_RANK
            print(f"eigen values for {leg}th leg at rank{MPI_RANK} are", e)
        MPI_COMM.barrier()

        #Tensor P_{U0, U1, U2, U3} = s_{a, b, c, d} V_{U0, a} V_{U1, b} V_{U2, c} V_{U3, d}

        all_vs = []
        for rank in range(MPI_SIZE):
            if rank == MPI_RANK:
                buf = vs
            else:
                buf = None
            buf = MPI_COMM.bcast(buf, root=rank)
            all_vs.append(buf)
        del vs
        MPI_COMM.barrier()

        vs = flatten_2dim_job_results(all_vs, 4, MPI_COMM)
        
        iteration = pure_gauge_slice(shape=(N, N, N, N), chunk=(N, chunk[0], chunk[1], chunk[2]))
        #Q0, Q1, Q2, Q3 = cp.conj(vs[0].T), cp.conj(vs[1].T), cp.conj(vs[2].T), cp.conj(vs[3].T)
        Q0, Q2 = cp.conj(vs[0].T), vs[2].T
        Q1, Q3 = cp.conj(vs[1].T), vs[3].T

        I02 = oe.contract("iU,jU->ij", Q0, Q2)
        I13 = oe.contract("iU,jU->ij", Q1, Q3)
        TrI02 = cp.trace(I02)
        normI02sqr = cp.linalg.norm(I02)**2
        TrI13 = cp.trace(I13)
        normI13sqr = cp.linalg.norm(I13)**2
        print(f"At rank {MPI_RANK}. Tr(V_{{U,c0}}V_{{U,a0}})={TrI02}, norm(V_{{U,c0}}V_{{U,a0}})={normI02sqr},Tr(V_{{U,d0}}V_{{U,b0}})={TrI13}, norm(V_{{U,d0}}V_{{U,b0}})={normI13sqr}")

        T_shape = (Q0.shape[0], Q1.shape[0], Q2.shape[0], Q3.shape[0])
        T_local = cp.zeros(shape=T_shape, dtype=complex)
        t0 = time.time() if MPI_RANK == 0 else None
        for n, i in enumerate(iteration):
            i0, i1, i2, i3 = i
            if n % MPI_SIZE == MPI_RANK:
        
                #TrP = oe.contract("abi,bcj,dck,adl->ijkl", cp.conj(U[:,:,i0]), cp.conj(U[:,:,i1]), U[:,:,i2], U[:,:,i3])
                TrP = oe.contract("dci,adj,abk,bcl->ijkl", cp.conj(U[:,:,i0]), cp.conj(U[:,:,i1]), U[:,:,i2], U[:,:,i3])
                
                if self.ε is not None:
                    norm, idx = admissibility_condition(TrP, self.ε)
                    P = (1 - 0.5*TrP.real) / (1 - (norm / self.ε))
                    P = cp.exp(-self.β * P)
                    P[idx] = 0.0
                else:
                    P = cp.exp(-self.β * (1 - 0.5*TrP.real))
                
                P = oe.contract("i,j,k,l,i,j,k,l,ijkl->ijkl", cp.sqrt(w[i0]), cp.sqrt(w[i1]), cp.sqrt(w[i2]), cp.sqrt(w[i3]), 
                                                              cp.sqrt(J[i0]), cp.sqrt(J[i1]), cp.sqrt(J[i2]), cp.sqrt(J[i3]), P)
                q0, q1, q2, q3 = Q0[:,i0], Q1[:,i1], Q2[:,i2], Q3[:,i3]
                T_local += oe.contract("ABCD,xA,YB,XC,yD->xYXy", P, q0, q1, q2, q3)
        
                if (n > 0) and (n % (25*MPI_SIZE) == 0) and (MPI_RANK == 0):
                    gpu_syn(self.use_gpu)
                    t1 = time.time() if MPI_RANK == 0 else None
                    print(f"n={n}", end=", ")
                    print(f"{n // MPI_SIZE} times finished. Time= {t1-t0:.2e} s. Size of P: {P.nbytes/(1024**3):.2e} Gbytes")
                    t0 = time.time() if MPI_RANK == 0 else None
        
        T = MPI_COMM.reduce(sendobj=T_local, op=MPI.SUM, root=0)
        gpu_syn(self.use_gpu)
        MPI_COMM.barrier()

        return T
    

    def atrg_tensor(self, 
                    chi_plaquette:int, 
                    chi_atrgtensor:int, 
                    chunk_plaquette:tuple, 
                    chunk_environment:tuple,
                    legs_to_hosvd:list, 
                    truncate_eps:float, 
                    degeneracy_eps:float,
                    chunk_atrg_rsvd:tuple,
                    n_oversample:int,
                    n_iters:int,
                    seed_atrg_rsvd=None,
                    verbose=False):
        
        if self.use_gpu:
            xp = cp
        else:
            xp = np

        MPI_COMM = self.comm
        MPI_RANK = MPI_COMM.Get_rank()

        P = self.plaquette_tensor(chi_plaquette, chunk_plaquette, legs_to_hosvd)
        Qs = [None, None]
        if MPI_RANK == 0:
            I0 = xp.diag(xp.ones(shape=P.shape[2], dtype=P.dtype))
            I1 = xp.diag(xp.ones(shape=P.shape[3], dtype=P.dtype))
            I2 = xp.diag(xp.ones(shape=P.shape[0], dtype=P.dtype))
            #I2 = cp.pad(I2, pad_width=((0, chi_plaquette - P.shape[0]), (0, 0)), mode='constant', constant_values=0)
            I3 = xp.diag(xp.ones(shape=P.shape[1], dtype=P.dtype))
            #I3 = cp.pad(I3, pad_width=((0, chi_plaquette - P.shape[1]), (0, 0)), mode='constant', constant_values=0)
            Q0 = oe.contract("Ωi,Ωj,Ωk,Ωl->ijkl", I0, I1, I2, I3)

            I0 = xp.diag(xp.ones(shape=P.shape[3], dtype=P.dtype))
            #I0 = cp.pad(I0, pad_width=((0, chi_plaquette - P.shape[3]), (0, 0)), mode='constant', constant_values=0)
            I1 = xp.diag(xp.ones(shape=P.shape[3], dtype=P.dtype))
            I2 = xp.diag(xp.ones(shape=P.shape[1], dtype=P.dtype))
            I3 = xp.diag(xp.ones(shape=P.shape[1], dtype=P.dtype))
            #I3 = cp.pad(I3, pad_width=((0, chi_plaquette - P.shape[1]), (0, 0)), mode='constant', constant_values=0)
            Q1 = oe.contract("Ωi,Ωj,Ωk,Ωl->ijkl", I0, I1, I2, I3)

            del I0, I1, I2, I3

            Qs = [Q0, Q1]
            
            
        gpu_syn(self.use_gpu)
        MPI_COMM.barrier()

        A, B, C, D = env_tensor_legswapping_3d_gauge(chi_plaquette, 
                                                     P, Qs, 
                                                     truncate_eps, 
                                                     degeneracy_eps, 
                                                     self.comm, 
                                                     self.use_gpu,
                                                     verbose=True)
        del Qs

        Q = None
        if MPI_RANK == 0:
            I0 = xp.diag(xp.ones(shape=P.shape[2], dtype=P.dtype))
            I1 = xp.diag(xp.ones(shape=P.shape[2], dtype=P.dtype))
            #I1 = cp.pad(I1, pad_width=((0, chi_plaquette - P.shape[2]), (0, 0)), mode='constant', constant_values=0)
            I2 = xp.diag(xp.ones(shape=P.shape[0], dtype=P.dtype))
            #I2 = cp.pad(I2, pad_width=((0, chi_plaquette - P.shape[0]), (0, 0)), mode='constant', constant_values=0)
            I3 = xp.diag(xp.ones(shape=P.shape[0], dtype=P.dtype))
            Q = oe.contract("Ωi,Ωj,Ωk,Ωl->ijkl", I0, I1, I2, I3)

            del I0, I1, I2, I3

        EnvT = env_tensor_3d_gauge(chi_plaquette, 
                                   A, B, C, D, P, Q,
                                   "T",
                                   truncate_eps,
                                   degeneracy_eps,
                                   chunk_environment,
                                   self.comm,
                                   self.use_gpu)
        Envt = env_tensor_3d_gauge(chi_plaquette, 
                                   A, B, C, D, P, Q,
                                   "t",
                                   truncate_eps,
                                   degeneracy_eps,
                                   chunk_environment,
                                   self.comm,
                                   self.use_gpu)
        
        
        EnvX = env_tensor_3d_gauge(chi_plaquette, 
                                   A, B, C, D, P, Q,
                                   "X",
                                   truncate_eps,
                                   degeneracy_eps,
                                   chunk_environment,
                                   self.comm,
                                   self.use_gpu)
        Envx = env_tensor_3d_gauge(chi_plaquette, 
                                   A, B, C, D, P, Q,
                                   "x",
                                   truncate_eps,
                                   degeneracy_eps,
                                   chunk_environment,
                                   self.comm,
                                   self.use_gpu)
        
        
        EnvY = env_tensor_3d_gauge(chi_plaquette, 
                                   A, B, C, D, P, Q,
                                   "Y",
                                   truncate_eps,
                                   degeneracy_eps,
                                   chunk_environment,
                                   self.comm,
                                   self.use_gpu)
        Envy = env_tensor_3d_gauge(chi_plaquette, 
                                   A, B, C, D, P, Q,
                                   "y",
                                   truncate_eps,
                                   degeneracy_eps,
                                   chunk_environment,
                                   self.comm,
                                   self.use_gpu)
        
        MPI_RANK = MPI_COMM.Get_rank()
        MPI_SIZE = MPI_COMM.Get_size()
        Envs = [[0, 'T', EnvT], [1, 't', Envt], [2, 'X', EnvX], [3, 'x', Envx], [4, 'Y', EnvY], [5, 'y', Envy]]
        Njobs = len(Envs)

        localjobs = []
        for job_id in range(Njobs):
            dest_rank = job_id % MPI_SIZE

            if MPI_RANK == 0:
                sendjob = Envs[job_id]

                if MPI_RANK == dest_rank:
                    localjobs.append(sendjob)
                else:
                    MPI_COMM.send(sendjob, dest=dest_rank, tag=dest_rank)
            else:
                if MPI_RANK == dest_rank:
                    localjobs.append(
                        MPI_COMM.recv(source=0, tag=MPI_RANK)
                    )
        
        results = []
        for job_id, direction, env in localjobs:
            k = min(env.shape[0]*env.shape[1], env.shape[2]*env.shape[3])
            e, u = eigh(env, shape=[[0,1], [2,3]], k=k, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps)
            
            if verbose:
                emax = xp.max(e)
                print(f"job_id:{job_id}, direction:{direction}, eigvals (λmax={emax:.6e}):\n", e[:chi_plaquette] / emax)

            if job_id % 2 == 0:
                R = oe.contract("i,abi->iab", xp.sqrt(e), xp.conj(u))
            else:
                R = oe.contract("abi,i->abi", u, xp.sqrt(e))

            results.append(R)
        
        results = MPI_COMM.gather(results, root=0)
        del localjobs
        gpu_syn(self.use_gpu)
        MPI_COMM.barrier()

        Rs = []
        if MPI_RANK == 0:
            results = flatten_2dim_job_results(results, job_size=Njobs, comm=MPI_COMM)
        else:
            results = [None for _ in range(len(Envs))]
        for i in range(0, len(results), 2):
            Rs.append([results[i], results[i+1]])
        del results, Envs

        
        Njobs = len(Rs)
        localjobs = []
        for job_id in range(Njobs):
            dest_rank = job_id % MPI_SIZE

            if MPI_RANK == 0:
                sendjob = Rs[job_id]
                if MPI_RANK == dest_rank:
                    localjobs.append(sendjob)
                else:
                    MPI_COMM.send(sendjob, dest=dest_rank, tag=dest_rank)
            else:
                if MPI_RANK == dest_rank:
                    localjobs.append(
                        MPI_COMM.recv(source=0, tag=MPI_RANK)
                    )
        
        results = []
        for Rl, Rr in localjobs:
            Vl, Vr = squeezer_3d_gauge(chi_atrgtensor, 
                                           Rl, Rr, 
                                           truncate_eps=truncate_eps, 
                                           degeneracy_eps=degeneracy_eps, 
                                           use_gpu=True, 
                                           verbose=verbose)
            results.append([Vl, Vr])
        results = MPI_COMM.gather(results, root=0)
        gpu_syn(self.use_gpu)
        MPI_COMM.barrier()

        Vs = []
        if MPI_RANK == 0:
            results = flatten_2dim_job_results(results, job_size=len(Rs), comm=MPI_COMM)
            for Vl, Vr in results:
                #PrPl = oe.contract("iab,abj->ij", Pr, Pl)
                #print(f"Tr(PrPl)={xp.trace(PrPl)}, |PrPl|^2={xp.linalg.norm(PrPl)**2}")
                Vs.append(Vl)
                Vs.append(Vr)
        del results

        U, S, VH = rsvd_3d_gauge(A, B, C, D, P, Q, Vs, 
                                 k = chi_atrgtensor, 
                                 p = n_oversample, 
                                 q = n_iters, 
                                 truncate_eps   = truncate_eps,
                                 degeneracy_eps = degeneracy_eps,
                                 chunk = chunk_atrg_rsvd, 
                                 comm = MPI_COMM, 
                                 seed = seed_atrg_rsvd, 
                                 use_gpu = self.use_gpu, 
                                 verbose=True)
        
        if MPI_RANK == 0:
            print(S)

        return U, S, VH
        
        







def admissibility_condition(TrP, ε:float):
    """
    Admissibility condition ||1 - U0† U1† U2 U3}|| < ε in Luscher's gauge action.

    Parameters
    ----------
    P : Gauge. numpy.ndarray or cupy.ndarray
    ε : Parameter of Luscher's gauge action

    ----------

    Retruns
    -------
    norm_{U0† U1† U2 U3} = ||1 - U0† U1† U2 U3}||, and bool index that satisfies ||1 - U0† U1† U2 U3}|| < ε

    -------
    """
    if type(TrP) == np.ndarray:
        sqrt = np.sqrt
        abs  = np.abs
    else:
        from cupy import sqrt, abs
    
    norm = 4 - 2 * TrP.real
    norm = sqrt(abs(norm))
    index = norm > ε
    return norm, index

def plaquette_contraction_for_hosvd(β:float, ε:float|None, U, w, J, leg_hosvd, iteration, 
                                    comm:MPI.Intercomm, use_gpu=False, verbose=False):
    MPI_COMM = comm
    MPI_RANK = MPI_COMM.Get_rank()
    MPI_SIZE = MPI_COMM.Get_size()
    info = MPI_COMM.Get_info()

    if use_gpu:
        from cupy import zeros, conj, exp, inf, sqrt
    else:
        from numpy import zeros, conj, exp, inf, sqrt
    
    N = U.shape[2]
    M_local = zeros(shape=(N, N), dtype=complex)

    subscripts = ["ijkl,Ijkl->iI", "ijkl,iJkl->jJ", "ijkl,ijKl->kK", "ijkl,ijkL->lL"]
    
    
    gpu_syn(use_gpu)
    MPI_COMM.barrier()

    t0 = time.time()
    t00 = time.time()

    for n, i in enumerate(iteration):
        i0, i1, i2, i3 = i
        if n % MPI_SIZE == MPI_RANK:

            #TrP = oe.contract("abi,bcj,dck,adl->ijkl", conj(U[:,:,i0]), conj(U[:,:,i1]), U[:,:,i2], U[:,:,i3])
            TrP = oe.contract("dci,adj,abk,bcl->ijkl", conj(U[:,:,i0]), conj(U[:,:,i1]), U[:,:,i2], U[:,:,i3])

            if ε is not None:
                norm, idx = admissibility_condition(TrP, ε)
                A = (1 - 0.5*TrP.real) / (1 - (norm / ε))
                A = exp(-β * A)
                A[idx] = 0.0
            else:
                A = exp(-β * (1 - 0.5*TrP.real))

            A = oe.contract("i,j,k,l,i,j,k,l,ijkl->ijkl", sqrt(w[i0]), sqrt(w[i1]), sqrt(w[i2]), sqrt(w[i3]), 
                                                          sqrt(J[i0]), sqrt(J[i1]), sqrt(J[i2]), sqrt(J[i3]), A)
            M_local += oe.contract(subscripts[leg_hosvd], A, conj(A))
            num_inf = len(M_local[M_local == inf])
            assert num_inf == 0, f"Overflow at {n}th iteration. Have {num_inf} infs."

            if verbose:
                if (n > 0) and (n % (25*MPI_SIZE) == 0) and (MPI_RANK == 0):
                    gpu_syn(use_gpu)
                    t1 = time.time() if MPI_RANK == 0 else None
                    print(f"Global iters:{n}. Local iters:{n // MPI_SIZE}. {(t1-t0) / (n // MPI_SIZE):.2e} sec/local_iter") #. Size of A: {A.nbytes/(1024**3):.2e} Gbytes")
                    t0 = time.time() if MPI_RANK == 0 else None

    t11 = time.time()
    if MPI_RANK == 0:
        print(f"Tot iteration {n+1}. Time= {t11-t00:.2e} s")

    M = MPI_COMM.reduce(sendobj=M_local, op=MPI.SUM, root=0)
    gpu_syn(use_gpu)
    MPI_COMM.barrier()

    return M

def env_tensor_legswapping_3d_gauge(chi, P, Qs:list, 
                                    truncate_eps:float, 
                                    degeneracy_eps:float, 
                                    comm:MPI.Intercomm, 
                                    use_gpu=False,
                                    verbose=False):
    if use_gpu:
        xp = cp
    else:
        xp = np

    MPI_RANK = comm.Get_rank()
    MPI_SIZE = comm.Get_size()

    _Qaxes_list = [(3, 0, 1, 2), (2, 3, 0, 1)]
    _Paxes_list = [(0, 1, 2, 3), (1, 2, 3, 0)]
    Njobs = 2

    localjob = []
    for job_id in range(Njobs):
        dest_rank = job_id % MPI_SIZE

        if MPI_RANK == 0:
            sendjob = [job_id, Qs[job_id], _Qaxes_list[job_id], P, _Paxes_list[job_id]]

            if MPI_RANK != dest_rank:
                comm.send(obj=sendjob, dest=dest_rank, tag=dest_rank)
            else:
                localjob.append(sendjob)
        else:
            if MPI_RANK == dest_rank:
                localjob.append(
                    comm.recv(source=0, tag=MPI_RANK)
                )
            else:
                pass

    results = []
    for job_id, _Q, _Qaxes, _P, _Paxes in localjob:
        dest_rank = job_id % MPI_SIZE

        if MPI_RANK == dest_rank:
            _Q = xp.transpose(_Q, _Qaxes)
            _P = xp.transpose(_P, _Paxes)
            if job_id == 0:
                #Q_{u1, x12, t'12, l2} = usQ_{u1, x12, i}, svhQ_{i, t'12, l2}
                #P_{l1, u1, r1, d1} = usP_{l1, u1, j}, svhP_{j, r1, d1}
                us_Q, svh_Q, _  = svd(_Q, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
                us_P, svh_P, sP = svd(_P, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
                #M_{x12, i, l1, j} = usQ_{x12, u1, i} usP_{l1, u1, j} = M_{i, l1, j, x12}
                M = oe.contract("uxi,luj->iljx", us_Q, us_P)
                #M_{i, l1, j, x12} = usM_{i, l1, k} svhM_{k, j, x12}
                usM, svhM, sM = svd(M, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
                #A_{t'12, l2, l1, k} = svhQ_{i, t'12, l2} usM_{i, l1, k}
                A = oe.contract("itL,ilk->tLlk", svh_Q, usM)
                #B_{k, d1, x12, r1} = svhM_{k, j, x12} svhP_{j, r1, d1}
                B = oe.contract("kjx,jrd->kdxr", svhM, svh_P)
                results.append([A, B])

                if verbose:
                    maxsP = xp.max(sP)
                    maxsM = xp.max(sM)
                    print(f"Leg swapping I, singular values of P(I) (s1={maxsP:.6e}):\n", sP/maxsP)
                    print(f"Leg swapping I, singular values of M(I) (s1={maxsM:.6e}):\n", sM/maxsM)

            elif job_id == 1:
                #Q_{y20, u0, d2, t'20} = usQ_{y20, u0, i}, svhQ_{i, d2, t'20}
                #P_{u0, r0, d0, l0} = usP_{u0, r0, j}, svhP_{j, d0, l0}
                us_Q, svh_Q, _ = svd(_Q, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
                us_P, svh_P, sP = svd(_P, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
                #M_{i, r0, j, y20} = usQ_{y20, u0, i} usP_{u0, r0, j} = M_{i, r0, j, y20}
                M = oe.contract("yui,urj->irjy", us_Q, us_P)
                #M_{i, r0, j, y20} = usM_{i, r0, k} svhM_{k, j, y20}
                usM, svhM, sM = svd(M, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
                #C_{k, l0, d0, y20} = svh_P_{d0, l0, j} svhM_{k, y20, j}
                C = oe.contract("kjy,jdl->kdly", svhM, svh_P)
                #D_{t'20, d2, r0, k} = svhQ_{i, d2, t'20} usM_{i, r0, k}
                D = oe.contract("idt,irk->tdrk", svh_Q, usM)
                results.append([C, D])

                if verbose:
                    maxsP = xp.max(sP)
                    maxsM = xp.max(sM)
                    print(f"Leg swapping II, singular values of P(II) (s1={maxsP:.6e}):\n", sP/maxsP)
                    print(f"Leg swapping II, singular values of M(II) (s1={maxsM:.6e}):\n", sM/maxsM)

    results = comm.gather(sendobj=results, root=0)
    if MPI_RANK == 0:
        results = flatten_2dim_job_results(results, job_size=Njobs, comm=comm)
        results_1dim = []
        for job_id in range(Njobs):
            for result in results[job_id]:
                results_1dim.append(result)
    else:
        results_1dim = [None, None, None, None]
    del results

    return results_1dim


def env_tensor_3d_gauge(chi, A, B, C, D, P, Q, 
                             direction:str, 
                             truncate_eps:float, 
                             degeneracy_eps:float,
                             Env_chunks:tuple,
                             comm:MPI.Intercomm, 
                             use_gpu=False):
    if use_gpu:
        xp = cp
    else:
        xp = np

    MPI_RANK = comm.Get_rank()
    MPI_SIZE = comm.Get_size()


    if direction == "T" or direction == "X" or direction == "Y":
        if MPI_RANK == 0:
            #Part(1)_{aAbB} = B_{a, d1, x12, r1}  Q_{x01, r1, l0, y01} C_{b, l0, y20, d0}
            #                B†_{A, d1, x12, R1} Q_{x01, R1, L0, y01} C†_{B, L0, y20, d0}
            part1 = oe.contract("icda,jcdA,eabf,eABf,kgbh,lgBh->ijkl", B, xp.conj(B), Q, Q, C, xp.conj(C))
        else:
            part1 = None
    elif direction == "t" or direction == "x" or direction == "y":
        if MPI_RANK == 0:
            #part(1)_(aAbB) =  A_{t'12, l1, l2, a}  P_{l2, u2, r2, d2}  D_{t'20, d2, r0, b}
            #                 A†_{t'12, l1, L2, A} P†_{L2, u2, r2, D2} D†_{t'20, D2, r0, B}
            part1 = oe.contract("cdai,cdAj,aefb,AefB,gbhk,gBhl->ijkl", A, xp.conj(A), P, xp.conj(P), D, xp.conj(D))
        else:
            part1 = None

    if direction == "T":
        if MPI_RANK == 0:
            #Part(2)_{l2, L2, d2, D2} = P_{l2, u2, r2, d2} P†_{L2, u2, r2, D2}
            part2 = oe.contract("lurd,LurD->lLdD", P, xp.conj(P))
            us, svh, s =svd(part1, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
            #print("sv of T",s/xp.max(s))
        else:
            part2 = None
            us, svh = None, None
        gpu_syn(use_gpu)
        comm.barrier()
        del part1

        ops = None
        if MPI_RANK == 0:
            ops  = [A, D, part2, us, svh]
        ops = comm.bcast(ops, root=0)
        A, D, part2, us, svh = ops
        _, _, chii = us.shape
        slicing = contract_slicer(shape=(chii,), 
                                  chunk=Env_chunks, 
                                  comm=comm)
        slicing = list(slicing)
        subs = "mlCA,olca,cCdD,nDrB,pdrb,aAi,ibB->mnop"

        for n, (i, ) in enumerate(slicing):
            if n == 0:
                path, _ = oe.contract_path(subs, 
                                           xp.conj(A), A, 
                                           part2, 
                                           xp.conj(D), D, 
                                           us[:,:,i], svh[i,:,:], 
                                           optimize='optimal')
                break
        
        chis = None
        if MPI_RANK == 0:
            chiT12, chiT20 = A.shape[0], D.shape[0]
            chis = chiT12, chiT20
        chis = comm.bcast(chis, root=0)
        chiT12, chiT20 = chis
        del chis

        Env = xp.zeros(shape=(chiT12, chiT20, chiT12, chiT20), dtype=A.dtype)
        t0 = time.time()
        for n, (i, ) in enumerate(slicing):
            if n % MPI_SIZE == MPI_RANK:
                Env += oe.contract(subs, 
                                   xp.conj(A), A, 
                                   part2, 
                                   xp.conj(D), D, 
                                   us[:,:,i], svh[i,:,:], 
                                   optimize=path)
        Env = comm.reduce(Env, op=MPI.SUM, root=0)
        gpu_syn(use_gpu)
        comm.barrier()


    elif direction == "X":
        if MPI_RANK == 0:
            #Part(2)_{l2,L2,a,A} = A_{t'12, l1, l2, a} A†_{t'12, l1, L2, A}
            part2 = oe.contract("tila,tiLA->lLaA", A, xp.conj(A))
            Env12 = oe.contract("lLaA,aAbB->lLbB", part2, part1)
            us, svh, s = svd(Env12, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
            del Env12, part1, part2
            #print("sv of X",s/xp.max(s))
        else:
            us, svh = None, None
        gpu_syn(use_gpu)
        comm.barrier()

        ops = None
        if MPI_RANK == 0:
            ops  = [P, D, us, svh]
        ops = comm.bcast(ops, root=0)
        P, D, us, svh = ops
        _, _, chii = us.shape
        slicing = contract_slicer(shape=(chii,), 
                                  chunk=Env_chunks, 
                                  comm=comm)
        slicing = list(slicing)
        subs = "LumD,luod,lLi,ibB,tDnB,tdpb->mnop"

        for n, (i, ) in enumerate(slicing):
            if n == 0:
                path, _ = oe.contract_path(subs, 
                                           xp.conj(P), P, 
                                           us[:,:,i], svh[i,:,:], 
                                           xp.conj(D), D, 
                                           optimize='optimal')
                break

        chis = None
        if MPI_RANK == 0:
            chir2, chir0 = P.shape[2], D.shape[2]
            chis = chir2, chir0
        chis = comm.bcast(chis, root=0)
        chir2, chir0 = chis
        del chis

        Env = xp.zeros(shape=(chir2, chir0, chir2, chir0), dtype=P.dtype)
        for n, (i, ) in enumerate(slicing):
            if n % MPI_SIZE == MPI_RANK:
                Env += oe.contract(subs, 
                                   xp.conj(P), P, 
                                   us[:,:,i], svh[i,:,:], 
                                   xp.conj(D), D, 
                                   optimize=path)
        Env = comm.reduce(Env, op=MPI.SUM, root=0)
        gpu_syn(use_gpu)
        comm.barrier()

    elif direction == "Y":
        if MPI_RANK == 0:
            #Part(2)_{b,B,d,D} = D_{t'20, d2, r0, b} D†_{t'20, D2, r0, B}
            part2 = oe.contract("tdrb,tDrB->bBdD", D, xp.conj(D))
            Env12 = oe.contract("aAbB,bBdD->aAdD", part1, part2)
            us, svh, s = svd(Env12, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=truncate_eps, split=True)
            del Env12, part1, part2
            #print("sv of Y",s/xp.max(s))
        else:
            us, svh = None, None
        gpu_syn(use_gpu)
        comm.barrier()

        ops = None
        if MPI_RANK == 0:
            ops  = [A, P, us, svh]
        ops = comm.bcast(ops, root=0)
        A, P, us, svh = ops
        _, _, chii = us.shape
        slicing = contract_slicer(shape=(chii,), 
                                  chunk=Env_chunks, 
                                  comm=comm)
        slicing = list(slicing)
        subs = "tmLA,tola,aAi,idD,LnrD,lprd->mnop"

        for n, (i, ) in enumerate(slicing):
            if n == 0:
                path, _ = oe.contract_path(subs, 
                                           xp.conj(A), A, 
                                           us[:,:,i], svh[i,:,:], 
                                           xp.conj(P), P, 
                                           optimize='optimal')
                break

        chis = None
        if MPI_RANK == 0:
            chil1, chiu2 = A.shape[1], P.shape[1]
            chis = chil1, chiu2
        chis = comm.bcast(chis, root=0)
        chil1, chiu2 = chis
        del chis

        Env = xp.zeros(shape=(chil1, chiu2, chil1, chiu2), dtype=P.dtype)
        for n, (i, ) in enumerate(slicing):
            if n % MPI_SIZE == MPI_RANK:
                Env += oe.contract(subs, 
                                   xp.conj(A), A, 
                                   us[:,:,i], svh[i,:,:], 
                                   xp.conj(P), P, 
                                   optimize=path)
        Env = comm.reduce(Env, op=MPI.SUM, root=0)
        gpu_syn(use_gpu)
        comm.barrier()


    elif direction == 't':
        if MPI_RANK == 0:
            #Part(2)_{r1,R1,l0,L0} = Q_{x01, r1, l0, y01} Q†_{x01, R1, Ll0, y01}
            part2 = oe.contract("xrly,xRLy->rRlL", Q, xp.conj(Q))
            us, svh, s = svd(part2, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=truncate_eps, split=True)
            del part2
            #print("sv of t",s/xp.max(s))
        else:
            us, svh = None, None
        gpu_syn(use_gpu)
        comm.barrier()

        ops = None
        if MPI_RANK == 0:
            ops  = [B, C, part1, us, svh]
        ops = comm.bcast(ops, root=0)
        B, C, part1, us, svh = ops
        _, _, chii = us.shape
        slicing = contract_slicer(shape=(chii,), 
                                  chunk=Env_chunks, 
                                  comm=comm)
        slicing = list(slicing)
        subs = "amxr,AoxR,aAbB,bnly,BpLy,rRi,ilL->mnop"

        for n, (i, ) in enumerate(slicing):
            if n == 0:
                path, _ = oe.contract_path(subs, 
                                           B, xp.conj(B), 
                                           part1, 
                                           C, xp.conj(C), 
                                           us[:,:,i], svh[i,:,:], 
                                           optimize='optimal')
                break
        
        chis = None
        if MPI_RANK == 0:
            chid1, chid0 = B.shape[1], C.shape[1]
            chis = chid1, chid0
        chis = comm.bcast(chis, root=0)
        chid1, chid0 = chis
        del chis

        Env = xp.zeros(shape=(chid1, chid0, chid1, chid0), dtype=B.dtype)
        t0 = time.time()
        for n, (i, ) in enumerate(slicing):
            if n % MPI_SIZE == MPI_RANK:
                Env += oe.contract(subs, 
                                   B, xp.conj(B), 
                                   part1, 
                                   C, xp.conj(C), 
                                   us[:,:,i], svh[i,:,:], 
                                   optimize=path)
        Env = comm.reduce(Env, op=MPI.SUM, root=0)
        gpu_syn(use_gpu)
        comm.barrier()
    

    elif direction == 'x':
        if MPI_RANK == 0:
            #Part(2)_{b, B, l0, L0} = C_{b, d0, l0, y20} C†_{B, d0, L0, y20}
            part2 = oe.contract("bdly,BdLy->bBlL", C, xp.conj(C))
            Env12 = oe.contract("aAbB,bBlL->aAlL", part1, part2)
            us, svh, s = svd(Env12, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=truncate_eps, split=True)
            del Env12, part1, part2
            #print("sv of x",s/xp.max(s))
        else:
            us, svh = None, None
        gpu_syn(use_gpu)
        comm.barrier()

        ops = None
        if MPI_RANK == 0:
            ops  = [B, Q, us, svh]
        ops = comm.bcast(ops, root=0)
        B, Q, us, svh = ops
        _, _, chii = us.shape
        slicing = contract_slicer(shape=(chii,), 
                                  chunk=Env_chunks, 
                                  comm=comm)
        slicing = list(slicing)
        subs = "admr,AdoR,aAi,ilL,nrly,pRLy->mnop"

        for n, (i, ) in enumerate(slicing):
            if n == 0:
                path, _ = oe.contract_path(subs, 
                                           B, xp.conj(B), 
                                           us[:,:,i], svh[i,:,:], 
                                           Q, Q, 
                                           optimize='optimal')
                break

        chis = None
        if MPI_RANK == 0:
            chil1, chiu2 = B.shape[2], Q.shape[0]
            chis = chil1, chiu2
        chis = comm.bcast(chis, root=0)
        chil1, chiu2 = chis
        del chis

        Env = xp.zeros(shape=(chil1, chiu2, chil1, chiu2), dtype=B.dtype)
        for n, (i, ) in enumerate(slicing):
            if n % MPI_SIZE == MPI_RANK:
                Env += oe.contract(subs, 
                                   B, xp.conj(B), 
                                   us[:,:,i], svh[i,:,:], 
                                   Q, Q, 
                                   optimize=path)
        Env = comm.reduce(Env, op=MPI.SUM, root=0)
        gpu_syn(use_gpu)
        comm.barrier()


    elif direction == 'y':
        if MPI_RANK == 0:
            #Part(2)_{rRaA} = B_{a, d1, x12, r1} B†_{A, d1, x12, R1}
            part2 = oe.contract("adxr,AdxR->rRaA", B, xp.conj(B))
            Env12 = oe.contract("rRaA,aAbB->rRbB", part2, part1)
            us, svh, s = svd(Env12, shape=[[0,1], [2,3]], k=chi, truncate_eps=truncate_eps, degeneracy_eps=degeneracy_eps, split=True)
            del Env12, part1, part2
            #print("sv of y",s/xp.max(s))
        else:
            us, svh = None, None
        gpu_syn(use_gpu)
        comm.barrier()

        ops = None
        if MPI_RANK == 0:
            ops  = [Q, C, us, svh]
        ops = comm.bcast(ops, root=0)
        Q, C, us, svh = ops
        _, _, chii = us.shape
        slicing = contract_slicer(shape=(chii,), 
                                  chunk=Env_chunks, 
                                  comm=comm)
        slicing = list(slicing)
        subs = "xrlm,xRLo,rRi,ibB,bdln,BdLp->mnop"

        for n, (i, ) in enumerate(slicing):
            if n == 0:
                path, _ = oe.contract_path(subs, 
                                           Q, Q, 
                                           us[:,:,i], svh[i,:,:], 
                                           C, xp.conj(C),
                                           optimize='optimal')
                break

        chis = None
        if MPI_RANK == 0:
            chiy01, chiy20 = Q.shape[3], C.shape[3]
            chis = chiy01, chiy20
        chis = comm.bcast(chis, root=0)
        chiy01, chiy20 = chis
        del chis

        Env = xp.zeros(shape=(chiy01, chiy20, chiy01, chiy20), dtype=C.dtype)
        for n, (i, ) in enumerate(slicing):
            if n % MPI_SIZE == MPI_RANK:
                Env += oe.contract(subs, 
                                   Q, Q, 
                                   us[:,:,i], svh[i,:,:], 
                                   C, xp.conj(C),
                                   optimize=path)
        Env = comm.reduce(Env, op=MPI.SUM, root=0)
        gpu_syn(use_gpu)
        comm.barrier()

    return Env

    

def squeezer_3d_gauge(Dcut, 
                      Rl, Rr, 
                      truncate_eps, 
                      degeneracy_eps, 
                      use_gpu=False, 
                      verbose=False):
    if use_gpu:
        xp = cp
    else:
        xp = np

    RlRr = oe.contract("iab,abj->ij", Rl, Rr)
    k = min(*RlRr.shape, Dcut)
    U, S, VH = svd(RlRr, shape=[[0], [1]], k=k, 
                   truncate_eps=truncate_eps, 
                   degeneracy_eps=degeneracy_eps)
    UH = xp.conj(U.T)
    V  = xp.conj(VH.T)
    Sinv = 1 / S

    if verbose:
        Smax = xp.max(S)
        print(f"singular values of squeezer (S0={Smax}):\n", S / Smax)

    del U, S, VH

    Pl = oe.contract("abj,ji,i->abi", Rr, V, xp.sqrt(Sinv))
    Pr = oe.contract("i,ij,jab->iab", xp.sqrt(Sinv), UH, Rl)

    return Pl, Pr


def rsvd_3d_gauge(A, B, C, D, P, Q, Vs, 
                  k:int, p:int, q:int, 
                  truncate_eps:float,
                  degeneracy_eps:float,
                  chunk:tuple, 
                  comm:MPI.Intercomm, 
                  seed=None, 
                  use_gpu=False, 
                  verbose=False):
    if use_gpu:
        xp = cp
    else:
        xp = np

    MPI_RANK = comm.Get_rank()
    MPI_SIZE = comm.Get_size()

    _Vs = comm.bcast(Vs, root=0)
    VT, Vt, VX, Vx, VY, Vy = _Vs
    del _Vs

    chis = None
    if MPI_RANK == 0:
        chia, chib = A.shape[3], D.shape[3]
        chiT, chiX, chiY = VT.shape[2], VX.shape[2], VY.shape[2]
        chit, chix, chiy = Vt.shape[0], Vx.shape[0], Vy.shape[0]
        chil = k + p
        chis = [chia, chib, chiT, chiX, chiY, chit, chix, chiy, chil]
    chis = comm.bcast(chis, root=0)
    chia, chib, chiT, chiX, chiY, chit, chix, chiy, chil = chis
    del chis

    shape = (chia, chib)
    slices = contract_slicer(shape, chunk, comm)
    slices = list(slices)

    Y = None
    if MPI_RANK == 0:
        rs = xp.random.RandomState(seed)
        Y  = rs.standard_normal(size=(chit, chix, chiy, chil), dtype=xp.float64)
        Y  = Y.astype(complex)
    Y = comm.bcast(Y, root=0)
    A = comm.bcast(A, root=0)
    B = comm.bcast(B, root=0)
    C = comm.bcast(C, root=0)
    D = comm.bcast(D, root=0)
    P = comm.bcast(P, root=0)
    Q = comm.bcast(Q, root=0)

    for n, (lega, legb) in enumerate(slices):
        if n == 0:
            optimize = 'optimal'
            #path of TY
            pathI , _ = oe.contract_path("acdi,dCbe,AeBj,aAT,bBX,cCY->TXYij", 
                                         A[:,:,:,lega], P, D[:,:,:,legb], 
                                         VT, VX, VY, 
                                         optimize=optimize)
            pathII, _ = oe.contract_path("iabd,Bdec,jAeC,taA,xbB,ycC->ijtxy", 
                                         B[lega,:,:,:], Q, C[legb,:,:,:], 
                                         Vt, Vx, Vy, 
                                         optimize=optimize)
            pathTY = [(1,2), (0,1)]

            #path of T†Y
            pathIIdag, _ = oe.contract_path("iabd,Bdec,jAeC,taA,xbB,ycC->txyij", 
                                            B[lega,:,:,:], Q, C[legb,:,:,:], 
                                            Vt, Vx, Vy, 
                                            optimize=optimize)
            pathIdag , _ = oe.contract_path("acdi,dCbe,AeBj,aAT,bBX,cCY->ijTXY", 
                                            A[:,:,:,lega], P, D[:,:,:,legb], 
                                            VT, VX, VY, 
                                            optimize=optimize)
            pathTdagY = [(1,2), (0,1)]

            #path of X_{j, t, x, y} = Q*_{t', x', y', j} IX_{t', x', y', a, b} IIX_{a, b, t, x, y}
            pathX = [(0,1), (0,1)]

            break
    gpu_syn(use_gpu)
    comm.barrier()
    

    def cal_TY(Y):
        TY = xp.zeros(shape=(chiT, chiX, chiY, chil), dtype=A.dtype)
        t0 = time.time()
        if verbose and MPI_RANK == 0:
            print("Cal Y_{{TXYl}} = I_{{TXYab}} II_{{abtxy}} Y_{{txyl}}:")
        for n, (lega, legb) in enumerate(slices):
            if n % MPI_SIZE == MPI_RANK:
                I  = oe.contract("acdi,dCbe,AeBj,aAT,bBX,cCY->TXYij", 
                                 A[:,:,:,lega], P, D[:,:,:,legb], 
                                 VT, VX, VY, 
                                 optimize=pathI )
                II = oe.contract("iabd,Bdec,jAeC,taA,xbB,ycC->ijtxy", 
                                 B[lega,:,:,:], Q, C[legb,:,:,:], 
                                 Vt, Vx, Vy, 
                                 optimize=pathII)
                TY += oe.contract("TXYij,ijtxy,txyl->TXYl", 
                                  I, II, Y, 
                                  optimize=pathTY)
                
            if verbose:
                globaliter = n + 1
                localiter  = n // MPI_SIZE + 1
                if (n % (chia // MPI_SIZE) == (chia // MPI_SIZE) - 1) and (MPI_RANK == 0):
                    t1 = time.time()
                    print(f"Global iters:{globaliter}. Local iters:{localiter}. Time= {t1-t0:.2e} s")
                    t0 = time.time()
        Y = comm.reduce(sendobj = TY, op = MPI.SUM, root = 0)
        del TY, I, II
        gpu_syn(use_gpu)
        comm.barrier()
        return Y

    def cal_TdagY(Y):
        TdagY = xp.zeros(shape=(chit, chix, chiy, chil), dtype=A.dtype)
        t0 = time.time()
        if verbose and MPI_RANK == 0:
            print("Cal Y_{{txyl}} = II†_{{txyab}} I†_{{abTXY}} Y_{{TXYl}}:")
        for n, (lega, legb) in enumerate(slices):
            if n % MPI_SIZE == MPI_RANK:
                II = oe.contract("iabd,Bdec,jAeC,taA,xbB,ycC->txyij", 
                                 B[lega,:,:,:], Q, C[legb,:,:,:], 
                                 Vt, Vx, Vy, 
                                 optimize=pathIIdag)
                I  = oe.contract("acdi,dCbe,AeBj,aAT,bBX,cCY->ijTXY", 
                                 A[:,:,:,lega], P, D[:,:,:,legb], 
                                 VT, VX, VY, 
                                 optimize=pathIdag)
                TdagY += oe.contract("txyij,ijTXY,TXYl->txyl", 
                                     xp.conj(II), xp.conj(I), Y, 
                                     optimize=pathTdagY)
                
            if verbose:
                globaliter = n + 1
                localiter  = n // MPI_SIZE + 1
                if (n % (chia // MPI_SIZE) == (chia // MPI_SIZE) - 1) and (MPI_RANK == 0):
                    t1 = time.time()
                    print(f"Global iters:{globaliter}. Local iters:{localiter}. Time= {t1-t0:.2e} s")
                    t0 = time.time()
        Y = comm.reduce(sendobj = TdagY, op = MPI.SUM, root = 0)
        del TdagY, IIdag, Idag
        gpu_syn(use_gpu)
        comm.barrier()
        return Y
    
    #Compute compressed tensor
    Y = cal_TY(Y)
    Q_qr = None
    if MPI_RANK == 0:
        Y = xp.reshape(Y, (chiT*chiX*chiY, chil))
        Q_qr, _ = xp.linalg.qr(Y)
        if Q_qr.shape[0] < chiT*chiX*chiY:
            Q_qr = np.pad(Q_qr, 
                          pad_width = ((0, chiT*chiX*chiY - Q_qr.shape[0]), (0, 0)),
                          mode      = 'constant',
                          constant_values = 0.0)
        if Q_qr.shape[1] < chil:
            Q_qr = np.pad(Q_qr, 
                          pad_width = ((0, 0), (0, chil - Q_qr.shape[1])),
                          mode      = 'constant',
                          constant_values = 0.0)
        Q_qr = xp.reshape(Q_qr, (chiT, chiX, chiY, chil))
    Q_qr = comm.bcast(Q_qr, root=0)
    del Y
    gpu_syn(use_gpu)
    comm.barrier()

    #Start QR iteration
    Y = None
    for iq in range(q):
        if verbose:
            if (iq == 0) and (MPI_RANK == 0):
                print(f"QR iteration start.")
        t0 = time.time()

        Y = cal_TdagY(Q_qr)
        if MPI_RANK == 0:
            Y = xp.reshape(Y, (chit*chix*chiy, chil))
            Q_qr, _ = xp.linalg.qr(Y)
            Q_qr = xp.reshape(Q_qr, (chit, chix, chiy, chil))
        Q_qr = comm.bcast(Q_qr, root=0)
        gpu_syn(use_gpu)
        comm.barrier()

        Y = cal_TY(Q_qr)
        if MPI_RANK == 0:
            Y = xp.reshape(Y, (chiT*chiX*chiY, chil))
            Q_qr, _ = xp.linalg.qr(Y)
            Q_qr = xp.reshape(Q_qr, (chiT, chiX, chiY, chil))
        Q_qr = comm.bcast(Q_qr, root=0)
        gpu_syn(use_gpu)
        comm.barrier()
        t1 = time.time()

        if verbose:
            if MPI_RANK == 0:
                t1 = time.time()
                print(f"{iq}-th QR iteration finished. Time= {t1-t0:.2e} s")
                t0 = time.time()
    del Y

    #Cal X
    localX = xp.zeros(shape=(chil, chit, chix, chiy), dtype=A.dtype)
    t0 = time.time()
    if verbose and MPI_RANK == 0:
        print("Cal X_{{jtxy}} = Q†_{{j,TXY}} I_{{TXYab}} II_{{abtxy}}:")
    for n, (lega, legb) in enumerate(slices):
        if n % MPI_SIZE == MPI_RANK:
            I  = oe.contract("acdi,dCbe,AeBj,aAT,bBX,cCY->TXYij", 
                             A[:,:,:,lega], P, D[:,:,:,legb], 
                             VT, VX, VY, 
                             optimize=pathI )
            II = oe.contract("iabd,Bdec,jAeC,taA,xbB,ycC->ijtxy", 
                             B[lega,:,:,:], Q, C[legb,:,:,:], 
                             Vt, Vx, Vy, 
                             optimize=pathII)
            localX += oe.contract("TXYj,TXYab,abtxy->jtxy", 
                                  xp.conj(Q_qr), I, II, 
                                  optimize=pathX)
            
        if verbose:
            globaliter = n + 1
            localiter  = n // MPI_SIZE + 1
            if (n % (chia // MPI_SIZE) == (chia // MPI_SIZE) - 1) and (MPI_RANK == 0):
                t1 = time.time()
                print(f"Global iters:{globaliter}. Local iters:{localiter}. Time= {t1-t0:.2e} s")
                t0 = time.time()
    X = comm.reduce(localX, root=0)
    del localX, I, II
    gpu_syn(use_gpu)
    comm.barrier()

    U, S, VH = None, None, None
    if MPI_RANK == 0:
        U, S, VH = svd(X, shape=[[0], [1,2,3]], k=k, 
                       truncate_eps=truncate_eps,
                       degeneracy_eps=degeneracy_eps)

        U = oe.contract("TXYj,ji->TXYi", Q_qr, U)

    return U, S, VH
