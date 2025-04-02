
import opt_einsum as oe
import numpy as np
import time
import cupy as cp

from mpi4py import MPI
#WORLD_MPI_COMM = MPI.COMM_WORLD
#node = MPI.Get_processor_name()

def SU2_matrix_double_exponential_formula(Kθ, hθ, Kα, Kβ, comm : MPI.Intercomm, to_cupy = False):
    WORLD_MPI_COMM = comm
    WORLD_MPI_RANK = WORLD_MPI_COMM.Get_rank()
    WORLD_MPI_SIZE = WORLD_MPI_COMM.Get_size()
    info = WORLD_MPI_COMM.Get_info()

    #Kθ = 2 * Nθ + 1
    Nθ = (Kθ - 1) // 2

    α, wa = np.arange(-1, 1, 2 / Kα), np.full(Kα, 1 / Kα)
    β, wb = np.arange(-1, 1, 2 / Kβ), np.full(Kβ, 1 / Kβ)
    α = np.asarray(np.pi * (α + 1))
    β = np.asarray(np.pi * (β + 1))

    t = np.linspace(-Nθ * hθ, Nθ * hθ, num = Kθ, endpoint = True)
    θ = (np.tanh((np.pi / 2) * np.sinh(t)) + 1) * (np.pi / 4)
    wt = hθ * (np.pi / 2) * np.cosh(t) / (np.cosh((np.pi / 2) * np.sinh(t))**2)

    epia = np.exp( 1j*α)
    epib = np.exp( 1j*β)
    emia = np.exp(-1j*α)
    emib = np.exp(-1j*β)
    st = np.sin(θ)
    ct = np.cos(θ)
    It = np.ones(shape=Kθ, dtype=complex)
    Ia = np.ones(shape=Kα, dtype=complex)
    Ib = np.ones(shape=Kβ, dtype=complex)

    #Uij = Uij(θ, α, β)
    U = np.zeros(shape=(2, 2, Kθ, Kα, Kβ), dtype=complex)
    subscript = "θ,α,β->θαβ"
    U[0,0] =  oe.contract(subscript, ct, epia, Ib)
    U[0,1] =  oe.contract(subscript, st, Ia, epib)
    U[1,0] = -oe.contract(subscript, st, Ia, emib)
    U[1,1] =  oe.contract(subscript, ct, emia, Ib)
    
    I = np.zeros_like(U)
    I[0,0] = oe.contract(subscript, It, Ia, Ib)
    I[1,1] = oe.contract(subscript, It, Ia, Ib)

    #w[0] = contract("α,α,α->α", cp.sin(theta), cp.cos(theta), w[0])
    #Jacobian = (π/8) * sin(θ)cos(θ)
    Jt = st*ct
    Ja = Ia
    Jb = Ib
    J = oe.contract(subscript, Jt, Ja, Jb) * (np.pi / 8)
    w = oe.contract(subscript, wt, wa, wb)
    #weight

    U = np.reshape(U, newshape=(2, 2, Kθ * Kα * Kβ))
    I = np.reshape(I, newshape=(2, 2, Kθ * Kα * Kβ))
    w = np.reshape(w, newshape=(Kθ * Kα * Kβ))
    J = np.reshape(J, newshape=(Kθ * Kα * Kβ))
    if to_cupy:
        import cupy as cp
        U = cp.asarray(U)
        I = cp.asarray(I)
        w = cp.asarray(w)
        J = cp.asarray(J)
    
    return U, w, J, I


def SU2_matrix_Gauss_Legendre_quadrature(Kθ:int, Kα:int, Kβ:int, comm:MPI.Intercomm, to_cupy=False):
    WORLD_MPI_COMM = comm
    WORLD_MPI_RANK = WORLD_MPI_COMM.Get_rank()
    WORLD_MPI_SIZE = WORLD_MPI_COMM.Get_size()
    info = WORLD_MPI_COMM.Get_info()
    
    from scipy.special import roots_legendre
    θ, wt = roots_legendre(Kθ)
    α, wa = roots_legendre(Kα)
    β, wb = roots_legendre(Kβ)
    θ = np.asarray(np.pi * (θ + 1) / 4)
    α = np.asarray(np.pi * (α + 1))
    β = np.asarray(np.pi * (β + 1))

    epia = np.exp( 1j*α)
    epib = np.exp( 1j*β)
    emia = np.exp(-1j*α)
    emib = np.exp(-1j*β)
    st = np.sin(θ)
    ct = np.cos(θ)
    It = np.ones(shape=Kθ, dtype=complex)
    Ia = np.ones(shape=Kα, dtype=complex)
    Ib = np.ones(shape=Kβ, dtype=complex)

    #Uij = Uij(θ, α, β)
    U = np.zeros(shape=(2, 2, Kθ, Kα, Kβ), dtype=complex)
    subscript = "θ,α,β->θαβ"
    U[0,0] =  oe.contract(subscript, ct, epia, Ib)
    U[0,1] =  oe.contract(subscript, st, Ia, epib)
    U[1,0] = -oe.contract(subscript, st, Ia, emib)
    U[1,1] =  oe.contract(subscript, ct, emia, Ib)
    
    I = np.zeros_like(U)
    I[0,0] = oe.contract(subscript, It, Ia, Ib)
    I[1,1] = oe.contract(subscript, It, Ia, Ib)

    #w[0] = contract("α,α,α->α", cp.sin(theta), cp.cos(theta), w[0])
    #Jacobian = (π/8) * sin(θ)cos(θ)
    Jt = st*ct
    Ja = Ia
    Jb = Ib
    J = oe.contract(subscript, Jt, Ja, Jb) * (np.pi / 8)
    w = oe.contract(subscript, wt, wa, wb)
    #weight

    U = np.reshape(U, newshape=(2, 2, Kθ * Kα * Kβ))
    I = np.reshape(I, newshape=(2, 2, Kθ * Kα * Kβ))
    w = np.reshape(w, newshape=(Kθ * Kα * Kβ))
    J = np.reshape(J, newshape=(Kθ * Kα * Kβ))
    if to_cupy:
        import cupy as cp
        U = cp.asarray(U)
        I = cp.asarray(I)
        w = cp.asarray(w)
        J = cp.asarray(J)
    
    return U, w, J, I

def norm_for_elements(x):
    if type(x) == np.ndarray:
        sqrt = np.sqrt
    else:
        import cupy as cp
        sqrt = cp.sqrt

    norm = sqrt((x.real)**2 + (x.imag)**2)
    return norm

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
    norm_{U0† U1† U2 U3} = ||1 - U0† U1† U2 U3}||, and bool indices that dose not satisfy ||1 - U0† U1† U2 U3}|| < ε

    -------
    """
    if type(TrP) == np.ndarray:
        sqrt = np.sqrt
        abs  = np.abs
    else:
        from cupy import sqrt, abs
    
    norm = 4 - 2 * TrP.real
    #assert linalgnorm(norm.imag) < 1e-12, "2 - 2*Re(P00) - 2*Re(P11) + Tr(PP†) must be real!"
    norm = sqrt(abs(norm))
    index = norm > ε
    return norm, index

def plaquette_contraction_for_hosvd(β:float, ε:float|None, U, w, J, leg_hosvd, iteration, 
                                    comm:MPI.Intercomm, use_gpu=False, verbose=False):
    WORLD_MPI_COMM = comm
    WORLD_MPI_RANK = WORLD_MPI_COMM.Get_rank()
    WORLD_MPI_SIZE = WORLD_MPI_COMM.Get_size()
    info = WORLD_MPI_COMM.Get_info()

    if use_gpu:
        from cupy import zeros, conj, exp, inf, sqrt
    else:
        from numpy import zeros, conj, exp, inf, sqrt
    
    N = U.shape[2]
    M_local = zeros(shape=(N, N), dtype=complex)

    subscripts = ["ijkl,Ijkl->iI", "ijkl,iJkl->jJ", "ijkl,ijKl->kK", "ijkl,ijkL->lL"]

    #WORLD_MPI_COMM.barrier()
    #if use_gpu:
    #    cp.cuda.get_current_stream().synchronize()
    t0 = time.time() if WORLD_MPI_RANK == 0 else None
    t00 = time.time() if WORLD_MPI_RANK == 0 else None
    for n, i in enumerate(iteration):
        i0, i1, i2, i3 = i
        if n % WORLD_MPI_SIZE == WORLD_MPI_RANK:

            TrP = oe.contract("dci,adj,abk,bcl->ijkl", conj(U[:,:,i0]), conj(U[:,:,i1]), U[:,:,i2], U[:,:,i3])
            
            if ε is not None:
                #P = oe.contract("abi,bcj,dck,edl->aeijkl", conj(U[:,:,i0]), conj(U[:,:,i1]), U[:,:,i2], U[:,:,i3])
                norm, idx = admissibility_condition(TrP, ε)
                A = (1 - 0.5*TrP.real) / (1 - norm / ε)
                #A[idx] = 1e+100
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
                if (n > 0) and (n % (25*WORLD_MPI_SIZE) == 0) and (WORLD_MPI_RANK == 0):
                    t1 = time.time() if WORLD_MPI_RANK == 0 else None
                    print(f"Global iters:{n}. Local iters:{n // WORLD_MPI_SIZE}. {(t1-t0) / (n // WORLD_MPI_SIZE):.2e} sec/local_iter") #. Size of A: {A.nbytes/(1024**3):.2e} Gbytes")
                    #print("norm(1-P)", norm)
                    t0 = time.time() if WORLD_MPI_RANK == 0 else None
    #WORLD_MPI_COMM.barrier()
    #if use_gpu:
    #    cp.cuda.get_current_stream().synchronize()
    t11 = time.time() if WORLD_MPI_RANK == 0 else None
    if WORLD_MPI_RANK == 0:
        print(f"Tot iteration {n+1}. Time= {t11-t00:.2e} s")

    if use_gpu:
        cp.cuda.get_current_stream().synchronize()
    M = WORLD_MPI_COMM.reduce(sendobj=M_local, op=MPI.SUM, root=0)
    WORLD_MPI_COMM.barrier()

    return M