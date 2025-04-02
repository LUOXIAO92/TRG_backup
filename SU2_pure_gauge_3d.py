import sys
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
info = comm.Get_info()
node = MPI.Get_processor_name()
size = comm.Get_size()



import numpy as np
import cupy as cp
import opt_einsum as oe
from itertools import product
import time 

if __name__ == "__main__":

    for i in range(size):
        if rank == i:
            print("node:",node,"rank:",rank,"size:",size,"info:",info)
        comm.Barrier()
    comm.Barrier()
    
    xloop = int(sys.argv[1])
    yloop = int(sys.argv[2])
    tloop = int(sys.argv[3])
    Dcut  = int(sys.argv[4])
    K     = int(sys.argv[5])

    beta  = float(sys.argv[6])
    eps   = float(sys.argv[7])

    quadrature = sys.argv[8]
    mesh_size  = float(sys.argv[9])

    gilt_eps = float(os.environ['GILT_EPS'])
    output_dir = os.environ['OUTPUT_DIR']
    if quadrature == "DE":
        Kt = 2 * (K // 2) + 1
    else:
        Kt = K

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    outputname = output_dir + '/result.dat'


    from tools.mpi_tools import use_gpu
    use_gpu(usegpu=True, comm=comm)

    from tensor_init.SU2_pure_gauge import SU2_pure_gauge
    eps = eps if eps > 1e-13 else None
    su2gauge = SU2_pure_gauge(dim        = 3         , 
                              Dcut       = Dcut      , 
                              Ks         = (Kt, K, K), 
                              β          = beta      , 
                              ε          = eps       , 
                              comm       = comm      , 
                              use_gpu    = True      ,
                              quadrature = quadrature,
                              mesh_size  = mesh_size )
    
    import measurement.SU2_pure_gauge_3d_measurement as su2_gauge_m

    time_start = time.time()
    lnZ_OVER_V = su2_gauge_m.ln_Z_over_V(su2gauge, xloop, yloop, tloop, comm)
    time_finish = time.time()

    if rank == 0:
        output = open(outputname, "w")
        print("β={:}, ε={:}, lnZ/V={:.12e}, time={:.2f}s".format(sys.argv[6], sys.argv[7],
                                                                 lnZ_OVER_V, time_finish-time_start))
        output.write("{:} {:19.12e}\n".format(beta, lnZ_OVER_V.real))
    
        output.close()