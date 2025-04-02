import sys
import os


import time

if __name__ == "__main__":
    
    xloop = int(sys.argv[1])
    yloop = int(sys.argv[2])
    Dcut  = int(sys.argv[3])
    K     = int(sys.argv[4])

    beta  = float(sys.argv[5])
    mu    = float(sys.argv[6])


    gilt_eps = float(os.environ['GILT_EPS'])
    output_dir = os.environ['OUTPUT_DIR']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    outputname = output_dir + '/result.dat'

    
    import measurement.O3_nlsm_measuremet as O3nlsm_measurement

    #time_start = time.time()
    #lnZ_OVER_V, n = O3nlsm_measurement.particle_number(beta, mu, Dcut, xloop, yloop)
    #time_finish = time.time()
    #
    #output = open(outputname, "w")
    #print("β={:}, μ={:}, lnZ/V={:.12e}, n={:12e}, time={:.2f}s".format(sys.argv[5], sys.argv[6], 
    #                                                                lnZ_OVER_V, n, time_finish-time_start))
    #output.write("{:} {:} {:19.12e} {:19.12e}\n".format(beta, mu, lnZ_OVER_V.real, n.real))

    time_start = time.time()
    lnZ_OVER_V, e = O3nlsm_measurement.internal_energy(beta, mu, Dcut, xloop, yloop)
    time_finish = time.time()
    
    output = open(outputname, "w")
    print("β={:}, μ={:}, lnZ/V={:.12e}, e={:12e}, time={:.2f}s".format(sys.argv[5], sys.argv[6], 
                                                                    lnZ_OVER_V, e, time_finish-time_start))
    output.write("{:} {:} {:19.12e} {:19.12e}\n".format(beta, mu, lnZ_OVER_V.real, e.real))


    output.close()
    