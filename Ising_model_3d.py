import sys
import os


import time

if __name__ == "__main__":
    
    xloop = int(sys.argv[1])
    yloop = int(sys.argv[2])
    tloop = int(sys.argv[3])
    Dcut  = int(sys.argv[4])

    beta  = float(sys.argv[5])
    h     = float(sys.argv[6])

    gilt_eps = float(os.environ['GILT_EPS'])
    output_dir = os.environ['OUTPUT_DIR']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    outputname = output_dir + '/result.dat'

    
    import measurement.Ising_model_3d as Ising3d_measurement

    #time_start = time.time()
    #lnZ_OVER_V = Ising3d_measurement.ln_Z_over_V(beta, h, Dcut, xloop, yloop, tloop)
    #time_finish = time.time()
    #
    #output = open(outputname, "w")
    #print("β={:}, h={:}, lnZ/V={:.12e}, time={:.2f}s".format(sys.argv[5], sys.argv[6], lnZ_OVER_V, time_finish-time_start))
    #output.write("{:} {:} {:19.12e}\n".format(beta, h, lnZ_OVER_V.real))
    #output.close()


    time_start = time.time()
    lnZ_OVER_V, m = Ising3d_measurement.magnetization(beta, h, Dcut, xloop, yloop, tloop)
    time_finish = time.time()
    
    output = open(outputname, "w")
    print("β={:}, h={:}, lnZ/V={:.12e}, m={:.12e}, time={:.2f}s".format(sys.argv[5], sys.argv[6], lnZ_OVER_V, m, time_finish-time_start))
    output.write("{:} {:} {:19.12e} {:19.12e}\n".format(beta, h, lnZ_OVER_V.real, m.real))
    output.close()


    #time_start = time.time()
    #lnZ_OVER_V, e = Ising3d_measurement.internal_energy(beta, h, Dcut, xloop, yloop, tloop)
    #time_finish = time.time()
    #
    #output = open(outputname, "w")
    #print("β={:}, h={:}, lnZ/V={:.12e}, e={:.12e}, time={:.2f}s".format(sys.argv[5], sys.argv[6], lnZ_OVER_V, e, time_finish-time_start))
    #output.write("{:} {:} {:19.12e} {:19.12e}\n".format(beta, h, lnZ_OVER_V.real, e.real))
    #output.close()
    