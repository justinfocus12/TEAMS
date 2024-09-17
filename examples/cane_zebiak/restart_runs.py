import os
import numpy as np
import random
from multiprocessing import Pool
import datetime


# Run CZ model with perturbed initial conditions
def run_model(inputs):
    t_find = inputs[0] # Timestep of long run that will be used as initial conditions
    run = inputs[1] # Run number (string, eg. '001')
   
    # Generate an input txt file specific to the run
    os.chdir('/n/home09/spackman/Out/base')
    input_txt = [t_find+'\n', 'outhst_1\n', 'outhst_'+run]
    with open('input_'+run+'.txt', 'w') as file:
        file.writelines(input_txt)
   
    # Perturb start file
    os.system('~/el_nino/CZ_model/modify_restart.exe < input_'+run+'.txt')
   
    # Move to appropriate directory
    os.system('cp ~/Out/base/outhst_'+run+ \
    ' ~/el_nino/CZ_model/Experiments/Standard_'+run+'/outhst_'+run)
   
    # Run model
    os.chdir('/n/home09/spackman/el_nino/CZ_model')
    os.system('./run_exp '+run+' Standard_'+run)

# Read one snapshot of CZ model output data
def read_one_line(file_path, rows, cols):

    # read all variables at this time step:
    sst = np.frombuffer(file.read(line_size), dtype=dtype).reshape(rows, cols)
    taux = np.frombuffer(file.read(line_size), dtype=dtype).reshape(rows, cols)
    tauy = np.frombuffer(file.read(line_size), dtype=dtype).reshape(rows, cols)
    uo = np.frombuffer(file.read(line_size), dtype=dtype).reshape(rows, cols)
    vo = np.frombuffer(file.read(line_size), dtype=dtype).reshape(rows, cols)
    h1 = np.frombuffer(file.read(line_size), dtype=dtype).reshape(rows, cols)
    u1 = np.frombuffer(file.read(line_size), dtype=dtype).reshape(rows, cols)
    v1 = np.frombuffer(file.read(line_size), dtype=dtype).reshape(rows, cols)
    qf = np.frombuffer(file.read(line_size), dtype=dtype).reshape(rows, cols)
    TT = np.frombuffer(file.read(line_size), dtype=dtype).reshape(rows, cols)

    return sst

rows=30
cols=34
dtype = np.float32  # assuming each entry in the array is a float
line_size = rows * cols * np.dtype(dtype).itemsize

N = 16 # number of ensemble members
M = 48 # number of start conditions
T = 1200 # length of time series (months)
T_grads = 100 * 12 * 3 # length of time series (10 day time steps)
gap = 2400 # space between initial conditions

# Perform runs
all_ts = np.empty(shape=(M,N,T_grads,6,12)) # for storing full time series
nino3 = np.empty(shape=(M,N,T)) # for storing nino3 only
t_find = 1200.5 - gap

for m in range(M):
    start = datetime.datetime.now()
   
    # Set up list of arguments to pass to function that runs the model
    t_find = t_find + gap
    args = []
    for n in range(N):
        run = str(n).zfill(3)
        args.append([str(t_find), run])
       
    # Perturb start file, run model
    result = Pool().map(run_model, args)
   
    # Extract time series data for later
    for i in range(N):
        run = str(i).zfill(3)
        file_path = '/n/home09/spackman/Out/'+run+'/grads_'+run+'.data'
        file = open(file_path,'rb')
       
        # Save nino3 data
        nino3_tmp = np.genfromtxt(fname='/n/home09/spackman/Out/'+run+'/nino3_'+run+'.dat')
        nino3[m,i,:] = nino3_tmp
       
        for t in range(T_grads):
            # Get one snapshot of SST data, flatten to 1D
            sst = read_one_line(file,rows,cols)[12:18,19:31]
           
            # Store data point
            all_ts[m,i,t,:,:] = sst
       
        file.close()
   
   # For keeping track of how long each ensemble takes  
    end = datetime.datetime.now() 
    print(str(m)+': ', end-start)

np.save('/n/home09/spackman/Out/Results/all_ts.npy', all_ts)
np.save('/n/home09/spackman/Out/Results/nino3.npy', nino3)
