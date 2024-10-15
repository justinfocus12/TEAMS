1. Run DNS
    a. In dns_lorenz96.py, in the block at the bottom 'if __name__ == "__main__"', change expt_supdir to your own directory (I constructed mine uising today's date and a sub-date experiment number).
    b. In the terminal, enter "dns_lorenz96.py single <i_expt>" where i_expt is an integer encoding the noise level, as an index in the list of available forcing levels F4s specified near the top in dns_multiparams(). This will run a DNS consisting of a sequence of chunks with a fixed duration, both of which you can modify in the function dns_paramset in the dictionary config_algo. 
    c. To run all four forcing levels in parallel, in the terminal enter "sbatch --array=0-4 dns.sbatch" (after modifying the desired output paths as needed in dns.sbatch).
    d. See the "plots" subdirectory of expt_supdir to see some summary plots
 
2. Run TEAMS
3. 
