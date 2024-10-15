1. Run DNS
    a. In dns_lorenz96.py, in 'if __name__ == "__main__"', change expt_supdir to your own directory (I constructed mine uising today's date and a sub-date experiment number).
    b. In the terminal, enter "dns_lorenz96.py single <i_expt>" where i_expt is an integer encoding the noise level, as an index in the list of available forcing levels F4s specified near the top in dns_multiparams(). This will run a DNS consisting of a sequence of chunks with a fixed duration, both of which you can modify in the function dns_paramset in the dictionary config_algo. 
    c. To run all four forcing levels in parallel, in the terminal enter "sbatch --array=0-4 dns.sbatch" (after modifying the desired output paths as needed in dns.sbatch).
    d. See the "plots" subdirectory of expt_supdir to see some summary plots
 
2. Run TEAMS
    a. In teams_lorenz96_coldstart.py, in 'if __name__ == "__main__"', change expt_supdir_dns to match what you have above, and change expt_supdir_teams to wherever you want the TEAMS output saved. (I have them in the same place; they'll create their own sub-directories). 
    b. In the terminal, enter "teams_lorenz96_coldstart.py single <i_expt>" where i_expt is an integer representing a flat index in the multi-dimensional array of parameter settings laid out in "teams_multiparams()":
        i. F4s: stochastic forcing strengths
        ii. deltas_phys: advance split time in *physical* units
        iii. drop_params: three simple specifications of the level-raising schedule. Each specification is an ordered pair (drop_sched,drop_rate)
            - ('frac',0.37): drop a fraction 0.37 of active members at each round.
            - ('num',3): drop 3 active members at each round.
            - ('frac_then_num',(0.5,2)): drop half the members at the first round, then 2 members in each subsequent round.
            - Feel free to add your own schedule designs. If you do, two additional code modifications are necessary in ../../algorithms.py:
                * TEAMS.raise_level_replenish_queue() (c. lines 1245-1253): add a new 'elif "your_new_schedule_name" == self.drop_sched' to the if-statements to compute a new num2drop, given the set of active scores. Make sure you arrive at an integer no smaller than 1. 
                * TEAMS.label_from_config (c. lines 1056-1087): add a new 'elif config["drop_sched"] == "your_new_schedule_name"' and specify a drop_label and a drop_abbrv for annotating figures and output directory names, respectively. 
    c. After it runs, a directory <expt_supdir_teams>/si<seed_inc> will be generated, under which "data" holds the trajectory output files; "plots" has some spaghetti and hovmoller diagrams; and "analysis" has nothing for the time being. 
    d. To do the above steps in a batch, run "sbatch --array=0-47 teams_run.sbatch" (1 F4 value x 2 delta values x 3 drop_params values x 8 seed inc values)
3. Analyze the output and compare 
    a. In the terminal, enter "teams_lorenz96_coldstart.py multiseed <i_expt>" where i_expt now represent a flat index in the same multi-dimensional array as above, *but excluding the seed array*. This will create a directory <expt_supdir_teams>/multiseed, and therein a return level vs. return period plot (one with error bars in each direction). 


Many other plotting utilities are available for post-analysis, but let's start here!
    
