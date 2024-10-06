# Instantiation of EnsembleMember class on Cane-Zebiak model 

import numpy as np
import xarray as xr
import f90nml
from matplotlib import pyplot as plt, rcParams 
rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
import os
from os.path import join, exists, basename
from os import mkdir, makedirs
import sys
import shutil
import glob
import subprocess
import resource
import pickle
import copy as copylib
import pprint

import sys
sys.path.append("../..")
import utils 
from dynamicalsystem import DynamicalSystem
import forcing

class CaneZebiak(DynamicalSystem):
    # TODO for SP: implement concrete versions of the abstract methods of DynamicalSystem, as stipulated in "../../dynamicalsystem.py". They are listed below with explanatory comments. 
    # The FriersonGCM class in ../frierson_gcm/frierson_gcm.py has some additional methods, like default_config() and default_namelist(), that might also be helpful here, but are not strictly required. 
    def __init__(self, cfg):
        self.dt_save = cfg['dt_save']
        # TODO for SP: optionally compute some other model parameters that might be variable between experiments, and store them as instance variables.
        return
    
    def generate_default_icandf(self,run,init_time,fin_time,seed=None): 
        """
        Inputs
        - init_time: time corresponding to an initial condition (should be an integer, specifying the number of time units since time 0)
        - fin_time: time (integer) when the integration should finish
        - seed: integer for random-number generator to create the random force

        Outputs
        - icandf of the format specified in the inputs to run_trajectory

        Side effects: none (unless needed to calculate outputs). 
        """
        ### Added as an input:
        ### run: an integer that will become this specific run's label. This is what will need to be changed to 
        ### create multiple unique runs, or it could be hard-coded into something. Can discuss Fri
        
        ## Make start files for new run
        # Copy everything from standard directory
        run_label = str(run).zfill(3)
        rundir = '~/el_nino/CZ_model/Experiments/Standard_'+run_label
        os.system('cp -r ~/el_nino/CZ_model/Experiments/Standard '+rundir)
        
        # Rename each file with correct label
        os.system('mv '+rundir+'/fc_1.data '+rundir+'/fc_'+run_label+'_tmp.data')
        os.system('mv '+rundir+'/modified_means.namelist_1 '+rundir+'/modified_means.namelist_'+run_label)
        os.system('mv '+rundir+'/out1 '+rundir+'/out_'+run_label)
        os.system('mv '+rundir+'/scales_EOF.namelist_1 '+rundir+'/scales_EOF.namelist_'+run_label)
        
        # Specify end time in fc file as fin_time
        os.chdir('/n/home09/spackman/el_nino/CZ_model/Experiments/Standard_'+run_label)
        with open('fc_'+run_label+'_tmp.data', 'r') as file:
            fc = file.readlines()
            fc[10] = 'TENDD  =  '+str(fin_time)+'\n'
            
        with open('fc_'+run_label+'.data', 'w') as ff:
            file.writelines(fc)
        
        ## Make perturbed start file at time init_time
        # Change to base directory (where 100000 yr run is stored)
        os.chdir('/n/home09/spackman/Out/base')
        
        # Make input text file for perturbation
        input_txt = [str(init_time)+'\n', 'outhst_1\n', 'outhst_'+run_label]
        with open('input_'+run_label+'.txt', 'w') as file:
            file.writelines(input_txt)
        
        # Perturb start file
        os.system('~/el_nino/CZ_model/modify_restart.exe < input_'+run_label+'.txt')
        
        # Move to directory with rest of start files
        os.system('cp ~/Out/base/outhst_'+run_label+' ~/el_nino/CZ_model/Experiments/Standard_'+run_label+\
                  '/outhst_'+run_label)

        ## Record-keeping
        default_icandf = dict({"run_label": run_label,
                               "output_dir" : "/n/home09/spackman/Out",
                               "t_find": init_time,
                               "t_end": fin_time})
        return default_icandf
    @staticmethod
    def get_timespan(metadata):
        pass
    @staticmethod
    def observable_props():
        """
        Convenience function for naming and plotting observable functions
        Inputs: none
        Outputs: a dictionary with keys corresponding to observable names and values corresponding to dictionaries of labels. A couple of rough examples are given below. 
        """
        obsprop = dict()
        obsprop["nino3"] = dict({
            "abbrv": "NINO3",
            "unit_symbol": "C",
            "label": "Nino-3 index",
            "cmap": "coolwarm",
            })
        obsprop["umax"] = dict({
            "abbrv": "UMAX",
            "unit_symbol": "m/s",
            "label": "Maximum zonal velocity",
            "cmap": "coolwarm",
            })
        return obsprop
    def compute_pairwise_observables(self, pair_funs, md0, md1list, root_dir):
        """
        Not needed now (helpful for comparison of two trajectories, e.g., Euclidean distance, mean)
        """
        return 
    def compute_observables(self, obs_funs, metadata, root_dir):
        # obs_names must correspond to class methods
        """
        Post-processing convenience function, not needed now (helpful for batch calculation of multiple low-dimensional observables without reading the output files repeatedly)
        """
        return
    def run_trajectory(self, icandf, obs_fun, saveinfo, root_dir):
        """
        Inputs
        - icandf (for 'initial conditions and forcing'): dictionary with following items
            a. init_cond: filename containing the initial condition, e.g., a restart leftover from a spinup run, BEFORE perturbation (because later ensemble members might want to launch from the same file with their own perturbation).
            b. frc: an object of type CaneZebiakForcing (see the class below). 
        - obs_fun: an "observable function" that the external calling function might want to use immediately for deciding what to do next, without having to read the cumbersome output files produced by the model. If not needed, just set obs_fun = lambda model_output: None
        - saveinfo: dictionary specifying output filenames, temporary directories, etc.; whatever is needed to specify where to store the output. In case you want to move the whole output directory elsewhere later, I suggest making these paths relative to the final argument, root_dir.
        - root_dir: the path relative to which saveinfo is specified. 

        Outputs: 
        - metadata: all info needed to recreate the output (suggested format below). 
        - observables: result of evaluating obs_fun on the integration.

        Side effects: model gets integrated and saved to the file system. 

        """
        ## Actually run model
        run_label = icandf["run_label"]
        os.chdir('/n/home09/spackman/el_nino/CZ_model')
        os.system('./run_exp '+run_label+' Standard_'+run_label)
        observables = dict({"nino_3" : np.genfromtxt(root_dir+run_label+'/nino3_'+run_label+'.dat')})
        
        metadata = dict({
            'icandf': icandf, 
            'filename_traj': join(root_dir,saveinfo['filename_traj']),
            'filename_restart': join(root_dir,saveinfo['filename_restart']),
            })
        return metadata, observables

class CaneZebiakForcing(forcing.Forcing):
    # TODO for SP: implement concrete methods implementing the abstract methods of Forcing, as stipulated in "../../forcing.py". (There's only one: get_forcing_times(), which is necessary for the algorithm to decide when and how to branch next). Beyond that, devise a compact data structure that fully specifies the times and shapes of perturbations, which can then be used in CaneZebiak.run_trajectory(). 
    def get_forcing_times():
        pass




if __name__ == "__main__":
    cfg = dict({
        # for any physical parameters you may want to systematically vary; could be none
        'dt_save': (1/12), # CZ model is set up in units of months
        }) 
    cz = CaneZebiak(cfg)
    # specify a timespan in physical units
    init_time_phys = 100
    fin_time_phys = 100
    # convert to integer save-out times 
    init_time = int(round(init_time_phys/cz.dt_save))+0.5 
    fin_time = int(round(fin_time_phys/cz.dt_save))+0.5 # quirk of CZ model
    seed = 98304
    run = 10
    run_label=str(run).zfill(3)
    icandf = cz.generate_default_icandf(10,init_time,fin_time,seed=seed)
    obs_fun = lambda model_output: None

    # create a place to store output
    root_dir = "/n/home09/spackman/Out/" 
    saveinfo = dict({
        'temp_directory': root_dir + run_label,
        'filename_traj': 'grads_'+run_label+'.data', 
        'filename_restart': 'outhst_'+run_label,
        })
        
    cz.run_trajectory(icandf, obs_fun, saveinfo, root_dir)
