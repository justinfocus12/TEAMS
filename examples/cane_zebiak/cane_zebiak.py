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
    def generate_default_icandf(self,init_time,fin_time,seed=None): 
        """
        Inputs
        - init_time: time corresponding to an initial condition (should be an integer, specifying the number of time units since time 0)
        - fin_time: time (integer) when the integration should finish
        - seed: integer for random-number generator to create the random force

        Outputs
        - icandf of the format specified in the inputs to run_trajectory

        Side effects: none (unless needed to calculate outputs). 
        """
        pass
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
            "unit_symbol": "K",
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
        'dt_save': 0.5, # years??
        }) 
    cz = CaneZebiak(cfg)
    # specify a timespan in physical units
    init_time_phys = 0.0
    fin_time_phys = 100.0
    # convert to integer save-out times 
    init_time = int(round(init_time_phys/cz.dt_save))
    fin_time = int(round(fin_time_phys/cz.dt_save))
    seed = 98304
    icandf = cz.generate_default_icandf(init_time,fin_time,seed=seed)
    obs_fun = lambda model_output: None

    # create a place to store output
    root_dir = "home/atmospheritas/elitas/sarahitas/canitas/zebiakitas" # Random question: does Harvard dining serve carnitas/sofritas?  
    saveinfo = dict({
        'temp_directory': 'mem0_tempdir',
        'filename_traj': 'mem0_history.nc', 
        'filename_restart': 'mem0_restart.gz',
        })
        
    cz.run_trajectory(icandf, obs_fun, saveinfo, root_dir)




