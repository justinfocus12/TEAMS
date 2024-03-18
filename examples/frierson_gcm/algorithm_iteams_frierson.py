i = 0
print(f'--------------Beginning imports-------------')
import numpy as np
print(f'{i = }'); i += 1
from numpy.random import default_rng
print(f'{i = }'); i += 1
import xarray as xr
print(f'{i = }'); i += 1
from matplotlib import pyplot as plt, rcParams 
print(f'{i = }'); i += 1
rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
import os
print(f'{i = }'); i += 1
from os.path import join, exists, basename, relpath
print(f'{i = }'); i += 1
from os import mkdir, makedirs
print(f'{i = }'); i += 1
import sys
print(f'{i = }'); i += 1
import shutil
print(f'{i = }'); i += 1
import glob
print(f'{i = }'); i += 1
import subprocess
print(f'{i = }'); i += 1
import resource
print(f'{i = }'); i += 1
import pickle
print(f'{i = }'); i += 1
import copy as copylib
print(f'{i = }'); i += 1
from importlib import reload

sys.path.append("../..")
print(f'Now starting to import my own modules')
import utils; reload(utils)
print(f'{i = }'); i += 1
import ensemble; reload(ensemble)
from ensemble import Ensemble
print(f'{i = }'); i += 1
import forcing; reload(forcing)
print(f'{i = }'); i += 1
import algorithms; reload(algorithms)
print(f'{i = }'); i += 1
import frierson_gcm; reload(frierson_gcm)
from frierson_gcm import FriersonGCM
print(f'{i = }'); i += 1

class FriersonGCMITEAMS(algorithms.ITEAMS):
    def derive_parameters(self, config):
        # Parameterize the score function in a simple way: the components will be area-averages of fields over specified regions. The combined score will be a linear combination.
        self.score_params = dict({
            'components': dict()
            })
        for compkey,compval in config['score_components'].items():
            sccomp = dict({'observable': compval['observable']}) # name of the observable function
            sccomp['roi'] = dict()
            for dim in ['lat','lon','pfull']:
                if dim in compval['roi'].keys():
                    if isinstance(compval['roi'][dim],list):
                        sccomp['roi'][dim] = slice(compval['roi'][dim][0],sccomp['roi'][dim][1])
                    else:
                        sccomp['roi'][dim] = compval['roi'][dim]
            sccomp['tavg'] = compval['tavg']
            sccomp['weight'] = compval['weight']
            self.score_params['components'][compkey] = sccomp.copy()
        super().derive_parameters(config)
        return
    def score_components(self, t, ds):
        scores = []
        for compkey,compval in self.score_params['components'].items():
            field = self.ens.dynsys.sel_from_roi(
                        getattr(self.ens.dynsys, compval['observable'])(ds),
                        compval['roi'])
            scores.append(field.mean(dim=set(field.dims) - {'time'}))
        return xr.concat(scores, dim='component').assign_coords(component=list(self.score_params['components'].keys()))
    def score_combined(self, sccomps):
        score = np.zeros(sccomps.time.size)
        total_weight = 0.0
        for compkey,compval in self.score_params['components'].items():
            conv = np.convolve(
                    np.ones(compval['tavg'])/compval['tavg'],
                    sccomps.sel(component=compkey).to_numpy(),
                    mode='full')[:sccomps['time'].size-(compval['tavg']-1)]
            conv[:compval['tavg']-1] = np.nan
            score += compval['weight']*conv
            total_weight += compval['weight']
        score /= total_weight
        return score
    @staticmethod
    def label_from_config(config):
        abbrv_population,label_population = algorithms.ITEAMS.label_from_config(config)
        obsprop = FriersonGCM.observable_props()
        comp_labels = []
        for compkey,compval in config['score_components'].items():
            roi_abbrv,roi_label = FriersonGCM.label_from_roi(compval['roi'])
            comp_label = r'%s%stavg%d'%(
                    obsprop[compval['observable']]['abbrv'],
                    roi_abbrv,
                    compval['tavg']
                    )
            comp_labels.append(comp_label)
        abbrv_score = '_'.join(comp_labels) 
        abbrv = '_'.join([
            'ITEAMS',
            abbrv_population,
            abbrv_score,
            ])
        label = ', '.join([
            label_population,
            ])
        return abbrv,label
    def generate_icandf_from_parent(self, parent, branch_time):
        # Replicate all parent seeds occurring before branch time
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        assert init_time_parent <= branch_time < init_time_parent + self.time_horizon + self.buffer_time == fin_time_parent
        print(f'{init_time_parent = }, {branch_time = }, {fin_time_parent = }')
        init_cond = self.ens.traj_metadata[parent]['icandf']['init_cond']
        init_time = init_time_parent
        fin_time = init_time + self.time_horizon #+ self.buffer_time
        new_seed = self.rng.integers(low=self.seed_min, high=self.seed_max)
        # TODO consider carefully whether we need to distinguish procedure based on SPPT vs. other kinds of forcing
        if init_time_parent < branch_time:
            pfrc = self.ens.traj_metadata[parent]['icandf']['frc']
            reseed_times = []
            seeds = []
            # Replicate parent's seeds up until the branch time
            for i_rst,rst in enumerate(pfrc.reseed_times):
                if rst < branch_time:
                    reseed_times.append(rst)
                    seeds.append(pfrc.seeds[i_rst])
            # Put in a new seed for the branch time
            reseed_times.append(branch_time)
            seeds.append(new_seed)
        else:
            reseed_times = [branch_time]
            seeds = [new_seed] # TODO if possible, when on trunk, continue the random number generator
        icandf = dict({
            'init_cond': init_cond,
            'frc': forcing.OccasionalReseedForcing(init_time, fin_time, reseed_times, seeds),
            })
        return icandf


def iteams(nproc,recompile,i_param,seed_inc):

    tododict = dict({
        'run_iteams':             1,
        'plot_spaghetti':         1,
        })
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-05"
    sub_date_str = "0/ITEAMS"
    config_gcm = FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)

    # Parameters to loop over
    pert_type_list = ['IMP']        + ['SPPT']*20
    std_sppt_list = [0.5]           + [0.5,0.3,0.1,0.05,0.01]*4
    tau_sppt_list = [6.0*3600]      + [6.0*3600]*5   + [6.0*3600]*5    + [24.0*3600]*5     + [96.0*3600]*5 
    L_sppt_list = [500.0*1000]      + [500.0*1000]*5 + [2000.0*1000]*5 + [500.0*1000]*5    + [500.0*1000]*5

    config_gcm['pert_type'] = pert_type_list[i_param]
    if config_gcm['pert_type'] == 'SPPT':
        config_gcm['SPPT']['tau_sppt'] = tau_sppt_list[i_param]
        config_gcm['SPPT']['std_sppt'] = std_sppt_list[i_param]
        config_gcm['SPPT']['L_sppt'] = L_sppt_list[i_param]
    config_gcm['remove_temp'] = 1
    param_abbrv_gcm,param_label_gcm = FriersonGCM.label_from_config(config_gcm)
    config_algo = dict({
        'autonomy': True,
        'num_levels_max': 5,
        'seed_min': 1000,
        'seed_max': 10000,
        'population_size': 4,
        'time_horizon_phys': 15,
        'buffer_time_phys': 0,
        'advance_split_time_phys': 3,
        'num2drop': 1,
        'score_components': dict({
            'rainrate': dict({
                'observable': 'total_rain',
                'roi': dict({
                    'lat': 45,
                    'lon': 180,
                    }),
                'tavg': 1,
                'weight': 1.0,
                }),
            }),
        })
    param_abbrv_algo,param_label_algo = FriersonGCMITEAMS.label_from_config(config_algo)

    dirdict = dict({
        'alg': join(scratch_dir, date_str, sub_date_str, param_abbrv_gcm, param_abbrv_algo)
        })
    dirdict['plots'] = join(dirdict['alg'], 'plots')
    for dirname in list(dirdict.values()):
        makedirs(dirname, exist_ok=True)
    root_dir = dirdict['alg']
                
    alg_filename = join(dirdict['alg'],'alg.pickle')
    # TODO write config to file, too 
    init_cond_dir = join(
            f'/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/2024-03-05/0/DNS/',
            param_abbrv_gcm)
    init_time = int(xr.open_mfdataset(join(init_cond_dir,'mem20.nc'),decode_times=False)['time'].load()[-1].item())
    init_cond = relpath(join(init_cond_dir,'restart_mem20.cpio'), root_dir)
        
    if tododict['run_iteams']:
        if exists(alg_filename):
            alg = pickle.load(open(alg_filename, 'rb'))
        else:
            gcm = FriersonGCM(config_gcm, recompile=recompile)
            ens = Ensemble(gcm, root_dir=root_dir)
            alg = FriersonGCMITEAMS(init_time, init_cond, config_algo, ens, gcm.seed_min+seed_inc)
            alg.set_init_cond(init_time,init_cond)

        alg.ens.dynsys.set_nproc(nproc)
        alg.ens.set_root_dir(root_dir)
        while not alg.terminate:
            mem = alg.ens.get_nmem()
            print(f'----------- Starting member {mem} ----------------')
            saveinfo = dict({
                # Temporary folder
                'temp_dir': f'mem{mem}',
                # Ultimate resulting filenames
                'filename_traj': f'mem{mem}.nc',
                'filename_restart': f'restart_mem{mem}.cpio',
                })
            alg.take_next_step(saveinfo)
            pickle.dump(alg, open(alg_filename, 'wb'))
    alg = pickle.load(open(alg_filename, 'rb'))
    if tododict['plot_spaghetti']:
        obsprop = alg.ens.dynsys.observable_props()
        obs_rois = [
                ('total_rain',dict(lat=45,lon=180)),
                ('column_water_vapor',dict(lat=45,lon=180)),
                ('temperature',dict(lat=45,lon=180,pfull=750)),
                ]
        for (obs_name,roi) in obs_rois:
            is_score = (obs_name in list(alg.score_params['components'].keys()))
            def obs_fun(dsmem):
                da = alg.ens.dynsys.sel_from_roi(getattr(alg.ens.dynsys, obs_name)(dsmem), roi)
                return da
            abbrv_obs = obsprop[obs_name]['abbrv']
            abbrv_roi,label_roi = alg.ens.dynsys.label_from_roi(roi)
            title = r'%s at %s'%(obsprop[obs_name]['label'],label_roi)
            abbrv = r'%s_%s'%(abbrv_obs,abbrv_roi)
            alg.plot_obs_spaghetti(obs_fun, dirdict['plots'], title=title, abbrv=abbrv, is_score=is_score)
    return

if __name__ == "__main__":
    procedure = 'run'
    print(f'Got into Main')
    if procedure == 'run':
        nproc = 4 
        recompile = 0 
        i_param = 2 #int(sys.argv[1])
        seed_inc = 0 #int(sys.argv[1])
        iteams(nproc,recompile,i_param,seed_inc)



