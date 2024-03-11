# Instantiation of EnsembleMember class on Frierson GCM

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
from ensemble import Ensemble
from dynamicalsystem import DynamicalSystem
import forcing
import precip_extremes_scaling

def boolstr(b):
    if b:
        bstr = ".true."
    else:
        bstr = ".false."
    return bstr

def print_comp_proc(compproc):
    # String representation of a completed process
    print("args: \n")
    print(compproc.args)
    print(f"returncode: {compproc.returncode}\n")
    print(f"stdout: \n")
    print(compproc.stdout.decode('utf-8'))
    #for line in compproc.stdout:
    #    if isinstance(line, bytes):
    #        print(line.decode('utf-8'))
    print(f"stderr: \n")
    print(compproc.stderr.decode('utf-8'))
    #for line in compproc.stderr:
    #    if isinstance(line, bytes):
    #        print(line.decode('utf-8'))
    print("\n")
    return 

class FriersonGCM(DynamicalSystem):
    def __init__(self, config, recompile=True):
        self.derive_parameters(config)
        self.configure_os_environment()
        if recompile:
            self.compile_mppnccombine()
            self.compile_model()
        self.nproc = 1
        return

    def set_nproc(self, nproc):
        self.nproc = nproc
        return

    def generate_default_icandf(self, init_time, fin_time):
        if self.pert_type == 'IMP':
            icandf = dict({
                'init_cond': None,
                'frc': forcing.OccasionalReseedForcing(init_time, fin_time, [], []),
                })
        elif self.pert_type == 'SPPT':
            # The time units here are in days, but will be converted to seconds for the namelist file. This basically restricts reseeding times to day boundaries.
            icandf = dict({
                'init_cond': None,
                'frc': forcing.OccasionalReseedForcing.reseed_from_start([init_time], [self.seed_min], fin_time),
                })
        return icandf
    @staticmethod
    def configure_os_environment():
        # OS stuff
        resource.setrlimit(resource.RLIMIT_STACK, (-1,-1)) # TODO should this really go here?
        return
    @classmethod
    def label_from_config(cls, config):
        abbrv_physconst = r'abs%g'%(config['abs'])
        abbrv_domain = r'res%s'%(config['resolution'])
        abbrv_pert = r'pert%s'%(config['pert_type'])
        if config['pert_type'] == 'SPPT':
            abbrv_pert = r'%s_std%g_clip%g_tau%gh_L%gkm'%(
                    abbrv_pert,
                    config['SPPT']['std_sppt'],
                    config['SPPT']['clip_sppt'],
                    config['SPPT']['tau_sppt']/3600,
                    config['SPPT']['L_sppt']/1000,
                    )
        elif config['pert_type'] == 'IMP':
            abbrv_pert = r'%s_frac%g'%(
                    abbrv_pert,
                    config['IMP']['pert_frac'],
                    )
        abbrv = (r'%s_%s_%s'%(abbrv_physconst,abbrv_domain,abbrv_pert)).replace('.','p')
        label = r"%s, $A=%g$"%(config['resolution'],config['abs'])
        return abbrv,label
    @classmethod
    def default_config(cls, source_dir_absolute, base_dir_absolute):
        config = dict({
            'resolution': 'T21',
            'abs': 1.0, # atmospheric absorption coefficient (larger means more greenhouse) 
            'nml_patches_misc': dict(),
            'source_dir_absolute': source_dir_absolute, # where the original source code comes from. Don't modify! 
            'base_dir_absolute': base_dir_absolute, # Copied from source_dir and then modified, compiled etc. 
            'platform': 'gnu',
            't_burnin': 0,
            'remove_temp': 1,
            'seed_min': 1000,
            'seed_max': 10000,
            # Stochastic perturbation parameters
            'pert_type': 'IMP', # options: SPPT, IMP
            })
        # sub-configs specific to perturbation type 
        config['SPPT'] = dict({
            'std_sppt': 0.5, # spectral standard deviation
            'clip_sppt': 2.0, # Number of standard deviations to clip
            'tau_sppt': 6.0 * 3600.0, # decorrelation timescale
            'L_sppt': 500000.0,
            })
        config['IMP'] = dict({
            'pert_frac': 0.001,
            })
        return config
    @classmethod
    def default_namelist(cls):
        # TODO integrate this namelist more flexibly with the default namelist
        # This goes on top of the base namelist
        nml = dict({
            "spectral_dynamics_nml": dict(
                lon_max = 64,
                lat_max = 32,
                num_fourier = 21,
                num_spherical = 22,
                ),
            "main_nml": dict(
                hours =  0,
                dt_atmos =  600
                ),
            "atmosphere_nml": dict(
                two_stream =  True,
                turb =  True,
                ldry_convection =  False,
                lwet_convection =  False, 
                rf_convection_flag =  False,
                rf_convcond_flag =  False,
                mixed_layer_bc =  True,
                do_virtual =  True,
                roughness_mom =  5e-3,
                roughness_heat =  1e-5,
                roughness_moist =  1e-5,
                ),
            "spectral_init_cond_nml": dict(
                initial_temperature      = 280.0,
                ),
            "radiation_nml": dict(
                albedo_value                  = 0.38,
                window                        = 0.0,
                linear_tau                    = 0.2,
                atm_abs                       = 0.22,
                wv_exponent                   = 4.0,
                solar_exponent                = 2.0,
                # The following three are the variable parameters from O'Gorman and Schneider 2008,9
                ir_tau_pole                   = 1.8,
                ir_tau_eq                     = 7.2,
                del_sol                       = 1.2,
                ),
            "mixed_layer_nml": dict(
                qflux_amp = 0.0,
                qflux_width = 16.0,
                depth = 0.5,
                evaporation = True,
                ),
            "dargan_bettsmiller_nml": dict(
                tau_bm                   = 7200.0,
                rhbm                     = 0.7,
                do_virtual               = True,
                do_bm_shift              = False,
                do_shallower             = True,
                ),
            "lscale_cond_nml": dict(
                do_evap                  = True,
                ),
            })
        return nml
    def compile_model(self):
        # Step 1: make the Makefile via mkmf
        print('About to mkmf')
        mkmf = join(self.base_dir_absolute,"bin","mkmf")
        template = join(self.base_dir_absolute,'bin',f'mkmf.template.{self.platform}')
        source = join(self.base_dir_absolute,'src')
        execdir = join(self.base_dir_absolute, f'exec_spectral.{self.platform}')
        print(f'{os.listdir(execdir) = }')
        pathnames = join(self.base_dir_absolute,'input','jf_spectral_pathnames')
        mkmf_output = subprocess.run(f'cd {execdir}; {mkmf} -p fms.x -t {template} -c "-Duse_libMPI -Duse_netCDF"  -a {source} {pathnames}', executable="/bin/csh", shell=True, capture_output=True)
        print(f"mkmf_output: \n{print_comp_proc(mkmf_output)}")

        # Step 2: compile the source code using the mkmf-generated Makefile
        print(f'About to compile source code')
        make_output = subprocess.run(f"cd {execdir}; make -f Makefile", executable="/bin/csh", shell=True, capture_output=True)
        print(f"make_output: \n{print_comp_proc(make_output)}")
        return
    def setup_directories(self, temp_dir): # To be called by an Ensemble object 
        # Prepare directories for a single ensemble member
        makedirs(temp_dir, exist_ok=False)
        work_dir = join(temp_dir,'work')
        output_dir = join(temp_dir,'output')
        makedirs(output_dir, exist_ok=False)
        makedirs(join(output_dir,'history'),exist_ok=False)
        makedirs(join(output_dir,'out_err_files'),exist_ok=False)
        print(f"Just set up the output directory {output_dir}")
        makedirs(work_dir, exist_ok=False)
        makedirs(join(work_dir,'INPUT'),exist_ok=False)
        makedirs(join(work_dir,'RESTART'),exist_ok=False)
        print(f"Just set up the output directory {work_dir}")
        # Copy the necessary code over
        shutil.copy2(join(self.base_dir_absolute,'input','jf_diag_table_precip'), join(work_dir, 'diag_table'))
        shutil.copy2(join(self.base_dir_absolute,'input','jf_spectral_field_table'), join(work_dir, 'field_table'))
        shutil.copy2(join(self.base_dir_absolute,f'exec_spectral.{self.platform}', 'fms.x'), join(work_dir,'fms.x'))
        return

    def derive_parameters(self, config):
        # Off the bat, save the whole config
        self.config = config

        # Basic dynamical systems attributes
        self.dt_save = 1.0 # days are the fundamental time unit
        self.t_burnin = config['t_burnin']

        # Directories containing source code and binaries
        self.base_dir_absolute = config['base_dir_absolute']
        self.platform = config['platform'] # probably gnu
        self.remove_temp = config['remove_temp'] # Set to False for debugging 

        nml = f90nml.read(join(self.base_dir_absolute, 'input', 'jf_spectral_namelist')).todict()
        nml_default = self.default_namelist() # actually a dictionary
        for key in nml_default.keys():
            if not (key in nml.keys()):
                nml[key] = dict()
            nml[key].update(nml_default[key])

        # Apply two patches: nmlpd with parameters derived from config, and then nml_misc (also from config) with explicit declarations for each extra variable.
        nmlpd = dict() # nml patch derived

        # Climate forcing parameters
        nmlpd['radiation_nml'] = dict({
            'ir_tau_pole': 1.8*config['abs'],
            'ir_tau_eq': 7.2*config['abs'],
            })

        # Resolution parameters
        resolution2gridspecs = dict({
            'T21': dict({
                'lon_max': 64,
                'lat_max': 32,
                'num_fourier': 21,
                'num_spherical': 22,
                }),
            'T42': dict({
                'lon_max': 128,
                'lat_max': 64,
                'num_fourier': 42,
                'num_spherical': 43,
                }),
            })
        nmlpd['spectral_dynamics_nml'] = resolution2gridspecs[config['resolution']]

        for section in nml.keys():
            for nmlpatch in [nmlpd,config['nml_patches_misc']]:
                if section in nmlpatch.keys():
                    nml[section].update(nmlpatch[section])

        self.nml_const = nml # Because this namelist only includes physical parameters, not duration and timestep as will 

        # Perturbation parameters
        self.pert_type = config['pert_type']
        self.seed_min = config['seed_min']
        self.seed_max = config['seed_max']

        return
        

    def compile_mppnccombine(self):
        # Compile mppnccombine
        mppnccombine_file = join(self.base_dir_absolute,'bin','mppnccombine.{self.platform}')
        if not exists(mppnccombine_file):
            mppnccombine_output = subprocess.run(
                    f"/home/software/gcc/6.2.0/bin/gcc -O -o {mppnccombine_file}"
                    f" -I/home/software/gcc/6.2.0/pkg/netcdf/4.6.3-c/include/"
                    #f" -I/usr/include"
                    f" -L/home/software/gcc/6.2.0/pkg/netcdf/4.6.3-c/lib/ -lnetcdf -lnetcdff"
                    f" -Wl,-rpath /home/software/gcc/6.2.0/pkg/openmpi/4.0.4/lib -Wl,-rpath /home/software/gcc/6.2.0/pkg/netcdf/4.6.3-c/lib/ -Wl,-rpath /home/software/gcc/6.2.0/pkg/openmpi/src/openmpi-2.1.1/ompi/mpi/fortran/base"
                    f" -L/home/software/hdf5/1.10.5-parallel/lib" 
                    f" -Wl,-rpath /home/software/hdf5/1.10.5-parallel/lib" 
                    f" -I/home/software/hdf5/1.10.5-parallel/include" 
                    f" {self.base_dir_absolute}/postprocessing/mppnccombine.c", shell=True, executable="/bin/csh", capture_output=True)
            print(f"mppnccombine output: \n{print_comp_proc(mppnccombine_output)}")
        else:
            print(f"No need to compile mppnccombine")
        return

    def load_member_ancestry(self, i_mem_leaf):
        # Return a Dask DataArray from all the netcdfs (1xday and 4xday) under histdir
        ds = self.mem_list[i_mem_leaf].load_history_selfmade()
        dt = ds["time"][:2].diff("time").item()
        for i_mem_twig in self.address_book[i_mem_leaf][::-1][1:]:
            start_time = ds["time"][0].item()
            ds_new = self.mem_list[i_mem_twig].load_history_selfmade()
            print(f"start_time - dt/10 = {start_time - dt/10}")
            ds = xr.concat([ds_new.sel(time=slice(None,start_time-dt/10)), ds], dim="time")
        return ds


    def run_trajectory(self, icandf, obs_fun, saveinfo, root_dir):
        self.setup_directories(join(root_dir,saveinfo['temp_dir']))
        wd = join(root_dir,saveinfo['temp_dir'],'work')
        od = join(root_dir,saveinfo['temp_dir'],'output')

        nml = self.nml_const.copy()
        if icandf['init_cond'] is not None:
            shutil.copy2(join(root_dir,icandf['init_cond']),join(wd,'INPUT',basename(icandf['init_cond'])))
            subprocess.run(f'cd {join(wd,"INPUT")}; cpio -iv < {basename(icandf["init_cond"])}', executable="/bin/csh", shell=True)
        else:
            nml['spectral_init_cond_nml'] = dict({
                'initial_temperature': 280.0,
                })

        # Augment the namelist with forcing information
        nml['main_nml']['days'] = icandf['frc'].fin_time - icandf['frc'].init_time
        numperts = len(icandf['frc'].reseed_times)
        assert numperts == len(icandf['frc'].seeds)
        if self.pert_type == 'IMP':
            nml['spectral_dynamics_nml'].update(dict({
                'do_sppt': False,
                'num_perturbations_actual': numperts,
                }))
            if numperts == 0:
                nml['spectral_dynamics_nml'].update(dict({
                    'do_perturbation': False,
                    'days_to_perturb': [-1],
                    'seed_values': [-1],
                    'perturbation_fraction': [0.0],
                    }))
            else:
                nml['spectral_dynamics_nml'].update(dict({
                    'do_perturbation': True,
                    'days_to_perturb': icandf['frc'].reseed_times,
                    'seed_values': icandf['frc'].seeds,
                    'perturbation_fraction': [self.config['IMP']['pert_frac'] for ipert in range(numperts)],
                    }))
        elif self.pert_type == 'SPPT':
            assert numperts > 0
            nml['spectral_dynamics_nml'].update(dict({ # TODO specify parameters from config
                'do_sppt': True,
                'std_sppt': self.config['SPPT']['std_sppt'],
                'clip_sppt': self.config['SPPT']['clip_sppt'],
                'tau_sppt': self.config['SPPT']['tau_sppt'], 
                'L_sppt': self.config['SPPT']['L_sppt'],
                'num_reseeds_sppt_actual': numperts,
                'reseed_times_sppt': [t for t in icandf['frc'].reseed_times],
                'seed_seq_sppt': icandf['frc'].seeds,
                }))
            # Also nullify the old kind of perturbation
            nml['spectral_dynamics_nml'].update(dict({
                'do_perturbation': False,
                'days_to_perturb': [-1],
                'seed_values': [-1],
                'perturbation_fraction': [0.0],
                }))

        print(f'nml = ')
        pprint.pprint(nml)
        f90nml.namelist.Namelist(nml,default_start_index=1).write(join(wd,'input.nml'))
        print(f'--------------- Starting MPIRUN --------')
        mpirun_output = subprocess.run(f'cd {wd}; /home/software/gcc/6.2.0/pkg/openmpi/4.0.4/bin/mpirun -np {self.nproc} fms.x', shell=True, executable='/bin/csh', capture_output=True)
        print(mpirun_output)
        print(f'--------------- Finished MPIRUN --------')

        # Move output files to output directory with informative names
        date_range_name = f'days{icandf["frc"].init_time}-{icandf["frc"].fin_time}'
        mkdir(join(od,'history',date_range_name))
        # netcdf files
        nc_files = glob.glob('*.nc*', root_dir=wd)
        for ncf in nc_files:
            shutil.move(join(wd,ncf), join(od, 'history', date_range_name, f'{date_range_name}.{ncf}'))
        # ascii files
        ascii_files = glob.glob('*.out', root_dir=wd)
        for ascf in ascii_files:
            shutil.move(join(wd,ascf), join(od,f'{date_range_name}.{ascf}'))
        # namelists, diagnostics
        for f in ['input.nml','diag_table']:
            shutil.copy2(join(wd,f),join(wd,'RESTART',f))
        restart_files = glob.glob('*.res*', root_dir=join(wd,'RESTART'))
        files2compress_str = " ".join(['input.nml','diag_table'] + restart_files)
        if len(restart_files) > 0:
            print(f'There are some restart files: \n{restart_files = }')
            compressed_restart_tail = f'{date_range_name}.cpio'
            subprocess.run(f'cd {wd}/RESTART; /bin/ls {files2compress_str} | cpio -ocv > {compressed_restart_tail}', executable='/bin/csh', shell=True)
            # Now move to a new restart directory
            makedirs(join(od,'restart'), exist_ok=True)
            shutil.move(join(wd,'RESTART',compressed_restart_tail), join(od,'restart',compressed_restart_tail)) 
        else:
            raise Exception(f'There are no restart files in {wd}')


        # TODO allow later possibility of running in multiple chunks, but for now just stick to one chunk per member. When doing that, keep the loop contained inside this same function here. 


        # Finally, aggregate trajectory output and also move (along with restart) to a common directory

        # Aggregate
        ds = dict()
        for freq in [1,4]:
            ds[freq] = self.resample_to_daily(
                    xr.open_mfdataset(
                        glob.glob(join(od,'history',date_range_name,f'*{freq}xday*.nc*')),
                        decode_times=False, 
                        preprocess=lambda ds: ds.drop_dims('latb')
                        )
                    ).compute()
            
            # Reample to daily
            ds[freq] = self.resample_to_daily(ds[freq])
        # Save the single netcdf
        ds = xr.merge(list(ds.values()), compat='override')
        ds.to_netcdf(join(root_dir,saveinfo['filename_traj']))

        # Compute any observables of interest
        observables = obs_fun(ds.time, ds)

        ds.close()
        # Save the single restart
        shutil.move(join(od,'restart',compressed_restart_tail),join(root_dir,saveinfo['filename_restart']))

        # Clean up the directories
        if self.remove_temp:
            shutil.rmtree(join(root_dir,saveinfo['temp_dir']))
        
        # TODO evaluate observable functions ...
        metadata = dict({
            'icandf': icandf, 
            'filename_traj': saveinfo['filename_traj'],
            'filename_restart': saveinfo['filename_restart'],
            })
        return metadata, observables
    @staticmethod
    def get_timespan(metadata):
        frc = metadata['icandf']['frc']
        return frc.init_time,frc.fin_time
    @staticmethod
    def resample_to_daily(da):
        day_end_tidx = np.where(np.mod(da["time"].to_numpy(), 1.0) == 0)[0]
        if len(day_end_tidx) > 1:
            steps_per_day = day_end_tidx[1] - day_end_tidx[0]
        else:
            steps_per_day = da.time.size 
        runavg = da.isel(time=day_end_tidx)
        for i_delay in range(1,steps_per_day):
            runavg += da.shift(time=i_delay).isel(time=day_end_tidx)
        runavg *= 1.0/steps_per_day
        return runavg
    # --------------- Distance functions ----------------------
    def compute_pairwise_observables(self, pairwise_funs, md0, md1list, root_dir): # Distance is the main application here 
        pairwise_fun_vals = [[] for pwf in pairwise_fun] # List of lists
        ds0 = xr.open_mfdataset(join(root_dir,md0['filename_traj']), decode_times=False)
        for i_md1,md1 in enumerate(md1list):
            ds1 = xr.open_mfdataset(join(root_dir,md1['filename_traj']), decode_times=False)
            pairwise_fun_vals.append([])
            for i_pwf,pwf in enumerate(pairwise_funs):
                pairwise_fun_vals[i_pwf].append(pwf(ds0,ds1))
        return pairwise_fun_vals
    def distance_props(self):
        obsprop = self.observable_props()
        distprop = dict()
        # 2D fields, Euclidean distance
        for field in ['total_rain','column_water_vapor','surface_pressure']:
            opf = obsprop[field]
            distkey = f'{field}_eucdist'
            distprop[distkey] = dict({
                'abbrv': r'%s_EUC'%(opf['abbrv']),
                'unit_symbol': opf['unit_symbol'],
                'label': r'Eucl. Dist. (%s)'%(opf['label']),
                'cmap': opf['cmap'],
                })
        return distprop



    # --------------- Observable functions ---------------------
    def compute_observables(self, obs_funs, metadata, root_dir):
        ds = xr.open_mfdataset(join(root_dir,metadata['filename_traj']), decode_times=False)
        obs_name = list(obs_funs.keys())
        obs = dict()
        for obs_name,obs_fun in obs_funs.items():
            obs[obs_name] = obs_fun(ds).compute()
        return obs
    def compute_stats_dns_rotsym(self, fxypt, lon_roll_step_requested, time_block_size, sel, bounds=None):
        # Given a physical input field f(x,y,t), augment it by rotations to compute return periods
        # constant parameters to adjust 
        time_block_size = 25 
        # Concatenate a long array of timeseries at different longitudes
        dlon = fxypt.lon[:2].diff('lon').item()
        nlon = fxypt['lon'].size
        lon_roll_step_idx = int(round(lon_roll_step_requested/dlon))
        idx_lon = np.arange(0, nlon, step=lon_roll_step_idx)
        # Clip the time axis to contain exactly an integer multiple of the block size
        clip_size = np.mod(fxypt['time'].size, time_block_size)
        f_subsel = fxypt.sel(sel,method='nearest').isel(time=slice(clip_size,None))
        fconcat = np.concatenate(tuple(f_subsel.isel(lon=i_lon).to_numpy() for i_lon in idx_lon))
        return utils.compute_returnstats_and_histogram(fconcat, time_block_size, bounds=bounds)
    def observable_props(self):
        obslib = dict()
        obslib['r_sppt_g'] = dict({
            "abbrv": "RSPPT",
            "unit_symbol": "",
            "label": r"$r_{\mathrm{SPPT}}$",
            "cmap": "coolwarm",
            "vmin": None,
            "vmax": None,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["effective_static_stability"] = dict({
            "abbrv": "ESS",
            "unit_symbol": "s$^{-2}$",
            "label": "Effective static stability",
            "cmap": "coolwarm",
            "vmin": None,
            "vmax": None,
            "clo": "gray",
            "chi": "yellow",
            })

        obslib["vertical_velocity"] = dict({
            "abbrv": "W",
            "unit_symbol": "Pa/s",
            "label": "Vertical velocity",
            "cmap": "coolwarm",
            "vmin": None,
            "vmax": None,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["meridional_velocity"] = dict({
            "abbrv": "V",
            "unit_symbol": "m/s",
            "label": "Meridional velocity",
            "cmap": "coolwarm",
            "vmin": None,
            "vmax": None,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["zonal_velocity"] = dict({
            "abbrv": "U",
            "unit_symbol": "m/s",
            "label": "Zonal velocity",
            "cmap": "coolwarm",
            "vmin": None,
            "vmax": None,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["exprec_scaling"] = dict({
            "abbrv": "XPS",
            "unit_symbol": "mm/day",
            "label": "Extreme precip. scaling",
            "cmap": "Blues",
            "vmin": 0.0,
            "vmax": 64.0,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["convection_rain"] = dict({
            "abbrv": "Rconv",
            "unit_symbol": "mm/day",
            "label": "Convection rain",
            "cmap": "Blues",
            "vmin": 0.0, 
            "vmax": 64.0,
            "clo": "gray", 
            "chi": "yellow",
            })
        obslib["condensation_rain"] = dict({
            "abbrv": "Rcond",
            "unit_symbol": "mm/day",
            "label": "Condensation rain",
            "cmap": "Blues",
            "vmin": 0.0, 
            "vmax": 64.0,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["total_rain"] = dict({
            "abbrv": "Rtot",
            "unit_symbol": "mm/day",
            "label": "Rain rate",
            "cmap": "Blues",
            "vmin": 0.0, 
            "vmax": 80.0,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["specific_humidity"] = dict({
            "abbrv": "Q",
            "unit_symbol": "kg/kg",
            "label": "Specific humidity",
            "cmap": "Blues",
            "vmin": None,
            "vmax": None,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["temperature"] = dict({
            "abbrv": "T",
            "unit_symbol": "K",
            "label": "Temperature",
            "cmap": "Reds",
            "vmin": 210.0, 
            "vmax": 350.0,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["column_water_vapor"] = dict({
            "abbrv": "CWV",
            "unit_symbol": r"kg m$^{-2}$",
            "label": "Column water vapor",
            "cmap": "Blues",
            "vmin": 0.0, 
            "vmax": 7.0,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["column_relative_humidity"] = dict({
            "abbrv": "CRH",
            "unit_symbol": r"fraction",
            "label": "Column relative humidity",
            "cmap": "Blues",
            "vmin": 0.0, 
            "vmax": 1.0,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["water_vapor_convergence"] = dict({
            "abbrv": "QCON",
            "unit_symbol": r"kg m$^{-2}$s$^{-1}$",
            "label": "Water vapor convergence",
            "cmap": "coolwarm_r",
            "vmin": None, 
            "vmax": None,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["vorticity"] = dict({
            "abbrv": "VOR",
            "unit_symbol": r"s$^{-1}$",
            "label": "Vorticity",
            "cmap": "coolwarm",
            "vmin": None, 
            "vmax": None,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["surface_pressure"] = dict({
            "abbrv": "PS",
            "unit_symbol": r"Pa",
            "label": "Surface pressure",
            "cmap": "rainbow",
            "vmin": 96.0e3, 
            "vmax": 103.0e3,
            "clo": "gray",
            "chi": "yellow",
            })
        return obslib
    @staticmethod
    def pressure(ds):
        # Return pressure with "pfull" as a vertical coordinate. 
        p_edge = ds["bk"]*ds["ps"] # Pascals
        p_cent = 0.5*(p_edge + p_edge.shift(phalf=-1)).isel(phalf=slice(None,-1)).rename({"phalf": "pfull"}).assign_coords({"pfull": ds["pfull"]})
        dp_dpfull = (p_edge.shift(phalf=-1) - p_edge)/(ds["phalf"].shift(phalf=-1) - ds["phalf"])
        dp_dpfull = dp_dpfull.isel(phalf=slice(None,-1)).rename({"phalf": "pfull"}).assign_coords({"pfull": ds["pfull"]})
        return p_cent, dp_dpfull
    @staticmethod
    def effective_static_stability(ds):
        # Compute effective static stability
        p,dp_dpfull = FriersonGCM.pressure(ds)
        temp = ds["temp"]
        lam = 1.0
        # Constants
        Rd = 287.04
        Rv = 461.5
        cpd = 1005.7
        cpv = 1870.0
        g = 9.80665
        p0 = 1e5
        kappa = Rd/cpd
        gc_ratio = Rd/Rv
    
        # Saturation vapor pressure
        Tc = temp - 273.15
        es = 611.2 * np.exp(17.67*Tc/(Tc + 243.5))
        
        # Latent heat of condensation
        L = (2.501-0.00237*Tc)*1e6
        
        # Saturation mixing ratio
        rs = gc_ratio*es/(p - es)
    
        # Saturation specific humidity
        qs = rs/(1 + rs)
        
        # Potential temperature
        exponent = kappa*(1 + rs/gc_ratio)/(1 + rs*cpv/cpd)
        theta = temp * (p0/p) ** exponent
    
        # derivative of potential temperature with respect to pressure
        dtheta_dp = theta.differentiate("pfull") / dp_dpfull
    
        # density
        temp_virtual = temp * (1 + rs/gc_ratio)/(1 + rs)
        rho = p/Rd/temp_virtual
    
        # moist adiabatic lapse rate
        malr = g/cpd * (1+rs) / (1+cpv/cpd*rs) * (1+L*rs/Rd/temp) / (1+L**2*rs*(1+rs/gc_ratio)/(Rv*temp**2*(cpd+rs*cpv)))
    
        # Derivative of potential temperature wrt pressure along a moist adiabat (neglects small contribution from vertical variations of exponent)
        dtemp_dp_ma = malr/g/rho
        dtheta_dp_ma = dtemp_dp_ma * theta/temp - exponent*theta/p
    
        # effective static stability (eq. 8 of O'Gorman JAS 2011)
        dtheta_dp_eff = dtheta_dp - lam*dtheta_dp_ma
    
        # Convert to buoyancy frequency
        dtheta_dz_eff = dtheta_dp_eff * (-rho*g) # K/m
        Nsq_eff = dtheta_dz_eff * g/theta # 1/s^2
        print(f"Nsq_eff coords = {Nsq_eff.coords}")
        return Nsq_eff 
    
    @staticmethod
    def r_sppt_g(ds):
        return ds['r_sppt_g']

    @staticmethod
    def column_water_vapor(ds):
        g = 9.806 
        p,dp_dpfull = FriersonGCM.pressure(ds)
        p = ds["bk"] * ds["ps"] # Pascals
        cwv = (ds["sphum"] * dp_dpfull).integrate("pfull")/g
        return cwv
    
    @staticmethod
    def water_vapor_convergence(ds):
        g = 9.806 
        p,dp_dpfull = FriersonGCM.pressure(ds)
        p = ds["bk"] * ds["ps"] # Pascals
        conv = -divergence(ds["ucomp"]*ds["sphum"], ds["vcomp"]*ds["sphum"])
        qcon = (conv * dp_dpfull).integrate("pfull")/g
        return qcon 
    
    @staticmethod
    def column_relative_humidity(ds):
        # CWV / max possible CWV
        p,dp_dpfull = FriersonGCM.pressure(ds)
        cwv_xg = (ds["sphum"] * dp_dpfull).integrate("pfull") # / g, but this cancels 
        qs = sat_spec_hum(ds)
        cwv_max_xg = (qs * dp_dpfull).integrate("pfull") # / g, but this cancels
        return cwv_xg / cwv_max_xg
    
    @staticmethod
    def vert_deriv_sat_sphum(ds):
        # Vertical derivative of saturation specific humidity at fixed saturation equivalent potential temperature
        pass
    
    @staticmethod
    def divergence(u, v):
        # Divergence in spherical coordinates
        a = 6371.0e3 # radius of earth
        coslat = np.cos(np.deg2rad(u["lat"]))
        div = 1.0/(a*coslat)*((v*coslat).differentiate("lat") + u.differentiate("lon")) * 180/np.pi
        return div
    
    @staticmethod
    def vorticity(ds):
        return curl(ds["ucomp"], ds["vcomp"])
    
    @staticmethod
    def curl(u, v):
        # Divergence in spherical coordinates (in the vertical direction)
        a = 6371.0e3 # radius of earth
        coslat = np.cos(np.deg2rad(u["lat"]))
        uxv = 1.0/(a*coslat)*(v.differentiate("lon") - (u*coslat).differentiate("lat")) * 180/np.pi
        return uxv
    
    @staticmethod
    def condensation_rain(ds):
        cond = ds["condensation_rain"] * 3600*24 # From kg/(m**2 * s) to kg/(m**2 * day) = mm/day
        cond.attrs["units"] = "mm/day"
        return cond
    @staticmethod
    def convection_rain(ds):
        if "convection_rain" in list(ds.data_vars.keys()):
            conv = ds["convection_rain"] * 3600*24 # From kg/(m**2 * s) to kg/(m**2 * day) = mm/day
        else:
            conv = xr.zeros_like(ds["condensation_rain"])
        conv.attrs["units"] = "mm/day"
        return conv
    @staticmethod
    def total_rain(ds):
        return FriersonGCM.condensation_rain(ds) + FriersonGCM.convection_rain(ds)
    @staticmethod
    def temperature(ds):
        return ds["temp"] 
    @staticmethod
    def specific_humidity(ds):
        return ds["sphum"] 
    @staticmethod
    def surface_pressure(ds):
        return ds["ps"]
    @staticmethod
    def vertical_velocity(ds):
        return ds["omega"]
    @staticmethod
    def zonal_velocity(ds):
        return ds["ucomp"]
    @staticmethod
    def meridional_velocity(ds):
        return ds["vcomp"]
    @staticmethod
    def exprec_scaling_wrapper(ds):
        # Swap the order of pressure to be increasing on all variables
        omega = ds["omega"].reindex(pfull=ds["pfull"][::-1])
        temp = ds["temp"].reindex(pfull=ds["pfull"][::-1])
        ps = ds["ps"]
        p, dp_dpfull = FriersonGCM.pressure(ds)
        p = p.reindex(pfull=ds["pfull"][::-1])
        dp_dpfull = dp_dpfull.reindex(pfull=ds["pfull"][::-1])
        scaling = precip_extremes_scaling.scaling(omega, temp, p, dp_dpfull, ps)
        scaling *= 3600 * 24
        return scaling



def dns(nproc,recompile,i_param):
    tododict = dict({
        'run':                            0,
        'summarize':                      1,
        'plot': dict({
            'snapshots':    0,
            'return_stats': 1,
            }),
        })
    # Create a small ensemble
    # Run three trajectories, each one picking up where the previous one left off
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-05"
    sub_date_str = "0/DNS"
    print(f'About to generate default config')

    config = FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)
    config['resolution'] = 'T21'

    pert_type_list = ['IMP']        + ['SPPT']*16
    std_sppt_list = [0.5]           + [0.5,0.1,0.05,0.01]*4
    tau_sppt_list = [6.0*3600]      + [6.0*3600]*4   + [6.0*3600]*4    + [24.0*3600]*4     + [96.0*3600]*4 
    L_sppt_list = [500.0*1000]      + [500.0*1000]*4 + [2000.0*1000]*4 + [500.0*1000]*4    + [500.0*1000]*4
    config['pert_type'] = pert_type_list[i_param]
    if config['pert_type'] == 'SPPT':
        config['SPPT']['tau_sppt'] = tau_sppt_list[i_param]
        config['SPPT']['std_sppt'] = std_sppt_list[i_param]
        config['SPPT']['L_sppt'] = L_sppt_list[i_param]



    label,display = FriersonGCM.label_from_config(config)
    expt_dir = join(scratch_dir,date_str,sub_date_str,label)
    makedirs(expt_dir,exist_ok=True)
    ens_filename = join(expt_dir,'ens.pickle')
    root_dir = expt_dir

    if tododict['run']:
        days_per_chunk = 100
        num_chunks = 21
        obs_fun = lambda t,x: None
        if exists(ens_filename):
            ens = pickle.load(open(ens_filename,'rb'))
            ens.set_root_dir(root_dir)
            n_mem = ens.memgraph.number_of_nodes()
            parent = n_mem-1
            _,init_time = ens.get_member_timespan(n_mem-1)
            init_cond = ens.traj_metadata[n_mem-1]['filename_restart']
        else:
            gcm = FriersonGCM(config,recompile=recompile)
            ens = Ensemble(gcm,root_dir=root_dir)
            n_mem = 0
            init_time = 0
            init_cond = None
            parent = None
        ens.dynsys.set_nproc(nproc)
        for mem in range(n_mem,n_mem+num_chunks):
            fin_time = init_time + days_per_chunk
            icandf = ens.dynsys.generate_default_icandf(init_time,fin_time) # For SPPT, this will restart the random seed. 
            icandf['init_cond'] = init_cond
            # saveinfo will have RELATIVE paths 
            saveinfo = dict({
                # Temporary folder
                'temp_dir': f'mem{mem}',
                # Ultimate resulting filenames
                'filename_traj': f'mem{mem}.nc',
                'filename_restart': f'restart_mem{mem}.cpio',
                })
            _ = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
            init_time = fin_time
            parent = mem
            init_cond = ens.traj_metadata[parent]['filename_restart']
            pickle.dump(ens, open(join(expt_dir,'ens.pickle'),'wb'))
    # Load the ensemble for further analysis
    ens = pickle.load(open(join(expt_dir,'ens.pickle'),'rb'))
    obsprop = ens.dynsys.observable_props()
    # Make the directory for analysis
    analysis_dir = join(expt_dir,'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    # Select regions of interest
    lat_target = 45.0
    pfull_target = 1000
    obs_roi = dict({
        'temperature': dict(lat=lat_target,pfull=pfull_target),
        'total_rain': dict(lat=lat_target),
        })
    if tododict['summarize']:
        spinup = 700
        nmem = ens.get_nmem()
        all_starts,all_ends = ens.get_all_timespans()
        mems2summarize = np.where(all_starts >= spinup)[0]
        time_block_size = 25
        for obs_name,roi in obs_roi.items():
            obs_fun = {obs_name: lambda dsmem: getattr(ens.dynsys, obs_name)(dsmem)}
            fxypt = xr.concat(ens.compute_observables(obs_fun,mems2summarize)[obs_name], dim='time')
            # ----------------- Mean and quantiles at various latitudes ----------
            roi = {dim: val for (dim,val) in obs_roi[obs_name].items() if dim not in ['lon','lat']}
            sf = np.array([0.5,0.1,0.01,0.001]) # complementary quantiles of interest
            coords_sf = dict({c: fxypt.coords[c].to_numpy() for c in set(fxypt.dims) - {'time','lon','pfull'}})
            coords_sf['sf'] = sf
            f_sf = xr.DataArray(coords=coords_sf, dims=tuple(coords_sf.keys()), data=np.nan)
            for i,sfval in enumerate(sf):
                f_sf.loc[dict(sf=sfval)] = fxypt.sel(
                        roi,method='nearest',drop=True).quantile(1-sfval, dim=['time','lon'])

            f_mean = fxypt.sel(roi,method='nearest',drop=True).mean(dim=['time','lon'])
            f_sf_mean = xr.Dataset(data_vars={'fmean': f_mean, 'fsf': f_sf})
            f_sf_mean.to_netcdf(join(analysis_dir,f'mean_sf_{obs_name}.nc'))
            f_sf_mean.close()

            # ----------------- Return period curves at a fixed latitude --------
            roi = {dim: val for (dim,val) in obs_roi[obs_name].items() if dim not in ['lon']}
            lon_roll_step_requested = 30
            bin_lows,hist,rtime,logsf = ens.dynsys.compute_stats_dns_rotsym(fxypt, lon_roll_step_requested, time_block_size, roi)
            location_suffix = '_'.join([r'%s%g'%(roikey,roival) for (roikey,roival) in roi.items()])
            np.save(join(analysis_dir,f'distn_{obs_name}_{location_suffix}.npy'),np.vstack((bin_lows,hist,logsf,rtime)))


    plot_dir = join(expt_dir,'plots')
    makedirs(plot_dir,exist_ok=True)
    if utils.find_true_in_dict(tododict['plot']):

        if tododict['plot']['return_stats']:
            for obs_name in ['temperature','total_rain']:
                # ------------------- Mean and quantiles at various latitudes ------------
                roi = {dim: val for (dim,val) in obs_roi[obs_name].items() if dim not in ['lon','lat']}
                location_suffix = ('_'.join([r'%s%g'%(roikey,roival) for (roikey,roival) in roi.items()])).replace('.','p')
                f_sf_mean = xr.open_dataset(join(analysis_dir,f'mean_sf_{obs_name}.nc'))
                print(f'{f_sf_mean.coords = }')
                print(f'{f_sf_mean["sf"].coords = }')
                fig,ax = plt.subplots()
                handles = []
                for i_sfval,sfval in enumerate(f_sf_mean['fsf'].coords['sf'].to_numpy()):
                    print(f'{i_sfval = }, {sfval = }')
                    xdata = f_sf_mean.lat.values
                    ydata = f_sf_mean['fsf'].isel(sf=i_sfval).to_numpy()
                    print(f'{xdata.shape = }')
                    print(f'{ydata.shape = }')
                    h, = ax.plot(xdata, ydata, label=f'{sfval}')
                    handles.append(h)
                h, = ax.plot(f_sf_mean['fmean'].lat.values, f_sf_mean['fmean'].values, color='black', linestyle='--', linewidth=2, label='mean')
                handles.append(h)
                ax.legend(handles=handles,title='Comp. quantiles')
                # Adjust y axis limits
                data4range = f_sf_mean['fsf'].sel(lat=slice(20,None))
                ax.set_ylim([data4range.min().item(), data4range.max().item()])
                ax.set_xlabel('Latitude')
                fig.savefig(join(plot_dir,f'mean_sf_{obs_name}_{location_suffix}.png'),**pltkwargs)
                plt.close(fig)


                # ------------------- Return period plots at a fixed latitude ---------
                roi = {dim: val for (dim,val) in obs_roi[obs_name].items() if dim not in ['lon']}
                location_suffix = ('_'.join([r'%s%g'%(roikey,roival) for (roikey,roival) in roi.items()])).replace('.','p')
                location_label = ', '.join([r'%s=%g'%(roikey,roival) for (roikey,roival) in roi.items()])
                bin_lows,hist,logsf,rtime = np.load(join(analysis_dir,f'distn_{obs_name}_{location_suffix}.npy'))
                print(f'{bin_lows[:3] = }')
                print(f'{hist[:3] = }')
                print(f'{rtime[:3] = }')
                bin_mids = bin_lows + 0.5*(bin_lows[1]-bin_lows[0])
                fig,axes = plt.subplots(ncols=2,figsize=(12,4),gridspec_kw={'wspace': 0.25})
                ax = axes[0]
                ax.plot(bin_lows,hist,color='black',marker='.')
                ax.set_xlabel(obsprop[obs_name]['label'])
                ax.set_ylabel('Prob. density')
                ax.set_yscale('log')
                ax = axes[1]
                ax.plot(rtime,bin_lows,color='black',marker='.')
                ax.set_ylim([bin_lows[np.argmax(rtime>0)],2*bin_lows[-1]-bin_lows[-2]])
                ax.set_xlabel('Return time')
                ax.set_ylabel('Return level')
                ax.set_xscale('log')
                ax.set_title(obsprop[obs_name]['label'])
                fig.suptitle(r'%s at %s'%(obsprop[obs_name]['label'],location_label))
                print(join(plot_dir,f'rtime_{obsprop[obs_name]["abbrv"]}.png'))
                fig.savefig(join(plot_dir,f'rtime_{obsprop[obs_name]["abbrv"]}_{location_suffix}.png'),**pltkwargs)
                plt.close(fig)


        if tododict['plot']['snapshots']:
            lat = 45.0
            lon = 180.0
            pfull = 1000.0
            obs_funs = dict()
            for obs_name in ['temperature']:
                obs_funs[obs_name] = lambda dsmem,obs_name=obs_name: getattr(ens.dynsys, obs_name)(dsmem).sel(pfull=pfull,method='nearest')
            for obs_name in ['r_sppt_g','total_rain','column_water_vapor','surface_pressure']:
                obs_funs[obs_name] = lambda dsmem,obs_name=obs_name: getattr(ens.dynsys, obs_name)(dsmem)
            mems2plot = [ens.get_nmem()-1]
            obs_vals = ens.compute_observables(obs_funs, mems2plot)

            for i_mem,mem in enumerate(mems2plot):
                for obs_name in list(obs_funs.keys()):
                    memobs = obs_vals[obs_name][i_mem].compute()
                    # Plot a few daily snapshots
                    for day in memobs.time.to_numpy()[:2]: #.astype(int):
                        fig,axes = plt.subplots(figsize=(12,5),ncols=2,sharey=True)
                        ax = axes[0]
                        xr.plot.pcolormesh(memobs.sel(time=day), x='lon', y='lat', cmap=obsprop[obs_name]['cmap'], ax=ax)
                        ax.set_title(r'%s [%s], mem. %d, day %d'%(obsprop[obs_name]['label'], obsprop[obs_name]['unit_symbol'], mem, day))
                        ax.set_xlabel('Longitude')
                        ax.set_ylabel('Latitude')
                        ax = axes[1]
                        hday, = xr.plot.plot(memobs.mean(dim=['time','lon']),y='lat',color='black',ax=ax,label=r'(zonal,time) avg')
                        havg, = xr.plot.plot(memobs.sel(time=day).mean(dim='lon'),y='lat',color='red',ax=ax,label=r'zonal avg')
                        ax.set_title("")
                        ax.set_xlabel(r'%s [%s]'%(obsprop[obs_name]['label'],obsprop[obs_name]['unit_symbol']))
                        ax.set_ylabel('')
                        ax.legend(handles=[hday,havg])

                        fig.savefig(join(plot_dir,r'%s_mem%d_day%d'%(obsprop[obs_name]['abbrv'],mem,day)),**pltkwargs)
                        plt.close(fig)
                    # Plot timeseries
                    fig,ax = plt.subplots()
                    xr.plot.plot(memobs.sel(lat=lat,lon=lon,method='nearest'), x='time', color='black')
                    ax.set_xlabel("time")
                    ax.set_ylabel(r'%s [%s]'%(obsprop[obs_name]['label'],obsprop[obs_name]['unit_symbol']))
                    ax.set_title(r'$(\lambda,\phi)=(%g,%g)$'%(lon,lat))
                    fig.savefig(join(plot_dir,r'%s_mem%d'%(obsprop[obs_name]['abbrv'],mem)),**pltkwargs)
                    plt.close(fig)
    return



def meta_analyze_dns():
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-05"
    sub_date_str = "0/DNS"
    expt_dir = join(scratch_dir,date_str,sub_date_str)
    meta_dir = join(expt_dir,'meta_analysis')
    makedirs(meta_dir, exist_ok=True)

    # -------- Specify which variables to fix and which to vary ---------
    params = dict()
    params['L_sppt'] = dict({
        'fun': lambda config: config['SPPT']['L_sppt'],
        'scale': 1000, # for display purposes
        'symbol': r'$L_{\mathrm{SPPT}}$',
        'unit_symbol': 'km',
        })
    params['tau_sppt'] = dict({
        'fun': lambda config: config['SPPT']['tau_sppt'],
        'scale': 3600, 
        'symbol': r'$\tau_{\mathrm{SPPT}}$',
        'unit_symbol': 'h',
        })
    params['std_sppt'] = dict({
        'fun': lambda config: config['SPPT']['std_sppt'],
        'scale': 1.0,
        'symbol': r'$\sigma_{\mathrm{SPPT}}$',
        })

    params2fix = ['L_sppt','tau_sppt']
    param2vary = 'std_sppt'

    # Specify the pool of files
    dnsdir_pattern = join(expt_dir,f"abs1_resT21_pertSPPT*/")
    dnsdirs = glob.glob(dnsdir_pattern)
    # Select regions of interest
    lat_target = 45.0
    pfull_target = 1000
    obs_roi = dict({
        'temperature': dict(lat=lat_target,pfull=pfull_target),
        'total_rain': dict(lat=lat_target),
        })
    # TODO compare percentile vs latitude plots
    for obs_name,roi in obs_roi.items():
        location_suffix = '_'.join([r'%s%g'%(roikey,roival) for (roikey,roival) in roi.items()])
        print(f'{dnsdirs = }')
        param_vals = dict({p: [] for p in params.keys()})
        return_stats = []
        for i_dnsdir,dnsdir in enumerate(dnsdirs):
            dynsys = pickle.load(open(join(dnsdir,'ens.pickle'),'rb')).dynsys
            for p in params.keys():
                param_vals[p].append(params[p]['fun'](dynsys.config))
            return_stats.append(np.load(join(dnsdir,'analysis',f'distn_{obs_name}_{location_suffix}.npy')))
            if i_dnsdir == 0:
                obsprop = dynsys.observable_props()

        # TODO Add special case to the dataset: non-SPPT
        ctrldir = glob.glob(join(expt_dir,f"abs1_resT21_pertIMP*/"))[0]

        bin_lows_ctrl,hist_ctrl,logsf_ctrl,rtime_ctrl = np.load(join(ctrldir,'analysis',f'distn_{obs_name}_{location_suffix}.npy'))
        # Enumerate all combinations of fixed parameters
        param_vals_fixed = list(zip(*(param_vals[p] for p in params2fix)))
        print(f'{param_vals_fixed = }')
        unique_param_vals_fixed = set(param_vals_fixed)
        for pvf in unique_param_vals_fixed:
            print(f'{pvf = }')
            fixed_param_abbrv = ('_'.join([r'%s%g%s'%(params2fix[i],pvf[i]/params[params2fix[i]]['scale'],params[params2fix[i]]['unit_symbol']) for i in range(len(params2fix))])).replace('.','p')
            print(f'{fixed_param_abbrv = }')
            fixed_param_label = ', '.join([
                r'%s $=%g$ %s'%(params[params2fix[i]]['symbol'],pvf[i]/params[params2fix[i]]['scale'],params[params2fix[i]]['unit_symbol']) 
                for i in range(len(params2fix))])
            idx = np.array([i for i in range(len(dnsdirs)) if (param_vals_fixed[i] == pvf)])
            order = np.argsort([param_vals[param2vary][i] for i in idx])
            idx = idx[order] 
            # 1. return period plots and histograms as function of variable parameter
            fig,axes = plt.subplots(ncols=2,figsize=(12,5))
            handles = []
            # Plot the SPPT statistics
            colors = plt.cm.Set1(np.arange(len(idx)))
            for ii,i in enumerate(idx):
                ax = axes[0]
                bin_lows,hist,logsf,rtime = return_stats[i]
                ax.plot(bin_lows,hist,color=colors[ii], marker='.')
                ax = axes[1]
                h, = ax.plot(rtime,bin_lows,color=colors[ii],marker='.',label=r'%g'%(param_vals[param2vary][i]))
                ax.set_xscale('log')
                handles.append(h)
            # Plot the control
            ax = axes[0]
            ax.plot(bin_lows_ctrl,hist_ctrl, color='black', marker='.', linestyle='--', linewidth=2)
            ax = axes[1]
            h, = ax.plot(rtime_ctrl,bin_lows_ctrl,color='black',marker='.', linestyle='--', linewidth=2, label=r'no SPPT')
            ax.set_ylim([bin_lows_ctrl[np.argmax(rtime_ctrl>0)],2*bin_lows[-1]-bin_lows[-2]])
            ax.set_xscale('log')
            handles.append(h)
            axes[0].set_xlabel(r'%s'%(obsprop[obs_name]['label']))
            axes[0].set_ylabel('Probability density')
            axes[0].set_yscale('log')
            axes[1].set_xlabel('Return time')
            axes[1].set_ylabel('Return level')
            axes[1].set_xscale('log')
            axes[1].legend(handles=handles, title=params[param2vary]['symbol'], loc=(1.05,0.05))
            fig.suptitle(fixed_param_label)
            fig.savefig(join(meta_dir,f'rtime_{obs_name}_{location_suffix}_asfunof_{param2vary}_{fixed_param_abbrv}.png'),**pltkwargs)
            plt.close(fig)
    


    # -------------------------------------------------------------------
    return


if __name__ == "__main__":
    print(f'Got into Main')
    nproc = 4
    recompile = False
    procedure = 'meta'
    if procedure == 'run':
        idx_param = [int(v) for v in sys.argv[1:]]
        for i_param in idx_param:
            dns(nproc,recompile,i_param)
    elif procedure == 'meta':
        meta_analyze_dns()
