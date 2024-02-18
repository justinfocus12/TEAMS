# Instantiation of EnsembleMember class on Frierson GCM

import numpy as np
from numpy.random import default_rng
from scipy.special import softmax
import xarray as xr
import dask
import f90nml
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
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

import sys
sys.path.append("../..")
from ensemble import Ensemble
from dynamicalsystem import DynamicalSystem
sys.path.append("/home/ju26596/jf_conv_gray_smooth/fms_analysis")
import forcing

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
    def __init__(self, config):
        self.derive_parameters(config)
        self.configure_os_environment()
        self.compile_mppnccombine()
        self.compile_model()
        return

    def generate_default_icandf(self, init_time, fin_time):
        nml = self.nml_const.copy()
        #nml['main_nml'] = dict({
        #    'days': fin_time - init_time,
        #    'hours': 0,
        #    'dt_atmos': 600,
        #    })
        #nml['spectral_init_cond_nml'] = dict({
        #    'initial_temperature': 280.0,
        #    })
        #nml['spectral_dynamics_nml'].update(dict({
        #    'do_perturbation': True,
        #    'num_perturbations_actual': 0,
        #    # TODO: can I actually have empty lists within the namelist?
        #    'days_to_perturb': [0],
        #    'seed_values': [84730],
        #    'perturbation_fraction': [1.0e-3],
        #    }))
        icandf = dict({
            'init_cond': None,
            'frc': forcing.ContinuousTimeForcing(init_time, fin_time, [], []),
            })
        return icandf
    @staticmethod
    def configure_os_environment():
        # OS stuff
        resource.setrlimit(resource.RLIMIT_STACK, (-1,-1)) # TODO should this really go here?
        return
    @classmethod
    def label_from_config(cls, config):
        label = (r"res%s_abs%g_pert%g"%(config['resolution'],config['abs'],config['pert_frac'])).replace('.','p')
        display = r"%s, $A=%g$"%(config['resolution'],config['abs'])
        return label, display
    @classmethod
    def default_config(cls, base_dir):
        config = dict({
            'resolution': 'T21',
            'abs': 1.0, # atmospheric absorption coefficient (larger means more greenhouse) 
            'pert_frac': 0.001,
            'nml_patches_misc': dict(),
            'base_dir': base_dir,
            'platform': 'gnu',
            })
        return config
    @classmethod
    def default_namelist(cls):
        # TODO integrate this namelist more flexibly with the default namelist
        # This goes on top of the base namelist
        nml = dict({
            "spectral_dynamics_nml": dict({
                #"do_perturbation": True,
                #"do_perturbation_eachday": False,
                #"num_perturbations_actual": 0,
                #"days_to_perturb": [0], 
                #"seed_values": [1234],
                #"perturbation_fraction": [1.0e-3],
                "lon_max": 64,
                "lat_max": 32,
                "num_fourier": 21,
                "num_spherical": 22,
                }),
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
        mkmf = join(self.base_dir,"bin","mkmf")
        template = join(self.base_dir,'bin','mkmf.template.{self.platform}')
        source = join(self.base_dir,'src')
        execdir = join(self.base_dir, f'exec_spectral.{self.platform}')
        print(f'{os.listdir(execdir) = }')
        pathnames = join(self.base_dir,'input','jf_spectral_pathnames')
        mkmf_output = subprocess.run(f'cd {execdir}; {mkmf} -p fms.x -t {template} -c "-Duse_libMPI -Duse_netCDF"  -a {source} {pathnames}', executable="/bin/csh", shell=True, capture_output=True)
        print(f"mkmf_output: \n{print_comp_proc(mkmf_output)}")

        # Step 2: compile the source code using the mkmf-generated Makefile
        print(f'About to compile source code')
        precall = f"set nproc = 4; set fms_version = jf_conv_gray_smooth; unset noclobber; set echo;  "
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
        shutil.copy2(join(self.base_dir,'input','jf_diag_table_precip'), join(work_dir, 'diag_table'))
        shutil.copy2(join(self.base_dir,'input','jf_spectral_field_table'), join(work_dir, 'field_table'))
        shutil.copy2(join(self.base_dir,f'exec_spectral.{self.platform}', 'fms.x'), join(work_dir,'fms.x'))
        return

    @staticmethod
    def cleanup_directories(base_dir, work_dir, output_dir, aggregate_output_flag=False):
        print(f"About to clean up")
        shutil.rmtree(work_dir)
        shutil.rmtree(join(output_dir,'out_err_files'))
        logfiles = glob.glob(join(output_dir,'*.out'))
        for f in logfiles:
            os.remove(f)
        print("About to aggregate")
        if aggregate_output:
            self.aggregate_output(load_immediately=True)
        print("Finished aggregating")

        # Remove the temporary and log files 
        shutil.rmtree(self.dirs["work"])
        shutil.rmtree(join(self.dirs["output"],"out_err_files"))
        logfiles = glob.glob(join(self.dirs["output"], "*.out"))
        for f in logfiles:
            os.remove(f)

        print(f"About to aggregate")
        if aggregate_output_flag:
            self.aggregate_output(load_immediately=True)
        print(f"Finished aggregating")
        return

    def aggregate_output(self, dt_chunk=None, overwrite=False, load_immediately=True):
        # TODO account for cases where the data size is too big
        # Merge the distributed netcdf files into a single one; downsample to daily 
        hist = self.load_history_selfmade_distributed(load_immediately=load_immediately)
        hist_agg = dict()
        # Downsample the 4xday fields to daily
        freqs = ["4xday","1xday"]
        for freq in freqs:
            hist_agg[freq] = transformations.resample_to_daily(hist[freq])
        # Combine into a single dataset
        hist_agg = xr.merge([hist_agg[freq] for freq in freqs], compat="override")
        # Split into chunks if necessary
        dt = hist_agg["time"][:2].diff("time").item() 
        Nt = hist_agg["time"].size
        if dt_chunk is None:
            dt_chunk = hist_agg["time"][-1].item() - hist_agg["time"][0].item() + 1.5*dt 
        t0 = hist_agg["time"][0].item()
        chunk_size = int(round(dt_chunk/dt))
        i0 = 0
        while i0 < Nt:
            i1 = min(i0 + chunk_size, Nt)
            print(f"i0 = {i0}, i1 = {i1}")
            hist_chunk = hist_agg.isel(time=slice(i0,i1))
            chunk_string = f"days{int(hist_chunk['time'][0].item()):04}-{int(hist_chunk['time'][-1].item()):04}"
            print(f"chunk {chunk_string}")
            filename = join(
                    self.dirs["output"], "history",
                    f"history_{chunk_string}.nc"
                    )
            print(f"filename = {filename}")
            print(f"Does the filename exist? {exists(filename)}")
            print(f"overwrite = {overwrite}")
            if (not exists(filename)) or overwrite:
                print(f"Starting to_netcdf...",end="")
                #hist_chunk.to_netcdf("/home/ju26596/rare_event_simulation/splitting/examples/frierson_gcm/test_hist.nc")
                hist_chunk.to_netcdf(filename, format="NETCDF3_64BIT")
                print(f"Done")
            i0 += chunk_size

        # Now delete the old chunk directories
        for freq in freqs:
            hist[freq].close()
        
        chunk_dirs = glob.glob(join(self.dirs["output"],"history","days*h00/"))
        for chd in chunk_dirs:
            shutil.rmtree(chd)
        return

    def derive_parameters(self, config):

        # Directories containing source code and binaries
        self.base_dir = config['base_dir']
        self.platform = config['platform'] # probably gnu

        nml = f90nml.read(join(self.base_dir, 'input', 'jf_spectral_namelist')).todict()
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
        self.pert_frac = config['pert_frac']

        return
        
    @classmethod
    def default_init(cls, base_paths, perturbation_specs, days_per_chunk, ensemble_size_limit, aggregate_output_flag=True, vbl_nml=None):
        resource.setrlimit(resource.RLIMIT_STACK, (-1,-1))
        phys_par = dict({"abs": 1.0})
        fms_version = "jf_conv_gray_smooth"
        platform = "gnu"
        dirs_ens = dict({
            "home": base_paths["home"],
            "base": join(base_paths["home"], fms_version), 
            "source": join(base_paths["home"], fms_version, "src"), 
            "script": join(base_paths["home"], fms_version, "scripts"), 
            "exec": join(base_paths["home"], fms_version, f"exec_spectral.{platform}"),
            "output": join(base_paths["output"], f"abs{phys_par['abs']}_smooth"),
            "work": join(base_paths["work"], f"abs{phys_par['abs']}_smooth/"),
            })
        
        # ---------- Specify the variable part of the namelist (but the same for all members) ------------
        # It should contain a flag for convection 
        if vbl_nml is None:
            vbl_nml = cls.default_namelist()
        vbl_nml["main_nml"]["days"] = days_per_chunk # each member will be able to run for a different number of chunks, however
        vbl_nml["radiation_nml"]["del_sol"] = 1.2
        vbl_nml["radiation_nml"]["ir_tau_eq"] = 7.2*phys_par["abs"]
        vbl_nml["radiation_nml"]["ir_tau_pole"] = 1.8*phys_par["abs"]

        model_params = dict({
            "nproc": 4,
            "parallel_flag": True,
            "fms_version": fms_version,
            "platform": "gnu",
            "phys_par": phys_par,
            "variable_namelist": vbl_nml,
            "perturbation_specs": perturbation_specs,
            "execdir": dirs_ens["exec"],
            "infiles": dict(
                template = os.path.join(dirs_ens["base"],"bin",f"mkmf.template.{platform}"),
                mkmf = os.path.join(dirs_ens["base"],"bin","mkmf"),
                pathnames = os.path.join(dirs_ens["base"],"input","jf_spectral_pathnames"),
                diagtable = os.path.join(dirs_ens["base"],"input","jf_diag_table_precip"),
                namelist = os.path.join(dirs_ens["base"],"input","jf_spectral_namelist"),
                fieldtable = os.path.join(dirs_ens["base"],"input","jf_spectral_field_table_fv"),
                mppnccombine = os.path.join(dirs_ens["base"],"bin",f"mppnccombine.{platform}"),
                time_stamp = os.path.join(dirs_ens["base"],"bin","time_stamp.csh"),
                )
            })
        model_params["precall"] = f"set nproc = {model_params['nproc']}; set fms_version = {model_params['fms_version']}; unset noclobber; set echo; " 
        model_params["aggregate_output_flag"] = aggregate_output_flag

        # -------- Prepare the ensemble --------
        ens = cls(dirs_ens, model_params, ensemble_size_limit) 

        return ens

    def setup_model(self): # To be called after __init__
        self.nproc = self.model_params["nproc"]
        self.platform = self.model_params["platform"]
        self.fms_version = self.model_params["fms_version"]
        self.infiles = self.model_params["infiles"].copy()
        self.vbl_nml = self.model_params["variable_namelist"].copy()
        self.perturbation_specs = self.model_params["perturbation_specs"]

        self.days_per_chunk = self.vbl_nml["main_nml"]["days"] # In other words, the restart interval
        
        self.precall = self.model_params["precall"]

        # Set the directories for base code and work

        if not np.all([dirname in self.dirs for dirname in ["base","exec","work"]]):
            raise Exception(f"The list of directories is incomplete: you passed dirs = {dirs}")

        if not np.all([f in self.infiles for f in ["template","mkmf","pathnames","diagtable","namelist","fieldtable","mppnccombine","time_stamp"]]):
            raise Exception(f"You gave an incomplete list of input files: you passed infiles = {infiles}")

        # Set up the directory for archiving the whole thing
        os.makedirs(self.dirs["output"], exist_ok=True) # This will be the permanent directory
        os.makedirs(self.dirs["work"], exist_ok=True)

        self.compile_mppnccombine()
        self.compile_model()
        
        # Initialize lists to store metadata about ensemble members and their relationships 
        self.mem_list = []
        self.address_book = []
        
        return

    def compile_mppnccombine(self):
        # Compile mppnccombine
        mppnccombine_file = join(self.base_dir,'bin','mppnccombine.{self.platform}')
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
                    f" {self.base_dir}/postprocessing/mppnccombine.c", shell=True, executable="/bin/csh", capture_output=True)
            print(f"mppnccombine output: \n{print_comp_proc(mppnccombine_output)}")
        else:
            print(f"No need to compile mppnccombine")
        return
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

    def compute_observables(self, fields2comp=None):
        for i_mem,mem in enumerate(self.mem_list):
            print(f"------------- Starting member {i_mem} out of {len(self.mem_list)} ------------")
            mem.compute_observables(fields2comp=fields2comp)
        return

    def compute_observables_altogether(self, fields2comp=None):
        # Only for when the ensemble members have disjoint times (e.g., all strung together)
        if fields2comp is None:
            fields2comp = dict({
                "1xday": [
                    "total_rain",
                    ],
                "4xday": [
                    "temperature",
                    "column_water_vapor",
                    "zonal_velocity",
                    "vertical_velocity",
                    "surface_pressure",
                    "vorticity",
                    ],
                })
        ds = dict({freq: [] for freq in fields2comp.keys()})
        for i_mem,mem in enumerate(self.mem_list):
            hist = mem.load_history_selfmade()
            for freq in fields2comp.keys():
                ds[freq].append(hist[freq])
            print(f"loaded history {i_mem} out of {len(self.mem_list)}")
        for freq in fields2comp.keys():
            ds[freq] = xr.concat(ds[freq], dim="time")
        print(f"Loaded the dataset")
        savefolder = join(self.dirs["output"], "observables")
        os.makedirs(savefolder, exist_ok=True)
        target_files = transformations.precompute_features(ds, fields2comp, savefolder, overwrite=False)
        self.target_files = target_files
        return 

    def run_trajectory(self, icandf, obs_fun, saveinfo, nproc=1):
        self.setup_directories(saveinfo['temp_dir'])
        wd = join(saveinfo['temp_dir'],'work')
        od = join(saveinfo['temp_dir'],'output')

        nml = self.nml_const.copy()
        if icandf['init_cond'] is not None:
            shutil.copy2(icandf['init_cond'],join(wd,'INPUT',basename(icandf['init_cond'])))
            subprocess.run(f'cd {join(wd,"INPUT")}; cpio -iv < {basename(icandf["init_cond"])}', executable="/bin/csh", shell=True)
        else:
            nml['spectral_init_cond_nml'] = dict({
                'initial_temperature': 280.0,
                })

        # Augment the namelist with forcing information
        nml['main_nml']['days'] = icandf['frc'].fin_time - icandf['frc'].init_time
        numperts = len(icandf['frc'].reseed_times)
        nml['spectral_dynamics_nml']['num_perturbations_actual'] = numperts
        if numperts == 0:
            nml['spectral_dynamics_nml']['do_perturbation'] = False
            nml['spectral_dynamics_nml']['days_to_perturb'] = [-1]
            nml['spectral_dynamics_nml']['seed_values'] = [-1]
            nml['spectral_dynamics_nml']['perturbation_fraction'] = [0.0]
        else:
            nml['spectral_dynamics_nml']['do_perturbation'] = True
            nml['spectral_dynamics_nml']['days_to_perturb'] = icandf['frc'].reseed_times
            nml['spectral_dynamics_nml']['seed_values'] = icandf['frc'].seeds
            nml['spectral_dynamics_nml']['perturbation_fraction'] = [self.pert_frac for ipert in range(numperts)]

        f90nml.namelist.Namelist(nml,default_start_index=1).write(join(wd,'input.nml'))
        mpirun_output = subprocess.run(f'cd {join(wd)}; /home/software/gcc/6.2.0/pkg/openmpi/4.0.4/bin/mpirun -np {nproc} fms.x', shell=True, executable='/bin/csh', capture_output=True)

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
            raise Exception(f'There are no restart files in {resdir_work}')


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
        ds.to_netcdf(saveinfo['filename_traj'])
        ds.close()
        # Save the single restart
        shutil.move(join(od,'restart',compressed_restart_tail),saveinfo['filename_restart'])

        # Clean up the directories
        shutil.rmtree(saveinfo['temp_dir'])
        
        # TODO evaluate observable functions ...
        metadata = dict({
            'icandf': icandf, 
            'filename_traj': saveinfo['filename_traj'],
            'filename_restart': saveinfo['filename_restart'],
            })
        observables = dict()
        return metadata, observables
    @staticmethod
    def resample_to_daily(da):
        day_end_tidx = np.where(np.mod(da["time"].to_numpy(), 1.0) == 0)[0]
        steps_per_day = day_end_tidx[1] - day_end_tidx[0]
        runavg = da.isel(time=day_end_tidx)
        for i_delay in range(1,steps_per_day):
            runavg += da.shift(time=i_delay).isel(time=day_end_tidx)
        runavg *= 1.0/steps_per_day
        return runavg

def dns_short_chain(nproc):
    # Run three trajectories, each one picking up where the previous one left off
    base_dir = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/frierson_gcm"
    date_str = "2024-02-16"
    sub_date_str = "1"
    print(f'About to generate default config')
    config = FriersonGCM.default_config(base_dir)
    gcm = FriersonGCM(config)

    expt_str = join(scratch_dir,date_str,sub_date_str)
    makedirs(expt_str,exist_ok=True)

    filename_warmstart = None
    init_time = 0
    for i_mem in range(3):
        fin_time = init_time + 8
        icandf = gcm.generate_default_icandf(init_time,fin_time)
        icandf['filename_warmstart'] = copylib.copy(filename_warmstart)
        saveinfo = dict({
            # Temporary folder
            'temp_dir': join(expt_str,f'mem{i_mem}'),
            # Ultimate resulting filenames
            'filename_traj': join(expt_str,f'mem{i_mem}.nc'),
            'filename_restart': join(expt_str,f'restart_mem{i_mem}.nc'),
            })
        obs_fun = None
        metadata,observable = gcm.run_trajectory(icandf, obs_fun, saveinfo, nproc=nproc)
        filename_warmstart = saveinfo['filename_restart']
        init_time = fin_time 
    return

def small_branching_ensemble(nproc):
    tododict = dict({
        'run':            0,
        'plot':           1,
        })
    # Create a small ensemble
    # Run three trajectories, each one picking up where the previous one left off
    base_dir = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/frierson_gcm"
    date_str = "2024-02-17"
    sub_date_str = "1"
    print(f'About to generate default config')
    config = FriersonGCM.default_config(base_dir)
    label,display = FriersonGCM.label_from_config(config)
    expt_dir = join(scratch_dir,date_str,sub_date_str,label)

    if tododict['run']:
        makedirs(expt_dir,exist_ok=True)
        obs_fun = lambda t,x: None

        gcm = FriersonGCM(config)
        ens = Ensemble(gcm)

        # Parent member: run for 8 days
        mem = 0
        init_time = 0
        fin_time = 8
        icandf = gcm.generate_default_icandf(init_time,fin_time)
        saveinfo = dict({
            # Temporary folder
            'temp_dir': join(expt_dir,f'mem{mem}'),
            # Ultimate resulting filenames
            'filename_traj': join(expt_dir,f'mem{mem}.nc'),
            'filename_restart': join(expt_dir,f'restart_mem{mem}.nc'),
            })
        _ = ens.branch_or_plant(icandf, obs_fun, saveinfo)


        # Branch off some children
        for (mem,seedval) in zip([1,2,3],[29183,48271,39183]):
            parent = 0
            mdp = ens.traj_metadata[parent]
            init_time = mdp['icandf']['frc'].fin_time
            icandf = dict({
                'init_cond': mdp['filename_restart'],
                'frc': forcing.ContinuousTimeForcing(init_time, init_time+10, [init_time+2], [seedval])
                })
            saveinfo = dict({
                'temp_dir': join(expt_dir,f'mem{mem}'),
                'filename_traj': join(expt_dir,f'mem{mem}.nc'),
                'filename_restart': join(expt_dir,f'restart_mem{mem}.nc'),
                })
            _ = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)

        # Save out the ensemble for later querying
        pickle.dump(ens, open(join(expt_dir,'ens.pickle'),'wb'))
    if tododict['plot']:
        plot_dir = join(expt_dir,'plots')
        makedirs(plot_dir,exist_ok=True)
        ens = pickle.load(open(join(expt_dir,'ens.pickle'),'rb'))
        lat = 45.0
        lon = 180.0
        fig,ax = plt.subplots(figsize=(20,5))
        handles = []
        for mem in range(ens.memgraph.number_of_nodes()):
            rtot = xr.open_mfdataset(ens.traj_metadata[mem]['filename_traj'], decode_times=False)['condensation_rain'] * 3600/24
            i_lat = np.argmin(np.abs(rtot.lat.values - lat))
            i_lon = np.argmin(np.abs(rtot.lon.values - lon))
            rtot = rtot.isel(lat=i_lat,lon=i_lon).compute()
            h, = xr.plot.plot(rtot, x='time', label=f'Member {mem}')
            handles.append(h)
        ax.legend(handles=handles)
        fig.savefig(join(plot_dir,'rain.png'),**pltlkwargs)

                    


    return






















class FriersonGCMEnsembleMember: #(EnsembleMember):
    def setup_directories(self):

        # Output directory
        output_dir = self.dirs["output"]
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(join(output_dir,"history"), exist_ok=True)
        os.makedirs(join(output_dir,"out_err_files"), exist_ok=True)
        print(f"Just set up the output directory {self.dirs['output']}")

        # Work directory
        work_dir = self.dirs["work"]
        if exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir, exist_ok=False)
        os.makedirs(join(work_dir,"INPUT"), exist_ok=True)
        os.makedirs(join(work_dir,"RESTART"), exist_ok=True)
        print(f"Just set up the work directory {self.dirs['work']}")
        return

    def cleanup_directories(self):
        print(f"About to clean up")

        # Remove the temporary and log files 
        shutil.rmtree(self.dirs["work"])
        shutil.rmtree(join(self.dirs["output"],"out_err_files"))
        logfiles = glob.glob(join(self.dirs["output"], "*.out"))
        for f in logfiles:
            os.remove(f)

        print(f"About to aggregate")
        if self.aggregate_output_flag:
            self.aggregate_output(load_immediately=True)
        print(f"Finished aggregating")
        return

    def aggregate_output(self, dt_chunk=None, overwrite=False, load_immediately=True):
        # TODO account for cases where the data size is too big
        # Merge the distributed netcdf files into a single one; downsample to daily 
        hist = self.load_history_selfmade_distributed(load_immediately=load_immediately)
        hist_agg = dict()
        # Downsample the 4xday fields to daily
        freqs = ["4xday","1xday"]
        for freq in freqs:
            hist_agg[freq] = transformations.resample_to_daily(hist[freq])
        # Combine into a single dataset
        hist_agg = xr.merge([hist_agg[freq] for freq in freqs], compat="override")
        # Split into chunks if necessary
        dt = hist_agg["time"][:2].diff("time").item() 
        Nt = hist_agg["time"].size
        if dt_chunk is None:
            dt_chunk = hist_agg["time"][-1].item() - hist_agg["time"][0].item() + 1.5*dt 
        t0 = hist_agg["time"][0].item()
        chunk_size = int(round(dt_chunk/dt))
        i0 = 0
        while i0 < Nt:
            i1 = min(i0 + chunk_size, Nt)
            print(f"i0 = {i0}, i1 = {i1}")
            hist_chunk = hist_agg.isel(time=slice(i0,i1))
            chunk_string = f"days{int(hist_chunk['time'][0].item()):04}-{int(hist_chunk['time'][-1].item()):04}"
            print(f"chunk {chunk_string}")
            filename = join(
                    self.dirs["output"], "history",
                    f"history_{chunk_string}.nc"
                    )
            print(f"filename = {filename}")
            print(f"Does the filename exist? {exists(filename)}")
            print(f"overwrite = {overwrite}")
            if (not exists(filename)) or overwrite:
                print(f"Starting to_netcdf...",end="")
                #hist_chunk.to_netcdf("/home/ju26596/rare_event_simulation/splitting/examples/frierson_gcm/test_hist.nc")
                hist_chunk.to_netcdf(filename, format="NETCDF3_64BIT")
                print(f"Done")
            i0 += chunk_size

        # Now delete the old chunk directories
        for freq in freqs:
            hist[freq].close()
        
        chunk_dirs = glob.glob(join(self.dirs["output"],"history","days*h00/"))
        for chd in chunk_dirs:
            shutil.rmtree(chd)
        return

    def set_run_params(self, model_params, warmstart_info):
        # This must be done only once for each ensemble member. Not at each restart.
        # TODO make the duration (days per chunk) variable, too. 
        self.precall = model_params["precall"]
        self.nproc = model_params["nproc"]
        self.perturbation_specs = model_params["perturbation_specs"]
        os.chdir(self.dirs["work"])
        print(f"Which files are present? \n{os.listdir()}")

        self.aggregate_output_flag = model_params["aggregate_output_flag"] # Yes if the time chunks aren't too long

        # --------- Modify the namelist --------------
        nml_file_input = model_params["infiles"]["namelist"]
        nml_file_output = "input.nml"

        nml = f90nml.read(nml_file_input).todict()
        #nml_patch = {**model_params["variable_namelist"], **warmstart_info["perturbation_namelist"]}
        #nml.patch(nml_patch)
        nml.patch(model_params["variable_namelist"])
        nml.patch(warmstart_info["perturbation_namelist"]) # This can override the duration parameter ('time' in main_nml)
        nml.end_comma = True
        nml.default_start_index = 1
        nml.write(nml_file_output)
        
        # ---------------------------------------------

        # Copy the files over to the working directory
        subprocess.run(f"{self.precall} cp {model_params['infiles']['diagtable']} diag_table", shell=True, executable="/bin/csh")
        subprocess.run(f"{self.precall} cp {model_params['infiles']['fieldtable']} field_table ", shell=True, executable="/bin/csh")
        subprocess.run(f"{self.precall} cp {model_params['execdir']}/fms.x fms.x", shell=True, executable="/bin/csh")

        self.days_per_chunk = nml["main_nml"]["days"] #model_params["variable_namelist"]["main_nml"]["days"]
        self.hourofday = nml["main_nml"]["hours"] #model_params["variable_namelist"]["main_nml"]["hours"]
        # TODO continue flexibilizing 

        # Record the time of birth of this trajectory 
        self.init_file_ancestral = warmstart_info["file"]
        self.init_time_ancestral = warmstart_info["time"]
        
        # Record where to start the next lifecycle from
        self.init_file = copylib.copy(self.init_file_ancestral)
        self.init_time = self.init_time_ancestral
        self.time_origin = warmstart_info["time_origin"]

        # Record the perturbation information 
        self.warmstart_info = warmstart_info

        # Initialize a list of restart files for the benefit of descendants
        # These will not include the ancestral restarts 
        self.term_file_list = []
        self.term_time_list = []

        return

    def run_one_cycle(self, verbose=False):
        # TODO: keep ability to perturb either outside or inside 
        os.chdir(self.dirs["work"])
        if self.init_file is not None:
            os.chdir("INPUT")
            subprocess.run(f"cp {self.init_file} {os.path.basename(self.init_file)}", shell=True, executable="/bin/csh")
            subprocess.run(f"{self.precall} cpio -iv < {os.path.basename(self.init_file)}", executable="/bin/csh", shell=True)
            os.remove(os.path.basename(self.init_file))

        # ------------ Perturbing the initial conditions -------------
        if self.warmstart_info["do_perturbation_externally"]:
            spec_dyn_res = Dataset("spectral_dynamics.res.nc",mode="r+",format="NETCDF4")
            for field in self.perturbation_specs["fields"]:
                num_wn_zon = spec_dyn_res[field].shape[3]
                for wn_zon in range(5,num_wn_zon):
                    for i_time in self.perturbation_specs["timelevels"]:
                        spec_dyn_res[field][i_time,:,wn_zon:,wn_zon] *= (1.0 + self.perturbation_specs["amplitude"] * (2*self.rng_perturb.choice([0.0,1.0],size=spec_dyn_res[field][i_time,:,wn_zon:,wn_zon].shape) - 1))

            spec_dyn_res.close()


        # ----------- Integrate from the (possibly perturbed) initial conditions -----
        # TODO: extract the time itself from the restart file 
        os.chdir(self.dirs["work"])
        print(f"About to run mpirun")
        mpirun_output = subprocess.run(f"{self.precall} /home/software/gcc/6.2.0/pkg/openmpi/4.0.4/bin/mpirun -np {self.nproc} {self.dirs['work']}/fms.x", shell=True, executable="/bin/csh", capture_output=True)
        print(f"mpirun_output: \n{print_comp_proc(mpirun_output)}\n")

        # Move the output files to the output directory
        date_name = f"days{self.init_time:04d}-{(self.init_time+self.days_per_chunk):04d}h{self.hourofday:02d}"

        # ------------ Move output files to their own directories (don't combine) --------
        local_files = os.listdir()
        os.mkdir(os.path.join(self.dirs["output"],"history",date_name)) #
        nc_files = [lf for lf in local_files if (lf.endswith(".nc") or lf[-8:-4]==".nc.")]
        for ncf in nc_files:
            shutil.move(ncf, join(self.dirs["output"],"history",date_name,f"{date_name}.{ncf}"))
        # ------------ Save ascii output files to local disk --------------
        local_files = os.listdir()
        out_files = [lf for lf in local_files if lf.endswith(".out")]
        for out in out_files:
            shutil.move(out, os.path.join(self.dirs["output"],f"{date_name}.{out}"))

        # ------------ Move restart files to output directory ----------
        os.chdir("RESTART")
        local_files = os.listdir()
        resfiles = [lf for lf in local_files if ".res" in lf]
        if len(resfiles) > 0:
            print(f"There are some restart files: \n{resfiles}")
            restart_dir = os.path.join(self.dirs["output"], "restart")
            os.makedirs(restart_dir, exist_ok=True)
            restart_file_tail = f"{date_name}.cpio"
            restart_file = os.path.join(self.dirs["output"], "restart", restart_file_tail)
            subprocess.run(f"{self.precall} cp {self.dirs['work']}/*.nml .", shell=True, executable="/bin/csh")
            subprocess.run(f"{self.precall} cp {self.dirs['work']}/diag_table .", shell=True, executable="/bin/csh")
            files = resfiles + ["input.nml", "diag_table"]
            # Set up a call to cpio
            filestr = " ".join(files)
            subprocess.run(f"{self.precall} /bin/ls {filestr} | cpio -ocv > {restart_file_tail}", executable="/bin/csh", shell=True)
            shutil.move(restart_file_tail, restart_file)
        else:
            print(f"Uh oh, there are no restart files... local_files = \n{local_files}")

            # TODO: decide if reload commands should be written to file, or if that's a task for the parent process

        # ---------- Get the next cycle ready ---------------
        # Out error files: not applicable here. 
        self.init_file = restart_file
        self.init_time = self.init_time + self.days_per_chunk

        self.term_file_list += [copylib.copy(self.init_file)]
        self.term_time_list += [self.init_time]
        print(f"After preparing next cycle, self.term_time_list = {self.term_time_list} and self.term_file_list = {self.term_file_list}")
        return 

    def load_history_selfmade(self):
        return self.load_history_selfmade_aggregated()

    def load_history_selfmade_aggregated(self):
        # To be called after aggregation
        files2open = glob.glob(join(self.dirs["output"], "history", "history*.nc"))
        ds = xr.open_mfdataset(files2open, decode_times=False)
        return ds

    def load_history_selfmade_distributed(self, verbose=False, load_immediately=False):
        print("---------- Inside load_history_selfmade ----------")
        freqs = ["1xday","4xday"]
        # Return a Dask DataArray from all the netcdfs (1xday and 4xday) under histdir
        chunk_dirs = glob.glob(join(self.dirs["output"],"history","d*h00"))
        if verbose:
            print(f"chunk_dirs = {chunk_dirs}")
        ds = self.ingest_timechunks(chunk_dirs, verbose=verbose, load_immediately=load_immediately)
        return ds

    def load_history_selfmade_observable(self, obs_name):
        # Assumes this observable has been computed and stored
        ds_obs = xr.open_dataarray(self.obs_files[obs_name], decode_times=False)
        return ds_obs

    
    def ingest_timechunks(self, chunk_dirs, verbose=False, load_immediately=False):
        freqs = ["1xday","4xday"]
        file_list = dict({freq: [] for freq in freqs})
        for chd in chunk_dirs:
            for freq in ["1xday","4xday"]:
                file_list[freq] += glob.glob(f"{chd}/*{freq}*.nc*")
        ds = dict()
        if verbose: print(f"file_list = \n{file_list}")
        for freq in freqs:
            ds[freq] = xr.open_mfdataset(file_list[freq], decode_times=False, preprocess=transformations.preprocess)
            if load_immediately:
                ds[freq] = ds[freq].load()
                print(f"loaded immediately indeed")
        return ds

    def compute_observables(self, fields2comp=None):
        if fields2comp is None:
            fields2comp = dict({
                "1xday": [
                    "total_rain",
                    ],
                "4xday": [
                    "temperature",
                    "column_water_vapor",
                    "zonal_velocity",
                    "vertical_velocity",
                    "surface_pressure",
                    "vorticity",
                    ],
                })
        ds = self.load_history_selfmade()
        savefolder = self.dirs["output"]
        target_files = transformations.precompute_features(ds, fields2comp, savefolder, overwrite=False)
        if not ("obs_files" in list(self.__dict__.keys())):
            self.obs_files = dict()
        self.obs_files.update(target_files)
        return



# ----------- Below are some standard test methods ------------
def old_dns():
    # Run a long integration from warmstart
    ensemble_size_limit = 1
    algo_params = dict({
        "days_per_chunk": 30,
        "num_chunks": 45,
        "resolution": "T21",
        })
    home_dir = "/home/ju26596"
    #scratch_dir = "/nobackup1c/users/ju26596/splitting/frierson_gcm"
    scratch_dir = "/pool001/ju26596/frierson_gcm"
    date_str = "2023-09-28"
    sub_date_str = "0"
    expt_str = (
            f"dns"
            f"_res{algo_params['resolution']}"
            f"_{algo_params['num_chunks']}chunks"
            f"_{algo_params['days_per_chunk']}days"
            ).replace(".","p")
    base_paths = dict({
        "home": home_dir,
        "work": join(scratch_dir, date_str, sub_date_str, expt_str, "fms_tmp"), 
        "output": join(scratch_dir, date_str, sub_date_str, expt_str, "fms_output"), 
        })
    perturbation_specs = None
    if exists(join(base_paths["output"], "abs1.0_smooth", "ens")):
        print(f"Adding to an existing ensemble")
        ens = pickle.load(open(join(base_paths["output"], "abs1.0_smooth", "ens"), "rb"))

        # --------- AUGMENT THE ENSEMBLE WITH THE NEW BELLS AND WHISTLES LEFT OVER FROM CODE DEVELOPMENT ----------- 
        ens.model_params["aggregate_output_flag"] = True

        # ----------------------------------------------------------------------------------------------------------
        init_file = ens.mem_list[-1].init_file
        init_time = ens.mem_list[-1].term_time_list[-1]
    else:
        print(f"Beginning a new ensemble")
        ens = FriersonGCMEnsemble.default_init(base_paths, perturbation_specs, algo_params["days_per_chunk"], ensemble_size_limit)
        #init_file = "/pool001/ju26596/fms_archive/2022-12-13/2/ctrl_noconv_21x100_8proc/abs1.0_smooth/mem_ctrl/restart/days0700-0800h00.cpio" 
        init_file = None
        init_time = 0

    # Whether warm- or cold-starting, make a new member
    warmstart_info = dict({
        "time": int(init_time), 
        "file": init_file,
        "time_origin": init_time,
        "do_perturbation_externally": False, 
        })
    if algo_params["resolution"] == "T21": 
        lon_max,lat_max,num_fourier,num_spherical = 64,32,21,22
    elif algo_params["resolution"] == "T42": 
        lon_max,lat_max,num_fourier,num_spherical = 128,64,42,43
    warmstart_info["perturbation_namelist"] = dict({
        "spectral_dynamics_nml": dict(
            do_perturbation = False,
            #do_perturbation_eachday = True,
            num_perturbations_actual = 1,
            days_to_perturb = [init_time],
            seed_values = [0],
            perturbation_fraction = [0.0],
            # Change resolution
            lon_max = lon_max,
            lat_max = lat_max,
            num_fourier = num_fourier,
            num_spherical = num_spherical,
            ),
        "main_nml": dict(
            days = int(algo_params["days_per_chunk"]),
            ),
        })

    ens.initialize_new_member(FriersonGCMEnsembleMember, warmstart_info.copy())

    num_chunks_per_mem = np.array([algo_params["num_chunks"]])
    #num_chunks_per_mem = np.array([15])
    memidx2run = np.array([len(ens.mem_list)-1])
    print(f"memidx2run = {memidx2run}")
    ens.run_batch(memidx2run,num_chunks_per_mem)

    # Save out the metadata
    pickle.dump(ens, open(join(ens.dirs["output"],"ens"), "wb"))
    return

def test_fortran_split():
    # Run a control simulation and then a second one, branching off in the middle
    ensemble_size_limit = 2
    home_dir = "/home/ju26596"
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/frierson_results"
    date_str = "2024-02-12"
    sub_date_str = "2"
    days_per_chunk = 5
    num_time_chunks = 1
    expt_str = f"fortran_split".replace(".","p")
    base_paths = dict({
        "home": home_dir,
        "work": join(scratch_dir, date_str, sub_date_str, expt_str, "fms_tmp"), 
        "output": join(scratch_dir, date_str, sub_date_str, expt_str, "fms_output"), 
        })
    perturbation_specs = dict({ # Shouldn't matter in this context
        "fields": ["vors_real","ts_real","ln_ps_real"],
        "amplitude": 1.0e-3,
        "timelevels": [0,1],
        })
    ens = FriersonGCMEnsemble.default_init(base_paths, perturbation_specs, days_per_chunk, ensemble_size_limit)
    start_time = 0 #800

    # Control run
    warmstart_info = dict({
        "time": start_time, 
        "file": None, #f"/pool001/ju26596/fms_archive/2022-12-13/2/ctrl_noconv_21x100_8proc/abs1.0_smooth/mem_ctrl/restart/days{start_time-100:04}-{start_time:04}h00.cpio", 
        "time_origin": start_time,
        "do_perturbation_externally": False,
        })
    warmstart_info["perturbation_namelist"] = dict({
        "spectral_dynamics_nml": dict(
            do_perturbation = False,
            #do_perturbation_eachday = False,
            num_perturbations_actual = 1,
            days_to_perturb = [-1],
            seed_values = [12345],
            perturbation_fraction = [perturbation_specs["amplitude"]],
            )
        })
    ens.initialize_new_member(FriersonGCMEnsembleMember, warmstart_info)
    memidx2run = np.array([0])
    num_chunks_per_mem = np.array([1])
    ens.run_batch(memidx2run, num_chunks_per_mem) #, perturb_fields=False)

    # Perturbed run 1: branch off 1/3 of the way through
    warmstart_info["perturbation_namelist"] = dict({
        "spectral_dynamics_nml": dict(
            do_perturbation = True,
            #do_perturbation_eachday = False,
            num_perturbations_actual = 1,
            days_to_perturb = [start_time + int(days_per_chunk / 3)],
            seed_values = [12345],
            perturbation_fraction = [perturbation_specs["amplitude"]],
            )
        })
    ens.initialize_new_member(FriersonGCMEnsembleMember, warmstart_info)
    memidx2run = np.array([1])
    num_chunks_per_mem = np.array([1])
    ens.run_batch(memidx2run, num_chunks_per_mem) #, perturb_fields=False)

    # Perturbed run 2: branch off 2/3 of the way through
    warmstart_info["perturbation_namelist"] = dict({
        "spectral_dynamics_nml": dict(
            do_perturbation = True,
            #do_perturbation_eachday = False,
            num_perturbations_actual = 2,
            days_to_perturb = [start_time+int(days_per_chunk/3), start_time+int(2*days_per_chunk/3)],
            #days_to_perturb = [start_time+int(2*days_per_chunk/3)],
            seed_values = [12345,30492],
            perturbation_fraction = [perturbation_specs["amplitude"]],
            )
        })
    ens.initialize_new_member(FriersonGCMEnsembleMember, warmstart_info)
    print(f'{len(ens.mem_list) = }')
    memidx2run = np.array([2])
    num_chunks_per_mem = np.array([1])
    ens.run_batch(memidx2run, num_chunks_per_mem)
    
    # Save out the ensemble
    pickle.dump(ens, open(join(ens.dirs["output"], "ens"), "wb"))
    return




def score_fun_instantaneous(ds):
    # This score fun must be positive, and in the algorithm trajectories will be scored by their running maximum, so that the scores are always increasing over time.
    score = (
           transformations.observable_from_name(ds["1xday"], "total_rain")
           .sel(lat=slice(49,52), lon=slice(179,181))
           .mean(dim=["lat","lon"]))

    return score 

if __name__ == "__main__":
    print(f'Got into Main')
    nproc = int(sys.argv[1])
    print(f'{nproc = }')
    small_branching_ensemble(nproc)
