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
    def __init__(self, config):
        self.derive_parameters(config)
        self.configure_os_environment()
        self.compile_mppnccombine()
        self.compile_model()
        self.nproc = 1
        return

    def set_nproc(self, nproc):
        self.nproc = nproc
        return

    def generate_default_icandf(self, init_time, fin_time):
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
        abbrv = (r"res%s_abs%g_pert%g"%(config['resolution'],config['abs'],config['pert_frac'])).replace('.','p')
        label = r"%s, $A=%g$"%(config['resolution'],config['abs'])
        return abbrv,label
    @classmethod
    def default_config(cls, source_dir_absolute, base_dir_absolute):
        config = dict({
            'resolution': 'T21',
            'abs': 1.0, # atmospheric absorption coefficient (larger means more greenhouse) 
            'pert_frac': 0.001,
            'nml_patches_misc': dict(),
            'source_dir_absolute': source_dir_absolute, # where the original source code comes from. Don't modify! 
            'base_dir_absolute': base_dir_absolute, # Copied from source_dir and then modified, compiled etc. 
            'platform': 'gnu',
            't_burnin': 5,
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
        mkmf = join(self.base_dir_absolute,"bin","mkmf")
        template = join(self.base_dir_absolute,'bin','mkmf.template.{self.platform}')
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

        # Basic dynamical systems attributes
        self.dt_save = 1.0 # days are the fundamental time unit
        self.t_burnin = config['t_burnin']

        # Directories containing source code and binaries
        self.base_dir_absolute = config['base_dir_absolute']
        self.platform = config['platform'] # probably gnu

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
        self.pert_frac = config['pert_frac']

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
        mpirun_output = subprocess.run(f'cd {wd}; /home/software/gcc/6.2.0/pkg/openmpi/4.0.4/bin/mpirun -np {self.nproc} fms.x', shell=True, executable='/bin/csh', capture_output=True)

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
        steps_per_day = day_end_tidx[1] - day_end_tidx[0]
        runavg = da.isel(time=day_end_tidx)
        for i_delay in range(1,steps_per_day):
            runavg += da.shift(time=i_delay).isel(time=day_end_tidx)
        runavg *= 1.0/steps_per_day
        return runavg
    # --------------- Observable functions ---------------------
    def observable_props(self):
        obslib = dict()
        obslib["effective_static_stability"] = dict({
            "abbrv": "ESS",
            "unit_symbol": "s$^{-2}$",
            "label": "Effective static stability",
            "cmap": "coolwarm",
            "cmin": None,
            "cmax": None,
            "clo": "gray",
            "chi": "yellow",
            })

        obslib["vertical_velocity"] = dict({
            "abbrv": "W",
            "unit_symbol": "Pa/s",
            "label": "Vertical velocity",
            "cmap": "coolwarm",
            "cmin": None,
            "cmax": None,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["meridional_velocity"] = dict({
            "abbrv": "V",
            "unit_symbol": "m/s",
            "label": "Meridional velocity",
            "cmap": "coolwarm",
            "cmin": None,
            "cmax": None,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["zonal_velocity"] = dict({
            "abbrv": "U",
            "unit_symbol": "m/s",
            "label": "Zonal velocity",
            "cmap": "coolwarm",
            "cmin": None,
            "cmax": None,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["exprec_scaling"] = dict({
            "abbrv": "XPS",
            "unit_symbol": "mm/day",
            "label": "Extreme precip. scaling",
            "cmap": "Blues",
            "cmin": 0.0,
            "cmax": 64.0,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["convection_rain"] = dict({
            "abbrv": "Rconv",
            "unit_symbol": "mm/day",
            "label": "Convection rain",
            "cmap": "Blues",
            "cmin": 0.0, 
            "cmax": 64.0,
            "clo": "gray", 
            "chi": "yellow",
            })
        obslib["condensation_rain"] = dict({
            "abbrv": "Rcond",
            "unit_symbol": "mm/day",
            "label": "Condensation rain",
            "cmap": "Blues",
            "cmin": 0.0, 
            "cmax": 64.0,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["total_rain"] = dict({
            "abbrv": "Rtot",
            "unit_symbol": "mm/day",
            "label": "Rain rate",
            "cmap": "Blues",
            "cmin": 0.0, 
            "cmax": 64.0,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["specific_humidity"] = dict({
            "abbrv": "Q",
            "unit_symbol": "kg/kg",
            "label": "Specific humidity",
            "cmap": "Blues",
            "cmin": None,
            "cmax": None,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["temperature"] = dict({
            "abbrv": "T",
            "unit_symbol": "K",
            "label": "Temperature",
            "cmap": "Reds",
            "cmin": 210.0, 
            "cmax": 350.0,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["column_water_vapor"] = dict({
            "abbrv": "CWV",
            "unit_symbol": r"kg m$^{-2}$",
            "label": "Column water vapor",
            "cmap": "Blues",
            "cmin": 0.0, 
            "cmax": 7.0,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["column_relative_humidity"] = dict({
            "abbrv": "CRH",
            "unit_symbol": r"fraction",
            "label": "Column relative humidity",
            "cmap": "Blues",
            "cmin": 0.0, 
            "cmax": 1.0,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["water_vapor_convergence"] = dict({
            "abbrv": "QCON",
            "unit_symbol": r"kg m$^{-2}$s$^{-1}$",
            "label": "Water vapor convergence",
            "cmap": "coolwarm_r",
            "cmin": None, 
            "cmax": None,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["vorticity"] = dict({
            "abbrv": "VOR",
            "unit_symbol": r"s$^{-1}$",
            "label": "Vorticity",
            "cmap": "coolwarm",
            "cmin": None, 
            "cmax": None,
            "clo": "gray",
            "chi": "yellow",
            })
        obslib["surface_pressure"] = dict({
            "abbrv": "PS",
            "unit_symbol": r"Pa",
            "label": "Surface pressure",
            "cmap": "rainbow",
            "cmin": 96.0e3, 
            "cmax": 103.0e3,
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
        return FriersonGCM.convection_rain(ds) + FriersonGCM.convection_rain(ds)
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

def dns_short_chain(nproc):
    # Run three trajectories, each one picking up where the previous one left off
    base_dir = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/frierson_gcm"
    date_str = "2024-02-16"
    sub_date_str = "1"
    print(f'About to generate default config')
    config = FriersonGCM.default_config(base_dir)
    gcm = FriersonGCM(config)
    gcm.set_nproc(nproc)

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

def dns_moderate(nproc):
    tododict = dict({
        'run':            1,
        'plot':           0,
        })
    # Create a small ensemble
    # Run three trajectories, each one picking up where the previous one left off
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/frierson_gcm"
    date_str = "2024-02-21"
    sub_date_str = "0/DNS"
    print(f'About to generate default config')
    config = FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)
    config['resolution'] = 'T21'
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
            gcm = FriersonGCM(config)
            ens = Ensemble(gcm,root_dir=root_dir)
            n_mem = 0
            init_time = 0
            init_cond = None
            parent = None
        ens.dynsys.set_nproc(nproc)
        for mem in range(n_mem,n_mem+num_chunks):
            fin_time = init_time + days_per_chunk
            icandf = ens.dynsys.generate_default_icandf(init_time,fin_time)
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
    if tododict['plot']:
        plot_dir = join(expt_dir,'plots')
        makedirs(plot_dir,exist_ok=True)

        ens = pickle.load(open(ens_filename,'rb'))
        ens.set_root_dir(root_dir)
        obslib = ens.dynsys.observable_props()
        ens = pickle.load(open(join(expt_dir,'ens.pickle'),'rb'))
        obs2plot = ['temperature','total_rain','column_water_vapor','surface_pressure']
        lat = 45.0
        lon = 180.0
        pfull = 1000.0

        # Plot full fields
        for mem in np.arange(ens.memgraph.number_of_nodes())[-1:]:
            dsmem = xr.open_mfdataset(join(ens.root_dir,ens.traj_metadata[mem]['filename_traj']), decode_times=False)
            print(f'{dsmem.coords = }')
            for obs in obs2plot:
                memobs = getattr(ens.dynsys, obs)(dsmem) #.sel(dict(lat=lat,lon=lon),method='nearest')
                if 'pfull' in memobs.dims:
                    memobs = memobs.sel(pfull=pfull,method='nearest')
                memobs = memobs.compute()
                print(f'{memobs.time = }')
                for day in memobs.time.to_numpy(): #.astype(int):
                    print(f'{day = }')
                    fig,axes = plt.subplots(figsize=(12,5),ncols=2,sharey=True)
                    ax = axes[0]
                    xr.plot.pcolormesh(memobs.sel(time=day), x='lon', y='lat', cmap=obslib[obs]['cmap'], ax=ax)
                    ax.set_title(r'%s [%s], mem. %d, day %d'%(obslib[obs]['label'], obslib[obs]['unit_symbol'], mem, day))
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    ax = axes[1]
                    hday, = xr.plot.plot(memobs.mean(dim=['time','lon']),y='lat',color='black',ax=ax,label=r'(zonal,time) avg')
                    havg, = xr.plot.plot(memobs.sel(time=day).mean(dim='lon'),y='lat',color='red',ax=ax,label=r'zonal avg')
                    ax.legend(handles=[hday,havg])

                    fig.savefig(join(plot_dir,r'%s_mem%d_day%d'%(obslib[obs]['abbrv'],mem,day)),**pltkwargs)
                    plt.close(fig)
    return


def small_branching_ensemble(nproc):
    tododict = dict({
        'run':            1,
        'plot':           1,
        })
    # Create a small ensemble
    # Run three trajectories, each one picking up where the previous one left off
    base_dir_absolute = '/home/ju26596/jf_conv_gray_smooth'
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/frierson_gcm"
    date_str = "2024-02-20"
    sub_date_str = "0"
    print(f'About to generate default config')
    config = FriersonGCM.default_config(base_dir_absolute,base_dir_absolute)
    label,display = FriersonGCM.label_from_config(config)
    expt_dir = join(scratch_dir,date_str,sub_date_str,label)

    if tododict['run']:
        makedirs(expt_dir,exist_ok=True)
        obs_fun = lambda t,x: None

        gcm = FriersonGCM(config)
        gcm.set_nproc(nproc)
        ens = Ensemble(gcm)

        seed_vals = [-1,29183,48271,39183,38383,88822,77612,22345]
        parent_duration = 15
        child_duration = 10

        # Parent member
        mem = 0
        init_time = 0
        fin_time = parent_duration
        icandf = gcm.generate_default_icandf(init_time,fin_time)
        saveinfo = dict({
            # Temporary folder
            'temp_dir': join(expt_dir,f'mem{mem}'),
            # Ultimate resulting filenames
            'filename_traj': join(expt_dir,f'mem{mem}.nc'),
            'filename_restart': join(expt_dir,f'restart_mem{mem}.cpio'),
            })
        _ = ens.branch_or_plant(icandf, obs_fun, saveinfo)


        # Branch off some children
        for mem in [1,2]:
            parent = 0
            mdp = ens.traj_metadata[parent]
            init_time = mdp['icandf']['frc'].fin_time
            icandf = dict({
                'init_cond': mdp['filename_restart'],
                'frc': forcing.ContinuousTimeForcing(init_time, init_time+child_duration, [init_time], [seed_vals[mem]])
                })
            saveinfo = dict({
                'temp_dir': join(expt_dir,f'mem{mem}'),
                'filename_traj': join(expt_dir,f'mem{mem}.nc'),
                'filename_restart': join(expt_dir,f'restart_mem{mem}.cpio'),
                })
            _ = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)

        # Another Parent member: run for 8 days
        mem = 3
        init_time = 0
        fin_time = parent_duration
        icandf = dict({
            'init_cond': None,
            'frc': forcing.ContinuousTimeForcing(init_time, fin_time, [init_time], [seed_vals[mem]])
            })
        saveinfo = dict({
            # Temporary folder
            'temp_dir': join(expt_dir,f'mem{mem}'),
            # Ultimate resulting filenames
            'filename_traj': join(expt_dir,f'mem{mem}.nc'),
            'filename_restart': join(expt_dir,f'restart_mem{mem}.cpio'),
            })
        _ = ens.branch_or_plant(icandf, obs_fun, saveinfo)


        # Branch off some children
        for mem in [4,5]:
            parent = 3
            mdp = ens.traj_metadata[parent]
            init_time = mdp['icandf']['frc'].fin_time
            icandf = dict({
                'init_cond': mdp['filename_restart'],
                'frc': forcing.ContinuousTimeForcing(init_time, init_time+child_duration, [init_time], [seed_vals[mem]])
                })
            saveinfo = dict({
                'temp_dir': join(expt_dir,f'mem{mem}'),
                'filename_traj': join(expt_dir,f'mem{mem}.nc'),
                'filename_restart': join(expt_dir,f'restart_mem{mem}.cpio'),
                })
            _ = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)

        # Save out the ensemble for later querying
        pickle.dump(ens, open(join(expt_dir,'ens.pickle'),'wb'))
    if tododict['plot']:
        plot_dir = join(expt_dir,'plots')
        makedirs(plot_dir,exist_ok=True)

        ens = pickle.load(open(join(expt_dir,'ens.pickle'),'rb'))
        obslib = ens.dynsys.observable_props()
        obs2plot = ['temperature','total_rain','column_water_vapor','surface_pressure'][1:]
        lat = 45.0
        lon = 180.0
        pfull = 500.0

        # Plot local observables for all members
        obs_vals = dict({obs: [] for obs in obs2plot})
        for mem in range(ens.memgraph.number_of_nodes()):
            dsmem = xr.open_mfdataset(ens.traj_metadata[mem]['filename_traj'], decode_times=False)
            i_lat = np.argmin(np.abs(dsmem.lat.values - lat))
            i_lon = np.argmin(np.abs(dsmem.lon.values - lon))
            for obs in obs2plot:
                memobs = getattr(ens.dynsys, obs)(dsmem).isel(lat=i_lat,lon=i_lon).compute()
                if 'pfull' in memobs.dims:
                    i_pfull = np.argmin(np.abs(dsmem.pfull.values - pfull))
                    memobs = memobs.isel(pfull=i_pfull)
                obs_vals[obs].append(memobs)
        for obs in obs2plot:
            fig,ax = plt.subplots(figsize=(20,5))
            handles = []
            for mem in range(ens.memgraph.number_of_nodes()):
                h, = xr.plot.plot(obs_vals[obs][mem], x='time', label=f'm{mem}', marker='o')
                handles.append(h)
            ax.legend(handles=handles)
            ax.set_title(obslib[obs]['label'])
            fig.savefig(join(plot_dir,f'{obslib[obs]["abbrv"]}.png'),**pltkwargs)


        # Plot full fields
        for mem in [0,4]: #range(ens.memgraph.number_of_nodes()):
            dsmem = xr.open_mfdataset(ens.traj_metadata[mem]['filename_traj'], decode_times=False)
            i_pfull = np.argmin(np.abs(dsmem.pfull.values - pfull))
            for obs in obs2plot:
                memobs = getattr(ens.dynsys, obs)(dsmem)
                if 'pfull' in memobs.dims:
                    memobs = memobs.isel(pfull=i_pfull)
                memobs = memobs.compute()
                for day in memobs.time.to_numpy().astype(int):
                    fig,ax = plt.subplots(figsize=(8,5))
                    xr.plot.pcolormesh(memobs.sel(time=day), x='lon', y='lat', cmap=obslib[obs]['cmap'], ax=ax)
                    ax.set_title(r'%s [%s], mem. %d, day %d'%(obslib[obs]['label'], obslib[obs]['unit_symbol'], mem, day))
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    fig.savefig(join(plot_dir,r'%s_mem%d_day%d'%(obslib[obs]['abbrv'],mem,day)),**pltkwargs)
                    plt.close(fig)

    return








# ----------- Below are some standard test methods ------------



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
    dns_moderate(nproc)
