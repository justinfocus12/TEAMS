
import numpy as np
from numpy.random import default_rng
from scipy.linalg import solve_sylvester
from scipy.special import softmax
import xarray as xr
import os
from os.path import join, exists
from os import mkdir, makedirs
from numpy.random import default_rng
import pickle
import copy as copylib

import sys
sys.path.append("../..")
from ensemble import Ensemble,EnsembleMember

def dot2p(decnumber):
    if isinstance(decnumber,int):
        numstr = f"{decnumber}"
    else:
        numstr = f"{decnumber:.2f}".replace(".","p")
    return numstr

class CrommelinEnsemble(Ensemble):
    def setup_model(self):
        return
    def load_member_ancestry(self, i_mem_leaf):
        hist_list = []
        ds = self.mem_list[i_mem_leaf].load_history_selfmade()
        dt = ds["time"][1].item() - ds["time"][0].item()
        for i_mem_twig in self.address_book[i_mem_leaf][::-1][1:]:
            start_time = ds["time"][0].item()
            ds_new = self.mem_list[i_mem_twig].load_history_selfmade()
            ds = xr.concat([ds_new.sel(time=slice(None,start_time-dt/10)), ds], dim="time")
        return ds
    @classmethod
    def derive_parameters(cls, fpd):
        n_max = 1
        m_max = 2
        xdim = 7
        q = dict({"fpd": fpd})
        q["year_length"] = fpd["year_length"]
        q["epsilon"] = 16*np.sqrt(2)/(5*np.pi)
        q["C"] = fpd["C"]
        q["b"] = fpd["b"]
        q["gamma_limits_fpd"] = fpd["gamma_limits"]
        q["xstar"] = np.array([fpd["x1star"],0,0,fpd["r"]*fpd["x1star"],0,0])
        q["alpha"] = np.zeros(m_max)
        q["beta"] = np.zeros(m_max)
        q["gamma_limits"] = np.zeros((2,m_max))
        q["gamma_tilde_limits"] = np.zeros((2,m_max))
        q["delta"] = np.zeros(m_max)
        for i_m in range(m_max):
            m = i_m + 1
            q["alpha"][i_m] = 8*np.sqrt(2)/np.pi*m**2/(4*m**2 - 1) * (fpd["b"]**2 + m**2 - 1)/(fpd["b"]**2 + m**2)
            q["beta"][i_m] = fpd["beta"]*fpd["b"]**2/(fpd["b"]**2 + m**2)
            q["delta"][i_m] = 64*np.sqrt(2)/(15*np.pi) * (fpd["b"]**2 - m**2 + 1)/(fpd["b"]**2 + m**2)
            for j in range(2):
                q["gamma_tilde_limits"][j,i_m] = fpd["gamma_limits"][j]*4*m/(4*m**2 - 1)*np.sqrt(2)*fpd["b"]/np.pi
                q["gamma_limits"][j,i_m] = fpd["gamma_limits"][j]*4*m**3/(4*m**2 - 1)*np.sqrt(2)*fpd["b"]/(np.pi*(fpd["b"]**2 + m**2))
        # Construct the forcing, linear matrix, and quadratic matrix, but leaving out the time-dependent parts 
        # 1. Constant term
        F = q["C"]*q["xstar"]
        # 2. Matrix for linear term
        L = -q["C"]*np.eye(xdim-1)
        #L[0,2] = q["gamma_tilde"][0]
        L[1,2] = q["beta"][0]
        #L[2,0] = -q["gamma"][0]
        L[2,1] = -q["beta"][0]
        #L[3,5] = q["gamma_tilde"][1]
        L[4,5] = q["beta"][1]
        L[5,4] = -q["beta"][1]
        #L[5,3] = -q["gamma"][1]
        # 3. Matrix for bilinear term
        B = np.zeros((xdim-1,xdim-1,xdim-1))
        B[1,0,2] = -q["alpha"][0]
        B[1,3,5] = -q["delta"][0]
        B[2,0,1] = q["alpha"][0]
        B[2,3,4] = q["delta"][0]
        B[3,1,5] = q["epsilon"]
        B[3,2,4] = -q["epsilon"]
        B[4,0,5] = -q["alpha"][1]
        B[4,3,2] = -q["delta"][1]
        B[5,0,4] = q["alpha"][1]
        B[5,3,1] = q["delta"][1]
        q["forcing_term"] = F
        q["linear_term"] = L
        q["bilinear_term"] = B
        q["state_dim"] = 7
        return q
    @classmethod
    def default_init(cls, expt_dir, model_params_patch, ensemble_size_limit):
        dirs_ens = dict({
            "work": join(expt_dir, "work"),
            "output": join(expt_dir, "output"),
            })
        fundamental_param_dict = dict({
            "b": 0.5, "beta": 1.25, "gamma_limits": [0.2, 0.2], 
            "C": 0.1, "x1star": 0.95, "r": -0.801, "year_length": 400.0,})
        model_params = cls.derive_parameters(fundamental_param_dict)
        model_params["dt_sim"] = 0.1
        model_params["dt_save"] = 0.5 
        model_params["dt_print"] = 500.0
        model_params.update(model_params_patch)
        ensemble_size_limit = ensemble_size_limit
        ens = cls(dirs_ens, model_params, ensemble_size_limit)
        return ens
    def default_coldstart(self, seed_init, seed_run, init_time=0.0):
        coldstart_info = dict({
            "init_time": init_time, 
            "init_cond": np.concatenate((self.model_params["xstar"], [init_time])),
            "time_origin": 0.0,
            "dt_restart": 400.0,
            "seeddict": dict({
                "perturb": 83640,
                "run": 743,
                }),
            "perturbation_magnitude": 1e-3,
            })
        return coldstart_info


class CrommelinEnsembleMember(EnsembleMember):
    def set_run_params(self, model_params, warmstart_info, **kwargs):

        self.par = model_params.copy()
        self.xdim = 7

        # Warmstart information (including perturbation)
        self.init_cond_ancestral = warmstart_info["init_cond"]
        self.init_time_ancestral = warmstart_info["init_time"]
        self.init_cond = self.init_cond_ancestral
        self.init_time = self.init_time_ancestral
        self.time_origin = warmstart_info["time_origin"]
        self.par["dt_restart"] = warmstart_info["dt_restart"]

        rng = default_rng(warmstart_info["seeddict"]["perturb"])
        idx2perturb = np.array([4,5]) # Only the smallest-scales
        self.init_cond[idx2perturb] *= (1.0 + warmstart_info["perturbation_magnitude"] * rng.normal(size=len(idx2perturb)))

        self.term_file_list = []
        self.term_time_list = []
        self.hist_file_list = [] # This is special to low-dimensional process, where we can save out all the history we need in single files. In more complex models the information will be stored differently.


        return

    def setup_directories(self):
        # Nothing to do here: no structure needed beyond the base directories 
        return

    def cleanup_directories(self):
        return

    def orography_cycle(self,t_abs):
        """
        Parameters
        ----------
        t_abs: numpy.ndarray
            The absolute time. Shape should be (Nx,) where Nx is the number of ensemble members running in parallel. 

        Returns 
        -------
        gamma_t: numpy.ndarray
            The gamma parameter corresponding to the given time of year. It varies sinusoidaly. Same is (Nx,m_max) where m_max is the maximum zonal wavenumber.
        gamma_tilde_t: numpy.ndarray
            The gamma_tilde parameter corresponding to the given time of year. Same shape as gamma_t.
        """
        q = self.par
        cosine = np.cos(2*np.pi*t_abs/q["year_length"])
        sine = np.sin(2*np.pi*t_abs/q["year_length"])
        gamma_t = cosine * (q["gamma_limits"][1,:] - q["gamma_limits"][0,:])/2 + (q["gamma_limits"][0,:] + q["gamma_limits"][1,:])/2
        gammadot_t = -sine * (q["gamma_limits"][1,:] - q["gamma_limits"][0,:])/2
        gamma_tilde_t = cosine * (q["gamma_tilde_limits"][1,:] - q["gamma_tilde_limits"][0,:])/2 + (q["gamma_tilde_limits"][0,:] + q["gamma_tilde_limits"][1,:])/2
        gammadot_tilde_t = -sine * (q["gamma_tilde_limits"][1,:] - q["gamma_tilde_limits"][0,:])/2 + (q["gamma_tilde_limits"][0,:] + q["gamma_tilde_limits"][1,:])/2
        gamma_fpd_t = cosine * (q["gamma_limits_fpd"][1] - q["gamma_limits_fpd"][0])/2 
        gammadot_fpd_t = -sine * (q["gamma_limits_fpd"][1] - q["gamma_limits_fpd"][0])/2
        return gamma_t,gamma_tilde_t,gamma_fpd_t,gammadot_t,gammadot_tilde_t,gammadot_fpd_t
    def tendency(self,x):
        """
        Parameters
        ----------
        x: numpy.ndarray
            shape (Nx,xdim) the current state of the dynamical system
        Returns
        -------
        xdot: numpy.ndarray
            shape (Nx,xdim) the time derivative of x
        """
        # ------------ Do the calculation as written -----------------
        xdot = self.tendency_forcing(x) + self.tendency_dissipation(x) + self.tendency_quadratic(x)
        return xdot
    def tendency_forcing(self,x):
        xdim = len(x)
        xdot = np.zeros(xdim)
        xdot[:-1] = self.par["forcing_term"]
        xdot[-1] = 1.0
        return xdot 
    def tendency_dissipation(self,x):
        xdim = len(x)
        diss = np.zeros(xdim)
        diss[:-1] = x[:-1].dot(self.par["linear_term"].T)
        # Modify the time-dependent components
        gamma_t,gamma_tilde_t,gamma_fpd_t,gammadot_t,gammadot_tilde_t,gammadot_fpd_t = self.orography_cycle(x[6])
        diss[0] += gamma_tilde_t[0]*x[2]
        diss[2] -= gamma_t[0]*x[0]
        diss[3] += gamma_tilde_t[1]*x[5]
        diss[5] -= gamma_t[1]*x[3]
        return diss
    def tendency_quadratic(self,x):
        """
        Compute the tendency according to only the nonlinear terms, in order to check conservation of energy and enstrophy.
        """
        xdot = np.zeros(self.xdim)
        # ----------- Do the calculation with the precomputed terms -------------
        for j in range(self.xdim-1):
            xdot[j] += np.sum(x[:-1] * (self.par["bilinear_term"][j] @ x[:-1]))
        return xdot

    def run_one_cycle(self):
        # Better version of Euler-Maruyama without timestep nonsense
        Nt_save = int(np.ceil(self.par["dt_restart"] / self.par["dt_save"])) + 1
        t_save = np.linspace(self.init_time, self.init_time + self.par["dt_restart"], Nt_save)
        sims_per_save = int(np.ceil(self.par["dt_save"]/self.par["dt_sim"]))
        saves_per_print = int(np.ceil(self.par["dt_print"]/self.par["dt_save"]))

        x_save = np.nan*np.ones((Nt_save,self.par["state_dim"]))
        x_save[0] = self.init_cond
        dt = self.par["dt_sim"]
        sqrtdt = np.sqrt(dt)
        for i_save in range(1,Nt_save):
            x = x_save[i_save-1].copy()
            t = t_save[i_save-1]
            for i_sim in range(sims_per_save):
                k1 = self.tendency(x)
                k2 = self.tendency(x+dt*k1/2)
                k3 = self.tendency(x+dt*k2/2)
                k4 = self.tendency(x+dt*k3)

                x += 1.0/6*dt*(k1 + 2*k2 + 2*k3 + k4)
                t += dt
                x[-1] = t
            x_save[i_save] = x
            if i_save % saves_per_print == 0:
                print(f"Integrated through time {t} out of {t_save[-1]}")
        # Save history
        hist_file = join(self.dirs["output"], f"history_{dot2p(self.init_time)}-{dot2p(t)}.nc")
        x_ds = xr.Dataset(
                data_vars = {"x": xr.DataArray(coords={"time": t_save[1:], "state_var": np.arange(x_save.shape[1])}, data=x_save[1:])})
        x_ds.to_netcdf(hist_file)
        # Save restart
        term_file = join(self.dirs["output"], f"restart_{dot2p(t)}.npy")
        np.save(term_file, x)
        # Update the lists
        self.term_file_list += [term_file]
        self.term_time_list += [t]
        self.hist_file_list += [hist_file]
        # Update state for the next round
        self.init_time = t_save[-1]
        self.init_file = term_file

        return

    def load_history_selfmade(self):
        ds = xr.open_mfdataset(self.hist_file_list, decode_times=False)["x"]
        ds.close()
        return ds
        
def run_long_integration():
    ensemble_size_limit = 8
    home_dir = "/home/jf4241"
    scratch_dir = "/scratch/jf4241/splitting/crommelin"
    date_str = "2023-01-26"
    sub_date_str = "0"
    dt_restart = 100000.0
    num_chunks = 1
    expt_str = f"ctrl_{num_chunks}x{dt_restart}".replace(".","p")
    expt_dir = join(scratch_dir, date_str, sub_date_str, expt_str)
    model_params_patch = {}
    ens = CrommelinEnsemble.default_init(expt_dir, model_params_patch, ensemble_size_limit) 
    seed_init = 6532
    seed_run = 10945
    coldstart_info = ens.default_coldstart(seed_init, seed_run)
    coldstart_info["dt_restart"] = dt_restart
    ens.initialize_new_member(CrommelinEnsembleMember, coldstart_info)
    memidx2run = np.array([0])
    ens.run_batch(memidx2run, np.array([num_chunks]))
    pickle.dump(ens, open(join(ens.dirs["output"],"ens"),"wb"))
    return


if __name__ == "__main__":
    run_long_integration()
