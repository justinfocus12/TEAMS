from abc import ABC, abstractmethod
import numpy as np
from numpy.random import default_rng
from scipy import sparse as sps
import forcing
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)

class DynamicalSystem(ABC):
    def __init__(self):
        return
    def list_required_instance_variables(self):
        return []
    @abstractmethod
    def generate_default_icandf(self,init_time,fin_time): # for spinup
        pass
    @abstractmethod
    def run_trajectory(self, icandf, obs_fun, saveinfo):
        # return some metadata sufficient to reconstruct the output, for example (1) a filename, (2) full numpy array of the output of an ODE solver. 
        # Optionally, return some observables passed as a dictionary of function handles
        # icandf stands for "initial conditions and forcing." It: could be e.g. 
        # 1. (a full state vector, a few impulses) (for a small ODESystem) 
        # 2. (a full state vector, a few reseeds) (for a small stochastic ODESystem) 
        # 3. (A filename containing a full state vector, a namelist) (for a big PDE system)
        pass 


# TODO make subclasses for ODESystem and SDESystem, each of which will have their own way to incorporate perturbations (apply_impulse vs run with white noise) and have their own way to generate a random sequence of standard inputs. 

class ODESystem(DynamicalSystem):
    # This is only SMALL systems --- small enough to fit a full-state trajectory in memory, avoiding the need for restart files. 
    def __init__(self, state_dim, config):
        self.state_dim = state_dim
        self.config = config
        self.derive_parameters(config) # This includes both physical and simulation parameters
        return
    @staticmethod
    @abstractmethod
    def label_from_config(config):
        pass
    @abstractmethod
    def derive_parameters(self, config):
        # convert raw configuration into class attributes for efficient integration of dynamics
        pass
    @staticmethod
    def list_required_instance_variables():
        varreq = super().list_required_instance_variables()
        varreq += ['dt_step','dt_save','impulse_matrix','impulse_dim']
        return varreq
    @abstractmethod
    def tendency(self, t, x): # aka 'drift' for an SDE
        pass
    def apply_impulse(self, t, x, imp):
        # apply the impulse perturbation from imp to the instantaneous state x, to get a perturbed state xpert
        return x + self.impulse_matrix @ imp
    @abstractmethod
    def generate_default_init_cond(self,init_time): # for spinup
        pass
    def generate_default_forcing_sequence(self,init_time,fin_time): 
        f = forcing.ImpulsiveForcing([init_time], [np.zeros(self.impulse_dim)], fin_time)
        return f
    def generate_default_icandf(self,init_time,fin_time):
        init_cond = self.generate_default_init_cond(init_time)
        f = self.generate_default_forcing_sequence(init_time,fin_time)
        icandf = dict({'init_cond': init_cond, 'frc': f})
        return icandf
    def assemble_metadata(self, icandf, method, saveinfo):
        md = dict({
            'icandf': icandf,
            'method': method,
            'filename': saveinfo['filename'],
            })
        return md
        
    def load_trajectory(self, metadata, tspan=None):
        traj = dict(np.load(metadata['filename']))
        if tspan is not None:
            idx0 = np.where(traj['t'] == tspan[0])[0][0]
            idx1 = np.where(traj['t'] == tspan[1])[0][0]
            traj['t'] = traj['t'][idx0:idx1+1]
            traj['x'] = traj['x'][idx0:idx1+1]
        return traj['t'],traj['x']
    # These are driven by impulses only
    def timestep_rk4(self, t, x): # physical time units
        k1 = self.dt_step * self.tendency(t,x)
        k2 = self.dt_step * self.tendency(t+self.dt_step/2, x+k1/2)
        k3 = self.dt_step * self.tendency(t+self.dt_step/2, x+k2/2)
        k4 = self.dt_step * self.tendency(t+self.dt_step, x+k3)
        xnew = x + (k1 + 2*(k2 + k3) + k4)/6
        return t+self.dt_step, xnew
    def timestep_euler(self, t, x): # physical time units
        k1 = self.dt_step * self.tendency(t,x)
        xnew = x + k1
        return t+self.dt_step, xnew
    def run_trajectory_unperturbed(self, init_cond, init_time, fin_time, method):
        assert(isinstance(init_time,int) and isinstance(fin_time,int))
        t_save = np.arange(init_time, fin_time+1, 1) # unitless
        tp_save = t_save * self.dt_save # physical (unitful)
        Nt_save = len(t_save)
        # Initialize the solution array
        x_save = np.zeros((Nt_save, self.state_dim))
        # special cases: endpoints
        if init_time % 1 == 0:
            x_save[0] = init_cond
            i_save = 1
        else:
            i_save = 0
        tp_save_next = tp_save[i_save]
        x = init_cond.copy()
        tp = init_time * self.dt_save # physical units
        timestep_fun_dict = {'rk4': self.timestep_rk4, 'euler': self.timestep_euler}
        timestep_fun = timestep_fun_dict[method]
        while tp < tp_save[-1]:
            tpnew,xnew = timestep_fun(tp, x)
            if tpnew > tp_save_next:
                new_weight = (tp_save_next - tp)/self.dt_step 
                # TODO: save out an observable instead of the full state? Depends on a specified frequency
                x_save[i_save] = (1-new_weight)*x + new_weight*xnew 
                i_save += 1
                if i_save < Nt_save:
                    tp_save_next = tp_save[i_save]
            x = xnew
            tp = tpnew
        return t_save,x_save
    def run_trajectory(self, icandf, obs_fun, saveinfo):
        init_cond_nopert,f = icandf['init_cond'],icandf['frc']
        assert(isinstance(f.init_time,int) and isinstance(f.fin_time,int))
        t = np.arange(f.init_time, f.fin_time+1)
        Nt = len(t)
        x = np.zeros((Nt, self.state_dim))
        # Need to set up three things: init_time, init_cond, fin_time
        init_time_temp = f.init_time
        method = 'rk4'
        # run one segment at a time, undisturbed, from one impulse to the next
        init_cond_temp = init_cond_nopert.copy()
        i_save = 0
        nimp = len(f.impulse_times)
        for i_imp in range(nimp):
            init_cond_temp = self.apply_impulse(init_time_temp, init_cond_temp, f.impulses[i_imp])
            if i_imp+1 < len(f.impulse_times):
                fin_time_temp = f.impulse_times[i_imp+1]
            else:
                fin_time_temp = f.fin_time
            print(f'{init_time_temp = }; {fin_time_temp = }')
            t_temp,x_temp = self.run_trajectory_unperturbed(init_cond_temp, init_time_temp, fin_time_temp, method)
            x[i_save:i_save+len(t_temp)] = x_temp
            init_time_temp = fin_time_temp
            init_cond_temp = x_temp[-1]
            i_save += len(t_temp) - 1
        # Return metadata and observables
        metadata = self.assemble_metadata(icandf, method, saveinfo)
        observables = obs_fun(t,x)
        # save full state out to saveinfo
        np.savez(saveinfo['filename'], t=t, x=x)
        return metadata,observables


class SDESystem(DynamicalSystem):
    def __init__(self, ode, config_sde):
        self.ode = ode
        self.dt_save = ode.dt_save
        self.config = config_sde
        self.derive_parameters(config_sde) # This includes both physical and simulation parameters
        return

    @staticmethod
    def list_required_instance_variables():
        varreq = super().list_required_instance_variables()
        varreq += ['sqrt_dt_step','seed_min','seed_max','white_noise_dim']
        return varreq
    @abstractmethod
    def derive_parameters(self, config):
        pass
    @abstractmethod
    def diffusion(self, t, x):
        pass
    def timestep_euler_maruyama(self, t, x, rng): # physical time units
        k1 = self.ode.dt_step * self.ode.tendency(t,x)
        sdw = self.sqrt_dt_step * self.diffusion(t,x) @ rng.normal(size=(self.white_noise_dim,))
        return t+self.ode.dt_step, x+k1+sdw
    def run_trajectory_unperturbed(self, init_cond, init_time, fin_time, method, rng):
        print(f'{init_time = }, {fin_time = }')
        #assert(isinstance(init_time,int) and isinstance(fin_time,int))
        t_save = np.arange(init_time, fin_time+1, 1) # unitless
        tp_save = t_save * self.ode.dt_save # physical (unitful)
        Nt_save = len(t_save)
        # Initialize the solution array
        x_save = np.zeros((Nt_save, self.ode.state_dim))
        # special cases: endpoints
        if init_time % 1 == 0:
            x_save[0] = init_cond
            i_save = 1
        else:
            i_save = 0
        tp_save_next = tp_save[i_save]
        x = init_cond.copy()
        tp = init_time * self.ode.dt_save # physical units
        if method == 'euler_maruyama':
            timestep_fun = self.timestep_euler_maruyama
        while tp < tp_save[-1]:
            tpnew,xnew = timestep_fun(tp, x, rng)
            if tpnew > tp_save_next:
                new_weight = (tp_save_next - tp)/self.ode.dt_step 
                # TODO: save out an observable instead of the full state? Depends on a specified frequency
                x_save[i_save] = (1-new_weight)*x + new_weight*xnew 
                i_save += 1
                if i_save < Nt_save:
                    tp_save_next = tp_save[i_save]
            x = xnew
            tp = tpnew
        return t_save,x_save
    def run_trajectory(self, icandf, obs_fun, saveinfo):
        init_cond_nopert,f = icandf['init_cond'],icandf['frc']
        t = np.arange(f.init_time, f.fin_time+1)
        Nt = len(t)
        x = np.zeros((Nt, self.ode.state_dim))
        # Need to set up three things: init_time, init_cond, fin_time
        init_time_temp = f.init_time
        method = 'euler_maruyama'
        init_cond_temp = init_cond_nopert.copy()
        i_save = 0
        frc_change_times = f.get_forcing_times()
        i_df_bytype = np.zeros(len(f.frc_list), dtype=int)
        for i_df in range(len(frc_change_times)):
            if i_df+1 < len(frc_change_times):
                fin_time_temp = frc_change_times[i_df+1]
            else:
                fin_time_temp = f.fin_time
            for i_comp in range(len(f.frc_list)):
                fcomp = f.frc_list[i_comp]
                if frc_change_times[i_df] in fcomp.get_forcing_times():
                    if isinstance(fcomp, forcing.ImpulsiveForcing):
                        init_cond_temp = self.ode.apply_impulse(init_time_temp, init_cond_temp, fcomp.impulses[i_df_bytype[i_comp]])
                    elif isinstance(fcomp, forcing.WhiteNoiseForcing):
                        rng = default_rng(fcomp.seeds[i_df_bytype[i_comp]])
                    i_df_bytype[i_comp] += 1
            print(f'{init_time_temp = }, {fin_time_temp = }')
            t_temp,x_temp = self.run_trajectory_unperturbed(init_cond_temp, init_time_temp, fin_time_temp, method, rng)
            x[i_save:i_save+len(t_temp)] = x_temp
            init_time_temp = fin_time_temp
            init_cond_temp = x_temp[-1]
            i_save += len(t_temp) - 1
        metadata = self.assemble_metadata(icandf, method, saveinfo)
        observables = obs_fun(t,x)
        # save full state out to saveinfo
        np.savez(saveinfo['filename'], t=t, x=x)
        return metadata,observables
    def generate_default_forcing_sequence(self,init_time,fin_time):
        f = forcing.WhiteNoiseForcing([init_time], [self.seed_min], fin_time)
        return f
    def generate_default_icandf(self,init_time,fin_time):
        f_white = self.generate_default_forcing_sequence(init_time, fin_time)
        icandf_imp = self.ode.generate_default_icandf(init_time,fin_time)
        icandf = dict({
            'init_cond': icandf_imp['init_cond'], 

            'frc': forcing.SuperposedForcing([icandf_imp['frc'],f_white]),
            })
        return icandf
    def assemble_metadata(self, icandf, method, saveinfo):
        md = dict({
            'icandf': icandf,
            'method': method,
            'filename': saveinfo['filename'],
            })
        return md
    def load_trajectory(self, metadata, tspan=None):
        return self.ode.load_trajectory(metadata, tspan)

class CoupledSystem(DynamicalSystem):
    # drive an ODE x with an SDE y (meaning an Ito diffusion)
    def __init__(self, ode, sde, config_coupling):
        self.ode = ode # X variables 
        self.sde = sde # Y variables
        # TODO: what if ode and sde have different save-out times? 
        assert(self.ode.dt_save == self.sde.dt_save)
        self.dt_save = ode.dt_save
        self.derive_parameters(config_coupling)
        return
    @abstractmethod
    def perturbed_tendency(self, t, x, xdot, y):
        # This should only depend on the current state variable, y. 
        pass
    def timestep_euler_maruyama(self, t, x, y, rng):
        dx = self.ode.dt_step * self.perturbed_tendency(
                t, x, self.ode.tendency(t, x), y)
        dy_drift = self.ode.dt_step * self.sde.ode.tendency(t, y) # TODO replace with self.sde.run_trajectory_unperturbed (need to generalize time units first to deal with fast-slow systems, for example)
        sdw += self.sde.sqrt_dt_step * self.sde.diffusion(t, y) @ rng.normal(size=(self.sde.white_noise_dim,))
        return t+self.ode.dt_step, x+dx, y+dy_drift+sdw
    def run_trajectory_unperturbed(self, init_cond_x, init_cond_y, init_time, fin_time, method, rng):
        print(f'{init_time = }, {fin_time = }')
        #assert(isinstance(init_time,int) and isinstance(fin_time,int))
        t_save = np.arange(init_time, fin_time+1, 1) # unitless
        tp_save = t_save * self.ode.dt_save # physical (unitful)
        Nt_save = len(t_save)
        # Initialize the solution array
        x_save = np.zeros((Nt_save, self.ode.state_dim))
        y_save = np.zeros((Nt_save, self.sde.state_dim))
        # special cases: endpoints
        if init_time % 1 == 0:
            x_save[0] = init_cond_x
            y_save[0] = init_cond_y
            i_save = 1
        else:
            i_save = 0
        tp_save_next = tp_save[i_save]
        x = init_cond_x.copy()
        y = init_cond_y.copy()
        tp = init_time * self.ode.dt_save # physical units
        if method == 'euler_maruyama':
            timestep_fun = self.timestep_euler_maruyama
        while tp < tp_save[-1]:
            tpnew,xnew,ynew = timestep_fun(tp, x, y, rng)
            if tpnew > tp_save_next:
                new_weight = (tp_save_next - tp)/self.ode.dt_step 
                # TODO: save out an observable instead of the full state? Depends on a specified frequency
                x_save[i_save] = (1-new_weight)*x + new_weight*xnew 
                y_save[i_save] = (1-new_weight)*y + new_weight*ynew 
                i_save += 1
                if i_save < Nt_save:
                    tp_save_next = tp_save[i_save]
            x = xnew
            y = ynew
            tp = tpnew
        return t_save,x_save,y_save
    def run_trajectory(self, icandf, obs_fun, saveinfo):
        init_cond_nopert_x,f_x = icandf['x']['init_cond'],icandf['x']['frc']
        init_cond_nopert_y,f_y = icandf['y']['init_cond'],icandf['y']['frc']
        t = np.arange(f_x.init_time, f_x.fin_time+1)
        Nt = len(t)
        x = np.zeros((Nt, self.ode.state_dim))
        y = np.zeros((Nt, self.sde.state_dim))
        # Need to set up three things: init_time, init_cond, fin_time
        init_time_temp = f_x.init_time
        method = 'euler_maruyama'
        init_cond_temp_x = init_cond_nopert_x.copy()
        init_cond_temp_y = init_cond_nopert_y.copy()
        i_save = 0
        frc_change_times_x = f_x.get_forcing_times()
        frc_change_times_y = f_y.get_forcing_times()
        i_df_x = np.zeros(len(f_x.frc_list), dtype=int)
        i_df_bytype_y = np.zeros(len(f_y.frc_list), dtype=int)
        frc_change_times = np.union1d(frc_change_times_x, frc_change_times_y)
        for i_df in range(len(frc_change_times)):
            if i_df+1 < len(frc_change_times):
                fin_time_temp = frc_change_times[i_df+1]
            else:
                fin_time_temp = f.fin_time
            # Forces on x can only be impulsive 
            if frc_change_times[i_df] in f_x.get_forcing_times():
                init_cond_temp_x = self.ode.apply_impulse(init_time_temp, init_cond_temp_x, f_x.impulses[i_df_x])
                i_df_x += 1
            if frc_change_times[i_dg] in f_y.get_forcing_times():
                for i_comp in range(len(f_y.frc_list)):
                    fcomp = f_y.frc_list[i_comp]
                    if frc_change_times[i_df] in fcomp.get_forcing_times():
                        if isinstance(fcomp, forcing.ImpulsiveForcing):
                            init_cond_temp_y = self.sde.ode.apply_impulse(init_time_temp, init_cond_temp_y, fcomp.impulses[i_df_bytype_y[i_comp]])
                        elif isinstance(fcomp, forcing.WhiteNoiseForcing):
                            rng = default_rng(fcomp.seeds[i_df_bytype_y[i_comp]])
                        i_df_bytype_y[i_comp] += 1
                print(f'{init_time_temp = }, {fin_time_temp = }')
            t_temp,x_temp,y_temp = self.run_trajectory_unperturbed(init_cond_temp_x, init_cond_temp_y, init_time_temp, fin_time_temp, method, rng)
            x[i_save:i_save+len(t_temp)] = x_temp
            y[i_save:i_save+len(t_temp)] = y_temp
            init_time_temp = fin_time_temp
            init_cond_temp = x_temp[-1]
            i_save += len(t_temp) - 1
        metadata = self.assemble_metadata(icandf, method, saveinfo)
        observables = obs_fun(t,x,y)
        # save full state out to saveinfo
        np.savez(saveinfo['filename'], t=t, x=x, y=y)
        return metadata,observables
    def assemble_metadata(self, icandf, method, saveinfo):
        md = dict({
            'icandf': icandf,
            'method': method,
            'filename': saveinfo['filename'],
            })
        return md
    def load_trajectory(self, metadata, tspan=None):
        traj = dict(np.load(metadata['filename']))
        if tspan is not None:
            idx0 = np.where(traj['t'] == tspan[0])[0][0]
            idx1 = np.where(traj['t'] == tspan[1])[0][0]
            traj['t'] = traj['t'][idx0:idx1+1]
            traj['x'] = traj['x'][idx0:idx1+1]
            traj['y'] = traj['y'][idx0:idx1+1]
        return traj['t'],traj['x'],traj['y']

