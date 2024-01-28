from abc import ABC, abstractmethod
import numpy as np
from numpy.random import default_rng
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
    #@abstractmethod
    #def run_trajectory(self, f, observables):
    #    # return some metadata sufficient to reconstruct the output, for example (1) a filename, (2) full numpy array of the output of an ODE solver. 
    #    # Optionally, return some observables passed as a dictionary of function handles
    #    pass 



class ODESystem(DynamicalSystem):
    def __init__(self, state_dim, config, *args, **kwargs):
        self.state_dim = state_dim
        self.config = config
        self.derive_parameters(config) # This includes both physical and simulation parameters
        return
    @abstractmethod
    def derive_parameters(self, config):
        # convert raw configuration into class attributes for efficient integration of dynamics
        pass
    @abstractmethod
    def tendency(self, t, x): # aka 'drift' for an SDE
        pass
    @staticmethod
    @abstractmethod
    def apply_impulse(t, x, imp):
        # apply the impulse perturbation from imp to the instantaneous state x, to get a perturbed state xpert
        pass
    @staticmethod
    def timestep_rk4(t, x, dt, tendency): # physical time units
        k1 = dt * tendency(t,x)
        k2 = dt * tendency(t+dt/2, x+k1/2)
        k3 = dt * tendency(t+dt/2, x+k2/2)
        k4 = dt * tendency(t+dt, x+k3)
        xnew = x + (k1 + 2*(k2 + k3) + k4)/6
        return t+dt, xnew
    @staticmethod
    def timestep_euler(t, x, dt, tendency): # physical time units
        k1 = dt * tendency(t,x)
        xnew = x + k1
        return t+dt, xnew
    @staticmethod
    def timestep_euler_maruyama(t, x, dt, drift, diffusion, rng): # physical time units
        k1 = dt * tendency(t,x)
        sdw = np.sqrt(dt) * diffusion(t,x) @ rng.normal(size=(self.noise_dim,))
        xnew = x + k1 + sdw
        return t+dt, xnew
    def run_trajectory_unperturbed(self, init_cond, init_time, fin_time, method, rng=None):
        t_save = np.arange(int(np.ceil(init_time)), int(np.floor(fin_time))+1, 1) # unitless
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
        x = init_cond
        tp = init_time * self.dt_save # physical units
        if method == 'rk4':
            timestep_fun = self.timestep_rk4
            args = ()
        elif method == 'euler':
            timestep_fun = self.timestep_euler
            args = ()
        elif method == 'euler_maruyama':
            assert rng is not None
            timestep_fun = self.timestep_euler_maruyama
            args = (self.diffusion,rng,)
        while tp < tp_save[-1]:
            tpnew,xnew = timestep_fun(tp, x, self.dt_step, self.tendency, *args)
            if tpnew > tp_save_next:
                new_weight = (tp_save_next - tp)/self.dt_step 
                x_save[i_save] = (1-new_weight)*x + new_weight*xnew 
                i_save += 1
                if i_save < Nt_save:
                    tp_save_next = tp_save[i_save]
            x = xnew
            tp = tpnew
        return t_save,x_save
    def run_trajectory(self, init_cond_nopert, f, observables, method):
        #assert(isinstance(f.init_time,int) * isinstance(f.fin_time,int))
        t = np.arange(int(np.ceil(f.init_time)), int(np.floor(f.fin_time))+1)
        print(f'{t = }')
        Nt = len(t)
        x = np.zeros((Nt, self.state_dim))
        # Need to set up three things: init_time, init_cond, fin_time
        init_time_temp = f.init_time
        if isinstance(f, forcing.ImpulsiveForcing):
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
        elif isinstance(f, forcing.WhiteNoiseForcing):
            init_cond = init_cond_nopert.copy()
            i_save = 0
            for i_seed in range(len(f.reseed_times)):
                assert init_time_temp == f.reseed_times[i_seed]
                if i_seed+1 < len(f.reseed_times):
                    fin_time_temp = f.reseed_times[i_seed+1]
                else:
                    fin_time_temp = f.fin_time
                rng = default_rng(f.seeds[i_seed])
                t_temp,x_temp = self.run_trajectory_unperturbed(init_cond_temp, init_time_temp, fin_time_temp, method, rng)
                x[i_save:i_save+len(t_temp)] = x_temp
                i_save += len(t_temp)
                init_time_temp = fin_time_temp
                init_cond_temp = x_temp[-1]

        return t,x




class Lorenz96(ODESystem):
    def __init__(self, config):
        state_dim = config['K']
        super().__init__(state_dim, config)
    def derive_parameters(self, config):
        self.K = config['K']
        self.F = config['F']
        self.dt_step = config['dt_step'] # physical
        self.dt_save = config['dt_save'] # physical
        # all time arrays will have integers as entries, for unambiguous time alignment
        return
    def tendency(self, t, x):
        return np.roll(x,1) * (np.roll(x, -1) - np.roll(x,2)) - x + self.F
    def apply_impulse(self, t, x, imp):
        return x + imp[0]*np.cos(2*np.pi*4*np.arange(self.K)/self.K) + imp[1]*np.sin(2*np.pi*4*np.arange(self.K)/self.K)
    


def test_Lorenz96():
    config = dict({'K': 40, 'F': 6.0, 'dt_step': 0.001, 'dt_save': 0.05})
    ode = Lorenz96(config)
    tu = ode.dt_save

    # Make a small set of simulations with branching perturbations
    # 0 ---------------------
    #        |
    # 1      o----x--------------
    #                      | 
    # 2                    x--------------
    # 
    # (In the above, 1 has to replicate trajectory of 0 from the beginning, since the state is not available at the time of splitting. Later, the Ensemble object will have methods for building this in)
    rng0 = default_rng(8888)
    init_cond = 0.001*rng0.normal(size=(config['K'],))
    init_time_phys = -4.0
    fin_time_phys = 15.0
    method = 'rk4'
    t_save_0,x_save_0 = ode.run_trajectory_unperturbed(init_cond, init_time_phys/tu, fin_time_phys/tu, method)

    fig,axes = plt.subplots(nrows=2,figsize=(10,10), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(t_save_0*tu, x_save_0[:,0])
    ax.set_xlabel('time')
    ax.set_ylabel('x0')
    ax = axes[1]
    im = ax.pcolormesh(t_save_0*tu, np.arange(ode.K)[::-1], x_save_0.T, shading='nearest', cmap='BrBG')
    ax.set_xlabel('time')
    ax.set_ylabel('Longitude $k$')
    fig.savefig('L96_x0', **pltkwargs)
    plt.close(fig)

    # now make a perturbation
    rng1 = default_rng(1928)
    impulse_times = [init_time_phys/tu,5.0/tu]
    impulses = [np.zeros(2),0.1*rng1.normal(size=(2,))]
    f = forcing.ImpulsiveForcing(impulse_times, impulses, (fin_time_phys+3)/tu)
    t_save_1,x_save_1 = ode.run_trajectory(init_cond, f, None, method)
    fig,axes = plt.subplots(nrows=2,figsize=(10,10), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(t_save_0*tu, x_save_0[:,0], color='black',linestyle='--')
    ax.plot(t_save_1*tu, x_save_1[:,0], color='red',linestyle='-')
    ax.set_xlabel('time')
    ax.set_ylabel('x0')
    ax = axes[1]
    im = ax.pcolormesh(t_save_1*tu, np.arange(ode.K)[::-1], x_save_1.T, shading='nearest', cmap='BrBG')
    ax.set_xlabel('time')
    ax.set_ylabel('Longitude $k$')
    fig.savefig('L96_x1', **pltkwargs)
    plt.close(fig)

    # now make a perturbation
    rng2 = default_rng(1928)
    impulse_times = [init_time_phys/tu,5.0/tu,10.0/tu]
    impulses = [np.zeros(2),0.1*rng2.normal(size=(2,)),0.2*rng2.normal(size=(2,))]
    f = forcing.ImpulsiveForcing(impulse_times, impulses, (fin_time_phys+3)/tu)
    t_save_2,x_save_2 = ode.run_trajectory(init_cond, f, None, method)
    fig,axes = plt.subplots(nrows=2,figsize=(10,10), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(t_save_0*tu, x_save_0[:,0], color='black',linestyle='--')
    ax.plot(t_save_1*tu, x_save_1[:,0], color='red',linestyle='-')
    ax.plot(t_save_2*tu, x_save_2[:,0], color='dodgerblue',linestyle='-')
    ax.set_xlabel('time')
    ax.set_ylabel('x2')
    ax = axes[1]
    im = ax.pcolormesh(t_save_1*tu, np.arange(ode.K)[::-1], x_save_1.T, shading='nearest', cmap='BrBG')
    ax.set_xlabel('time')
    ax.set_ylabel('Longitude $k$')
    fig.savefig('L96_x2', **pltkwargs)
    plt.close(fig)

    return

if __name__ == "__main__":
    test_Lorenz96()



