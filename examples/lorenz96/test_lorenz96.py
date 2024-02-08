import numpy as np
from numpy.random import default_rng
import pickle
from scipy import sparse as sps
from os.path import join, exists
from os import makedirs
import copy as copylib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
from lorenz96 import Lorenz96
from ensemble import Ensemble
import forcing


def test_Lorenz96(frc_type):
    # Same test whether white or impulsive

    tododict = dict({
        'run_ensemble':          1,
        'check_overlap':         1,
        'plot_ensemble':         1,
        })

    config = dict({'K': 40, 'F': 6.0, 'dt_step': 0.001, 'dt_save': 0.05,})
    config['t_burnin'] = int(10/config['dt_save'])
    config['frc'] = dict({
        'type': frc_type,
        'impulsive': dict({
            'wavenumbers': [1,4],
            'wavenumber_magnitudes': [0.1,0.1],
            'sites': [20,30],
            'site_magnitudes': [0.5, 0.5],
            }),
        'white': dict({
            'wavenumbers': [4],
            'wavenumber_magnitudes': [1.0],
            'sites': [],
            'site_magnitudes': [],
            }),
        })
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/lorenz96"
    date_str = "2024-02-06"
    sub_date_str = "0"
    param_abbrv,param_label = Lorenz96.label_from_config(config)
    ensdir = join(scratch_dir, date_str, sub_date_str, param_abbrv)

    if tododict['run_ensemble']: 

        ode = Lorenz96(config)
        tu = ode.dt_save

        ens = Ensemble(ensdir, ode)

        frc_times_phys = [-4.0, 5.0, 25.0]
        frc_times = [int(itp/tu) for itp in frc_times_phys]
        fin_time_phys = 40.0
        fin_time = int(fin_time_phys/tu)
        if frc_type == 'impulsive':
            rng = default_rng(8764)
            frcs = rng.normal(size=(len(frc_times), ode.impulse_dim))
            ForcingType = forcing.ImpulsiveForcing
        elif frc_type == 'white':
            frcs = [8764, 3910, 6789]
            ForcingType = forcing.WhiteNoiseForcing

        # Decide which observables to track 
        obs_names = ['t','x0','E0','E','Emax']
        obs_dict = dict({name: [] for name in obs_names})

        def obs_fun(t,x):
            obs = dict({name: ode.observable(t,x,name) for name in obs_names})
            return obs

        def append_obs_dict(od, odnew):
            for name in list(od.keys()):
                od[name].append(odnew[name])
            return


        # 0
        icandf = dict({
            'init_cond': 0.001*np.sin(2*np.pi*np.arange(config['K'])/config['K']), 
            'frc': ForcingType(frc_times[:1], frcs[:1], fin_time)
            })
        saveinfo = dict(filename = join(ensdir, 'mem0.npz'))
        obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo)
        append_obs_dict(obs_dict, obs_val_new)

        # 1 
        parent = 0
        icandf = dict({
            'init_cond': 0.001*np.sin(2*np.pi*np.arange(config['K'])/config['K']), 
            'frc': ForcingType(frc_times[:2], frcs[:2], fin_time)
            })
        saveinfo = dict(filename = join(ensdir, 'mem1.npz'))
        obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
        append_obs_dict(obs_dict, obs_val_new)

        # 2 
        # This time branch directly off the parent
        # TODO devise a more elegant solution to the problem of branching from a specific point
        parent = 0
        t_parent,x_parent = ode.load_trajectory(ens.traj_metadata[parent], tspan=[frc_times[1]]*2)
        print(f"{t_parent = }; {frc_times = }")
        icandf = dict({
            'init_cond': x_parent,
            'frc': ForcingType(frc_times[1:3], frcs[1:3], fin_time)
            })
        saveinfo = dict(filename = join(ensdir, 'mem2.npz'))
        obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
        append_obs_dict(obs_dict, obs_val_new)

        # Save the observables; this will be the 'save_state' for future managers
        pickle.dump(obs_dict,open(join(ensdir,'obs_dict.pickle'),'wb'))
        pickle.dump(ens, open(join(ensdir,'ens.pickle'),'wb')
        

    
    ens = pickle.load(open(join(ensdir, 'ens.pickle'), 'rb'))
    obs_dict = pickle.load(open(join(ensdir, 'obs_dict.pickle'),'rb'))
    ode = ens.dynsys
    tu = ode.dt_save

    if tododict['check_overlap']:
        # Do all three trajectories match? 
        t0,x0 = ode.load_trajectory(ens.traj_metadata[0])
        t1,x1 = ode.load_trajectory(ens.traj_metadata[1])
        t2,x2 = ode.load_trajectory(ens.traj_metadata[2])

        x0_neq_x1 = np.where(np.any(x0 != x1, axis=1))[0]
        if len(x0_neq_x1) > 0:
            print(f"{t0[x0_neq_x1[0]]*tu = }")
        else:
            print(f"x1 never diverges from x0")

        print(f"{t0[[0,-1]] = }; {t1[[0,-1]] = }; {t2[[0,-1]] = };")
        idxshift = t2[0] - t1[0]
        x1_neq_x2 = np.where(np.any(x1[idxshift:] != x2, axis=1))[0]
        if len(x1_neq_x2) > 0:
            print(f"{t2[x1_neq_x2[0]]*tu = }")
        else:
            print(f"x2 never diverges from x1")

    if tododict['plot_ensemble']:
        # Plot the observables
        mems = [0,1,2]
        #kt = obs_dict['names'].index('t')
        ts = obs_dict['t'] #[obs_dict['vals'][mem][kt].astype(int) for mem in mems]
        #print(f'{ts = }')
        for obs_name in ['x0','E0','E','Emax']:
            #ko = obs_dict['names'].index(obs_name)
            obsvals2plot = obs_dict[obs_name] #[mem][ko] for mem in mems]
            #print(f'{obsvals2plot = }')
            fig,axes = ens.plot_observables(mems, ts, obsvals2plot)
            fig.savefig(join(ensdir, f"obs_{obs_name}.png"), **pltkwargs)
            plt.close(fig)
    return

if __name__ == "__main__":
    test_Lorenz96('impulsive')
