import numpy as np
from numpy.random import default_rng
import pickle
from scipy import sparse as sps
from os.path import join, exists
from os import makedirs
import sys
import copy as copylib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
from lorenz96 import Lorenz96ODE,Lorenz96SDE
from ensemble import Ensemble
import forcing

def test_Lorenz96_ode():

    tododict = dict({
        'run_ensemble':          1,
        'check_overlap':         1,
        'plot_ensemble':         1,
        })

    config = dict({'K': 40, 'F': 6.0, 'dt_step': 0.001, 'dt_save': 0.05,})
    config['t_burnin'] = int(10/config['dt_save'])
    config['frc'] = dict({
        'type': 'impulsive',
        'impulsive': dict({
            'wavenumbers': [1,4],
            'wavenumber_magnitudes': [0.1,0.1],
            'sites': [20,30],
            'site_magnitudes': [0.5, 0.5],
            }),
        })
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/lorenz96"
    date_str = "2024-02-20"
    sub_date_str = "0"
    param_abbrv,param_label = Lorenz96ODE.label_from_config(config)
    ensdir = join(scratch_dir, date_str, sub_date_str, param_abbrv)
    makedirs(ensdir, exist_ok=True)

    if tododict['run_ensemble']: 
        ode = Lorenz96ODE(config)
        tu = ode.dt_save

        ens = Ensemble(ode)

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


        # -------- 0 ----------
        #                     | --------- 1 ---------
        #                     | --------- 2 ---------
        init_time_phys = -4.0
        fin_time_phys = 40.0
        split_times_phys = [5.0, 20.0]
        rng = default_rng(87654)

        init_time = int(init_time_phys/tu)
        fin_time = int(fin_time_phys/tu)
        split_times = [int(stp/tu) for stp in split_times_phys]

        # 0
        icandf = ode.generate_default_icandf(init_time, fin_time)
        print(f'For mem 0, {icandf = }')
        saveinfo = dict(filename = join(ensdir, 'mem0.npz'))
        obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo)
        append_obs_dict(obs_dict, obs_val_new)

        # 1 
        parent = 0
        md_parent = ens.traj_metadata[parent]
        icandf = copylib.deepcopy(md_parent['icandf'])
        icandf['frc'].impulse_times.append(split_times[0])
        icandf['frc'].impulses.append(rng.normal(size=ode.impulse_dim))
        saveinfo = dict(filename = join(ensdir, 'mem1.npz'))
        print(f'For mem 1, {icandf = }\n{icandf["frc"].impulse_times = }')
        obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
        append_obs_dict(obs_dict, obs_val_new)

        # 2 
        parent = 1
        md_parent = ens.traj_metadata[parent]
        icandf = copylib.deepcopy(md_parent['icandf'])
        icandf['frc'].impulse_times.append(split_times[1])
        icandf['frc'].impulses.append(rng.normal(size=ode.impulse_dim))
        saveinfo = dict(filename = join(ensdir, 'mem2.npz'))
        print(f'For mem 2, {icandf = }\n{icandf["frc"].impulse_times = }')
        obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
        append_obs_dict(obs_dict, obs_val_new)

        # Save the observables; this will be the 'save_state' for future managers
        pickle.dump(obs_dict,open(join(ensdir,'obs_dict.pickle'),'wb'))
        pickle.dump(ens, open(join(ensdir,'ens.pickle'),'wb'))
        

    
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

def test_Lorenz96_sde():

    tododict = dict({
        'run_ensemble':          1,
        'check_overlap':         1,
        'plot_ensemble':         1,
        })

    # Configure the ODE firt
    config_ode = dict({'K': 40, 'F': 6.0, 'dt_step': 0.001, 'dt_save': 0.05,})
    config_ode['t_burnin'] = int(10/config_ode['dt_save'])
    config_ode['frc'] = dict({
        'type': 'impulsive',
        'impulsive': dict({
            'wavenumbers': [],
            'wavenumber_magnitudes': [],
            'sites': [20,30],
            'site_magnitudes': [0.5, 0.5],
            }),
        })
    # Configure the SDE
    config_sde = dict({
        'seed_min': 1000,
        'seed_max': 100000,
        'frc': dict({
            'type': 'white',
            'white': dict({
                'wavenumbers': [1,4],
                'wavenumber_magnitudes': [0.1,0.1],
                'sites': [],
                'site_magnitudes': [],
                }),
            }),
        })
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/lorenz96"
    date_str = "2024-02-20"
    sub_date_str = "0"
    param_abbrv,param_label = Lorenz96SDE.label_from_config(config_ode,config_sde)
    ensdir = join(scratch_dir, date_str, sub_date_str, param_abbrv)
    makedirs(ensdir, exist_ok=True)

    if tododict['run_ensemble']: 
        ode = Lorenz96ODE(config_ode)
        sde = Lorenz96SDE(ode, config_sde)
        tu = sde.ode.dt_save

        ens = Ensemble(sde)

        # Decide which observables to track 
        obs_names = ['t','x0','E0','E','Emax']
        obs_dict = dict({name: [] for name in obs_names})

        def obs_fun(t,x):
            obs = dict({name: sde.ode.observable(t,x,name) for name in obs_names})
            return obs

        def append_obs_dict(od, odnew):
            for name in list(od.keys()):
                od[name].append(odnew[name])
            return


        # -------- 0 ----------
        #                     | --------- 1 ---------
        #                     | --------- 2 ---------
        init_time_phys = -4.0
        fin_time_phys = 40.0
        split_times_phys = [5.0, 20.0]
        rng = default_rng(87654)

        init_time = int(init_time_phys/tu)
        fin_time = int(fin_time_phys/tu)
        split_times = [int(stp/tu) for stp in split_times_phys]

        # 0
        icandf = sde.generate_default_icandf(init_time, fin_time)
        print(f'For mem 0, {icandf = }')
        saveinfo = dict(filename = join(ensdir, 'mem0.npz'))
        obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo)
        append_obs_dict(obs_dict, obs_val_new)

        # 1 
        parent = 0
        md_parent = ens.traj_metadata[parent]
        icandf = copylib.deepcopy(md_parent['icandf'])
        icandf['frc'].frc_list[1].reseed_times.append(split_times[0])
        icandf['frc'].frc_list[1].seeds.append(rng.integers(low=sde.seed_min,high=sde.seed_max))
        saveinfo = dict(filename = join(ensdir, 'mem1.npz'))
        obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
        append_obs_dict(obs_dict, obs_val_new)

        # 2 
        parent = 1
        md_parent = ens.traj_metadata[parent]
        icandf = copylib.deepcopy(md_parent['icandf'])
        icandf['frc'].frc_list[1].reseed_times.append(split_times[1])
        icandf['frc'].frc_list[1].seeds.append(rng.integers(low=sde.seed_min,high=sde.seed_max))
        saveinfo = dict(filename = join(ensdir, 'mem2.npz'))
        obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
        append_obs_dict(obs_dict, obs_val_new)

        # Save the observables; this will be the 'save_state' for future managers
        pickle.dump(obs_dict,open(join(ensdir,'obs_dict.pickle'),'wb'))
        pickle.dump(ens, open(join(ensdir,'ens.pickle'),'wb'))
        

    
    ens = pickle.load(open(join(ensdir, 'ens.pickle'), 'rb'))
    obs_dict = pickle.load(open(join(ensdir, 'obs_dict.pickle'),'rb'))
    sde = ens.dynsys
    tu = sde.ode.dt_save

    if tododict['check_overlap']:
        # Do all three trajectories match? 
        t0,x0 = sde.load_trajectory(ens.traj_metadata[0])
        t1,x1 = sde.load_trajectory(ens.traj_metadata[1])
        t2,x2 = sde.load_trajectory(ens.traj_metadata[2])

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
    sysarg2test = dict({'0': test_Lorenz96_ode, '1': test_Lorenz96_sde, })
    for arg in sys.argv[1:]:
        sysarg2test[arg]()
