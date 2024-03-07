def meta_analyze_dns():
    scratch_dir = "/net/bstor002.ib/pog/001/ju26596/TEAMS/examples/frierson_gcm/"
    date_str = "2024-03-05"
    sub_date_str = "0/DNS"

    # -------- Specify which variables to fix and which to vary ---------
    params = dict()
    params['L_sppt'] = dict({
        'fun': lambda config: config['L_sppt'],
        'scale': 1000, # for display purposes
        'symbol': r'$L_{\mathrm{SPPT}}$ [km]',
        })
    params['tau_sppt'] = dict({
        'fun': lambda config: config['tau_sppt'],
        'scale': 3600, 
        'symbol': r'$\tau_{\mathrmm{SPPT}}$ [h]',
        })
    params['std_sppt'] = dict({
        'fun': lambda config: config['std_sppt'],
        'scale': 1.0,
        'symbol': r'$\sigma_{\mathrm{SPPT}}$',
        })

    params2fix = {'L_sppt','tau_sppt'}
    params2vary = {'std_sppt'}
        'L_sppt': lambda o'tau_sppt'} # For each unique value of these, make a new plot
    param_vbl = 'std_sppt'

    # -------------------------------------------------------------------

