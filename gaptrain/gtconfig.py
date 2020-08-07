class GTConfig:

    n_cores = 1

    dftb_path = '/u/fd/ball4935/opt/dftbplus-20.1.x86_64-linux'
    dftb_exe = f'{dftb_path}/bin/dftb+'
    dftb_data = f'{dftb_path}/recipes/slakos/download/3ob-3-1'

    dftb_filenames = ('detailed.out',
                      'dftb_in.hsd',
                      'dftb.out',
                      'dftb_pin.hsd',
                      'geo_end.gen',
                      'band.out')

    # ----------------------- GAP -------------------------------
    gap_fit_command = ['singularity', 'exec',
                       '/u/fd/ball4935/opt/QUIP.sif', 'teach_sparse']

    # Default parameters for a GAP potential
    gap_default_params = {'sigma_E': 10**(-2.5),        # eV
                          'sigma_F': 10**(-1)}          # eV Å-1

    gap_default_2b_params = {'cutoff': 5.5,             # Å
                             'n_sparse': 30,
                             'delta': 2.0}

    gap_default_soap_params = {'cutoff': 3.0,           # Å
                               'n_sparse': 100,
                               'order': 6,
                               'delta': 0.01}
