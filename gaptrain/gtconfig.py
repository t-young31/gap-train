import os


# ----------------------- GPAW -------------------------------
os.environ['GPAW_SETUP_PATH'] = '/u/fd/ball4935/opt/anaconda3/envs/gpaw/share/gpaw-setups-0.9.20000'

# WARNING: Calling GPAW asynchronously on a set of data will NOT respect
# os.environ['OMP_NUM_THREADS'] = '1' so this environment variable must be
# set before calling any gaptrain python script

# ----------------------- DFTB+ -------------------------------
dftb_path = '/u/fd/ball4935/opt/dftbplus-20.1.x86_64-linux'

os.environ['DFTB_PREFIX'] = f'{dftb_path}/recipes/slakos/download/3ob-3-1'
os.environ['DFTB_COMMAND'] = f'{dftb_path}/bin/dftb+'


class GTConfig:

    n_cores = 1

    # ----------------------- DFTB+ -------------------------------
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

    # Default two-body parameters
    gap_default_2b_params = {'cutoff': 5.5,             # Å
                             'n_sparse': 30,
                             'delta': 2.0}

    # Default SOAP parameters
    gap_default_soap_params = {'cutoff': 3.0,           # Å
                               'n_sparse': 100,
                               'order': 6,
                               'delta': 0.01}
