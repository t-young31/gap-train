"""
gap-train makes use of the following environment variables

1. GPAW_SETUP_PATH
2. DFTB_PATH or. DFTB_PREFIX/DFTB_COMMAND
3. QUIP_CONT

if they're not set some defaults are given but it's probably useful to set them
for development with e.g.

export QUIP_CONT = $HOME/QUIP.sif

in a bash shell
"""
import os


# ----------------------- GPAW -------------------------------
if 'GPAW_SETUP_PATH' not in os.environ:
    os.environ['GPAW_SETUP_PATH'] = '/u/fd/ball4935/opt/anaconda3/envs/gpaw/share/gpaw-setups-0.9.20000'

# WARNING: Calling GPAW asynchronously on a set of data will NOT respect
# os.environ['OMP_NUM_THREADS'] = '1' so this environment variable must be
# set before calling any gaptrain python script

# ----------------------- DFTB+ -------------------------------
if 'DFTB_PATH' in os.environ:
    dftb_path = os.environ['DFTB_PATH']

else:
    dftb_path = '/u/fd/ball4935/opt/dftbplus-20.1.x86_64-linux'
    # dftb_path = '/u/fd/ball4935/.local/dftbplus-20.1.x86_64-linux'

if 'DFTB_PREFIX' not in os.environ:
    os.environ['DFTB_PREFIX'] = f'{dftb_path}/recipes/slakos/download/3ob-3-1'

if 'DFTB_COMMAND' not in os.environ:
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
    if 'QUIP_CONT' not in os.environ:
        os.environ['QUIP_CONT'] = '/u/fd/ball4935/opt/QUIP.sif'

    gap_fit_command = ['singularity', 'exec', os.environ['QUIP_CONT'],
                       'teach_sparse']

    quippy_gap_command = ['singularity', 'exec',  os.environ['QUIP_CONT'],
                          '/usr/local/bin/python']

    # Default parameters for a GAP potential
    gap_default_params = {'sigma_E': 10**(-3.5),        # eV
                          'sigma_F': 10**(-1.0)}        # eV Å-1

    # Default two-body parameters
    gap_default_2b_params = {'cutoff': 5.5,             # Å
                             'n_sparse': 30,
                             'delta': 1.0               # eV
                             }

    # Default SOAP parameters
    gap_default_soap_params = {'cutoff': 3.0,           # Å
                               'n_sparse': 100,
                               'order': 6,
                               'sigma_at': 0.5,         # Å
                               'delta': 0.1             # eV
                               }
 