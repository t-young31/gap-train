"""
gap-train makes use of the following environment variables

1. GPAW_SETUP_PATH
2. DFTB_PATH or. DFTB_PREFIX/DFTB_COMMAND
3. QUIP_CONT

if they're not set some defaults are given but it's probably useful to set them
for development with e.g.

export GPAW_SETUP_PATH = $HOME/gpaw/share/

in a bash shell.
"""
import os
import shutil

# WARNING: Calling GPAW asynchronously on a set of data will NOT respect
# os.environ['OMP_NUM_THREADS'] = '1' so this environment variable must be
# set before calling any gaptrain python script

# ----------------------- DFTB+ -------------------------------
if 'DFTB_PATH' in os.environ:
    dftb_path = os.environ['DFTB_PATH']

    if 'DFTB_PREFIX' not in os.environ:
        os.environ['DFTB_PREFIX'] = f'{dftb_path}/recipes/slakos/download/3ob-3-1'

    if 'DFTB_COMMAND' not in os.environ:
        os.environ['DFTB_COMMAND'] = f'{dftb_path}/bin/dftb+'
elif shutil.which('dftb+') is None:
    print("WARNING: $DFTB_PATH not set, dftb+ not available")


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

    # Commands must be lists
    gap_fit_command = [shutil.which('gap_fit')]
    quip_command = [shutil.which('quip')]

    # Default parameters for a GAP potential
    gap_default_params = {'sigma_E': 10**(-3.5),        # eV
                          'sigma_F': 10**(-1.5)}        # eV Å-1

    # Default two-body parameters
    gap_default_2b_params = {'cutoff':   5.5,           # Å
                             'n_sparse': 30,
                             'delta':    1.0
                             }

    # Default SOAP parameters
    gap_default_soap_params = {'cutoff':   4.0,         # Å
                               'n_sparse': 1000,
                               'order':    6,
                               'sigma_at': 0.5,         # Å
                               'delta':    0.1
                               }

    # ----------------------- ORCA -------------------------------
    # Keywords to use for an ORCA calculation
    orca_keywords = None

    # ---------------------- Gaussian -----------------------------
    gaussian_keywords = None

    # ----------------------- CP2K -------------------------------
    cp2k_command = ['srun', '--cpu-bind=cores', 'cp2k.popt']
