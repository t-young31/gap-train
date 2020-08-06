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
