color='\033[0;32m'
no_color='\033[0m'

function print {
    echo -e "$color$1$no_color"
    }

print "Installing gap-train from scratch"
print "Checking for conda install..."

if ! command -v conda &> /dev/null; then
    print "                          ...not found"
    print "Installing conda..."
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
    rm miniconda.sh
    eval "$("$HOME"/miniconda/bin/conda shell.bash hook)"
    conda init bash
    print "                ...done\n\n"
    print "Installed miniconda to $HOME/miniconda"
fi
print "conda install             ...found"

# -----------------------------------------------------------------------------

print "Creating new Python 3 environment for gap-train module (named 'gap')..."
conda create -n gap --yes
source activate gap
print "                                                                    ...done\n\n"
# -----------------------------------------------------------------------------

if [ -d "gap-train" ]; then
    print "gap-train directory exists - skipping"
else
    print "Cloning the gap-train repository and installing..."

    if command -v git &> /dev/null; then
      print "Found git install"
      git clone https://github.com/t-young31/gap-train.git

    else
      print "No git found, using wget instead"
      wget https://github.com/t-young31/gap-train/archive/refs/heads/master.zip

      if command -v unzip &> /dev/null; then
        print "Could not clone the repository. Must have either git or unzip installed."
        exit
      fi

      unzip master.zip
      mv gap-train-master/ gap-train/
    fi
fi

cd gap-train || { print -e "gap-train directory must exist"; exit 1; }
conda install --file requirements.txt --channel conda-forge --yes
python setup.py install
# -----------------------------------------------------------------------------

print "Installing QUIP..."
pip install quippy-ase
print "               ...done\n\n"
# -----------------------------------------------------------------------------

print "Installing electronic structure packages.\n
Note: ORCA cannot be installed automatically as the EULA must be accepted individually.
Go to https://orcaforum.kofo.mpg.de/index.php, sign in and go to 'downloads' to download and install."

read -p "Install gpaw? ([y]/n)" -r install_gpaw

if [ "$install_gpaw" == "y" ] || [ "$install_gpaw" == "" ]; then
    print "Installing gpaw..."
    conda install -c conda-forge gpaw --yes
    print "               ...done\n\n"
fi
# -----------------------------------------------------------------------------

read -p "Install xtb? ([y]/n)" -r install_xtb

if [ "$install_xtb" == "y" ] || [ "$install_xtb" == "" ]; then
    print "Installing xtb..."
    conda install -c conda-forge xtb --yes
    print "              ...done\n\n"
fi
# -----------------------------------------------------------------------------

read -p "Install DFTB+? ([y]/n)" -r install_dftb

if [ "$install_dftb" == "y" ] || [ "$install_dftb" == "" ]; then
    print "Installing DFTB+..."
    conda install -c conda-forge dftbplus --yes
    print "                ...done\n\n"


    if [ -d "$HOME/.local/dftb/slakos" ]; then
      print "Parameter files found - skipping"
    else
      print "Installing DFTB+ parameter files (3ob only)"
      print "Downloading..."
      wget -N https://dftb.org/fileadmin/DFTB/public/slako/3ob/3ob-3-1.tar.xz
      wget -N https://dftb.org/fileadmin/DFTB/public/slako/3ob/wfc.3ob-3-1.hsd
      print "Downloading...done\n\n"

      param_dir="$HOME/.local/dftb/slakos"
      print "Copying parameter files to $param_dir"
      tar -xf 3ob-3-1.tar.xz
      mkdir -p "$param_dir"
      mv 3ob-3-1 "$param_dir"
      rm 3ob-3-1.tar.xz

      mv wfc.3ob-3-1.hsd "$param_dir/3ob-3-1/"
    fi

    # Set $DFTB_PREFIX if it is not already present
    if ! grep -Fxq "export DFTB_PREFIX" "$HOME/.bashrc"; then
      print "Modifying ~/.bashrc to include DFTB parameter path"
      echo "export DFTB_PREFIX=$HOME/.local/dftb/slakos/3ob-3-1/" >> "$HOME/.bashrc"
    fi

    print "Installing DFTB+ parameter files          ...done\n\n"
fi
# -----------------------------------------------------------------------------

print "Done!   Logout and log back in for the changes to take effect\n
Note: The gap environment will need to be activated each time a terminal is opened with:\n
       conda activate gap\n\n"
