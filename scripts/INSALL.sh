echo "Installing gap-train from scratch"
echo "Checking for conda install..."

if ! command -v conda &> /dev/null
then
    echo "                          ...not found"
    echo "Installing conda..."
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
    rm miniconda.sh
    echo "                ...done"
    echo "Installed miniconda to $HOME/miniconda"
fi
echo "                          ...found"
# -----------------------------------------------------------------------------

echo "Creating new Python 3 environment for gap-train module (named 'gap')..."
conda conda create -n gap --yes
conda conda activate gap
echo "                                                                    ...done"
# -----------------------------------------------------------------------------

echo "Cloning the gap-train repository and installing..."

if command -v git &> /dev/null
then
  echo "Found git install"
  git clone https://github.com/t-young31/gap-train.git

else
  echo "No git found, using wget instead"
  wget https://github.com/t-young31/gap-train/archive/refs/heads/master.zip

  if command -v unzip &> /dev/null
  then
    echo "Could not clone the repository. Must have either git or unzip installed."
    exit
  fi

  unzip master.zip
  mv gap-train-master/ gap-train/
fi

cd gap-train || { echo "gap-train directory must exist"; exit 1; }
conda conda install --file requirements.txt --channel conda-forge --yes
python setup.py install
# -----------------------------------------------------------------------------

echo "Installing QUIP..."
pip install quippy-ase
echo "               ...done"
# -----------------------------------------------------------------------------

echo "Install gpaw? ([y]/n)"
read -r install_gpaw

if "${install_gpaw:=y}" == "y"
  then
    echo "Installing gpaw..."
    conda install -c conda-forge gpaw --yes
    echo "               ...done"
fi
# -----------------------------------------------------------------------------

echo ""


printf "Note: The gap environment will need to be activated each time a terminal
        is opened with\n
        conda activate gap\n"
