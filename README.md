# gap-train

## Introduction

This repository contains the _gaptrain_ module for generating datasets, training
GAPs and running simulations using Atomic Simulation Environment (ASE) to 
drive dynamics and electronic structure packages for reference energies and forces. 

## Installation

To install _gaptrain_ in one line:
```
wget https://raw.githubusercontent.com/t-young31/gap-train/master/scripts/INSTALL.sh && bash INSTALL.sh
```
> **_NOTE:_**  ORCA cannot be installed with this script.

Otherwise, from source:

```bash
git clone https://github.com/t-young31/gap-train.git
cd gap-train
conda install --file requirements.txt -c conda-forge
python setup.py install
```
where the electronic structure packages ([GPAW](https://wiki.fysik.dtu.dk/gpaw/install.html),
[DFTB+](https://dftbplus.org), [XTB](https://github.com/grimme-lab/xtb), [ORCA](https://sites.google.com/site/orcainputlibrary/)) 
and [QUIP](https://github.com/libAtoms/QUIP) need to be installed manually.


## Usage

See _examples_ for a selection of examples using active learning. A minimal 
example to train a GAP on 10 random configurations of a water box:

```python
import gaptrain as gt

system = gt.System(box_size=[10, 10, 10])
system.add_solvent('h2o', n=20)

training_data = gt.Data()
for i in range(10):
    training_data += system.random()

training_data.parallel_dftb()
gap = gt.GAP(name='random', system=system)
gap.train(training_data)
```

> **_NOTE:_**  This will not be a stable water potential!

## Configuration

Different environments are handled with `gt.GTConfig` and/or environment variables.
DFTB+ requires an executable path and a parameter path, which can be set (in bash) with:

```bash
export DFTB_COMMAND=/path/to/dftb-install/bin/dftb+
export DFTB_PREFIX=/path/to/dftb-params/
```

GPAW similarly requires a parameter path with e.g.

```bash
export GPAW_SETUP_PATH=/path/to/gpaw-install/share/gpaw-setups-0.9.20000
```

To train GAPs requires a teach_sparse (or gap_fit) command to be available which
can be set in a Python script with

```python
gt.GTConfig.gap_fit_command = '/path/to/gap/train/executable'
```

To drive ASE dynamics using a GAP potential requires a Python with
quippy installed

```python
gt.GTConfig.quippy_gap_command = '/path/to/quippy/python'
```


## Testing

_gaptrain_ is unit tested with [pytest](https://docs.pytest.org/en/stable/), run
them with

```bash
py.test
```

in the top level gap-train directory. To run the DFTB+, GAP and GROMACS tests
set the $GT_DFTB, $GT_GAP, $GT_GMX environment variables to True as appropriate (if they're installed)

```bash
export GT_DFTB=True
export GT_ORCA=True
export GT_XTB=True
export GT_GAP=True
export GT_GMX=True
```


## Citation

If _gaptrain_ is used in a publication please consider citing the [paper](https://doi.org/10.1039/d1sc01825f):

```
@article{GAPTraining2021,
  doi = {10.1039/d1sc01825f},
  url = {https://doi.org/10.1039/d1sc01825f},
  year = {2021},
  publisher = {Royal Society of Chemistry ({RSC})},
  author = {Tom Young and Tristan Johnston-Wood and Volker L. Deringer and Fernanda Duarte},
  title = {A transferable active-learning strategy for reactive molecular force fields},
  journal = {Chemical Science}
}
```
