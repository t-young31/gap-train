# gap-train

## Introduction

This repository contains the _gaptrain_ module for generating datasets, training
GAPs and running simulations.


## Installation

To install _gaptrain_ first satisfy the Python dependencies, install electronic
structure methods and QUIP, then install the module:

1. Install [autode](https://github.com/duartegroup/autodE)
2. Install remaining Pyhton dependencies
```
conda install ase coloredlogs dscribe
```
3. If required install:

    a. [GPAW](https://wiki.fysik.dtu.dk/gpaw/install.html)

    b. [DFTB+](https://dftbplus.org) and the appropriate [parameters](https://dftb.org/parameters/download)

    c. [GROMACS](http://www.gromacs.org)

    d. [XTB](https://github.com/grimme-lab/xtb)

    e. [ORCA](https://sites.google.com/site/orcainputlibrary/)

4. Install [QUIP](https://github.com/libAtoms/QUIP) with [GAP](http://www.libatoms.org/gap/gap_download.html).
the easiest way to install these is to use the Docker or Singularity containers

5. Install _gaptrain_
```
git clone https://github.com/t-young31/gap-train.git
cd gap-train
python setup.py install
```


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

## Configuration

Different environments handled with gt.GTConfig and/or environment variables.
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
