## Examples
Here we'll walk through how to generate a stable bulk water ML potential from scratch
in just an of hour or so, using a DFTB 'ground truth'. A few other minimal examples
are also in this directory

#### Contents 
***
(1) benzene.py, a gas phase benzene molecule

(2) water.py, bulk water

(3) water_methane.py, training a 'solute-solvent' GAP for methane in water

(4) water_methane_md.py, running molecular dynamics with the water_methane model


> **_NOTE:_**  These examples assume a working DFTB+ installation

### Water
***

First import `gaptrian` and set the number of available processing cores for
training. For such a fast ground truth method four cores is sufficent

```python
import gaptrain as gt
gt.GTConfig.n_cores = 4
```

define the system as a cubic box with a side length of 10 Å, then populate it 
with 20 water molecules

```python
h2o = gt.System(box_size=[10, 10, 10])
h2o.add_solvent('h2o', n=20)
```

and train using intra+inter molecular decomposition using active learning with 
all the defaults, which comprise a 1 ps maximum active learning time, a 1 kcal 
mol-1 per molecule energy threshold for adding a configuration among others.

```python
data, gap = gt.active.train_ii(h2o, method_name='dftb')
```

Once the training is complete a GAP object is returned along with the configrations
(data) that is was trained on. To run a dynamics with ASE using the potential 

```python
traj = gt.md.run_gapmd(configuration=h2o.random(),
                       gap=gap,
                       temp=300,     # Kelvin
                       dt=0.5,       # femtoseconds
                       interval=5,   # frames
                       ps=1,         # picoseconds
                       n_cores=4)

traj.save(filename='traj.xyz')
```

where the configuration is generated from the system by placing molecules
at random positions and orientations with a minimum intermolecular distance of
 1.7 Å.


![water](common/dftb_water.gif)


