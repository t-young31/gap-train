# Scaling Behaviour

In the paper[1] a selection of GAPs were trained but how the number of required
calculations scales with complexity was not explicitly outlined. This scaling 
is presented below for a selection of systems.

### Alkanes

As a semi-ideal case, linear alkanes are first considered. The number of 
total evaluations (*n*<sub>eval</sub>), number of training configurations 
(*n*<sub>train</sub>) and the resulting τ<sub>acc</sub>(E<sub>l</sub> = 1 kcal 
mol<sup>-1</sup>, E<sub>T</sub>=10E<sub>l</sub>, T=300 K) are plotted as a 
function of the carbon chain length (*alkane_scaling.py*).

<img src="alkane/alkane_scaling.png" width="500">

Despite the symmetry, the required number of calculations for τ<sub>acc</sub> > 1 ps
consistently is roughly linear up to pentane at which point the maximum number of AL 
iterations (100) precludes further increases. Note that while the number of training 
configurations hits the maximum at 1000 they are CUR selected within the gap_fit 
code down to 500 (default `gt.GTConfig.gap_default_soap_params['n_sparse'] = 500` 
at the time of writing).


### Small Organic Molecules

For a more diverse set of molecules (organic solvents)

<img src="solvents/solvents.png" width="600">

plotting the number of configurations required to generate a GAP with τ<sub>acc</sub> > 1 ps 
(parameters as above)

<img src="solvents/solvent_scaling.png" width="900">

suggests no strong correlation, only that - as expected - larger molecules
require more training evaluations. Note that the number of training configurations shown above 
is not representative of the number required to generate a stable potential for that molecule, as
the RMSE is << 1 kcal mol<sub>-1</sub> over configurations accessed in a 300 K simulation. Furthermore,
the AL is performed at 500 K, exceeding the temperature of the desired simulation and increasing 
the effective space to learn. For example, reducing the AL temperature to 300 K for glycine (*30*) 
halves *n*<sub>eval</sub> and *n*<sub>train</sub> while maintaining a τ<sub>acc</sub> > 1 ps (parameters as above).


### References

[1] T. Young et. al, *Chem. Sci.*, 2021.


