
### Ludwig

A lattice Boltzmann code for complex fluids

Ludwig is a parallel code for the simulation of complex fluids, which
include mixtures, colloidal suspensions, gels, and liquid crystals.
It takes its name from Ludwig Boltzmann, as it uses a lattice Boltzmann
method as a basis for numerical solution of the Navier Stokes equations
for hydrodynamics. It typically combines hydrodynamics with a coarse-grained
order parameter (or order parameters) to represent the "complex" part
in a free energy picture.

The code is written in standard ANSI C, and uses the Message Passing
Interface for distributed memory parallelism. Threaded parallelism is
also available via a lightweight abstraction layer ("Target Data Parallel"
or "TargetDP") which currently supports either OpenMP or CUDA (NVIDIA GPUs)
from a single source.

#### For the impatient

Copy a config file from the config directory to
the top level directory and make any changes required. E.g.,
```
$ cp config/lunix-gcc-default.mk config.mk
$ cd tests
$ make compile-mpi-d3q19
```
This will produce an execuatable `./src/Ludwig.exe`

#### Help

For bug reports, problems, and other issues, please open a new issue.

#### Contributing


If you would like to contribute, please consider a pull request.


