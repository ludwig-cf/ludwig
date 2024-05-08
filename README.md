### Ludwig

A lattice Boltzmann code for complex fluids

[![Build Status](https://travis-ci.com/ludwig-cf/ludwig.svg?branch=develop)](https://travis-ci.com/ludwig-cf/ludwig)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1998/badge)](https://bestpractices.coreinfrastructure.org/projects/1998)


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
or "TargetDP") which currently supports OpenMP, CUDA (NVIDIA GPUs) or
HIP (AMD GPUs) from a single source.

#### Installation

Copy a config file from the config directory to
the top level directory and make any changes required. E.g.,

```
$ cp config/unix-gcc-default.mk config.mk
$ make serial
$ make
$ make test
```
Note that the tests expect standard C assertions to be active; for
production runs, one should add the standard preprocessor option
`-DNDEBUG` to the compiler options in the `config.mk` file.

If a parallel build is wanted omit the serial step, for example,
```
$ cp config/unix-mpicc-default.mk config.mk
$ make
$ make test
```


Full details of the build process, and tutorials on how to
use the code are available at
<a href = "https://ludwg.epcc.ed.ac.uk/">https://ludwig.epcc.ed.ac.uk/</a>.

#### Background

Background documentation on the LB model and various free energy choices
is available in the `docs` directory.
```
$ cd docs
$ make
```
will produce a pdf version of the LaTeX source.

#### Contributing

If you would like to contribute, please consider a pull request.
See `CONTRIBUTING.md` for further details of testing and
development.


#### Credits

[![DOI](https://zenodo.org/badge/137508275.svg)](https://zenodo.org/badge/latestdoi/137508275)

Recent release versions have a Zenodo-provided DOI. Please consider using the
appropriate DOI as a reference if you use Ludwig in your publications.

From Version 0.19.0 we have included a copy of `cJSON` which is released
under an MIT license by Gave Gamble at https://github.com/DaveGamble/cJSON.

#### Help


For bug reports, problems, and other issues, please open a new issue.



## For MSc projects
Our goal is to inspect the code in-depth, locate the root cause of poor performance of some part of the code, and implement necessary fixes, which may require a complete reimplementation of the GPU version of the code. Ultimately, our objective is to create a well-optimized GPU version of the 'Lees Edwards BC' method, which will enable us to achieve significant performance enhancements and good scalability.

Student: Andong Xiao 

Supervisor: Kevin Stanford

Relevant documents including meeting blog, questions and some assessments materials can be seen from:  [Wiki](https://git.ecdf.ed.ac.uk/s2329216/msc_projects_document.git), which is maintained in another repository.



