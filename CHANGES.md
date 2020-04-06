
### Changes

version 0.10.0
- Added an option to fix colloid position or velocity on a per-direction
  basis, e.g.
    colloid_isfixedrxyz  1_1_0
  allows movement in z-direction only. Any value is overridden by
  colloid_isfixedr. An analogous option colloid_isfixedvxyz is available.
- Added target thread model information to output
- Refactored d_ij and e_ijk from char to int8_t to avoid potential
  pitfalls with default unsigned char.

version 0.9.3
- Allow stress relaxation option in bare liquid crystal free energy

version 0.9.2
- Moved input section in porous media docs to online version only

version 0.9.1
- Disallow porous media files using "status_with_h" as erroneous.
  Use "status_with_c_h" instead.
- Bug fix: allow colloid wetting by default.

version 0.9.0
- The build process has changed to try to move all the configuration
  to the config.mk file. Please see updated examples in the ./config
  directory. The configs are either serial or parallel (not both).
  The build process should now be from the top level and is via
  "make && make test". Serial builds should do "make serial" first.

- You should be able to type "make" in any directory and the local
  default target will be built.

- Executables in utils are built via "make" to be consistent with
  other source directories

- Added input colloid_rebuild_freq (with default 1) to allow discrete
  rebuild to be done less often than every time step

- Regression tests have been re-organised into different directories
  and are run on a per-directory basis (see tests/Makefile)

- The default test is regression/d3q19-short 

- A link to new build and test instructions has been made available
  from the README

- UPDATE TUTORIALS

- Added travis .travis.yml and relevant config file

- Fixed gcc -Wformat-overflow and a number of other warnings

version 0.8.16
- add option for uniform composition via "phi_initialisation uniform"
- fix composition replacement bug (was dependent on charge psi)

version 0.8.15
- fix "weight = 0" problem in replacement of fluid for binary order
  parameter and add test (issue 30)

version 0.8.14
- add html placeholder

version 0.8.13
- add option for density output via "rho" commands in input

version 0.8.12
- allow force divergence method to see porous media

version 0.8.11
- updated util/length_from_sk.c to take double input to be consistent
  with extract output

version 0.8.10
- Add vtk format output for composition, velocity, in extract;
  automatically detect input format
- Replace dubious assertion in stats_symmetric_length()

version 0.8.09
- Avoid zero-sized allocations in wall.c

version 0.8.08
- Repaired util/colloid_init.c
- Separate vtk scalar order / director / biaxial order files

version 0.8.07
- Updated the extract program to include vtk headers
- Added some Bond number test examples regression/d3q19/serial-bond-c01.inp

Version 0.8.05
- Added the option to specify fixed liquid crystal anchoring orientation
  from the input.

Version 0.8.04
- Fixed bug in device halo swap.

Version 0.8.0

- The constraint that the number of MPI tasks divide exactly the system
  size in each direction has been relaxed. Logically rectangular, but
  uneven decompositions are computed automatically if required.

- Output format in parallel. Files for a given I/O group now appear with
  data in a format which is independent of parallel decomposition. This
  means an 'extract' step is no longer required. It also means files
  generated with previous versions will no longer work as input.

- Different collision relaxation time schemes are now available by using
  the 'lb_relaxation_scheme' input either ('bgk', 'trt', or 'm10'). The
  default is unchanged ('m10').

- An option for a 'split' treatment of symmetric and antisymmetric stress
  arising from the thermodynamic sector has been introduced. This is
  via the input key 'fe_use_stress_relaxation yes'. This introduces the
  symmetric part of the stress as a relaxation in the collision, while
  the anti-symmetric part is via the force (via divergence). The default
  is still to treat the whole stress via the divergence.

- A 'second active stress' is now available for active liquid crystals.
