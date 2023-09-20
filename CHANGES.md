
### Changes


version 0.21.0

- Ellipsoids. A feature which allows ellipsoidal particles to be
  used has been added by Sumesh Thampi.

- "colloid_type" has been split into a number of more general
  properties including: boundary condition (bc) either bbl or
  subgrid; shale either "disk", "sphere", or "ellipsoid";
  active (logical); magnetic (logical).
  The default is: bc = bbl, shape = sphere, active = no, magnetic = no.
  There are some conditions, e.g., "disk" expects a 2D D2Q9 model.
  This affects the way colloid details are specified in the input.
  Older colloid state files should still work.
  See https://ludwig.epcc.ed.ac.uk

- Issue 268: if using wetting information read from porous media files,
             the form of the key words in the input file has been
	     adjusted. See https://ludwig.epcc.ed.ac.uk/inputs/porous.html

version 0.20.1
- Issue 271: missing stub prevents compilation at some compiler optimisation
             levels.

version 0.20.0

- IMPORTANT: The input file can no longer be specified as a command
  line argument. It must be called "input" in the current directory.
- The electrokinetics sector has been updated and information is
  available at
  https://ludwig.epcc.ed.ac.uk/tutorials/electrokinetics/electrokinetics.html
- A D3Q27 model is available
- Added coverage via https://about.codecov.io/
- The free energy is now reported at t = 0 for the initial state.
- An extra "first touch" option has been added. For details, see
  https://ludwig.epcc.ed.ac.uk/inputs/parallel.html
- Various minor changes and code quality improvements.

version 0.19.1
- Fix bug in io_subfile to prevent failure at iogrid > 1.

version 0.19.0

- There has been a significant change to the way that i/o is undertaken.
  See https://ludwig.epcc.ed.ac.uk/outputs/fluid.html
  It is the intention to replace completely the 'old' I/O mechanism
  by release v0.21.0.
- The extract_colloids.c utility has been updated so that it takes
  only one command line argument for new I/O metadata.

- Input associated with the choice of force arising from the free energy
  sector has been made more general. Specifically, input keys
  `fd_force_divergence` and `fe_use_stress_relaxation` are replaced.
  - For scalar order parameters an extra version of the "phi_gradmu"
    approach is available" "phi_gradmu_correction".
  - See https://ludwig.epcc.ed.ac.uk/inputs/force.html for details.
- Order parameter statistics are now reported as "phi" for scalars
  [Px,Py,Pz] for vectors, and [Qxx, Qxy, Qxz, Qyy, Qyz] for liquid
  crystal. The computation of the total has been improved by using a
  compensated sum, which is more robust to threads/MPI.
- Added compiler option information and git commit information at run
  time.
- The LaTeX tutorials document has been replaced by html tutorials
  at https://ludwig.epcc.ed.ac.uk/ to reflect new I/O and to add a
  number of new topics.
- LTGM.com analysis has been retired as the service has closed. The
  two outstanding recommendations are covered by CodeQL notes.
- Various minor updates.

version 0.18.0

- Added a lubrication correction offset to allow an option for keeping
  particles clear of plane walls.
  See https://ludwig.epcc.ed.ac.uk/inputs/colloid.html
- Added options for arranging 'first touch' on allocation of memory
  for LB data and field componenents.
  See https://ludwig.epcc.ed.ac.uk/inputs/parallel.html

- Various minor code improvements.

version 0.17.2

- Bug fix (issue #204) prevent crashes with s7_anchoring related to
  proximity of wall/colloid or colloid/colloid. Advice added to
  documentation on avoiding such close approaches.

version 0.17.1

- Bug fix (issue #197). The liquid crystal reduced field strength was reported
  incorrectly in the output (always zero). Thanks to Oliver H. for spotting.

version 0.17.0

- add liquid crystal anchoring "fd_gradient_calculation s7_anchoring"
  - this is a replcement for "3d_7pt_fluid" and does a slightly better
    job at the edges and corners by using a consistent surface normal.
    The anchoring properties are now specified in a slightly different
    way.
  - For walls, see https://ludwig.epcc.ed.ac.uk/inputs/walls.html
  - For colloids, see https://ludwig.epcc.ed.ac.uk/inputs/colloid.html

  - The existing fd_gradient_calculation 3d_7pt_solid is retained, and
    existing input keys for anchoring will be recognised.

- add option for rectilinear grid format vtk output "extract -l"
- add option for 2d random nematic "lc_q_initialisation random_xy"

- A functional AMD GPU version is now available using HIP.
  - See https://ludwig.epcc.ed.ac.uk/building/index.html

- Various minor improvements

version 0.16.1

- And get the version number right!

version 0.16.0

- Improved host halo swaps are available.
  The implementation of the reduced distribution halo has been replaced
  with one that will work in all circumstances for a single distribution.
  See https://ludwig.epcc.ed.ac.uk/inputs/index.html Parallelism for details.
  No user action is required if you are not interested.

- Reinstated the boundary (wall) - colloid soft sphere potential.
  See https://ludwig.epcc.ed.ac.uk/inputs/colloid.html
  Thanks to Rishish Mishra for spotting this problem.

- Various minor updates.

version 0.15.0

- Active stress implementation is updated to conform to the documented
  case; active emulsion stress is available.
- Add ability to rotate BPI and BPII liquid crystal initial conditions
  Thanks to Oliver H. for this. See Section 3 of web documentation.
- A diagnostic computation and output of the force breakdown on each
  colloid has been added. This is currently via a static switch in
  stats_colloid_force_split.c
- An option for a "lap timer" is now provided.
- Some simple open boundary conditions for fluid and binary composition
  are available. See the "Open boundaries" section at
  https://ludwig.epcc.ed.ac.uk/
- Some refactoring of the lattice Boltzmann basis information has been
  performed in order to be able to move to a more flexible approach.
  This should have no practical impact at present.

version 0.14.0

- Add a ternary free energy. Thanks to Shan Chen and Sergios Granados Leyva.
  - See https://ludwig.epcc.ed.ac.uk/inputs/fe.html
  - Added various initial ternary configurations.
  - Allowed uniform wetting from input via free energy parameters.
  - Added various porous media style initialisations from input;
    input handling has changed slightly.
  - Updated the util/capillary.c code to use the standard map structure,
    and standard output functionality.
- Add back an operational surfactant free energy. There is no
  dynamics available yet.
- Add unit tests for the the same.
- Add a description of how to add a free energy to free_energy.h
- Unified CPU/GPU short regression tests.
- Compilation of target HIP updated for AMD platform. See config/unix-hpcc.mk.

version 0.13.0

- Add report on compiler and start/end times.
- Add report on key/value pairs which appear in input but are not used
  at end of execution.
- Added compensated sums for binary order parameter sum statistic to
  improve robustness of result to round-off. Additional compensation
  in time evolution for Cahn-Hilliard update.
- Add pair_ss_cut_ij interaction: a cut-and-shoft soft sphere potential
  with pair-dependent parameters. Thanks to Qi Kai.
- Added subgrid offset parameter; this replaces ah in the computation
  of the drag force (typically aL >> ah).

version 0.12.0

- Allow user to specify a linear combination of slip and no-slip for
  plane walls. This was originally implemented by Katrin Wolff when
  at Edinburgh, and has been resurrected with the help of Ryan Keogh
  and Tyler Shendruk. See https://ludwig.epcc.ed.ac.uk/inputs/walls.html
- Various minor code quality improvements
- Extended target abstraction layer to include HIP (only tested via
  __HIP_PLATFORM_NVCC__ so far). Thanks to Nikola Vasilev for this.
- Various minor code quality improvements

version 0.11.0
- Add external chemical potential gradient in Cahn Hilliard for
  free energy symmetric. Thanks to Jurij Sablic (jurij.sablic@gmail.com).
- Add Arrhenius viscosity model for compositional order parameter
- Add the ability to run both subgrid and fully resolved particles at
  the same time. Thanks to Qi Kai (kai.qi@epfl.ch) for this.
- Various code quality updates

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
