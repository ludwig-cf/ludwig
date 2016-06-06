/*****************************************************************************
 *
 *  psi_rt.c
 *
 *  Run time initialisation of electrokinetics stuff.
 *
 *  At the moment the number of species is set to 2 automatically
 *  if the electrokinetics is switched on.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Oliver Henrich  (ohenrich@epcc.ed.ac.uk)
 *
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "psi_rt.h"
#include "psi_init.h"
#include "io_harness.h"

/*****************************************************************************
 *
 *  psi_rt_param_init
 *
 *****************************************************************************/

int psi_rt_init_param(psi_t * obj) {

  int n;
  int nk;

  int valency[2] = {+1, -1};  /* Valencies (should be +/-!)*/
  double diffusivity[2] = {0.01, 0.01};

  double eunit = 1.0;         /* Unit charge */
  double temperature, beta;   /* Temperature (set by fluctuations) */
  double epsilon = 0.0;       /* Reference permittivity */
  double lbjerrum;            /* Bjerrum length; derived, not input */
  double tolerance;           /* Numerical tolerance for SOR and Krylov subspace solver */
  int    niteration;          /* Max. number of iterations */ 

  int io_grid[3] = {1,1,1};
  int io_format_in = IO_FORMAT_DEFAULT;
  int io_format_out = IO_FORMAT_DEFAULT;
  char value[BUFSIZ] = "BINARY";

  int multisteps;             /* Number of substeps in NPE */
  int skipsteps;              /* Poisson equation solved every skipstep timesteps */ 
  double diffacc;             /* Relative accuracy of diffusion in NPE */

  psi_nk(obj, &nk);
  assert(nk == 2); /* nk must be two for the time being */

  n = RUN_get_int_parameter("electrokinetics_z0", valency);
  n = RUN_get_int_parameter("electrokinetics_z1", valency + 1);
  n = RUN_get_double_parameter("electrokinetics_d0", diffusivity);
  n = RUN_get_double_parameter("electrokinetics_d1", diffusivity + 1);

  for (n = 0; n < nk; n++) {
    psi_valency_set(obj, n, valency[n]);
    psi_diffusivity_set(obj, n, diffusivity[n]);
  }

  n = RUN_get_double_parameter("electrokinetics_eunit", &eunit);
  n = RUN_get_double_parameter("electrokinetics_epsilon", &epsilon);

  psi_unit_charge_set(obj, eunit);
  psi_epsilon_set(obj, epsilon);
  psi_epsilon2_set(obj, epsilon); /* Default is no dielectric contrast */

  n = RUN_get_double_parameter("temperature", &temperature);

  if (n == 0 || temperature <= 0.0) {
    fatal("Please set a temperature to use electrokinetics\n");
  }

  beta = 1.0/temperature;

  psi_beta_set(obj, beta);
  psi_bjerrum_length(obj, &lbjerrum);

  info("Electrokinetic species:    %2d\n", nk);
  info("Boltzmann factor:          %14.7e (T = %14.7e)\n", beta, temperature);
  info("Unit charge:               %14.7e\n", eunit);
  info("Permittivity:              %14.7e\n", epsilon);
  info("Bjerrum length:            %14.7e\n", lbjerrum);

  for (n = 0; n < nk; n++) {
    info("Valency species %d:         %2d\n", n, valency[n]);
    info("Diffusivity species %d:     %14.7e\n", n, diffusivity[n]);
  }

  /* Multisteps and diffusive accuracy in NPE */

  n = RUN_get_int_parameter("electrokinetics_multisteps", &multisteps);
  if (n == 1) psi_multisteps_set(obj, multisteps);
  n = RUN_get_int_parameter("electrokinetics_skipsteps", &skipsteps);
  if (n == 1) psi_skipsteps_set(obj, skipsteps);
  n = RUN_get_double_parameter("electrokinetics_diffacc", &diffacc);
  if (n == 1) psi_diffacc_set(obj, diffacc);

  psi_multisteps(obj, &multisteps);
  info("Number of multisteps:       %d\n", multisteps);
  info("Number of skipsteps:        %d\n", psi_skipsteps(obj));
  psi_diffacc(obj, &diffacc);
  info("Diffusive accuracy in NPE: %14.7e\n", diffacc);

  /* Tolerances and Iterations */

  n = RUN_get_double_parameter("electrokinetics_rel_tol", &tolerance);
  if (n == 1) psi_reltol_set(obj, tolerance);
  n = RUN_get_double_parameter("electrokinetics_abs_tol", &tolerance);
  if (n == 1) psi_abstol_set(obj, tolerance);
  n = RUN_get_int_parameter("electrokinetics_maxits", &niteration);
  if (n == 1) psi_maxits_set(obj, niteration);

  psi_reltol(obj, &tolerance);
  info("Relative tolerance:  %20.7e\n", tolerance);
  psi_abstol(obj, &tolerance);
  info("Absolute tolerance:  %20.7e\n", tolerance);
  psi_maxits(obj, &niteration);
  info("Max. no. of iterations:  %16d\n", niteration);

  /* I/O */

  n = RUN_get_int_parameter_vector("default_io_grid", io_grid);
  n = RUN_get_string_parameter("psi_format", value, BUFSIZ);

  if (strcmp(value, "ASCII") == 0) {
    io_format_in = IO_FORMAT_ASCII;
    io_format_out = IO_FORMAT_ASCII;
  }

  info("I/O decomposition:          %d %d %d\n", io_grid[0], io_grid[1],
       io_grid[2]);
  info("I/O format:                 %s\n", value);

  psi_init_io_info(obj, io_grid, io_format_in, io_format_out);

  return 0;
}

/*****************************************************************************
 *
 *  psi_rt_init_rho
 *
 *  Initial configurations of the charge density.
 *
 *  - Gouy Chapman test (flow between two parallel plates)
 *  - "Liquid junction" test
 *  - uniform charge densities
 *
 *****************************************************************************/

int psi_rt_init_rho(psi_t * obj, map_t * map) {

  int n;
  char value[BUFSIZ];
  char filestub[FILENAME_MAX];

  double rho_el;              /* Charge density */
  double delta_el;            /* Relative difference in charge densities */
  double sigma;               /* Surface charge density */
  double ld;                  /* Debye length */
  double ld2;                 /* Second Debye length for dielectric contrast */
  double eps1, eps2;          /* Dielectric permittivities */

  io_info_t * iohandler;

  /* Initial charge densities */

  info("\n");
  info("Initial charge densities\n");
  info("------------------------\n");

  n = RUN_get_string_parameter("electrokinetics_init", value, BUFSIZ);

  if (strcmp(value, "gouy_chapman") == 0) {
    info("Initial conditions:         %s\n", "Gouy Chapman");

    n = RUN_get_double_parameter("electrokinetics_init_rho_el", &rho_el);
    if (n == 0) fatal("... please set electrokinetics_init_rho_el\n");
    info("Initial condition rho_el:  %14.7e\n", rho_el);
    psi_debye_length(obj, rho_el, &ld);
    info("Debye length:              %14.7e\n", ld);

    n = RUN_get_double_parameter("electrokinetics_init_sigma", &sigma);
    if (n == 0) fatal("... please set electrokinetics_init_sigma\n");
    info("Initial condition sigma:   %14.7e\n", sigma);

    psi_init_gouy_chapman_set(obj, map, rho_el, sigma);
  }

  if (strcmp(value, "liquid_junction") == 0) {
    info("Initial conditions:         %s\n", "Liquid junction");

    n = RUN_get_double_parameter("electrokinetics_init_rho_el", &rho_el);
    if (n == 0) fatal("... please set electrokinetics_init_rho_el\n");
    info("Initial condition rho_el: %14.7e\n", rho_el);
    psi_debye_length(obj, rho_el, &ld);
    info("Debye length:             %14.7e\n", ld);

    n = RUN_get_double_parameter("electrokinetics_init_delta_el", &delta_el);
    if (n == 0) fatal("... please set electrokinetics_init_delta_el\n");
    info("Initial condition delta_el: %14.7e\n", delta_el);

    psi_init_liquid_junction_set(obj, rho_el, delta_el);
  }

  if (strcmp(value, "uniform") == 0) {
    info("Initial conditions:         %s\n", "Uniform");

    n = RUN_get_double_parameter("electrokinetics_init_rho_el", &rho_el);
    if (n == 0) fatal("... please set electrokinetics_init_rho_el\n");
    info("Initial condition rho_el: %14.7e\n", rho_el);
    psi_debye_length(obj, rho_el, &ld);
    info("Debye length:             %14.7e\n", ld);

    /* Call permittivities and check for dielectric contrast */

    psi_epsilon(obj, &eps1);
    psi_epsilon2(obj, &eps2);

    if (eps1 != eps2) {
      psi_debye_length2(obj, rho_el, &ld2);
      info("Second Debye length:      %14.7e\n", ld2);
    }

    psi_init_uniform(obj, rho_el);
  }

  if (strcmp(value, "from_file") == 0) {
    sprintf(filestub, "%s", "psi-00000000");
    info("Initialisation requested from file %s.001-001\n", filestub);
    psi_io_info(obj, &iohandler);
    io_read_data(iohandler, filestub, obj);
  }

  return 0;
}
