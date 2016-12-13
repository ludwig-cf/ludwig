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
 *  (c) 2012-2016 The University of Edinburgh
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

int psi_rt_init_param(pe_t * pe, rt_t * rt, psi_t * obj) {

  int n;
  int nk;
  int nfreq;

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

  assert(pe);
  assert(rt);

  psi_nk(obj, &nk);
  assert(nk == 2); /* nk must be two for the time being */

  n = rt_int_parameter(rt, "electrokinetics_z0", valency);
  n = rt_int_parameter(rt, "electrokinetics_z1", valency + 1);
  n = rt_double_parameter(rt, "electrokinetics_d0", diffusivity);
  n = rt_double_parameter(rt, "electrokinetics_d1", diffusivity + 1);

  for (n = 0; n < nk; n++) {
    psi_valency_set(obj, n, valency[n]);
    psi_diffusivity_set(obj, n, diffusivity[n]);
  }

  n = rt_double_parameter(rt, "electrokinetics_eunit", &eunit);
  n = rt_double_parameter(rt, "electrokinetics_epsilon", &epsilon);

  psi_unit_charge_set(obj, eunit);
  psi_epsilon_set(obj, epsilon);
  psi_epsilon2_set(obj, epsilon); /* Default is no dielectric contrast */

  n = rt_double_parameter(rt, "temperature", &temperature);

  if (n == 0 || temperature <= 0.0) {
    pe_fatal(pe, "Please set a temperature to use electrokinetics\n");
  }

  beta = 1.0/temperature;

  psi_beta_set(obj, beta);
  psi_bjerrum_length(obj, &lbjerrum);

  pe_info(pe, "Electrokinetic species:    %2d\n", nk);
  pe_info(pe, "Boltzmann factor:          %14.7e (T = %14.7e)\n", beta, temperature);
  pe_info(pe, "Unit charge:               %14.7e\n", eunit);
  pe_info(pe, "Permittivity:              %14.7e\n", epsilon);
  pe_info(pe, "Bjerrum length:            %14.7e\n", lbjerrum);

  for (n = 0; n < nk; n++) {
    pe_info(pe, "Valency species %d:         %2d\n", n, valency[n]);
    pe_info(pe, "Diffusivity species %d:     %14.7e\n", n, diffusivity[n]);
  }

  /* Multisteps and diffusive accuracy in NPE */

  n = rt_int_parameter(rt, "electrokinetics_multisteps", &multisteps);
  if (n == 1) psi_multisteps_set(obj, multisteps);
  n = rt_int_parameter(rt, "electrokinetics_skipsteps", &skipsteps);
  if (n == 1) psi_skipsteps_set(obj, skipsteps);
  n = rt_double_parameter(rt, "electrokinetics_diffacc", &diffacc);
  if (n == 1) psi_diffacc_set(obj, diffacc);

  psi_multisteps(obj, &multisteps);
  pe_info(pe, "Number of multisteps:       %d\n", multisteps);
  pe_info(pe, "Number of skipsteps:        %d\n", psi_skipsteps(obj));
  psi_diffacc(obj, &diffacc);
  pe_info(pe, "Diffusive accuracy in NPE: %14.7e\n", diffacc);

  /* Tolerances and Iterations */

  n = rt_double_parameter(rt, "electrokinetics_rel_tol", &tolerance);
  if (n == 1) psi_reltol_set(obj, tolerance);
  n = rt_double_parameter(rt, "electrokinetics_abs_tol", &tolerance);
  if (n == 1) psi_abstol_set(obj, tolerance);
  n = rt_int_parameter(rt, "electrokinetics_maxits", &niteration);
  if (n == 1) psi_maxits_set(obj, niteration);

  psi_reltol(obj, &tolerance);
  pe_info(pe, "Relative tolerance:  %20.7e\n", tolerance);
  psi_abstol(obj, &tolerance);
  pe_info(pe, "Absolute tolerance:  %20.7e\n", tolerance);
  psi_maxits(obj, &niteration);
  pe_info(pe, "Max. no. of iterations:  %16d\n", niteration);

  /* Output */

  n = 0;
  n += rt_int_parameter(rt, "freq_statistics", &nfreq);
  n += rt_int_parameter(rt, "freq_psi_resid", &nfreq);
  if (n > 0) psi_nfreq_set(obj, nfreq);;


  /* I/O */

  n = rt_int_parameter_vector(rt, "default_io_grid", io_grid);
  n = rt_string_parameter(rt, "psi_format", value, BUFSIZ);

  if (strcmp(value, "ASCII") == 0) {
    io_format_in = IO_FORMAT_ASCII;
    io_format_out = IO_FORMAT_ASCII;
  }

  pe_info(pe, "I/O decomposition:          %d %d %d\n",
	  io_grid[0], io_grid[1],
       io_grid[2]);
  pe_info(pe, "I/O format:                 %s\n", value);

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

int psi_rt_init_rho(pe_t * pe, rt_t * rt, psi_t * obj, map_t * map) {

  int n;
  char value[BUFSIZ];
  char filestub[FILENAME_MAX];

  double rho_el;              /* Charge density */
  double delta_el;            /* Relative difference in charge densities */
  double sigma;               /* Surface charge density */
  double ld;                  /* Debye length */
  double ld2;                 /* Second Debye length for dielectric contrast */
  double eps1, eps2;          /* Dielectric permittivities */

  assert(pe);
  assert(rt);

  /* Initial charge densities */

  pe_info(pe, "\n");
  pe_info(pe, "Initial charge densities\n");
  pe_info(pe, "------------------------\n");

  n = rt_string_parameter(rt, "electrokinetics_init", value, BUFSIZ);

  if (strcmp(value, "gouy_chapman") == 0) {
    pe_info(pe, "Initial conditions:         %s\n", "Gouy Chapman");

    n = rt_double_parameter(rt, "electrokinetics_init_rho_el", &rho_el);
    if (n == 0) pe_fatal(pe, "... please set electrokinetics_init_rho_el\n");
    pe_info(pe, "Initial condition rho_el:  %14.7e\n", rho_el);
    psi_debye_length(obj, rho_el, &ld);
    pe_info(pe, "Debye length:              %14.7e\n", ld);

    n = rt_double_parameter(rt, "electrokinetics_init_sigma", &sigma);
    if (n == 0) fatal("... please set electrokinetics_init_sigma\n");
    pe_info(pe, "Initial condition sigma:   %14.7e\n", sigma);

    psi_init_gouy_chapman(obj, map, rho_el, sigma);
  }

  if (strcmp(value, "liquid_junction") == 0) {
    pe_info(pe, "Initial conditions:         %s\n", "Liquid junction");

    n = rt_double_parameter(rt, "electrokinetics_init_rho_el", &rho_el);
    if (n == 0) pe_fatal(pe, "... please set electrokinetics_init_rho_el\n");
    pe_info(pe, "Initial condition rho_el: %14.7e\n", rho_el);
    psi_debye_length(obj, rho_el, &ld);
    pe_info(pe, "Debye length:             %14.7e\n", ld);

    n = rt_double_parameter(rt, "electrokinetics_init_delta_el", &delta_el);
    if (n == 0) pe_fatal(pe, "... please set electrokinetics_init_delta_el\n");
    pe_info(pe, "Initial condition delta_el: %14.7e\n", delta_el);

    psi_init_liquid_junction(obj, rho_el, delta_el);
  }

  if (strcmp(value, "uniform") == 0) {
    pe_info(pe, "Initial conditions:         %s\n", "Uniform");

    n = rt_double_parameter(rt, "electrokinetics_init_rho_el", &rho_el);
    if (n == 0) pe_fatal(pe, "... please set electrokinetics_init_rho_el\n");
    pe_info(pe, "Initial condition rho_el: %14.7e\n", rho_el);
    psi_debye_length(obj, rho_el, &ld);
    pe_info(pe, "Debye length:             %14.7e\n", ld);

    /* Call permittivities and check for dielectric contrast */

    psi_epsilon(obj, &eps1);
    psi_epsilon2(obj, &eps2);

    if (eps1 != eps2) {
      psi_debye_length2(obj, rho_el, &ld2);
      pe_info(pe, "Second Debye length:      %14.7e\n", ld2);
    }

    psi_init_uniform(obj, rho_el);
  }

  if (strcmp(value, "from_file") == 0) {

    pe_info(pe, "Initial conditions:        %s\n", "Charge from file");

    n = rt_double_parameter(rt, "electrokinetics_init_rho_el", &rho_el);
    if (n == 0) pe_fatal(pe, "... please set electrokinetics_init_rho_el\n");
    pe_info(pe, "Initial condition rho_el: %14.7e\n", rho_el);
    psi_debye_length(obj, rho_el, &ld);
    pe_info(pe, "Debye length:             %14.7e\n", ld);

    /* Call permittivities and check for dielectric contrast */
    psi_epsilon(obj, &eps1);
    psi_epsilon2(obj, &eps2);

    if (eps1 != eps2) {
      psi_debye_length2(obj, rho_el, &ld2);
      pe_info(pe, "Second Debye length:      %14.7e\n", ld2);
    }
    /* Set background charge densities */
    psi_init_uniform(obj, rho_el);

    /* Set surface charge */
    n = rt_string_parameter(rt, "porous_media_file", filestub, FILENAME_MAX);
    if (n == 0) pe_fatal(pe, " ... please provide porous media file\n");
    pe_info(pe, "\nInitialisation of charge from file %s.001-001\n", filestub);
    psi_init_sigma(obj,map);
  }

  return 0;
}
