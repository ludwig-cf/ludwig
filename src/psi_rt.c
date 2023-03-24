/*****************************************************************************
 *
 *  psi_rt.c
 *
 *  Run time initialisation of electrokinetics stuff.
 *
 *  At the moment the number of species is set to 2 automatically
 *  if the electrokinetics is switched on.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Oliver Henrich  (ohenrich@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "io_info_args_rt.h"
#include "psi_rt.h"
#include "psi_init.h"
#include "util_bits.h"

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

  psi_options_t opts = {.e = obj->e, .beta = obj->beta,
			.epsilon1 = obj->epsilon, .epsilon2 = obj->epsilon2};

  /* Initial charge densities */

  pe_info(pe, "\n");
  pe_info(pe, "Initial charge densities\n");
  pe_info(pe, "------------------------\n");

  rt_string_parameter(rt, "electrokinetics_init", value, BUFSIZ);

  if (strcmp(value, "gouy_chapman") == 0) {
    double ld_actual = 0.0;
    pe_info(pe, "Initial conditions:         %s\n", "Gouy Chapman");

    n = rt_double_parameter(rt, "electrokinetics_init_rho_el", &rho_el);
    if (n == 0) pe_fatal(pe, "... please set electrokinetics_init_rho_el\n");

    n = rt_double_parameter(rt, "electrokinetics_init_sigma", &sigma);
    if (n == 0) pe_fatal(pe, "... please set electrokinetics_init_sigma\n");

    psi_debye_length1(&opts, rho_el, &ld);
    psi_init_gouy_chapman(obj, map, rho_el, sigma);

    {
      /* We want a real Debye length from the actual charge/countercharge
	 densities that have been initialisated in the fluid. */
      int index = cs_index(obj->cs, 2, 1, 1);
      double rho_actual = 0.0;
      psi_ionic_strength(obj, index, &rho_actual);
      psi_debye_length1(&opts, rho_actual, &ld_actual);
    }
    pe_info(pe, "Initial condition rho_el:  %14.7e\n", rho_el);
    pe_info(pe, "Debye length:              %14.7e\n", ld);
    pe_info(pe, "Debye length (actual):     %14.7e\n", ld_actual);
    pe_info(pe, "Initial condition sigma:   %14.7e\n", sigma);
  }

  if (strcmp(value, "liquid_junction") == 0) {
    pe_info(pe, "Initial conditions:         %s\n", "Liquid junction");

    n = rt_double_parameter(rt, "electrokinetics_init_rho_el", &rho_el);
    if (n == 0) pe_fatal(pe, "... please set electrokinetics_init_rho_el\n");
    pe_info(pe, "Initial condition rho_el: %14.7e\n", rho_el);
    psi_debye_length1(&opts, rho_el, &ld);
    pe_info(pe, "Debye length:             %14.7e\n", ld);

    n = rt_double_parameter(rt, "electrokinetics_init_delta_el", &delta_el);
    if (n == 0) pe_fatal(pe, "... please set electrokinetics_init_delta_el\n");
    pe_info(pe, "Initial condition delta_el: %14.7e\n", delta_el);

    {
      /* psi_p is the saturation potential */
      double beta   = obj->beta;
      double e      = obj->e;
      double dplus  = obj->diffusivity[0];
      double dminus = obj->diffusivity[1];
      double psi_p  = dplus*dminus*delta_el/(beta*e*(dplus + dminus)*rho_el);
      double tau_e  = obj->epsilon/(beta*e*e*(dplus + dminus)*rho_el);
      pe_info(pe, "Saturation potential:        %14.7e\n", psi_p);
      pe_info(pe, "Saturation timescale:        %14.7e\n", tau_e);
    }

    psi_init_liquid_junction(obj, rho_el, delta_el);
  }

  if (strcmp(value, "uniform") == 0) {
    pe_info(pe, "Initial conditions:         %s\n", "Uniform");

    n = rt_double_parameter(rt, "electrokinetics_init_rho_el", &rho_el);
    if (n == 0) pe_fatal(pe, "... please set electrokinetics_init_rho_el\n");
    pe_info(pe, "Initial condition rho_el: %14.7e\n", rho_el);
    psi_debye_length1(&opts, rho_el, &ld);
    pe_info(pe, "Debye length:             %14.7e\n", ld);

    /* Call permittivities and check for dielectric contrast */

    psi_epsilon(obj, &eps1);
    psi_epsilon2(obj, &eps2);

    /* Unless really the same number ... */
    if (0 == util_double_same(eps1, eps2)) {
      psi_debye_length2(&opts, rho_el, &ld2);
      pe_info(pe, "Second Debye length:      %14.7e\n", ld2);
    }

    psi_init_uniform(obj, rho_el);
  }

  if (strcmp(value, "from_file") == 0) {
    io_event_t event1 = {0};
    io_event_t event2 = {0};
    pe_info(pe, "Initialisation requested from file(s)\n");
    field_io_read(obj->psi, 0, &event1);
    field_io_read(obj->rho, 0, &event2);
  }

  if (strcmp(value, "point_charges") == 0) {

    pe_info(pe, "Initial conditions:        %s\n", "Point or surface charges from file");

    n = rt_double_parameter(rt, "electrokinetics_init_rho_el", &rho_el);
    if (n == 0) pe_fatal(pe, "... please set electrokinetics_init_rho_el\n");
    pe_info(pe, "Initial condition rho_el: %14.7e\n", rho_el);
    psi_debye_length1(&opts, rho_el, &ld);
    pe_info(pe, "Debye length:             %14.7e\n", ld);

    /* Call permittivities and check for dielectric contrast */
    psi_epsilon(obj, &eps1);
    psi_epsilon2(obj, &eps2);

    /* Unless really the same number... */
    if (0 == util_double_same(eps1, eps2)) {
      psi_debye_length1(&opts, rho_el, &ld2);
      pe_info(pe, "Second Debye length:      %14.7e\n", ld2);
    }
    /* Set background charge densities */
    psi_init_uniform(obj, rho_el);

    /* Set surface charge */
    n = rt_string_parameter(rt, "porous_media_file", filestub, FILENAME_MAX);
    if (n == 0) pe_fatal(pe, " ... please provide porous media file\n");
    pe_info(pe, "\nInitialisation of point or surface charges from file %s.001-001\n", filestub);
    psi_init_sigma(obj,map);
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_options_rt
 *
 *****************************************************************************/

int psi_options_rt(pe_t * pe, cs_t * cs, rt_t * rt, psi_options_t * popts) {

  psi_options_t opts = psi_options_default(cs->param->nhalo);

  assert(pe);
  assert(cs);
  assert(rt);

  /* Physics */
  /* The Boltzmann factor comes from the temperature */
  /* Can use "epsilon" or specific keys "epsilon1" and "epsilon2" */

  rt_double_parameter(rt, "electrokinetics_eunit", &opts.e);

  {
    double t = -1.0;
    rt_double_parameter(rt, "temperature", &t);
    if (t <= 0.0) pe_fatal(pe, "Please use a +ve temperature for electro\n");
    opts.beta = 1.0/t;
  }

  rt_double_parameter(rt, "electrokinetics_epsilon", &opts.epsilon1);
  rt_double_parameter(rt, "electrokinetics_epsilon", &opts.epsilon2);
  rt_double_parameter(rt, "electrokinetics_epsilon1", &opts.epsilon1);
  rt_double_parameter(rt, "electrokinetics_epsilon2", &opts.epsilon2);

  rt_double_parameter_vector(rt, "electric_e0", opts.e0);

  rt_double_parameter(rt, "electrokinetics_d0", &opts.diffusivity[0]);
  rt_double_parameter(rt, "electrokinetics_d1", &opts.diffusivity[1]);

  rt_int_parameter(rt,    "electrokinetics_z0", &opts.valency[0]);
  rt_int_parameter(rt,    "electrokinetics_z1", &opts.valency[1]);

  /* Poisson solver */
  /* There are two possible sources of nfreq */

  {
    /* Solver type */
    char str[BUFSIZ] = {0};
    strcpy(str, psi_poisson_solver_to_string(psi_poisson_solver_default()));
    rt_string_parameter(rt, "electrokinetics_solver_type", str, BUFSIZ);
    opts.solver.psolver = psi_poisson_solver_from_string(str);
    switch (opts.solver.psolver) {
    case (PSI_POISSON_SOLVER_INVALID):
      /* Not recognised */
      pe_info(pe, "electrokinetics_solver_type: %s\n", str);
      pe_info(pe, "is not recongnised\n");
      pe_fatal(pe, "Please check and try again!\n");
      break;
    case (PSI_POISSON_SOLVER_PETSC):
      /* Check Petsc is actually available */
      {
	int isAvailable = 0;
	PetscInitialised(&isAvailable);
	if (isAvailable == 0) {
	  pe_info(pe, "electrokinetics_solver_type:  petsc\n");
	  pe_info(pe, "Petsc has not been compiled in this build\n");
	  pe_info(pe, "Please use the `sor` solver.\n");
	  pe_fatal(pe, "Please check the input and try again!\n");
	}
      }
      break;
    default:
      ; /* ok */
    }
  }

  /* Stencil must be available. */
  if (rt_int_parameter(rt, "electrokinetics_solver_stencil",
		       &opts.solver.nstencil)) {
    stencil_t * s = NULL;
    int ifail = stencil_create(opts.solver.nstencil, &s);
    if (ifail != 0) {
      pe_info(pe, "electroknietics_solver_stencil: %d\n", opts.solver.nstencil);
      pe_fatal(pe, "Not supported. Please check and try again!\n");
    }
    if (s) stencil_free(&s);
  }

  rt_int_parameter(rt, "electrokinetics_maxits",  &opts.solver.maxits);
  rt_int_parameter(rt, "freq_statistics", &opts.solver.nfreq);
  rt_int_parameter(rt, "freq_psi_resid",  &opts.solver.nfreq);

  rt_double_parameter(rt, "electrokinetics_rel_tol", &opts.solver.reltol);
  rt_double_parameter(rt, "electrokinetics_abs_tol", &opts.solver.abstol);

  /* NPE time splitting and criteria */

  rt_int_parameter(rt, "electrokinetics_multisteps", &opts.nsmallstep);
  rt_double_parameter(rt, "electrokinetics_diffacc", &opts.diffacc);

  /* Field quantites */
  /* At the moment there are two fields (potential and charge densities)
   * but only one input key "psi" involved */

  {
    opts.psi = field_options_ndata_nhalo(1, cs->param->nhalo);
    opts.rho = field_options_ndata_nhalo(opts.nk, cs->param->nhalo);

    io_info_args_rt(rt, RT_FATAL, "psi", IO_INFO_READ_WRITE, &opts.psi.iodata);
    io_info_args_rt(rt, RT_FATAL, "psi", IO_INFO_READ_WRITE, &opts.rho.iodata);

    if (opts.psi.iodata.input.mode != IO_MODE_MPIIO) {
      pe_fatal(pe, "Electrokinetics i/o must use psi_io_mode mpiio\n");
    }
  }

  *popts = opts;

  return 0;
}

/*****************************************************************************
 *
 *  psi_info
 *
 *  Could be moved elsewhere.
 *
 *****************************************************************************/

int psi_info(pe_t * pe, const psi_t * psi) {

  double lbjerrum = 0.0;

  assert(pe);
  assert(psi);

  /* Refactoring has casued a slight awkwardnesss here... */
  {
    psi_options_t opts = {.e = psi->e, .beta = psi->beta,
			  .epsilon1 = psi->epsilon};
    psi_bjerrum_length1(&opts, &lbjerrum);
  }

  /* Information */

  pe_info(pe, "Electrokinetic species:    %2d\n", psi->nk);
  pe_info(pe, "Boltzmann factor:          %14.7e (T = %14.7e)\n",
	  psi->beta, 1.0/psi->beta);
  pe_info(pe, "Unit charge:               %14.7e\n", psi->e);
  pe_info(pe, "Permittivity:              %14.7e\n", psi->epsilon);
  pe_info(pe, "Bjerrum length:            %14.7e\n", lbjerrum);

  for (int n = 0; n < psi->nk; n++) {
    pe_info(pe, "Valency species %d:         %2d\n", n, psi->valency[n]);
    pe_info(pe, "Diffusivity species %d:     %14.7e\n", n, psi->diffusivity[n]);
  }

  /* Add full information ... */
  pe_info(pe, "Solver type:         %20s\n",
	  psi_poisson_solver_to_string(psi->solver.psolver));
  pe_info(pe, "Solver stencil points:   %16d\n", psi->solver.nstencil);
  pe_info(pe, "Relative tolerance:  %20.7e\n",   psi->solver.reltol);
  pe_info(pe, "Absolute tolerance:  %20.7e\n",   psi->solver.abstol);
  pe_info(pe, "Max. no. of iterations:  %16d\n", psi->solver.maxits);

  pe_info(pe, "Number of multisteps:       %d\n", psi->multisteps);
  pe_info(pe, "Diffusive accuracy in NPE: %14.7e\n", psi->diffacc);

  return 0;
}
