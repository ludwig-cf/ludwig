/*****************************************************************************
 *
 *  ludwig.c
 *
 *  A lattice Boltzmann code for complex fluids.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Coordinate system, general */
#include "pe.h"
#include "runtime.h"
#include "ran.h"
#include "timer.h"
#include "coords_rt.h"
#include "coords.h"
#include "leesedwards.h"
#include "control.h"
#include "util.h"

#include "model.h"
#include "model_le.h"
#include "bbl.h"

#include "collision.h"
#include "propagation.h"
#include "propagation_ode.h"
#include "distribution_rt.h"
#include "collision_rt.h"

#include "map.h"
#include "wall.h"
#include "interaction.h"
#include "physics.h"

#include "hydro_rt.h"

#include "io_harness.h"
#include "phi_stats.h"
#include "phi_force.h"
#include "phi_force_colloid.h"
#include "phi_fluctuations.h"
#include "phi_lb_coupler.h"

/* Order parameter fields */
#include "field.h"
#include "field_grad.h"
#include "gradient_rt.h"

/* Free energy */
#include "symmetric.h"
#include "brazovskii.h"
#include "polar_active.h"
#include "blue_phase.h"
#include "symmetric_rt.h"
#include "brazovskii_rt.h"
#include "polar_active_rt.h"
#include "blue_phase_rt.h"

/* Dynamics */
#include "phi_cahn_hilliard.h"
#include "leslie_ericksen.h"
#include "blue_phase_beris_edwards.h"

/* Colloids */
#include "colloids.h"
#include "colloids_Q_tensor.h"
#include "cio.h"
#include "subgrid.h"

#include "advection_rt.h"

/* Electrokinetics */
#include "psi.h"
#include "psi_rt.h"
#include "psi_sor.h"
#include "psi_stats.h"
#include "psi_force.h"
#include "psi_colloid.h"
#include "nernst_planck.h"

/* Statistics */
#include "stats_colloid.h"
#include "stats_turbulent.h"
#include "stats_surfactant.h"
#include "stats_rheology.h"
#include "stats_free_energy.h"
#include "stats_distribution.h"
#include "stats_calibration.h"
#include "stats_velocity.h"
#include "stats_sigma.h"
#include "stats_symmetric.h"

#include "ludwig.h"

typedef struct ludwig_s ludwig_t;
struct ludwig_s {
  hydro_t * hydro;          /* Hydrodynamic quantities */
  field_t * phi;            /* Scalar order parameter */
  field_t * p;              /* Vector order parameter */
  field_t * q;              /* Tensor order parameter */
  field_grad_t * phi_grad;  /* Gradients for phi */
  field_grad_t * p_grad;    /* Gradients for p */
  field_grad_t * q_grad;    /* Gradients for q */
  psi_t * psi;              /* Electrokinetics */
  map_t * map;              /* Site map for fluid/solid status etc. */
};

static int ludwig_rt(ludwig_t * ludwig);
static int ludwig_report_momentum(ludwig_t * ludwig);
int free_energy_init_rt(ludwig_t * ludwig);
int map_init_rt(map_t ** map);
int symmetric_rt_initial_conditions(field_t * phi);
int symmetric_init_drop(field_t * fphi, double radius, double xi0);

/*****************************************************************************
 *
 *  ludwig_rt
 *
 *  Digest the run-time arguments for different parts of the code.
 *
 *****************************************************************************/

static int ludwig_rt(ludwig_t * ludwig) {

  int form;
  int n, nstat;
  char filename[FILENAME_MAX];
  char subdirectory[FILENAME_MAX];
  char value[BUFSIZ];
  int io_grid_default[3] = {1, 1, 1};
  int io_grid[3];

  io_info_t * iohandler = NULL;

  assert(ludwig);
  
  TIMER_init();
  TIMER_start(TIMER_TOTAL);

  /* Initialise free-energy related objects, and the coordinate
   * system (the halo extent depends on choice of free energy). */

  free_energy_init_rt(ludwig);

  init_control();
  init_physics();

  if (is_propagation_ode()) propagation_ode_init();

  distribution_run_time();
  collision_run_time();
  map_init_rt(&ludwig->map);

  ran_init();

  psi_init_rt(&ludwig->psi, ludwig->map);
  hydro_rt(&ludwig->hydro);

  /* PHI I/O */

  RUN_get_int_parameter_vector("default_io_grid", io_grid_default);
  for (n = 0; n < 3; n++) {
    io_grid[n] = io_grid_default[n];
  }
  RUN_get_int_parameter_vector("phi_io_grid", io_grid);

  form = IO_FORMAT_DEFAULT;
  n = RUN_get_string_parameter("phi_format", value, BUFSIZ);
  if (n != 0 && strcmp(value, "ASCII") == 0) {
    form = IO_FORMAT_ASCII;
  }


  /* All the same I/O grid  */

  if (ludwig->phi) field_init_io_info(ludwig->phi, io_grid, form, form);
  if (ludwig->p) field_init_io_info(ludwig->p, io_grid, form, form);
  if (ludwig->q) field_init_io_info(ludwig->q, io_grid, form, form);

  if (ludwig->phi || ludwig->p || ludwig->q) {
    info("\n");
    info("Order parameter I/O\n");
    info("-------------------\n");
    
    info("Order parameter I/O format:   %s\n", value);
    info("I/O decomposition:            %d %d %d\n", io_grid[0], io_grid[1],
	 io_grid[2]);
    advection_run_time();
  }

  /* Can we move this down to t = 0 initialisation? */
  if (ludwig->phi) symmetric_rt_initial_conditions(ludwig->phi);

  wall_init(ludwig->map);
  COLL_init(ludwig->map);

  /* NOW INITIAL CONDITIONS */

  pe_subdirectory(subdirectory);

  if (get_step() == 0) {
    n = 0;
    distribution_rt_initial_conditions();
    RUN_get_int_parameter("LE_init_profile", &n);

    if (n != 0) model_le_init_shear_profile();
  }
  else {
    /* Distributions */

    sprintf(filename, "%sdist-%8.8d", subdirectory, get_step());
    info("Re-starting simulation at step %d with data read from "
	 "config\nfile(s) %s\n", get_step(), filename);

    io_read(filename, distribution_io_info());

    /* Restart t != 0 for order parameter */

    if (ludwig->phi) {
      sprintf(filename, "%sphi-%8.8d", subdirectory, get_step());
      info("files(s) %s\n", filename);
      field_io_info(ludwig->phi, &iohandler);
      io_read_data(iohandler, filename, ludwig->phi);
    }
    if (ludwig->p) {
      sprintf(filename, "%sp-%8.8d", subdirectory, get_step());
      info("files(s) %s\n", filename);
      field_io_info(ludwig->p, &iohandler);
      io_read_data(iohandler, filename, ludwig->p);
    }
    if (ludwig->q) {
      sprintf(filename, "%sqs-%8.8d", subdirectory, get_step());
      info("files(s) %s\n", filename);
      field_io_info(ludwig->q, &iohandler);
      io_read_data(iohandler, filename, ludwig->q);
    }
    if (ludwig->hydro) {
      sprintf(filename, "%svel-%8.8d", subdirectory, get_step());
      info("hydro files(s) %s\n", filename);
      hydro_io_info(ludwig->hydro, &iohandler);
      io_read_data(iohandler, filename, ludwig->hydro);
    }
  }

  /* gradient initialisation for field stuff */

  if (ludwig->phi) gradient_rt_init(ludwig->phi_grad, ludwig->map);
  if (ludwig->p) gradient_rt_init(ludwig->p_grad, ludwig->map);
  if (ludwig->q) gradient_rt_init(ludwig->q_grad, ludwig->map);

  stats_rheology_init();
  stats_turbulent_init();
  collision_init();

  /* Calibration statistics for ah required? */

  nstat = 0;
  n = RUN_get_string_parameter("calibration", filename, FILENAME_MAX);
  if (n == 1 && strcmp(filename, "on") == 0) nstat = 1;
  stats_calibration_init(nstat);

  /* Calibration of surface tension required (symmetric only) */

  nstat = 0;
  n = RUN_get_string_parameter("calibration_sigma", filename, FILENAME_MAX);
  if (n == 1 && strcmp(filename, "on") == 0) nstat = 1;

  if (get_step() == 0) {
    stats_sigma_init(ludwig->phi, nstat);
    if (distribution_ndist() == 2) phi_lb_from_field(ludwig->phi); 
  }

  collision_init();

  /* Initial Q_ab field required, apparently More GENERAL? */

  if (get_step() == 0 && ludwig->p) {
    polar_active_rt_initial_conditions(ludwig->p);
  }
  if (get_step() == 0 && ludwig->q) {
    blue_phase_rt_initial_conditions(ludwig->q);
  }

  /* Electroneutrality */

  if (get_step() == 0 && ludwig->psi) {
    psi_colloid_rho_set(ludwig->psi);
    psi_colloid_electroneutral(ludwig->psi);
  }

  return 0;
}

/*****************************************************************************
 *
 *  ludwig_run
 *
 *****************************************************************************/

void ludwig_run(const char * inputfile) {

  char    filename[FILENAME_MAX];
  char    subdirectory[FILENAME_MAX];
  int     step = 0;
  int     is_subgrid = 0;
  double  fzero[3] = {0.0, 0.0, 0.0};
  io_info_t * iohandler = NULL;

  ludwig_t * ludwig = NULL;

  ludwig = calloc(1, sizeof(ludwig_t));
  assert(ludwig);

  pe_init();
  RUN_read_input_file(inputfile);

  ludwig_rt(ludwig);

  /* Report initial statistics */

  pe_subdirectory(subdirectory);

  info("Initial conditions.\n");

  stats_distribution_print(ludwig->map);

  if (distribution_ndist() == 2) phi_lb_to_field(ludwig->phi);
  if (ludwig->phi) stats_field_info(ludwig->phi, ludwig->map);
  if (ludwig->p)   stats_field_info(ludwig->p, ludwig->map);
  if (ludwig->q)   stats_field_info(ludwig->q, ludwig->map);
  if (ludwig->psi) psi_stats_info(ludwig->psi);
  ludwig_report_momentum(ludwig);

  /* Main time stepping loop */

  info("\n");
  info("Starting time step loop.\n");
  subgrid_on(&is_subgrid);

  while (next_step()) {

    TIMER_start(TIMER_STEPS);
    step = get_step();
    if (ludwig->hydro) hydro_f_zero(ludwig->hydro, fzero);
    COLL_update(ludwig->hydro, ludwig->map, ludwig->phi, ludwig->p, ludwig->q);

    /* Electrokinetics */

    if (ludwig->psi) {
      psi_colloid_rho_set(ludwig->psi);
      psi_halo_psi(ludwig->psi);
      /* Sum force for this step before update */
      psi_force_grad_mu(ludwig->psi, ludwig->hydro);
      psi_sor_poisson(ludwig->psi);
      psi_halo_rho(ludwig->psi);
      if (ludwig->hydro) hydro_u_halo(ludwig->hydro);
      nernst_planck_driver(ludwig->psi, ludwig->hydro, ludwig->map);
    }

    /* Order parameter */

    TIMER_start(TIMER_PHI_GRADIENTS);

    /* if symmetric_lb store phi to field */
    if (distribution_ndist() == 2) phi_lb_to_field(ludwig->phi);

    if (ludwig->phi) {
      field_halo(ludwig->phi);
      field_grad_compute(ludwig->phi_grad);
    }
    if (ludwig->p) {
      field_halo(ludwig->p);
      field_grad_compute(ludwig->p_grad);
      }
    if (ludwig->q) {
      field_halo(ludwig->q);
      field_grad_compute(ludwig->q_grad);
      blue_phase_redshift_compute(); /* if redshift required? */
    }

    TIMER_stop(TIMER_PHI_GRADIENTS);

    /* order parameter dynamics (not if symmetric_lb) */

    if (distribution_ndist() == 2) {
      /* dynamics are dealt with at the collision stage (below) */
    }
    else {

      TIMER_start(TIMER_FORCE_CALCULATION);

      if (colloid_ntotal() == 0) {
	phi_force_calculation(ludwig->phi, ludwig->hydro);
      }
      else {
	phi_force_colloid(ludwig->hydro, ludwig->map);
      }
      TIMER_stop(TIMER_FORCE_CALCULATION);

      if (ludwig->phi) phi_cahn_hilliard(ludwig->phi, ludwig->hydro);
      if (ludwig->p) leslie_ericksen_update(ludwig->p, ludwig->hydro);
      if (ludwig->q) blue_phase_beris_edwards(ludwig->q, ludwig->hydro,
					      ludwig->map);
    }

    /* Collision stage */

    TIMER_start(TIMER_COLLIDE);
    collide(ludwig->hydro, ludwig->map);
    TIMER_stop(TIMER_COLLIDE);

    model_le_apply_boundary_conditions();

    TIMER_start(TIMER_HALO_LATTICE);
    distribution_halo();
    TIMER_stop(TIMER_HALO_LATTICE);

    /* Colloid bounce-back applied between collision and
     * propagation steps. */

    if (is_subgrid) {
      subgrid_update(ludwig->hydro);
    }
    else {
      TIMER_start(TIMER_BBL);
      wall_update();
      bounce_back_on_links();
      wall_bounce_back(ludwig->map);
      TIMER_stop(TIMER_BBL);
    }

    /* There must be no halo updates between bounce back
     * and propagation, as the halo regions are active */

    TIMER_start(TIMER_PROPAGATE);

    if(is_propagation_ode()) {
      propagation_ode();
    }
    else {
      propagation();
    }

    TIMER_stop(TIMER_PROPAGATE);

    TIMER_stop(TIMER_STEPS);

    /* Configuration dump */

    if (is_config_step()) {
      info("Writing distribution output at step %d!\n", step);
      sprintf(filename, "%sdist-%8.8d", subdirectory, step);
      io_write(filename, distribution_io_info());
    }

    /* is_measurement_step() is here to prevent 'breaking' old input
     * files; it should really be removed. */

    if (is_config_step() || is_measurement_step() || is_colloid_io_step()) {
      if (colloid_ntotal() > 0) {
	info("Writing colloid output at step %d!\n", step);
	sprintf(filename, "%s%s%8.8d", subdirectory, "config.cds", step);
	colloid_io_write(filename);
      }
    }

    if (is_phi_output_step() || is_config_step()) {

      if (ludwig->phi) {
	field_io_info(ludwig->phi, &iohandler);
	info("Writing phi file at step %d!\n", step);
	sprintf(filename,"%sphi-%8.8d", subdirectory, step);
	io_write_data(iohandler, filename, ludwig->phi);
      }
      if (ludwig->q) {
	field_io_info(ludwig->q, &iohandler);
	info("Writing qs file at step %d!\n", step);
	sprintf(filename,"%sqs-%8.8d", subdirectory, step);
	io_write_data(iohandler, filename, ludwig->q);
      }
    }

    if (is_psi_output_step()) {
      if (ludwig->psi) {
	psi_io_info(ludwig->psi, &iohandler);
	info("Writing psi file at step %d!\n", step);
	sprintf(filename,"%spsi-%8.8d", subdirectory, step);
	io_write_data(iohandler, filename, ludwig->psi);
      }
    }

    /* Measurements */

    if (is_measurement_step()) {
      stats_sigma_measure(ludwig->phi, step);
    }

    if (is_shear_measurement_step()) {
      stats_rheology_stress_profile_accumulate(ludwig->hydro);
    }

    if (is_shear_output_step()) {
      sprintf(filename, "%sstr-%8.8d.dat", subdirectory, step);
      stats_rheology_stress_section(filename);
      stats_rheology_stress_profile_zero();
    }

    if (is_vel_output_step() || is_config_step()) {
      hydro_io_info(ludwig->hydro, &iohandler);
      info("Writing velocity output at step %d!\n", step);
      sprintf(filename, "%svel-%8.8d", subdirectory, step);
      io_write_data(iohandler, filename, ludwig->hydro);
    }

    /* Print progress report */

    if (is_statistics_step()) {
      stats_distribution_print(ludwig->map);

      if (distribution_ndist() == 2) phi_lb_to_field(ludwig->phi);
      if (ludwig->phi) stats_field_info(ludwig->phi, ludwig->map);
      if (ludwig->p)   stats_field_info(ludwig->p, ludwig->map);
      if (ludwig->q)   stats_field_info(ludwig->q, ludwig->map);
      stats_free_energy_density(ludwig->q, ludwig->map);
      if (ludwig->psi) {
	psi_stats_info(ludwig->psi);
      }
      ludwig_report_momentum(ludwig);
      if (ludwig->hydro) stats_velocity_minmax(ludwig->hydro, ludwig->map);

      collision_stats_kt(ludwig->map);

      info("\nCompleted cycle %d\n", step);
    }

    stats_calibration_accumulate(step, ludwig->hydro, ludwig->map);

    /* Next time step */
  }

  /* To prevent any conflict between the last regular dump, and
   * a final dump, there's a barrier here. */

  MPI_Barrier(pe_comm()); 

  /* Dump the final configuration if required. */

  if (is_config_at_end()) {
    sprintf(filename, "%sdist-%8.8d", subdirectory, step);
    io_write(filename, distribution_io_info());
    sprintf(filename, "%s%s%8.8d", subdirectory, "config.cds", step);
    colloid_io_write(filename);

    if (ludwig->phi) {
      field_io_info(ludwig->phi, &iohandler);
      info("Writing phi file at step %d!\n", step);
      sprintf(filename,"%sphi-%8.8d", subdirectory, step);
      io_write_data(iohandler, filename, ludwig->phi);
    }
    if (ludwig->q) {
      field_io_info(ludwig->q, &iohandler);
      info("Writing qs file at step %d!\n", step);
      sprintf(filename,"%sqs-%8.8d", subdirectory, step);
      io_write_data(iohandler, filename, ludwig->q);
    }
    /* Only strictly required if have order parameter dynamics */ 
    if (ludwig->hydro) {
      hydro_io_info(ludwig->hydro, &iohandler);
      info("Writing velocity output at step %d!\n", step);
      sprintf(filename, "%svel-%8.8d", subdirectory, step);
      io_write_data(iohandler, filename, ludwig->hydro);
    }
  }

  /* Shut down cleanly. Give the timer statistics. Finalise PE. */

  stats_rheology_finish();
  stats_turbulent_finish();
  stats_calibration_finish();
  colloids_finish();
  wall_finish();

  if (ludwig->phi_grad) field_grad_free(ludwig->phi_grad);
  if (ludwig->p_grad)   field_grad_free(ludwig->p_grad);
  if (ludwig->q_grad)   field_grad_free(ludwig->q_grad);
  if (ludwig->phi)      field_free(ludwig->phi);
  if (ludwig->p)        field_free(ludwig->p);
  if (ludwig->q)        field_free(ludwig->q);

  TIMER_stop(TIMER_TOTAL);
  TIMER_statistics();

  pe_finalise();

  return;
}

/*****************************************************************************
 *
 *  ludwig_report_momentum
 *
 *  Tidy report of the current momentum of the system.
 *
 *****************************************************************************/

static int ludwig_report_momentum(ludwig_t * ludwig) {

  int n;

  double g[3];         /* Fluid momentum (total) */
  double gc[3];        /* Colloid momentum (total) */
  double gwall[3];     /* Wall momentum (for accounting purposes only) */
  double gtotal[3];

  for (n = 0; n < 3; n++) {
    gtotal[n] = 0.0;
    g[n] = 0.0;
    gc[n] = 0.0;
    gwall[n] = 0.0;
  }

  stats_distribution_momentum(ludwig->map, g);
  stats_colloid_momentum(gc);
  if (wall_present()) wall_net_momentum(gwall);

  for (n = 0; n < 3; n++) {
    gtotal[n] = g[n] + gc[n] + gwall[n];
  }

  info("\n");
  info("Momentum - x y z\n");
  info("[total   ] %14.7e %14.7e %14.7e\n", gtotal[X], gtotal[Y], gtotal[Z]);
  info("[fluid   ] %14.7e %14.7e %14.7e\n", g[X], g[Y], g[Z]);
  if (colloid_ntotal()) {
    info("[colloids] %14.7e %14.7e %14.7e\n", gc[X], gc[Y], gc[Z]);
  }
  if (wall_present()) {
    info("[walls   ] %14.7e %14.7e %14.7e\n", gwall[X], gwall[Y], gwall[Z]);
  }

  return 0;
}

/******************************************************************************
 *
 *  free_energy_init_rt
 *
 *  This is to get over the rather awkward initialisation of the
 *  order parameter, free energy, and coordinate system, which
 *  are inter-related.
 *
 *  The choice of free energy sets the maximum halo width required,
 *  and what order parameter is required. This in turn determines
 *  the extent of the coordinate system, which must be initialised
 *  before the order parameter field, which is required to compute
 *  the free energy.
 *
 *  No free energy is appropriate for a single phase fluid.
 *
 *  This is currently rather repetative, so some rationalisation
 *  is required.
 *
 *  TODO: force divergence method alters halo requirement?
 *
 *****************************************************************************/

int free_energy_init_rt(ludwig_t * ludwig) {

  int n = 0;
  int p;
  int nf;
  int ngrad;
  int nhalo;
  double value;
  char description[BUFSIZ];
  unsigned int seed;

  assert(ludwig);

  n = RUN_get_string_parameter("free_energy", description, BUFSIZ);

  if (strcmp(description, "none") == 0) {
    /* Appropriate for single fluid */
    info("\n");
    info("No free energy selected\n");
    phi_force_required_set(0); /* Could reverse the default */

    nhalo = 1;
    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();
  }
  else if (strcmp(description, "symmetric") == 0 ||
	   strcmp(description, "symmetric_noise") == 0) {

    /* Symmetric free energy via finite difference */

    nf = 1;      /* 1 scalar order parameter */
    nhalo = 2;   /* Require stress divergence. */
    ngrad = 2;   /* \nabla^2 required */

    /* Noise requires additional stencil point for Cahn Hilliard */

    if (strcmp(description, "symmetric_noise") == 0) {
      nhalo = 3;
    }

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();

    field_create(nf, "phi", &ludwig->phi);
    field_init(ludwig->phi, nhalo);
    field_grad_create(ludwig->phi, ngrad, &ludwig->phi_grad);

    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");
    symmetric_run_time();
    symmetric_phi_set(ludwig->phi, ludwig->phi_grad);

    info("\n");
    info("Using Cahn-Hilliard finite difference solver.\n");

    RUN_get_double_parameter("mobility", &value);
    phi_cahn_hilliard_mobility_set(value);
    info("Mobility M            = %12.5e\n", phi_cahn_hilliard_mobility());

    /* This could be set via symmetric_noise, no? Or check consistent */
    p = 0;
    RUN_get_int_parameter("fd_phi_fluctuations", &p);
    info("Order parameter noise = %3s\n", (p == 0) ? "off" : " on");
    if (p != 0) phi_fluctuations_on_set(p);

    if (phi_fluctuations_on()) {
      if (nhalo != 3) fatal("Fluctuations: use symmetric_noise\n");
      RUN_get_int_parameter("fd_phi_fluctuations_seed", &p);
      seed = 0;
      if (p > 0) {
	seed = p;
	info("Order parameter noise seed: %u\n", seed);
      }
      phi_fluctuations_init(seed);
    }

    /* Force */

    p = 1; /* Default is to use divergence method */
    RUN_get_int_parameter("fd_force_divergence", &p);
    info("Force calculation:      %s\n",
         (p == 0) ? "phi grad mu method" : "divergence method");
    phi_force_divergence_set(p);

  }
  else if (strcmp(description, "symmetric_lb") == 0) {

    /* Symmetric free energy via full lattice kintic equation */

    nf = 1;      /* 1 scalar order parameter */
    nhalo = 1;   /* Require one piont for LB. */
    ngrad = 2;   /* \nabla^2 required */

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();

    field_create(nf, "phi", &ludwig->phi);
    field_init(ludwig->phi, nhalo);
    field_grad_create(ludwig->phi, ngrad, &ludwig->phi_grad);

    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");
    symmetric_run_time();
    symmetric_phi_set(ludwig->phi, ludwig->phi_grad);

    info("\n");
    info("Using full lattice Boltzmann solver for Cahn-Hilliard:\n");
    phi_force_required_set(0);

    RUN_get_double_parameter("mobility", &value);
    phi_cahn_hilliard_mobility_set(value);
    info("Mobility M            = %12.5e\n", phi_cahn_hilliard_mobility());

  }
  else if (strcmp(description, "brazovskii") == 0) {

    /* Brazovskii (always finite difference). */

    nf = 1;      /* 1 scalar order parameter */
    nhalo = 3;   /* Required for stress diveregnce. */
    ngrad = 4;   /* (\nabla^2)^2 required */

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();

    field_create(nf, "phi", &ludwig->phi);
    field_init(ludwig->phi, nhalo);
    field_grad_create(ludwig->phi, ngrad, &ludwig->phi_grad);

    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");
    brazovskii_run_time();
    brazovskii_phi_set(ludwig->phi, ludwig->phi_grad);

    info("\n");
    info("Using Cahn-Hilliard solver:\n");

    RUN_get_double_parameter("mobility", &value);
    phi_cahn_hilliard_mobility_set(value);
    info("Mobility M            = %12.5e\n", phi_cahn_hilliard_mobility());

    p = 1;
    RUN_get_int_parameter("fd_force_divergence", &p);
    info("Force caluclation:      %s\n",
         (p == 0) ? "phi grad mu method" : "divergence method");
    phi_force_divergence_set(p);


  }
  else if (strcmp(description, "surfactant") == 0) {
    /* Disable surfactant for the time being */
    info("Surfactant free energy is disabled\n");
    assert(0);
  }
  else if (strcmp(description, "lc_blue_phase") == 0) {

    /* Brazovskii (always finite difference). */

    nf = NQAB;   /* Tensor order parameter */
    nhalo = 2;   /* Required for stress diveregnce. */
    ngrad = 2;   /* (\nabla^2) required */

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();

    field_create(nf, "q", &ludwig->q);
    field_init(ludwig->q, nhalo);
    field_grad_create(ludwig->q, ngrad, &ludwig->q_grad);

    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");

    blue_phase_run_time();
    blue_phase_q_set(ludwig->q, ludwig->q_grad);

    info("\n");
    info("Using Beris-Edwards solver:\n");

    p = RUN_get_double_parameter("lc_Gamma", &value);
    if (p != 0) {
      blue_phase_be_set_rotational_diffusion(value);
      info("Rotational diffusion constant = %12.5e\n", value);
    }

  }
  else if (strcmp(description, "polar_active") == 0) {

    /* Polar active. */

    nf = NVECTOR;/* Vector order parameter */
    nhalo = 2;   /* Required for stress diveregnce. */
    ngrad = 2;   /* (\nabla^2) required */

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();

    field_create(nf, "p", &ludwig->p);
    field_init(ludwig->p, nhalo);
    field_grad_create(ludwig->p, ngrad, &ludwig->p_grad);

    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");

    polar_active_run_time();
    polar_active_p_set(ludwig->p, ludwig->p_grad);

    RUN_get_double_parameter("leslie_ericksen_gamma", &value);
    leslie_ericksen_gamma_set(value);
    info("Rotational diffusion     = %12.5e\n", value);

    RUN_get_double_parameter("leslie_ericksen_swim", &value);
    leslie_ericksen_swim_set(value);
    info("Self-advection parameter = %12.5e\n", value);
  }
  else {
    if (n == 1) {
      /* The user has put something which hasn't been recognised,
       * suggesting a spelling mistake */
      info("free_energy %s not recognised.\n", description);
      fatal("Please check and try again.\n");
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  map_init_rt
 *
 *  Could do more work trapping duff input keys.
 *
 *****************************************************************************/

int map_init_rt(map_t ** pmap) {

  int is_porous_media = 0;
  int ndata = 0;
  int form_in = IO_FORMAT_DEFAULT;
  int form_out = IO_FORMAT_DEFAULT;
  int grid[3] = {1, 1, 1};

  char status[BUFSIZ] = "";
  char format[BUFSIZ] = "";
  char filename[FILENAME_MAX];

  io_info_t * iohandler = NULL;
  map_t * map = NULL;

  is_porous_media = RUN_get_string_parameter("porous_media_file", filename,
					     FILENAME_MAX);
  if (is_porous_media) {

    RUN_get_string_parameter("porous_media_type", status, BUFSIZ);

    if (strcmp(status, "status_only") == 0) ndata = 0;
    if (strcmp(status, "status_with_h") == 0) ndata = 1;

    RUN_get_string_parameter("porous_media_format", format, BUFSIZ);

    if (strcmp(format, "ASCII") == 0) form_in = IO_FORMAT_ASCII_SERIAL;
    if (strcmp(format, "BINARY") == 0) form_in = IO_FORMAT_BINARY_SERIAL;

    info("\n");
    info("Porous media\n");
    info("------------\n");
    info("Porous media file requested:  %s\n", filename);
    info("Porous media file type:       %s\n", status);
    info("Porous media format (serial): %s\n", format);
  }

  map_create(ndata, &map);
  map_init_io_info(map, grid, form_in, form_out);
  map_io_info(map, &iohandler);

  if (is_porous_media) io_read_data(iohandler, filename, map);
  map_halo(map);

  *pmap = map;

  return 0;
}

/*****************************************************************************
 *
 *  symmetric_rt_initial_conditions
 *
 *  To be moved to symmtric_rt.c
 *
 *****************************************************************************/

int symmetric_rt_initial_conditions(field_t * phi) {

  int p;
  int ic, jc, kc, index;
  int ntotal[3], nlocal[3];
  int offset[3];
  double phi0, phi1;
  double noise0 = 0.1; /* Default value. */
  char value[BUFSIZ];

  assert(phi);

  coords_nlocal(nlocal);
  coords_nlocal_offset(offset);
  ntotal[X] = N_total(X);
  ntotal[Y] = N_total(Y);
  ntotal[Z] = N_total(Z);

  phi0 = get_phi0();
  RUN_get_double_parameter("noise", &noise0);

  /* Default initialisation (always?) Note serial nature of this,
   * which could be replaced. */

  for (ic = 1; ic <= ntotal[X]; ic++) {
    for (jc = 1; jc <= ntotal[Y]; jc++) {
      for (kc = 1; kc <= ntotal[Z]; kc++) {

	phi1 = phi0 + noise0*(ran_serial_uniform() - 0.5);

	/* For computation with single fluid and no noise */
	/* Only set values if within local box */
	if ( (ic > offset[X]) && (ic <= offset[X] + nlocal[X]) &&
	     (jc > offset[Y]) && (jc <= offset[Y] + nlocal[Y]) &&
	     (kc > offset[Z]) && (kc <= offset[Z] + nlocal[Z]) ) {

	    index = coords_index(ic-offset[X], jc-offset[Y], kc-offset[Z]);
	    field_scalar_set(phi, index, phi1);
	    
	}
      }
    }
  }

  p = RUN_get_string_parameter("phi_initialisation", value, BUFSIZ);

  if (p != 0 && strcmp(value, "block") == 0) {
    info("Initialisng phi as block\n");
    assert(0);
    phi_init_block(symmetric_interfacial_width());
  }

  if (p != 0 && strcmp(value, "bath") == 0) {
    info("Initialising phi for bath\n");
    assert(0);
    phi_init_bath();
  }

  if (p != 0 && strcmp(value, "drop") == 0) {
    info("Initialising droplet\n");
    symmetric_init_drop(phi, 0.4*L(X), symmetric_interfacial_width());
  }

  if (p != 0 && strcmp(value, "from_file") == 0) {
    info("Initial order parameter requested from file\n");
    info("Reading phi from serial file\n");

    assert(0);
    /* need to do something! */
  }

  if (distribution_ndist() == 2) phi_lb_from_field(phi);

  return 0;
}

int symmetric_init_drop(field_t * fphi, double radius, double xi0) {

  int nlocal[3];
  int noffset[3];
  int index, ic, jc, kc;

  double position[3];
  double centre[3];
  double phi, r, rxi0;

  assert(fphi);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);

  rxi0 = 1.0/xi0;

  centre[X] = 0.5*L(X);
  centre[Y] = 0.5*L(Y);
  centre[Z] = 0.5*L(Z);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
        position[X] = 1.0*(noffset[X] + ic) - centre[X];
        position[Y] = 1.0*(noffset[Y] + jc) - centre[Y];
        position[Z] = 1.0*(noffset[Z] + kc) - centre[Z];

        r = sqrt(dot_product(position, position));

        phi = tanh(rxi0*(r - radius));
	field_scalar_set(fphi, index, phi);
      }
    }
  }

  return 0;
}
