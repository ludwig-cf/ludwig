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
#include "noise.h"
#include "timer.h"
#include "coords_rt.h"
#include "coords.h"
#include "leesedwards.h"
#include "control.h"
#include "util.h"

#include "model.h"
#include "model_le.h"
#include "lb_model_s.h"
#include "bbl.h"

#include "collision.h"
#include "propagation.h"
#include "distribution_rt.h"
#include "collision_rt.h"

#include "map.h"
#include "wall.h"
#include "interaction.h"
#include "physics_rt.h"

#include "hydro_rt.h"

#include "io_harness.h"
#include "phi_stats.h"
#include "phi_force.h"
#include "phi_force_colloid.h"
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
#include "lc_droplet_rt.h"
#include "lc_droplet.h"
#include "fe_electro.h"
#include "fe_electro_symmetric.h"

/* Dynamics */
#include "phi_cahn_hilliard.h"
#include "leslie_ericksen.h"
#include "blue_phase_beris_edwards.h"

/* Colloids */
#include "colloids_rt.h"
#include "colloid_sums.h"
#include "colloids_halo.h"
#include "colloids_Q_tensor.h"
#include "build.h"
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
#include "psi_petsc.h"

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
  physics_t * param;        /* Physical parameters */
  lb_t * lb;                /* Lattice Botlzmann */
  hydro_t * hydro;          /* Hydrodynamic quantities */
  field_t * phi;            /* Scalar order parameter */
  field_t * p;              /* Vector order parameter */
  field_t * q;              /* Tensor order parameter */
  field_grad_t * phi_grad;  /* Gradients for phi */
  field_grad_t * p_grad;    /* Gradients for p */
  field_grad_t * q_grad;    /* Gradients for q */
  psi_t * psi;              /* Electrokinetics */
  map_t * map;              /* Site map for fluid/solid status etc. */
  noise_t * noise_rho;      /* Lattice fluctuation generator (rho) */
  noise_t * noise_phi;      /* Binary fluid noise generation (fluxes) */
  f_vare_t epsilon;         /* Variable epsilon function for Poisson solver */

  colloids_info_t * collinfo;  /* Colloid information */
  colloid_io_t * cio;          /* Colloid I/O harness */
  ewald_t * ewald;             /* Ewald sum for dipoles */
  interact_t * interact;       /* Colloid-colloid interaction handler */
  bbl_t * bbl;                 /* Bounce-back on links boundary condition */
};

static int ludwig_rt(ludwig_t * ludwig);
static int ludwig_report_momentum(ludwig_t * ludwig);
static int ludwig_colloids_update(ludwig_t * ludwig);
int free_energy_init_rt(ludwig_t * ludwig);
int map_init_rt(map_t ** map);

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

  physics_init_rt(ludwig->param);
  physics_info(ludwig->param);

  lb_run_time(ludwig->lb);
  collision_run_time(ludwig->noise_rho);
  map_init_rt(&ludwig->map);

  noise_init(ludwig->noise_rho, 0);
  ran_init();
  hydro_rt(&ludwig->hydro);

  /* PHI I/O */

  RUN_get_int_parameter_vector("default_io_grid", io_grid_default);
  for (n = 0; n < 3; n++) {
    io_grid[n] = io_grid_default[n];
  }
  RUN_get_int_parameter_vector("phi_io_grid", io_grid);

  form = IO_FORMAT_DEFAULT;
  strcpy(value, ""); /* Really need a way to get string from "form" */
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
  if (ludwig->phi) {
    symmetric_rt_initial_conditions(ludwig->phi);
    lb_ndist(ludwig->lb, &n);
    if (n == 2) phi_lb_from_field(ludwig->phi, ludwig->lb);
  }

  /* To be called before wall_init() */
  if (ludwig->psi) {
    advection_run_time();
    psi_init_rho_rt(ludwig->psi, ludwig->map);
  }

  wall_init(ludwig->lb, ludwig->map);
  colloids_init_rt(&ludwig->collinfo, &ludwig->cio, &ludwig->interact,
		   ludwig->map);
  colloids_init_ewald_rt(ludwig->collinfo, &ludwig->ewald);
  colloids_q_cinfo_set(ludwig->collinfo);

  bbl_create(ludwig->lb, &ludwig->bbl);
  bbl_active_set(ludwig->bbl, ludwig->collinfo);

  /* NOW INITIAL CONDITIONS */

  pe_subdirectory(subdirectory);

  if (get_step() == 0) {
    n = 0;
    lb_rt_initial_conditions(ludwig->lb, ludwig->param);

    RUN_get_int_parameter("LE_init_profile", &n);
    if (n != 0) lb_le_init_shear_profile(ludwig->lb);
  }
  else {
    /* Distributions */

    sprintf(filename, "%sdist-%8.8d", subdirectory, get_step());
    info("Re-starting simulation at step %d with data read from "
	 "config\nfile(s) %s\n", get_step(), filename);

    lb_io_info(ludwig->lb, &iohandler);
    io_read_data(iohandler, filename, ludwig->lb);

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
      sprintf(filename, "%sq-%8.8d", subdirectory, get_step());
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
    if (ludwig->psi) {
      psi_io_info(ludwig->psi, &iohandler);
      sprintf(filename,"%spsi-%8.8d", subdirectory, get_step());
      info("electrokinetics files(s) %s\n", filename);
      io_read_data(iohandler, filename, ludwig->psi);
    }
  }

  /* gradient initialisation for field stuff */

  if (ludwig->phi) gradient_rt_init(ludwig->phi_grad, ludwig->map);
  if (ludwig->p) gradient_rt_init(ludwig->p_grad, ludwig->map);
  if (ludwig->q) gradient_rt_init(ludwig->q_grad, ludwig->map);

  stats_rheology_init();
  stats_turbulent_init();

  /* Calibration statistics for ah required? */

  nstat = 0;
  n = RUN_get_string_parameter("calibration", filename, FILENAME_MAX);
  if (n == 1 && strcmp(filename, "on") == 0) nstat = 1;

  stats_calibration_init(ludwig->collinfo, nstat);

  /* Calibration of surface tension required (symmetric only) */

  nstat = 0;
  n = RUN_get_string_parameter("calibration_sigma", filename, FILENAME_MAX);
  if (n == 1 && strcmp(filename, "on") == 0) nstat = 1;

  if (get_step() == 0) {
    stats_sigma_init(ludwig->phi, nstat);
    lb_ndist(ludwig->lb, &n);
    if (n == 2) phi_lb_from_field(ludwig->phi, ludwig->lb); 
  }

  /* Initial Q_ab field required, apparently More GENERAL? */

  if (get_step() == 0 && ludwig->p) {
    polar_active_rt_initial_conditions(ludwig->p);
  }
  if (get_step() == 0 && ludwig->q) {
    blue_phase_rt_initial_conditions(ludwig->q);
  }

  /* Electroneutrality */

  if (get_step() == 0 && ludwig->psi) {
    info("Arranging initial charge neutrality.\n\n");
    psi_colloid_rho_set(ludwig->psi, ludwig->collinfo);
    psi_colloid_electroneutral(ludwig->psi, ludwig->collinfo);
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
  int     is_porous_media = 0;
  int     step = 0;
  int     is_subgrid = 0;
  int     is_pm = 0;
  int ncolloid = 0;
  double  fzero[3] = {0.0, 0.0, 0.0};
  double  uzero[3] = {0.0, 0.0, 0.0};
  int     im, multisteps;
  int	  flag;

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
  wall_pm(&is_porous_media);

  stats_distribution_print(ludwig->lb, ludwig->map);

  lb_ndist(ludwig->lb, &im);
  if (im == 2) {
    phi_lb_to_field(ludwig->phi, ludwig->lb);
    stats_field_info_bbl(ludwig->phi, ludwig->map, ludwig->bbl);
  }
  else {
    if (ludwig->phi) stats_field_info(ludwig->phi, ludwig->map);
  }
  if (ludwig->p)   stats_field_info(ludwig->p, ludwig->map);
  if (ludwig->q)   stats_field_info(ludwig->q, ludwig->map);
  if (ludwig->psi) {
    psi_stats_info(ludwig->psi);
  }
  ludwig_report_momentum(ludwig);



#ifdef CUDAHOST  //temporary optimisation specific to GPU code for benchmarking 
    int nlocal[3];
    coords_nlocal(nlocal);
    int nhalo = coords_nhalo();
    int Nall[3];
    Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;
    int nSites=Nall[X]*Nall[Y]*Nall[Z];
    int nFields=NVEL*ludwig->lb->ndist;
    copyToTarget(ludwig->lb->t_f,ludwig->lb->f,nSites*nFields*sizeof(double));  
#endif
  
  /* Main time stepping loop */

  info("\n");
  info("Starting time step loop.\n");
  subgrid_on(&is_subgrid);

  while (next_step()) {


    TIMER_start(TIMER_STEPS);
    step = get_step();
    if (ludwig->hydro) hydro_f_zero(ludwig->hydro, fzero);

    colloids_info_ntotal(ludwig->collinfo, &ncolloid);
    ludwig_colloids_update(ludwig);

    /* Order parameter gradients */

    TIMER_start(TIMER_PHI_GRADIENTS);

    /* if symmetric_lb store phi to field */

    lb_ndist(ludwig->lb, &im);

    if (im == 2) phi_lb_to_field(ludwig->phi, ludwig->lb);

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

    /* Electrokinetics (including electro/symmetric requiring above
     * gradients for phi) */

    if (ludwig->psi) {
      /* Set charge distribution according to updated map */     
      psi_colloid_rho_set(ludwig->psi, ludwig->collinfo);

      /* Poisson solve */

      if(get_step() % psi_skipsteps(ludwig->psi) == 0){
	TIMER_start(TIMER_ELECTRO_POISSON);
#ifdef PETSC
	psi_petsc_solve(ludwig->psi, ludwig->epsilon);
#else
	psi_sor_solve(ludwig->psi, ludwig->epsilon);
#endif
	TIMER_stop(TIMER_ELECTRO_POISSON);
      }

      if (ludwig->hydro) {
	TIMER_start(TIMER_HALO_LATTICE);
	hydro_u_halo(ludwig->hydro);
	TIMER_stop(TIMER_HALO_LATTICE);
      }

      /* Time splitting for high electrokinetic diffusions in Nernst Planck */

      psi_multisteps(ludwig->psi, &multisteps);
      
      for (im = 0; im < multisteps; im++) {

	TIMER_start(TIMER_HALO_LATTICE);
	psi_halo_psi(ludwig->psi);
	psi_halo_rho(ludwig->psi);
	TIMER_stop(TIMER_HALO_LATTICE);

	/* Force for this step before update. Note that nhalo = 1
	 * is indicating grad mu method and nhalo = 2 the divergence
	 * method. Once per large time step with dt = 1.0. */

	if (im == 0) {

	  TIMER_start(TIMER_FORCE_CALCULATION);
	  psi_force_is_divergence(&flag);

          /* Force input as body force and momentum correction */
	  if (flag == 0) {
	    psi_force_gradmu(ludwig->psi, ludwig->hydro,
				  ludwig->map, ludwig->collinfo);
	    hydro_correct_momentum(ludwig->hydro);
	  }

          /* Force calculation as divergence of stress tensor */
	  if (flag == 1) {
	    psi_force_divstress_d3qx(ludwig->psi, ludwig->hydro,
				  ludwig->map, ludwig->collinfo);
	  }
	  TIMER_stop(TIMER_FORCE_CALCULATION);

	}

	TIMER_start(TIMER_ELECTRO_NPEQ);
	nernst_planck_driver_d3qx(ludwig->psi, ludwig->hydro, ludwig->map, ludwig->collinfo);
	TIMER_stop(TIMER_ELECTRO_NPEQ);

      }

      nernst_planck_adjust_multistep(ludwig->psi);

      if (is_statistics_step()) info("%d multisteps\n",im);

      psi_sor_offset(ludwig->psi);

    }

    /* order parameter dynamics (not if symmetric_lb) */

    lb_ndist(ludwig->lb, &im);
    if (im == 2) {
      /* dynamics are dealt with at the collision stage (below) */
    }
    else {

      TIMER_start(TIMER_FORCE_CALCULATION);

      if (ludwig->psi) {
	/* Force in electrokinetic models is computed above */
      }
      else {
	if (ncolloid == 0) {
	  /* Force calculation as divergence of stress tensor */
	  phi_force_calculation(ludwig->phi, ludwig->hydro);
	  /* LC-droplet requires partial body force input and momentum correction */
	  if (ludwig->q && ludwig->phi) {
	    lc_droplet_bodyforce(ludwig->hydro);
	    hydro_correct_momentum(ludwig->hydro);
	  }
	}
	else {
	  phi_force_colloid(ludwig->collinfo, ludwig->hydro, ludwig->map);
	}
      }

      TIMER_stop(TIMER_FORCE_CALCULATION);

      TIMER_start(TIMER_ORDER_PARAMETER_UPDATE);

      if (ludwig->phi) phi_cahn_hilliard(ludwig->phi, ludwig->hydro,
					 ludwig->map, ludwig->noise_phi);
      if (ludwig->p) leslie_ericksen_update(ludwig->p, ludwig->hydro);
      if (ludwig->q) {
	if (ludwig->hydro) hydro_u_halo(ludwig->hydro);
	colloids_fix_swd(ludwig->collinfo, ludwig->hydro, ludwig->map);
	blue_phase_beris_edwards(ludwig->q, ludwig->hydro,
				 ludwig->map, ludwig->noise_rho);
      }

      TIMER_stop(TIMER_ORDER_PARAMETER_UPDATE);
    }


    if (ludwig->hydro) {
      /* Zero velocity field here, as velocity at collision is used
       * at next time step for FD above. Strictly, we only need to
       * do this if velocity output is required in presence of
       * colloids to present non-zero u inside particles. */
      hydro_u_zero(ludwig->hydro, uzero);

      /* Collision stage */

      TIMER_start(TIMER_COLLIDE);
      lb_collide(ludwig->lb, ludwig->hydro, ludwig->map, ludwig->noise_rho);
      TIMER_stop(TIMER_COLLIDE);

      /* Boundary conditions */

      lb_le_apply_boundary_conditions(ludwig->lb);

      TIMER_start(TIMER_HALO_LATTICE);
      lb_halo(ludwig->lb);
      TIMER_stop(TIMER_HALO_LATTICE);

      /* Colloid bounce-back applied between collision and
       * propagation steps. */

      if (is_subgrid) {
	subgrid_update(ludwig->collinfo, ludwig->hydro);
      }
      else {
	TIMER_start(TIMER_BBL);
	wall_set_wall_velocity(ludwig->lb);
	bounce_back_on_links(ludwig->bbl, ludwig->lb, ludwig->collinfo);
	wall_bounce_back(ludwig->lb, ludwig->map);
	TIMER_stop(TIMER_BBL);
      }
    }
    else {
      /* No hydrodynamics, but update colloids in response to
       * external forces. */

      bbl_update_colloids(ludwig->bbl, ludwig->collinfo);
    }

    /* There must be no halo updates between bounce back
     * and propagation, as the halo regions are active */

    TIMER_start(TIMER_PROPAGATE);
    lb_propagation(ludwig->lb);
    TIMER_stop(TIMER_PROPAGATE);

    TIMER_stop(TIMER_STEPS);

    
#ifdef CUDAHOST //temporary optimisation specific to GPU code for benchmarking
    copyFromTarget(ludwig->lb->f,ludwig->lb->t_f,nSites*nFields*sizeof(double));
#endif




    /* Configuration dump */

    if (is_config_step()) {
      info("Writing distribution output at step %d!\n", step);
      sprintf(filename, "%sdist-%8.8d", subdirectory, step);
      lb_io_info(ludwig->lb, &iohandler);
      io_write_data(iohandler, filename, ludwig->lb);
    }

    /* is_measurement_step() is here to prevent 'breaking' old input
     * files; it should really be removed. */

    if (is_config_step() || is_measurement_step() || is_colloid_io_step()) {
      if (ncolloid > 0) {
	info("Writing colloid output at step %d!\n", step);
	sprintf(filename, "%s%s%8.8d", subdirectory, "config.cds", step);
	colloid_io_write(ludwig->cio, filename);
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
	info("Writing q file at step %d!\n", step);
	sprintf(filename,"%sq-%8.8d", subdirectory, step);
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
      stats_rheology_stress_profile_accumulate(ludwig->lb, ludwig->hydro);
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

    if (fe_set() && is_fed_output_step()) {
      fed_io_info(&iohandler);
      info("Writing free energy density output at step %d!\n", step);
      sprintf(filename, "%sfed-%8.8d", subdirectory, step);
      io_write_data(iohandler, filename, NULL);
    }

    /* Print progress report */

    if (is_statistics_step()) {

      stats_distribution_print(ludwig->lb, ludwig->map);
      lb_ndist(ludwig->lb, &im);
      if (im == 2) {
	phi_lb_to_field(ludwig->phi, ludwig->lb);
	stats_field_info_bbl(ludwig->phi, ludwig->map, ludwig->bbl);
      }
      else {
	if (ludwig->phi) stats_field_info(ludwig->phi, ludwig->map);
      }
      if (ludwig->p)   stats_field_info(ludwig->p, ludwig->map);
      if (ludwig->q)   stats_field_info(ludwig->q, ludwig->map);

      if (ludwig->psi) {
	double psi_zeta;
	psi_colloid_rho_set(ludwig->psi, ludwig->collinfo);
	psi_stats_info(ludwig->psi);
	/* Zeta potential for one colloid only to follow psi_stats()*/
	psi_colloid_zetapotential(ludwig->psi, ludwig->collinfo, &psi_zeta);
	if (ncolloid == 1) info("[psi_zeta] %14.7e\n",  psi_zeta);
      }

      stats_free_energy_density(ludwig->q, ludwig->map, ncolloid);
//      blue_phase_stats(ludwig->q, ludwig->q_grad, ludwig->map, step);
      ludwig_report_momentum(ludwig);

      if (ludwig->hydro) {
	wall_pm(&is_pm);
	stats_velocity_minmax(ludwig->hydro, ludwig->map, is_pm);
      }

      lb_collision_stats_kt(ludwig->lb, ludwig->noise_rho, ludwig->map);

      info("\nCompleted cycle %d\n", step);
    }

    stats_calibration_accumulate(ludwig->collinfo, step, ludwig->hydro,
				 ludwig->map);




    /* Next time step */
  }

  /* To prevent any conflict between the last regular dump, and
   * a final dump, there's a barrier here. */

  MPI_Barrier(pe_comm()); 

  /* Dump the final configuration if required. */

  if (is_config_at_end()) {
    sprintf(filename, "%sdist-%8.8d", subdirectory, step);
    lb_io_info(ludwig->lb, &iohandler);
    io_write_data(iohandler, filename, ludwig->lb);
    sprintf(filename, "%s%s%8.8d", subdirectory, "config.cds", step);

    if (ncolloid > 0) colloid_io_write(ludwig->cio, filename);

    if (ludwig->phi) {
      field_io_info(ludwig->phi, &iohandler);
      info("Writing phi file at step %d!\n", step);
      sprintf(filename,"%sphi-%8.8d", subdirectory, step);
      io_write_data(iohandler, filename, ludwig->phi);
    }

    if (ludwig->q) {
      field_io_info(ludwig->q, &iohandler);
      info("Writing q file at step %d!\n", step);
      sprintf(filename,"%sq-%8.8d", subdirectory, step);
      io_write_data(iohandler, filename, ludwig->q);
    }
    /* Only strictly required if have order parameter dynamics */ 
    if (ludwig->hydro) {
      hydro_io_info(ludwig->hydro, &iohandler);
      info("Writing velocity output at step %d!\n", step);
      sprintf(filename, "%svel-%8.8d", subdirectory, step);
      io_write_data(iohandler, filename, ludwig->hydro);
    }
    if (ludwig->psi) {
      psi_io_info(ludwig->psi, &iohandler);
      info("Writing psi file at step %d!\n", step);
      sprintf(filename,"%spsi-%8.8d", subdirectory, step);
      io_write_data(iohandler, filename, ludwig->psi);
    }
  }

  /* Shut down cleanly. Give the timer statistics. Finalise PE. */
#ifdef PETSC
  psi_petsc_finish();
#endif

  stats_rheology_finish();
  stats_turbulent_finish();
  stats_calibration_finish();

  wall_finish();

  if (ludwig->phi_grad) field_grad_free(ludwig->phi_grad);
  if (ludwig->p_grad)   field_grad_free(ludwig->p_grad);
  if (ludwig->q_grad)   field_grad_free(ludwig->q_grad);
  if (ludwig->phi)      field_free(ludwig->phi);
  if (ludwig->p)        field_free(ludwig->p);
  if (ludwig->q)        field_free(ludwig->q);

  bbl_free(ludwig->bbl);
  colloids_info_free(ludwig->collinfo);

  if (ludwig->noise_phi) noise_free(ludwig->noise_phi);
  if (ludwig->noise_rho) noise_free(ludwig->noise_rho);

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
  int ncolloid;
  int is_pm;

  double g[3];         /* Fluid momentum (total) */
  double gc[3];        /* Colloid momentum (total) */
  double gwall[3];     /* Wall momentum (for accounting purposes only) */
  double gtotal[3];

  wall_pm(&is_pm);

  for (n = 0; n < 3; n++) {
    gtotal[n] = 0.0;
    g[n] = 0.0;
    gc[n] = 0.0;
    gwall[n] = 0.0;
  }

  stats_distribution_momentum(ludwig->lb, ludwig->map, g);
  stats_colloid_momentum(ludwig->collinfo, gc);
  colloids_info_ntotal(ludwig->collinfo, &ncolloid);

  if (wall_present() || is_pm) wall_net_momentum(gwall);

  for (n = 0; n < 3; n++) {
    gtotal[n] = g[n] + gc[n] + gwall[n];
  }

  info("\n");
  info("Momentum - x y z\n");
  info("[total   ] %14.7e %14.7e %14.7e\n", gtotal[X], gtotal[Y], gtotal[Z]);
  info("[fluid   ] %14.7e %14.7e %14.7e\n", g[X], g[Y], g[Z]);
  if (ncolloid > 0) {
    info("[colloids] %14.7e %14.7e %14.7e\n", gc[X], gc[Y], gc[Z]);
  }
  if (wall_present() || is_pm) {
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
  int nk;
  int ngrad;
  int nhalo;
  int noise_on = 0;
  double value;
  char description[BUFSIZ];

  assert(ludwig);

  physics_ref(&ludwig->param);

  lb_create(&ludwig->lb);

  noise_create(&ludwig->noise_rho);

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
    le_info();
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
    le_info();

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
    physics_mobility_set(value);
    info("Mobility M            = %12.5e\n", value);

    /* Order parameter noise */

    RUN_get_int_parameter("fd_phi_fluctuations", &noise_on);
    info("Order parameter noise = %3s\n", (noise_on == 0) ? "off" : " on");

    if (noise_on) {
      noise_create(&ludwig->noise_phi);
      noise_init(ludwig->noise_phi, 0);
      noise_present_set(ludwig->noise_phi, NOISE_PHI, noise_on);
      if (nhalo != 3) fatal("Fluctuations: use symmetric_noise\n");
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

    lb_ndist_set(ludwig->lb, 2);

    nf = 1;      /* 1 scalar order parameter */
    nhalo = 1;   /* Require one piont for LB. */
    ngrad = 2;   /* \nabla^2 required */

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();
    le_info();

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
    physics_mobility_set(value);
    info("Mobility M            = %12.5e\n", value);

  }
  else if (strcmp(description, "brazovskii") == 0) {

    /* Brazovskii (always finite difference). */

    nf = 1;      /* 1 scalar order parameter */
    nhalo = 3;   /* Required for stress diveregnce. */
    ngrad = 4;   /* (\nabla^2)^2 required */

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();
    le_info();

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
    physics_mobility_set(value);
    info("Mobility M            = %12.5e\n", value);

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

    /* Liquid crystal (always finite difference). */

    nf = NQAB;   /* Tensor order parameter */
    nhalo = 2;   /* Required for stress diveregnce. */
    ngrad = 2;   /* (\nabla^2) required */

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();
    le_info();

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
      physics_lc_gamma_rot_set(value);
      info("Rotational diffusion const = %14.7e\n", value);
    }

    p = 0;
    RUN_get_int_parameter("lc_noise", &p);
    noise_present_set(ludwig->noise_rho, NOISE_QAB, p);
    info("LC fluctuations:           =  %s\n", (p == 0) ? "off" : "on");

  }
  else if (strcmp(description, "polar_active") == 0) {

    /* Polar active. */

    nf = NVECTOR;/* Vector order parameter */
    nhalo = 2;   /* Required for stress diveregnce. */
    ngrad = 2;   /* (\nabla^2) required */

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();
    le_info();

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
  else if(strcmp(description, "lc_droplet") == 0){
    
    /* liquid crystal droplet */
    info("\nliquid crystal droplet free energy chosen\n");
    
    /* first do the symmetric */
    nf = 1;      /* 1 scalar order parameter */
    nhalo = 2;   /* Require stress divergence. */
    ngrad = 3;   /* \nabla^2 and d_a d_b required */

    /* Noise requires additional stencil point for Cahn Hilliard */
    if (strcmp(description, "symmetric_noise") == 0) {
      nhalo = 3;
    }

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();
    le_info();
        
    field_create(nf, "phi", &ludwig->phi);
    field_init(ludwig->phi, nhalo);
    field_grad_create(ludwig->phi, ngrad, &ludwig->phi_grad);

    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");
    symmetric_run_time();
    
    lc_droplet_phi_set(ludwig->phi, ludwig->phi_grad);

    info("\n");
    info("Using Cahn-Hilliard finite difference solver.\n");

    RUN_get_double_parameter("mobility", &value);
    physics_mobility_set(value);
    info("Mobility M            = %12.5e\n", value);

    /* Force */

    p = 1; /* Default is to use divergence method */
    RUN_get_int_parameter("fd_force_divergence", &p);
    info("Force calculation:      %s\n",
         (p == 0) ? "phi grad mu method" : "divergence method");
    phi_force_divergence_set(p);
  
    /* Liquid crystal part */
    nf = NQAB;   /* Tensor order parameter */
    nhalo = 2;   /* Required for stress diveregnce. */
    ngrad = 2;   /* (\nabla^2) required */
    
    field_create(nf, "q", &ludwig->q);
    field_init(ludwig->q, nhalo);
    field_grad_create(ludwig->q, ngrad, &ludwig->q_grad);

    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");

    blue_phase_run_time();
    lc_droplet_q_set(ludwig->q, ludwig->q_grad);

    info("\n");
    info("Using Beris-Edwards solver:\n");

    p = RUN_get_double_parameter("lc_Gamma", &value);
    if (p != 0) {
      physics_lc_gamma_rot_set(value);
      info("Rotational diffusion constant = %12.5e\n", value);
    }
    
    /* finalise with the droplet specific init*/
    lc_droplet_run_time();

  }
  else if(strcmp(description, "fe_electro") == 0) {

    nk = 2;    /* Number of charge densities always 2 for now */

    /* Single fluid electrokinetic free energy */

    /* Default method is divergence of stress tensor */
    p = 1;
    RUN_get_int_parameter("fd_force_divergence", &p);
    psi_force_divergence_set(p);

    if (p == 1) phi_force_required_set(1);

    if (p == 0) nhalo = 1;
    if (p == 1) nhalo = 2;

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();
    le_info();

    info("\n");
    info("Free energy details\n");
    info("-------------------\n\n");
    info("Electrokinetics (single fluid) selected\n");

    info("\n");
    info("Parameters:\n");

    psi_create(nk, &ludwig->psi);
    psi_init_param_rt(ludwig->psi);

    /* P should still hold ... */
    info("Force calculation:          %s\n",
         (p == 0) ? "psi grad mu method" : "Divergence method");

#ifdef PETSC
    psi_petsc_init(ludwig->psi, ludwig->epsilon);
#endif

    /* Free energy object */

    fe_electro_create(ludwig->psi);

  }
  else if(strcmp(description, "fe_electro_symmetric") == 0) {

    double e1, e2;
    double mu[2];
    double lbjerrum;

    /* Binary fluid plus electrokinetics */

    nf = 1;      /* Single scalar order parameter phi */
    nk = 2;      /* Two charge species */
    nhalo = 2;   /* Require stress divergence. */
    ngrad = 2;   /* \nabla^2 phi */

    /* Default method is divergence of stress tensor */
    p = 1;
    RUN_get_int_parameter("fd_force_divergence", &p);
    psi_force_divergence_set(p);

    if (p == 1) phi_force_required_set(1);

    info("Force calculation:          %s\n",
         (p == 0) ? "psi grad mu method" : "Divergence method");

    /* First, the symmetric part. */

    coords_nhalo_set(nhalo);
    coords_run_time();
    le_init();
    le_info();

    field_create(nf, "phi", &ludwig->phi);
    field_init(ludwig->phi, nhalo);
    field_grad_create(ludwig->phi, ngrad, &ludwig->phi_grad);

    info("\n");
    info("Charged binary fluid 'Electrosymmetric' free energy\n");
    info("---------------------------------------------------\n");

    info("\n");
    info("Symmetric part\n");
    info("--------------\n\n");
    symmetric_run_time();
    symmetric_phi_set(ludwig->phi, ludwig->phi_grad);

    info("\n");
    info("Using Cahn-Hilliard finite difference solver.\n");

    RUN_get_double_parameter("mobility", &value);
    physics_mobility_set(value);
    info("Mobility M            = %12.5e\n", value);

    /* Electrokinetic part */

    info("\n");
    info("Electrokinetic part\n");
    info("-------------------\n\n");

    info("Parameters:\n");

    psi_create(nk, &ludwig->psi);
    psi_init_param_rt(ludwig->psi);

#ifdef PETSC
    psi_petsc_init(ludwig->psi, ludwig->epsilon);
#endif

    /* Coupling part */

    info("\n");
    info("Coupling part\n");
    info("-------------\n");

    fe_es_create(ludwig->phi, ludwig->phi_grad, ludwig->psi);

    /* Dielectric contrast */

    /* Call permittivities, e1=e2 has been set as default */
    psi_epsilon(ludwig->psi, &e1);
    psi_epsilon2(ludwig->psi, &e2);

    /* Read the second permittivity */
    n = RUN_get_double_parameter("electrosymmetric_epsilon2", &e2);
    if (n == 1) psi_epsilon2_set(ludwig->psi, e2);

    fe_es_epsilon_set(e1, e2);

    /* Solvation free energy difference: nk = 2 */

    mu[0] = 0.0;
    mu[1] = 0.0;

    RUN_get_double_parameter("electrosymmetric_delta_mu0", mu);
    RUN_get_double_parameter("electrosymmetric_delta_mu1", mu + 1);

    fe_es_deltamu_set(nk, mu);

    psi_bjerrum_length(ludwig->psi, &lbjerrum);

    info("Second permittivity:      %15.7e\n", e2);
    info("Dielectric average:       %15.7e\n", 0.5*(e1 + e2));
    info("Dielectric contrast:      %15.7e\n", (e1-e2)/(e1+e2));
    info("Average Bjerrum length:   %15.7e\n", lbjerrum);
    info("Solvation dmu species 0:  %15.7e\n", mu[0]);
    info("Solvation dmu species 1:  %15.7e\n", mu[1]);

    /* f_vare_t function */

    info("Poisson solver:           %15s\n",
	 (e1 == e2) ? "uniform" : "heterogeneous");
    if (e1 != e2) ludwig->epsilon = fe_es_var_epsilon;

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

    RUN_get_int_parameter_vector("porous_media_io_grid", grid);

    info("\n");
    info("Porous media\n");
    info("------------\n");
    info("Porous media file requested:  %s\n", filename);
    info("Porous media file type:       %s\n", status);
    info("Porous media format (serial): %s\n", format);
    info("Porous media io grid:         %d %d %d\n", grid[X], grid[Y], grid[Z]);
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
 *  ludwig_colloids_update
 *
 *  Driver for update called at start of timestep loop.
 *
 *****************************************************************************/

int ludwig_colloids_update(ludwig_t * ludwig) {

  int ndist;
  int ncolloid;
  int iconserve;         /* switch for finite-difference conservation */
  int is_subgrid = 0;    /* subgrid particle switch */

  assert(ludwig);

  colloids_info_ntotal(ludwig->collinfo, &ncolloid);
  if (ncolloid == 0) return 0;

  subgrid_on(&is_subgrid);

  lb_ndist(ludwig->lb, &ndist);
  iconserve = (ludwig->psi || (ludwig->phi && ndist == 1));

  TIMER_start(TIMER_PARTICLE_HALO);

  colloids_info_position_update(ludwig->collinfo);
  colloids_info_update_cell_list(ludwig->collinfo);
  colloids_halo_state(ludwig->collinfo);
  colloids_info_update_lists(ludwig->collinfo);

  TIMER_stop(TIMER_PARTICLE_HALO);

  if (is_subgrid) {
    interact_compute(ludwig->interact, ludwig->collinfo, ludwig->map,
		     ludwig->psi, ludwig->ewald);
    subgrid_force_from_particles(ludwig->collinfo, ludwig->hydro);    
  }
  else {

    /* Removal or replacement of fluid requires a lattice halo update */

    TIMER_start(TIMER_HALO_LATTICE);
    lb_halo(ludwig->lb);
    TIMER_stop(TIMER_HALO_LATTICE);

    TIMER_start(TIMER_FREE1);
    if (iconserve && ludwig->phi) field_halo(ludwig->phi);
    if (iconserve && ludwig->psi) psi_halo_rho(ludwig->psi);
    TIMER_stop(TIMER_FREE1);

    TIMER_start(TIMER_REBUILD);

    build_update_map(ludwig->collinfo, ludwig->map);
    build_remove_replace(ludwig->collinfo, ludwig->lb, ludwig->phi, ludwig->p,
			 ludwig->q, ludwig->psi);
    build_update_links(ludwig->collinfo, ludwig->map);

    TIMER_stop(TIMER_REBUILD);

    TIMER_start(TIMER_FREE1);
    if (iconserve) {
      colloid_sums_halo(ludwig->collinfo, COLLOID_SUM_CONSERVATION);
      build_conservation(ludwig->collinfo, ludwig->phi, ludwig->psi);
    }
    TIMER_stop(TIMER_FREE1);

    TIMER_start(TIMER_FORCES);

    interact_compute(ludwig->interact, ludwig->collinfo, ludwig->map,
		     ludwig->psi, ludwig->ewald);

    TIMER_stop(TIMER_FORCES);
  }

  return 0;
}
