/*****************************************************************************
 *
 *  colloids_rt.c
 *
 *  Run time initialisation of colloid information.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014 The University of Edinburgh
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "control.h"

#include "lubrication.h"
#include "pair_ss_cut.h"
#include "pair_lj_cut.h"
#include "pair_yukawa.h"

#include "colloids_halo.h"
#include "colloids_init.h"
#include "colloid_io_rt.h"
#include "colloids_rt.h"

#include "bbl.h"
#include "build.h"
#include "subgrid.h"

int lubrication_init(interact_t * inter);
int pair_ss_cut_init(interact_t * inter);
int pair_yukawa_init(interact_t * inter);
int pair_lj_cut_init(interact_t * inter);

/*****************************************************************************
 *
 *  colloids_init_rt
 * 
 *  Driver routine for colloid initialisation.
 *
 *  (a) Read source (input file, or external file)
 *  (b) Work out how many colloids are present, if any
 *  (c) Initialise objects always required
 *  (d) Initialise single-particle information
 *  (e) Initialise pairwise etc interactions
 *
 *****************************************************************************/

int colloids_init_rt(colloids_info_t ** pinfo, colloid_io_t ** pcio,
		     interact_t ** interact, map_t * map) {

  int n, ncheck;
  int init_from_file;
  int init_random;
  int ncell[3] = {2, 2,2};
  int gravity;
  char filename[FILENAME_MAX];
  char stub[FILENAME_MAX];
  char subdirectory[FILENAME_MAX];
  char keyvalue[128];
  double dh = 0.0;
  double width;
  double g[3] = {0.0, 0.0, 0.0};

  colloid_state_t * state0;

  /* Default position: no colloids */

  init_random = 0;
  init_from_file = 0;

  RUN_get_string_parameter("colloid_init", keyvalue, 128);

  if (strcmp(keyvalue, "random") == 0) init_random = 1;
  if (strcmp(keyvalue, "from_file") == 0) init_from_file = 1;

  if ((init_random || init_from_file)) {

    /* Set the list list width. */
    n = RUN_get_double_parameter("colloid_cell_min", &width);
    if (n == 0) {
      info("Must set a minimum cell width for colloids colloid_cell_min\n");
      fatal("Stop.\n");
    }
    ncell[X] = L(X) / (cart_size(X)*width);
    ncell[Y] = L(Y) / (cart_size(Y)*width);
    ncell[Z] = L(Z) / (cart_size(Z)*width);

    if (ncell[X] < 2 || ncell[Y] < 2 || ncell[Z] < 2) {
      info("[Error  ] Please check the cell width (cell_list_lmin).\n");
      fatal("[Stop] Must be at least two cells in each direction.\n");
    }
  }

  info("\n");
  info("Colloid information\n");
  info("-------------------\n");

  colloids_info_create(ncell, pinfo);

  if (init_random || init_from_file) {

    pe_subdirectory(subdirectory);

    /* Initialisation section. */

    colloid_io_run_time(*pinfo, pcio);

    if (init_from_file) {
      strcpy(stub, "config.cds.init");
      RUN_get_string_parameter("colloid_file_stub", stub, FILENAME_MAX);

      if (get_step() == 0) {
	sprintf(filename, "%s%s", subdirectory, stub);
      }
      else {
	sprintf(filename, "%s%s%8.8d", subdirectory, "config.cds", get_step());
      }

      colloid_io_read(*pcio, filename);
    }

    if (init_random) {
      state0 = (colloid_state_t *) calloc(1, sizeof(colloid_state_t));
      assert(state0 != NULL);

      /* Minimal error testing here at the moment. */
      RUN_get_int_parameter("colloid_random_no", &n);
      RUN_get_double_parameter("colloid_random_a0", &state0->a0);
      RUN_get_double_parameter("colloid_random_ah", &state0->ah);
      RUN_get_double_parameter_vector("colloid_random_r0", state0->r);
      RUN_get_double_parameter_vector("colloid_random_v0", state0->v);
      RUN_get_double_parameter_vector("colloid_random_w0", state0->w);
      RUN_get_double_parameter_vector("colloid_random_s0", state0->s);
      RUN_get_double_parameter_vector("colloid_random_m0", state0->m);
      RUN_get_double_parameter("colloid_random_c", &state0->c);
      RUN_get_double_parameter("colloid_random_h", &state0->h);
      RUN_get_double_parameter("colloid_random_b1", &state0->b1);
      RUN_get_double_parameter("colloid_random_b2", &state0->b2);
      RUN_get_double_parameter("colloid_random_dh", &dh);

      RUN_get_int_parameter("colloid_random_isfixedr", &state0->isfixedr);
      RUN_get_int_parameter("colloid_random_isfixedv", &state0->isfixedv);

      RUN_get_double_parameter("colloid_random_q0", &state0->q0);
      RUN_get_double_parameter("colloid_random_q1", &state0->q1);
      RUN_get_double_parameter("colloid_random_epsilon", &state0->epsilon);

      RUN_get_string_parameter("colloid_type", keyvalue, 128);
      if (strcmp(keyvalue, "active") == 0) state0->type = COLLOID_TYPE_ACTIVE;
      if (strcmp(keyvalue, "subgrid") == 0) state0->type = COLLOID_TYPE_SUBGRID;

      colloids_init_random(*pinfo, n, state0, dh);
      colloids_info_ntotal(*pinfo, &ncheck);

      info("Requested   %d colloid%s from input\n", n, (n > 1) ? "s" : "");
      info("Initialised %d colloid%s\n", ncheck, (ncheck == 1) ? "" : "s");
      info("Colloid  radius a0 = %le\n", state0->a0);
      info("Hydrodyn radius ah = %le\n", state0->ah);
      info("Colloid charges q0 = %le    q1 = %le\n", state0->q0, state0->q1);

      free(state0);
    }

    /* At this point, we should know number of colloids */

    interact_create(interact);
    assert(*interact);

    lubrication_init(*interact);
    pair_ss_cut_init(*interact);
    pair_lj_cut_init(*interact);
    pair_yukawa_init(*interact);

    colloids_init_halo_range_check(*pinfo);
    interact_range_check(*interact, *pinfo);

    colloids_info_map_init(*pinfo);

    /* Transfer any particles in the halo regions, initialise the
     * colloid map and build the particles for the first time. */

    colloids_halo_state(*pinfo);

    /* TODO. Instead, look to see what particle types are present. */
    /* Active */
    RUN_get_string_parameter("colloid_type", keyvalue, 128);
    if (strcmp(keyvalue, "active") == 0) bbl_active_on_set();

    if (strcmp(keyvalue, "subgrid") == 0) {
      subgrid_on_set();
    }
    else  {
      build_update_map(*pinfo, map);
      build_update_links(*pinfo, map);
    }

    /* Information */

    n = RUN_get_double_parameter_vector("colloid_gravity", g);
    if (n != 0) physics_fgrav_set(g);
    gravity = 0;
    gravity = (g[X] != 0.0 || g[Y] != 0.0 || g[Z] != 0.0);

    if (gravity) {
      info("Sedimentation force on:   yes\n");
      info("Sedimentation force:      %14.7e %14.7e %14.7e", g[X], g[Y], g[Z]);
    }

    info("\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_init_ewald_rt
 *
 *****************************************************************************/

int colloids_init_ewald_rt(colloids_info_t * cinfo, ewald_t ** pewald) {

  int ncolloid;
  int iarg;
  int is_required = 0;
  double mu;               /* Dipole strength */
  double rc;               /* Real space cut off */

  assert(cinfo);

  colloids_info_ntotal(cinfo, &ncolloid);
  if (ncolloid < 2) return 0;

  RUN_get_int_parameter("ewald_sum", &is_required);

  if (is_required) {

    iarg = RUN_get_double_parameter("ewald_mu", &mu);
    if (iarg == 0) fatal("Ewald sum requires dipole strength input\n");
    iarg = RUN_get_double_parameter("ewald_rc", &rc);
    if (iarg == 0) fatal("Ewald sum requires a real space cut off\n");

    ewald_create(mu, rc, cinfo, pewald);
    assert(*pewald);
    ewald_info(*pewald); 
  }

  return 0;
}

/*****************************************************************************
 *
 *  lubrication_init_rt
 *
 *  Initialise the parameters for corrections to the lubrication
 *  forces between colloids.
 *
 *****************************************************************************/

int lubrication_init(interact_t * inter) {

  int n, on = 0;
  double rcnorm = 0.0;
  double rctang = 0.0;
  lubr_t * lubr = NULL;

  n = RUN_get_int_parameter("lubrication_on", &on);

  if (on) {
    info("\nColloid-colloid lubrication corrections\n");
    info("Lubrication corrections are switched on\n");
    n = RUN_get_double_parameter("lubrication_normal_cutoff", &rcnorm);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Normal force cutoff is %f\n", rcnorm);
    
    n = RUN_get_double_parameter("lubrication_tangential_cutoff", &rctang);
    info((n == 0) ? "[Default] " : "[User   ] ");
    info("Tangential force cutoff is %f\n", rctang);

    lubrication_create(&lubr);
    lubrication_rch_set(lubr, LUBRICATION_SS_FNORM, rcnorm);
    lubrication_rch_set(lubr, LUBRICATION_SS_FTANG, rctang);
    lubrication_register(lubr, inter);
  }

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_init
 *
 *  Initialise the parameters for the soft-sphere interaction between
 *  colloids.
 *
 *****************************************************************************/

int pair_ss_cut_init(interact_t * inter) {

  int n;
  int on = 0;
  double epsilon ;
  double sigma;
  int nu;
  double kt;
  double cutoff;

  pair_ss_cut_t * pair = NULL;

  physics_kt(&kt);

  n = RUN_get_int_parameter("soft_sphere_on", &on);

  if (on) {

    n = RUN_get_double_parameter("soft_sphere_epsilon", &epsilon);
    if (n == 0) fatal("Please define soft_sphere_epsilon in input\n");

    n = RUN_get_double_parameter("soft_sphere_sigma", &sigma);
    if (n == 0) fatal("Please define soft_sphere_sigme in input\n");

    n = RUN_get_int_parameter("soft_sphere_nu", &nu);
    if (n == 0) fatal("Please check soft_sphere_nu appears in input\n");
    if (nu <= 0) fatal("Please check soft_sphere_nu is positive\n");

    n = RUN_get_double_parameter("soft_sphere_cutoff", &cutoff);
    if (n == 0) fatal("Please check soft_sphere_cutoff appears in input\n");

    pair_ss_cut_create(&pair);
    pair_ss_cut_param_set(pair, epsilon, sigma, nu, cutoff);
    pair_ss_cut_register(pair, inter);
    pair_ss_cut_info(pair);
  }

  return 0;
}

/*****************************************************************************
 *
 *  pair_yukawa_init
 *
 *****************************************************************************/

int pair_yukawa_init(interact_t * interact) {

  int n;
  int on = 0;
  double epsilon;
  double kappa;
  double cutoff;

  pair_yukawa_t * yukawa = NULL;

  assert(interact);

  n = RUN_get_int_parameter("yukawa_on", &on);

  if (on) {
    n = RUN_get_double_parameter("yukawa_epsilon", &epsilon);
    if (n == 0) fatal("Please check yukawa_epsilon appears in input\n");
    n = RUN_get_double_parameter("yukawa_kappa", &kappa);
    if (n == 0) fatal("Please check yukawa_kappa appears in input");
    n = RUN_get_double_parameter("yukawa_cutoff", &cutoff);
    if (n == 0) fatal("Please check yukawa_cutoff appears in input\n");

    pair_yukawa_create(&yukawa);
    pair_yukawa_param_set(yukawa, epsilon, kappa, cutoff);
    pair_yukawa_register(yukawa, interact);
    pair_yukawa_info(yukawa);
  }

  return 0;
}

/*****************************************************************************
 *
 *  pair_lj_cut_init
 *
 *****************************************************************************/

int pair_lj_cut_init(interact_t * inter) {

  int n;
  int on = 0;
  double epsilon;
  double sigma;
  double cutoff;

  pair_lj_cut_t * lj = NULL;

  assert(inter);

  n = RUN_get_int_parameter("lennard_jones_on", &on);

  if (on) {
    n = RUN_get_double_parameter("lj_epsilon", &epsilon);
    if (n == 0) fatal("Please set lj_epsilon in input for LJ potential\n");
    n = RUN_get_double_parameter("lj_sigma", &sigma);
    if (n == 0) fatal("Please set lj_sigma in input for LJ potential\n");
    n = RUN_get_double_parameter("lj_cutoff", &cutoff);
    if (n == 0) fatal("Please set lj_cutoff in input for LJ potential\n");

    pair_lj_cut_create(&lj);
    pair_lj_cut_param_set(lj, epsilon, sigma, cutoff);
    pair_lj_cut_register(lj, inter);
    pair_lj_cut_info(lj);
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_init_halo_range_check
 *
 *  Examine the current system (e.g., from user input) and check
 *  message passing for BBL based on input radii a0. This is indepedent
 *  of any colloid-colloid interaction considerations.
 *
 *  1) 2a0 < nlocal - 1 ensures links for a single particle are
 *                      limited to at most 2 adjacent subdomains;
 *  2) lcell > a0       ensures relevant particles join the halo
 *                      swap in relevant directions (i.e., a particle
 *                      that requires halo swap information cannot be
 *                      stranded in a cell which is not at the edge
 *                      of the sub-domain);
 *  3) ncell >= 2       must have at least two cells to separate
 *                      'left-going' and 'right-going' communications.
 *
 *****************************************************************************/

int colloids_init_halo_range_check(colloids_info_t * cinfo) {

  int ifail = 0;
  int ncolloid;
  int ncell[3];
  int nlocal[3];
  int nhalo = 1;       /* Always, for purpose of BBL. */

  double a0max = 0.0;  /* Maximum colloid a0 present */
  double lcell[3];

  assert(cinfo);

  colloids_info_ntotal(cinfo, &ncolloid);
  if (ncolloid == 0) return 0;

  coords_nlocal(nlocal);
  colloids_info_ncell(cinfo, ncell);
  colloids_info_lcell(cinfo, lcell);

  colloids_info_a0max(cinfo, &a0max);

  /* Information */
  /* info("Colloid BBL check\"); */
  /* info("Cells:  ); */
  /* Cell widths */
  /* a0max */

  if (2.0*a0max >= 1.0*(nlocal[X] - nhalo)) ifail = 1;
  if (2.0*a0max >= 1.0*(nlocal[Y] - nhalo)) ifail = 1;
  if (2.0*a0max >= 1.0*(nlocal[Z] - nhalo)) ifail = 1;
  if (ifail == 1) {
      fatal("Particle diameter larger than (nlocal - 1) domain size\n");
    }

  if (lcell[X] <= a0max) ifail = 1;
  if (lcell[Y] <= a0max) ifail = 1;
  if (lcell[Z] <= a0max) ifail = 1;
  if (ifail == 1) {
    fatal("Particle a0 > cell width breaks BBL message passing\n");
  }

  if (ncell[X] < 2) ifail = 1;
  if (ncell[Y] < 2) ifail = 1;
  if (ncell[Z] < 2) ifail = 1;

  if (ifail == 1) {
    fatal("Must have two cells minimum\n");
  }

  return ifail;
}
