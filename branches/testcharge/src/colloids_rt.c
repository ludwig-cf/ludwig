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
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "runtime.h"
#include "control.h"

#include "lubrication.h"
#include "pair_ss_cut.h"
#include "pair_lj_cut.h"
#include "pair_yukawa.h"
#include "bond_fene.h"
#include "angle_cosine.h"

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
int bond_fene_init(interact_t * interact);
int angle_cosine_init(interact_t * interact);

int colloids_rt_dynamics(colloids_info_t * cinfo, map_t * map);
int colloids_rt_gravity(colloids_info_t * cinfo);
int colloids_rt_init_few(colloids_info_t * cinfo, int nc);
int colloids_rt_init_from_file(colloids_info_t * cinfo, colloid_io_t * cio);
int colloids_rt_init_random(colloids_info_t * cinfo);
int colloids_rt_state_stub(colloids_info_t * cinfo, const char * stub,
			   colloid_state_t * state);
int colloids_rt_cell_list_checks(colloids_info_t ** pinfo,
				 interact_t * interact);

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
  int nc;
  int init_one = 0;
  int init_two = 0;
  int init_three = 0;
  int init_from_file = 0;
  int init_random = 0;
  int ncell[3] = {2, 2, 2};
  char keyvalue[BUFSIZ];

  /* Colloid info object always created with ncell = 2;
   * later we check if this is ok and adjust if necesaary/possible. */

  colloids_info_create(ncell, pinfo);

  RUN_get_string_parameter("colloid_init", keyvalue, BUFSIZ);

  if (strcmp(keyvalue, "input_one") == 0) init_one = 1;
  if (strcmp(keyvalue, "input_two") == 0) init_two = 1;
  if (strcmp(keyvalue, "input_three") == 0) init_three = 1;
  if (strcmp(keyvalue, "input_random") == 0) init_random = 1;
  if (strcmp(keyvalue, "from_file") == 0) init_from_file = 1;

  /* Trap old input files */
  if (strcmp(keyvalue, "random") == 0) fatal("check input file: random\n");
  
  if ((init_one + init_two + init_three + init_random + init_from_file) < 1)
    return 0;

  info("\n");
  info("Colloid information\n");
  info("-------------------\n");

  colloid_io_run_time(*pinfo, pcio);

  if (init_one) colloids_rt_init_few(*pinfo, 1);
  if (init_two) colloids_rt_init_few(*pinfo, 2);
  if (init_three) colloids_rt_init_few(*pinfo, 3);
  if (init_from_file) colloids_rt_init_from_file(*pinfo, *pcio);
  if (init_random) colloids_rt_init_random(*pinfo);

  /* At this point, we know number of colloids */

  colloids_info_ntotal_set(*pinfo);
  colloids_info_ntotal(*pinfo, &nc);

  info("\n");
  info("Initialised %d colloid%s\n", nc, (nc == 1) ? "" : "s");

  interact_create(interact);
  assert(*interact);

  lubrication_init(*interact);
  pair_ss_cut_init(*interact);
  pair_lj_cut_init(*interact);
  pair_yukawa_init(*interact);
  bond_fene_init(*interact);
  angle_cosine_init(*interact);

  colloids_rt_cell_list_checks(pinfo, *interact);
  colloids_init_halo_range_check(*pinfo);
  if (nc > 1) interact_range_check(*interact, *pinfo);

  /* As the cell list has potentially changed, update I/O reference */

  colloid_io_info_set(*pcio, *pinfo);

  /* Transfer any particles in the halo regions, initialise the
   * colloid map and build the particles for the first time. */

  colloids_info_map_init(*pinfo);
  colloids_halo_state(*pinfo);

  colloids_rt_dynamics(*pinfo, map);

  colloids_rt_gravity(*pinfo);
  info("\n");
  
  return 0;
}

/*****************************************************************************
 *
 *  colloids_rt_dynamics
 *
 *****************************************************************************/

int colloids_rt_dynamics(colloids_info_t * cinfo, map_t * map) {

  int nsubgrid_local = 0;
  int nsubgrid;

  assert(cinfo);

  colloids_info_count_local(cinfo, COLLOID_TYPE_SUBGRID, &nsubgrid_local);

  MPI_Allreduce(&nsubgrid_local, &nsubgrid, 1, MPI_INT, MPI_SUM, pe_comm());

  if (nsubgrid > 0) {
    subgrid_on_set();
  }
  else {
    build_update_map(cinfo, map);
    build_update_links(cinfo, map);
  }  

  return 0;
}

/*****************************************************************************
 *
 *  colloids_rt_init_few
 *
 *  Few means "one", "two" or "three".
 *
 *****************************************************************************/

int colloids_rt_init_few(colloids_info_t * cinfo, int nc) {

  colloid_t * pc = NULL;
  colloid_state_t * state1 = NULL;
  colloid_state_t * state2 = NULL;
  colloid_state_t * state3 = NULL;

  assert(cinfo);
  assert(nc == 1 || nc == 2 || nc == 3);

  if (nc >= 1) {
    info("Requested one colloid via input:\n");
    state1 = (colloid_state_t *) calloc(1, sizeof(colloid_state_t));
    assert(state1 != NULL);
    colloids_rt_state_stub(cinfo, "colloid_one", state1);
    colloids_info_add_local(cinfo, 1, state1->r, &pc);
    state1->index = 1;
    if (pc) pc->s = *state1;
    free(state1);
  }

  if (nc >= 2) {
    info("Requested second colloid via input:\n");
    state2 = (colloid_state_t *) calloc(1, sizeof(colloid_state_t));
    assert(state2 != NULL);
    colloids_rt_state_stub(cinfo, "colloid_two", state2);
    colloids_info_add_local(cinfo, 2, state2->r, &pc);
    state2->index = 2;
    if (pc) pc->s = *state2;
    free(state2);
  }

  if (nc >= 3) {
    info("Requested third colloid via input:\n");
    state3 = (colloid_state_t *) calloc(1, sizeof(colloid_state_t));
    assert(state3 != NULL);
    colloids_rt_state_stub(cinfo, "colloid_three", state3);
    colloids_info_add_local(cinfo, 3, state3->r, &pc);
    state3->index = 3;
    if (pc) pc->s = *state3;
    free(state3);
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_rt_init_from_file
 *
 *****************************************************************************/

int colloids_rt_init_from_file(colloids_info_t * cinfo, colloid_io_t * cio) {

  char subdirectory[FILENAME_MAX];
  char filename[FILENAME_MAX];
  char stub[FILENAME_MAX];

  assert(cinfo);
  assert(cio);

  pe_subdirectory(subdirectory);

  strcpy(stub, "config.cds.init");
  RUN_get_string_parameter("colloid_file_stub", stub, FILENAME_MAX);

  if (get_step() == 0) {
    sprintf(filename, "%s%s", subdirectory, stub);
  }
  else {
    strcpy(stub, "config.cds");
    RUN_get_string_parameter("colloid_file_stub", stub, FILENAME_MAX);
    sprintf(filename, "%s%s%8.8d", subdirectory, stub, get_step());
  }

  colloid_io_read(cio, filename);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_rt_init_random
 *
 *****************************************************************************/

int colloids_rt_init_random(colloids_info_t * cinfo) {

  int nc;
  double dh = 0.0;
  colloid_state_t * state0 = NULL;

  assert(cinfo);

  state0 = (colloid_state_t *) calloc(1, sizeof(colloid_state_t));
  assert(state0 != NULL);

  colloids_rt_state_stub(cinfo, "colloid_random", state0);

  RUN_get_int_parameter("colloid_random_no", &nc);
  RUN_get_double_parameter("colloid_random_dh", &dh);

  colloids_init_random(cinfo, nc, state0, dh);

  info("Requested   %d colloid%s at random\n", nc, (nc > 1) ? "s" : "");
  info("Colloid  radius a0 = %le\n", state0->a0);
  info("Hydrodyn radius ah = %le\n", state0->ah);
  info("Colloid charges q0 = %le    q1 = %le\n", state0->q0, state0->q1);

  free(state0);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_rt_state_stub
 *
 *  Read details of a colloid state form the input via keys
 *    stub_a0
 *    stub_ah
 *    ...
 *
 *****************************************************************************/

int colloids_rt_state_stub(colloids_info_t * cinfo, const char * stub,
			   colloid_state_t * state) {
  int nrt;
  char key[BUFSIZ];
  char value[BUFSIZ];

  const char * format_i1 = "%-28s  %d\n";
  const char * format_e1 = "%-28s %14.7e\n";
  const char * format_e3 = "%-28s %14.7e %14.7e %14.7e\n";
  const char * format_s1 = "%-28s  %s\n";

  assert(cinfo);
  assert(stub);

  state->rebuild = 1;

  sprintf(key, "%s_%s", stub, "nbonds");
  nrt = RUN_get_int_parameter(key, &state->nbonds);
  if (nrt) info(format_i1, key, state->nbonds);

  if (state->nbonds > 0) {
    sprintf(key, "%s_%s", stub, "bond1");
    nrt = RUN_get_int_parameter(key, &state->bond[0]);
    if (nrt) info(format_i1, key, state->bond[0]);
  }

  if (state->nbonds > 1) {
    sprintf(key, "%s_%s", stub, "bond2");
    nrt = RUN_get_int_parameter(key, &state->bond[1]);
    if (nrt) info(format_i1, key, state->bond[1]);
  }

  sprintf(key, "%s_%s", stub, "nangles");
  nrt = RUN_get_int_parameter(key, &state->nangles);
  if (nrt) info(format_i1, key, state->nangles);

  sprintf(key, "%s_%s", stub, "isfixedr");
  nrt = RUN_get_int_parameter(key, &state->isfixedr);
  if (nrt) info(format_i1, key, state->isfixedr);

  sprintf(key, "%s_%s", stub, "isfixedv");
  nrt = RUN_get_int_parameter(key, &state->isfixedv);
  if (nrt) info(format_i1, key, state->isfixedv);

  sprintf(key, "%s_%s", stub, "isfixedw");
  nrt = RUN_get_int_parameter(key, &state->isfixedw);
  if (nrt) info(format_i1, key, state->isfixedw);

  sprintf(key, "%s_%s", stub, "isfixeds");
  nrt = RUN_get_int_parameter(key, &state->isfixeds);
  if (nrt) info(format_i1, key, state->isfixeds);

  sprintf(key, "%s_%s", stub, "type");
  nrt = RUN_get_string_parameter(key, value, BUFSIZ);

  state->type = COLLOID_TYPE_DEFAULT;
  if (strcmp(value, "active") == 0) state->type = COLLOID_TYPE_ACTIVE;
  if (strcmp(value, "subgrid") == 0) state->type = COLLOID_TYPE_SUBGRID;
  if (nrt) info(format_s1, stub, value);

  sprintf(key, "%s_%s", stub, "a0");
  nrt = RUN_get_double_parameter(key, &state->a0);
  if (nrt) info(format_e1, key, state->a0);

  sprintf(key, "%s_%s", stub, "ah");
  nrt = RUN_get_double_parameter(key, &state->ah);
  if (nrt) info(format_e1, key, state->ah);

  sprintf(key, "%s_%s", stub, "r");
  nrt = RUN_get_double_parameter_vector(key, state->r);
  if (nrt) info(format_e3, key, state->r[X], state->r[Y], state->r[Z]);

  sprintf(key, "%s_%s", stub, "v");
  nrt = RUN_get_double_parameter_vector(key, state->v);
  if (nrt) info(format_e3, key, state->v[X], state->v[Y], state->v[Z]);

  sprintf(key, "%s_%s", stub, "w");
  nrt = RUN_get_double_parameter_vector(key, state->w);
  if (nrt) info(format_e3, key, state->w[X], state->w[Y], state->w[Z]);

  sprintf(key, "%s_%s", stub, "s");
  nrt = RUN_get_double_parameter_vector(key, state->s);
  if (nrt) info(format_e3, key, state->s[X], state->s[Y], state->s[Z]);

  sprintf(key, "%s_%s", stub, "m");
  nrt = RUN_get_double_parameter_vector(key, state->m);
  if (nrt) info(format_e3, key, state->m[X], state->m[Y], state->m[Z]);

  sprintf(key, "%s_%s", stub, "b1");
  nrt = RUN_get_double_parameter(key, &state->b1);
  if (nrt) info(format_e1, key, state->b1);

  sprintf(key, "%s_%s", stub, "b2");
  nrt = RUN_get_double_parameter(key, &state->b2);
  if (nrt) info(format_e1, key, state->b2);

  sprintf(key, "%s_%s", stub, "c");
  nrt = RUN_get_double_parameter(key, &state->c);
  if (nrt) info(format_e1, key, state->c);

  sprintf(key, "%s_%s", stub, "h");
  nrt = RUN_get_double_parameter(key, &state->h);
  if (nrt) info(format_e1, key, state->h);

  sprintf(key, "%s_%s", stub, "q0");
  nrt = RUN_get_double_parameter(key, &state->q0);
  if (nrt) info(format_e1, key, state->q0);

  sprintf(key, "%s_%s", stub, "q1");
  nrt = RUN_get_double_parameter(key, &state->q1);
  if (nrt) info(format_e1, key, state->q1);

  sprintf(key, "%s_%s", stub, "epsilon");
  nrt = RUN_get_double_parameter(key, &state->epsilon);
  if (nrt) info(format_e1, key, state->epsilon);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_rt_gravity
 *
 *****************************************************************************/

int colloids_rt_gravity(colloids_info_t * cinfo) {

  int nc;
  int isgrav = 0;
  double g[3] = {0.0, 0.0, 0.0};

  assert(cinfo);

  colloids_info_ntotal(cinfo, &nc);
  if (nc == 0) return 0;

  nc = RUN_get_double_parameter_vector("colloid_gravity", g);
  if (nc != 0) physics_fgrav_set(g);

  isgrav = (g[X] != 0.0 || g[Y] != 0.0 || g[Z] != 0.0);

  if (isgrav) {
    info("\n");
    info("Sedimentation force on:       yes\n");
    info("Sedimentation force:         %14.7e %14.7e %14.7e\n",
	 g[X], g[Y], g[Z]);
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_rt_cell_list_checks
 *
 *  For given set of colloids in the default cell list, and given
 *  interactions, work out what the best cell list size is.
 *
 *  The cell width should be as small as possible to prevent
 *  unnecessary halo transfers.
 *
 *****************************************************************************/

int colloids_rt_cell_list_checks(colloids_info_t ** pinfo,
				 interact_t * interact) {
  int nc;
  int nlocal[3];
  int nbest[3];

  double a0max, ahmax;  /* maximum radii */
  double rcmax, hcmax;  /* Interaction ranges */
  double rmax;          /* Maximum interaction range */
  double wcell[3];      /* Final cell widths */

  assert(pinfo);
  assert(*pinfo);

  colloids_info_ntotal(*pinfo, &nc);

  if (nc == 0) return 0;

  coords_nlocal(nlocal);
  colloids_info_a0max(*pinfo, &a0max);

  /* The 0.5 is required for BBL to identify fluid-solid links,
   * and a0max cannot drop below 1.0 (e.g., for subgrid the
   * net minimum is 1.5 for drange = 1). */

  a0max = dmax(1.0, a0max);

  nbest[X] = (int) floor(1.0*(nlocal[X]) / (a0max + 0.5));
  nbest[Y] = (int) floor(1.0*(nlocal[Y]) / (a0max + 0.5));
  nbest[Z] = (int) floor(1.0*(nlocal[Z]) / (a0max + 0.5));

  info("\n");
  info("Colloid cell list information\n");
  info("-----------------------------\n");
  info("Input radius maximum:        %14.7e\n", a0max);

  if (nc > 1) {
    /* Interaction case */
    colloids_info_ahmax(*pinfo, &ahmax);
    interact_rcmax(interact, &rcmax);
    interact_hcmax(interact, &hcmax);
    rmax = dmax(2.0*ahmax + hcmax, rcmax);
    rmax = dmax(rmax, 1.5); /* subgrid particles again */
    nbest[X] = (int) floor(1.0*nlocal[X] / rmax);
    nbest[Y] = (int) floor(1.0*nlocal[Y] / rmax);
    nbest[Z] = (int) floor(1.0*nlocal[Z] / rmax);

    info("Hydrodynamic radius maximum: %14.7e\n", ahmax);
    info("Surface-surface interaction: %14.7e\n", hcmax);
    info("Centre-centre interaction:   %14.7e\n", rcmax);
  }

  /* Transfer colloids to new cell list if required */

  if (nbest[X] > 2 || nbest[Y] > 2 || nbest[Z] > 2) {
    colloids_info_recreate(nbest, pinfo);
  }

  colloids_info_lcell(*pinfo, wcell);
  info("Final cell list:              %d %d %d\n",
       nbest[X], nbest[Y], nbest[Z]);
  info("Final cell lengths:          %14.7e %14.7e %14.7e\n",
       wcell[X], wcell[Y], wcell[Z]);


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
 *  bond_fene_init
 *
 *****************************************************************************/

int bond_fene_init(interact_t * interact) {

  int n;
  int on = 0;
  double kappa;
  double r0;

  bond_fene_t * fene = NULL;

  assert(interact);

  n = RUN_get_int_parameter("bond_fene_on", &on);

  if (on) {
    n = RUN_get_double_parameter("bond_fene_k", &kappa);
    if (n == 0) fatal("Please set bond_fene_k int input for fene bond\n");
    n = RUN_get_double_parameter("bond_fene_r0", &r0);
    if (n == 0) fatal("Please set bond_fene_r0 in input for fene bond\n");

    bond_fene_create(&fene);
    bond_fene_param_set(fene, kappa, r0);
    bond_fene_register(fene, interact);
    bond_fene_info(fene);
  }

  return 0;
}

/*****************************************************************************
 *
 *  angle_cosine_init
 *
 *****************************************************************************/

int angle_cosine_init(interact_t * interact) {

  int n;
  int on = 0;
  double kappa;

  angle_cosine_t * angle = NULL;

  assert(interact);

  n = RUN_get_int_parameter("angle_cosine_on", &on);

  if (on) {
    n = RUN_get_double_parameter("angle_cosine_k", &kappa);
    if (n == 0) fatal("Please set anagle_cosine_k int input for angle\n");

    angle_cosine_create(&angle);
    angle_cosine_param_set(angle, kappa);
    angle_cosine_register(angle, interact);
    angle_cosine_info(angle);
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
