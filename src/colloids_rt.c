/*****************************************************************************
 *
 *  colloids_rt.c
 *
 *  Run time initialisation of colloid information.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "util.h"
#include "util_bits.h"
#include "util_ellipsoid.h"
#include "util_vector.h"
#include "coords.h"
#include "runtime.h"
#include "physics.h"

#include "lubrication.h"
#include "pair_ss_cut.h"
#include "pair_ss_cut_ij.h"
#include "pair_lj_cut.h"
#include "pair_yukawa.h"
#include "bond_fene.h"
#include "angle_cosine.h"
#include "wall_ss_cut.h"

#include "colloids_halo.h"
#include "colloids_init.h"
#include "colloid_io_rt.h"
#include "colloids_rt.h"

#include "build.h"

int lubrication_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * inter);
int pair_ss_cut_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * inter);
int pair_yukawa_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * inter);
int pair_lj_cut_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * inter);
int bond_fene_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * interact);
int angle_cosine_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * interact);
int pair_ss_cut_ij_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * intrct);

int wall_ss_cut_init(pe_t * pe, cs_t * cs, rt_t * rt, wall_t * wall,
		     interact_t * inter);

int colloids_rt_dynamics(cs_t * cs, colloids_info_t * cinfo, wall_t * wall,
			 map_t * map, const lb_model_t * model);
int colloids_rt_gravity(pe_t * pe, rt_t * rt, colloids_info_t * cinfo);
int colloids_rt_init_few(pe_t * pe, rt_t * rt, colloids_info_t * cinfo, int nc);
int colloids_rt_init_from_file(pe_t * pe, rt_t * rt, colloids_info_t * cinfo,
			       colloid_io_t * cio);
int colloids_rt_init_random(pe_t * pe, cs_t * cs, rt_t * rt, wall_t * wall,
			    colloids_info_t * cinfo);
int colloids_rt_state_stub(pe_t * pe, rt_t * rt, colloids_info_t * cinfo,
			   const char * stub,
			   colloid_state_t * state);
int colloids_rt_cell_list_checks(pe_t * pe, cs_t * cs,
				 const lb_model_t * model,
				 colloids_info_t ** pinfo,
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

int colloids_init_rt(pe_t * pe, rt_t * rt, cs_t * cs, colloids_info_t ** pinfo,
		     colloid_io_t ** pcio,
		     interact_t ** interact, wall_t * wall, map_t * map,
		     const lb_model_t * model) {
  int nc;
  int init_one = 0;
  int init_two = 0;
  int init_three = 0;
  int init_from_file = 0;
  int init_random = 0;
  int ncell[3] = {2, 2, 2};
  char keyvalue[BUFSIZ] = "";

  assert(pe);
  assert(cs);
  assert(rt);

  /* Colloid info object always created with ncell = 2;
   * later we check if this is ok and adjust if necesaary/possible. */

  colloids_info_create(pe, cs, ncell, pinfo);

  rt_string_parameter(rt, "colloid_init", keyvalue, BUFSIZ);

  if (strcmp(keyvalue, "input_one") == 0) init_one = 1;
  if (strcmp(keyvalue, "input_two") == 0) init_two = 1;
  if (strcmp(keyvalue, "input_three") == 0) init_three = 1;
  if (strcmp(keyvalue, "input_random") == 0) init_random = 1;
  if (strcmp(keyvalue, "from_file") == 0) init_from_file = 1;

  /* Trap old input files */
  if (strcmp(keyvalue, "random") == 0) pe_fatal(pe, "check input file: random\n");

  if ((init_one + init_two + init_three + init_random + init_from_file) < 1)
    return 0;

  pe_info(pe, "\n");
  pe_info(pe, "Colloid information\n");
  pe_info(pe, "-------------------\n");

  colloid_io_run_time(pe, rt, cs, *pinfo, pcio);

  if (init_one) colloids_rt_init_few(pe, rt, *pinfo, 1);
  if (init_two) colloids_rt_init_few(pe, rt, *pinfo, 2);
  if (init_three) colloids_rt_init_few(pe, rt, *pinfo, 3);
  if (init_from_file) colloids_rt_init_from_file(pe, rt, *pinfo, *pcio);
  if (init_random) colloids_rt_init_random(pe, cs, rt, wall, *pinfo);

  /* At this point, we know number of colloids */

  colloids_info_ntotal_set(*pinfo);
  colloids_info_ntotal(*pinfo, &nc);

  pe_info(pe, "\n");
  pe_info(pe, "Initialised %d colloid%s\n", nc, (nc == 1) ? "" : "s");

  interact_create(pe, cs, interact);
  assert(*interact);

  lubrication_init(pe, cs, rt, *interact);
  pair_ss_cut_init(pe, cs, rt, *interact);
  pair_lj_cut_init(pe, cs, rt, *interact);
  pair_yukawa_init(pe, cs, rt, *interact);
  bond_fene_init(pe, cs, rt, *interact);
  angle_cosine_init(pe, cs, rt, *interact);

  pair_ss_cut_ij_init(pe, cs, rt, *interact);

  wall_ss_cut_init(pe, cs, rt, wall, *interact);

  colloids_rt_cell_list_checks(pe, cs, model, pinfo, *interact);
  colloids_init_halo_range_check(pe, cs, *pinfo);
  if (nc > 1) interact_range_check(*interact, *pinfo);

  /* As the cell list has potentially changed, update I/O reference */

  colloid_io_info_set(*pcio, *pinfo);

  /* Transfer any particles in the halo regions, initialise the
   * colloid map and build the particles for the first time. */

  colloids_info_map_init(*pinfo);
  colloids_halo_state(*pinfo);

  colloids_rt_dynamics(cs, *pinfo, wall, map, model);
  colloids_rt_gravity(pe, rt, *pinfo);

  /* Set the update frequency and report (non-default values) */

  {
    int isfreq = 0;
    int nfreq = 1;

    isfreq = rt_int_parameter(rt, "colloid_rebuild_freq", &nfreq);
    if (nfreq <= 0) pe_fatal(pe, "colloids_rebuild_freq must be >= 1\n");

    if (isfreq) {
      colloids_info_rebuild_freq_set(*pinfo, nfreq);
      pe_info(pe, "Colloid rebuild freq:         %d\n", nfreq);
    }
  }

  pe_info(pe, "\n");

  return 0;
}

/*****************************************************************************
 *
 *  colloids_rt_dynamics
 *
 *****************************************************************************/

int colloids_rt_dynamics(cs_t * cs, colloids_info_t * cinfo, wall_t * wall,
			 map_t * map, const lb_model_t * model) {

  int nsubgrid_local = 0;
  int nsubgrid = 0;
  MPI_Comm comm;
  colloid_t * pc = NULL;

  assert(cs);
  assert(cinfo);

  /* Find out if we have any sub-grid particles */

  colloids_info_local_head(cinfo, &pc);

  for ( ; pc; pc = pc->nextlocal) {
    if (pc->s.bc == COLLOID_BC_SUBGRID) nsubgrid_local += 1;
  }

  cs_cart_comm(cs, &comm);
  MPI_Allreduce(&nsubgrid_local, &nsubgrid, 1, MPI_INT, MPI_SUM, comm);

  cinfo->nsubgrid = nsubgrid;

  /* Assume there are always fully-resolved particles */

  build_update_map(cs, cinfo, map);
  build_update_links(cs, cinfo, wall, map, model);
  colloids_memcpy(cinfo, tdpMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_rt_init_few
 *
 *  Few means "one", "two" or "three".
 *
 *****************************************************************************/

int colloids_rt_init_few(pe_t * pe, rt_t * rt, colloids_info_t * cinfo,
			 int nc) {

  colloid_t * pc = NULL;
  colloid_state_t * state1 = NULL;
  colloid_state_t * state2 = NULL;
  colloid_state_t * state3 = NULL;

  assert(pe);
  assert(rt);
  assert(cinfo);

  if (nc >= 1) {
    pe_info(pe, "Requested one colloid via input:\n");
    state1 = (colloid_state_t *) calloc(1, sizeof(colloid_state_t));
    assert(state1 != NULL);

    colloids_rt_state_stub(pe, rt, cinfo, "colloid_one", state1);
    colloids_info_add_local(cinfo, 1, state1->r, &pc);
    state1->index = 1;
    if (pc) pc->s = *state1;
    free(state1);
  }

  if (nc >= 2) {
    pe_info(pe, "Requested second colloid via input:\n");
    state2 = (colloid_state_t *) calloc(1, sizeof(colloid_state_t));
    assert(state2 != NULL);
    colloids_rt_state_stub(pe, rt, cinfo, "colloid_two", state2);
    colloids_info_add_local(cinfo, 2, state2->r, &pc);
    state2->index = 2;
    if (pc) pc->s = *state2;
    free(state2);
  }

  if (nc >= 3) {
    pe_info(pe, "Requested third colloid via input:\n");
    state3 = (colloid_state_t *) calloc(1, sizeof(colloid_state_t));
    assert(state3 != NULL);
    colloids_rt_state_stub(pe, rt, cinfo, "colloid_three", state3);
    colloids_info_add_local(cinfo, 3, state3->r, &pc);
    state3->index = 3;
    if (pc) pc->s = *state3;
    free(state3);
  }

  if (nc >= 4) {
    pe_fatal(pe, "Cannot specify more than 3 colloids with a file\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_rt_init_from_file
 *
 *****************************************************************************/

int colloids_rt_init_from_file(pe_t * pe, rt_t * rt, colloids_info_t * cinfo,
			       colloid_io_t * cio) {

  int ntstep;
  char filename[BUFSIZ] = {0};
  physics_t * phys = NULL;

  assert(pe);
  assert(rt);
  assert(cinfo);
  assert(cio);

  physics_ref(&phys);
  ntstep = physics_control_timestep(phys);

  if (ntstep == 0) {
    snprintf(filename, BUFSIZ-1, "config.cds.init");
  }
  else {
    snprintf(filename, BUFSIZ-1, "config.cds%8.8d", ntstep);
  }

  colloid_io_read(cio, filename);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_rt_init_random
 *
 *****************************************************************************/

int colloids_rt_init_random(pe_t * pe, cs_t * cs, rt_t * rt, wall_t * wall,
			    colloids_info_t * cinfo) {

  int nc;
  double dh = 0.0;
  colloid_state_t * state0 = NULL;

  assert(pe);
  assert(rt);
  assert(cinfo);

  state0 = (colloid_state_t *) calloc(1, sizeof(colloid_state_t));
  assert(state0 != NULL);

  colloids_rt_state_stub(pe, rt, cinfo, "colloid_random", state0);

  rt_int_parameter(rt, "colloid_random_no", &nc);
  rt_double_parameter(rt, "colloid_random_dh", &dh);

  colloids_init_random(pe, cs, cinfo, nc, state0, wall, dh);

  pe_info(pe, "Requested   %d colloid%s at random\n", nc, (nc > 1) ? "s" : "");
  pe_info(pe, "Colloid  radius a0 = %le\n", state0->a0);
  pe_info(pe, "Hydrodyn radius ah = %le\n", state0->ah);
  pe_info(pe, "Colloid charges q0 = %le    q1 = %le\n", state0->q0, state0->q1);

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

int colloids_rt_state_stub(pe_t * pe, rt_t * rt, colloids_info_t * cinfo,
			   const char * stub,
			   colloid_state_t * state) {
  int nrt, nrt1;
  char key[BUFSIZ] = "";
  char key1[BUFSIZ] = "";
  char value[BUFSIZ] = "";

  const char * format_i1 = "%-28s  %d\n";
  const char * format_i3 = "%-28s  %d %d %d\n";
  const char * format_e1 = "%-28s %14.7e\n";
  const char * format_e3 = "%-28s %14.7e %14.7e %14.7e\n";
  const char * format_s1 = "%-28s  %s\n";

  /* For ellipsoids */
  int nrteuler = 0;
  int nrtv1    = 0;
  int nrtv2    = 0;
  double elev1[3] = {0};
  double elev2[3] = {0};
  double euler[3] = {0};

  assert(pe);
  assert(rt);
  assert(cinfo);
  assert(stub);
  assert(state);
  PI_DOUBLE(pi);

  state->rebuild = 1;

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "nbonds");
  nrt = rt_int_parameter(rt, key, &state->nbonds);
  if (nrt) pe_info(pe, format_i1, key, state->nbonds);

  if (state->nbonds > 0) {
    snprintf(key, BUFSIZ-1, "%s_%s", stub, "bond1");
    nrt = rt_int_parameter(rt, key, &state->bond[0]);
    if (nrt) pe_info(pe, format_i1, key, state->bond[0]);
  }

  if (state->nbonds > 1) {
    snprintf(key, BUFSIZ-1, "%s_%s", stub, "bond2");
    nrt = rt_int_parameter(rt, key, &state->bond[1]);
    if (nrt) pe_info(pe, format_i1, key, state->bond[1]);
  }

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "nangles");
  nrt = rt_int_parameter(rt, key, &state->nangles);
  if (nrt) pe_info(pe, format_i1, key, state->nangles);

  snprintf(key1, BUFSIZ-1, "%s_%s", stub, "isfixedrxyz");
  nrt1 = rt_int_parameter_vector(rt, key1, state->isfixedrxyz);
  /* Defer output until isfxiedr is known */

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "isfixedr");
  nrt = rt_int_parameter(rt, key, &state->isfixedr);
  if (nrt) {
    pe_info(pe, format_i1, key, state->isfixedr);
    /* Override any previous value of rxyz */
    state->isfixedrxyz[X] = state->isfixedr;
    state->isfixedrxyz[Y] = state->isfixedr;
    state->isfixedrxyz[Z] = state->isfixedr;
  }
  if (nrt1) pe_info(pe, format_i3, key1, state->isfixedrxyz[X],
		    state->isfixedrxyz[Y], state->isfixedrxyz[Z]);

  snprintf(key1, BUFSIZ-1, "%s_%s", stub, "isfixedvxyz");
  nrt1 = rt_int_parameter_vector(rt, key1, state->isfixedvxyz);
  /* Defer output until isfixedv is known */

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "isfixedv");
  nrt = rt_int_parameter(rt, key, &state->isfixedv);
  if (nrt) {
    pe_info(pe, format_i1, key, state->isfixedv);
    /* Override any previous value of vxyz */
    state->isfixedvxyz[X] = state->isfixedv;
    state->isfixedvxyz[Y] = state->isfixedv;
    state->isfixedvxyz[Z] = state->isfixedv;
  }
  if (nrt1) pe_info(pe, format_i3, key1, state->isfixedvxyz[X],
		    state->isfixedvxyz[Y], state->isfixedvxyz[Z]);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "isfixedw");
  nrt = rt_int_parameter(rt, key, &state->isfixedw);
  if (nrt) pe_info(pe, format_i1, key, state->isfixedw);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "isfixeds");
  nrt = rt_int_parameter(rt, key, &state->isfixeds);
  if (nrt) pe_info(pe, format_i1, key, state->isfixeds);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "type");
  nrt = rt_string_parameter(rt, key, value, BUFSIZ);

  if (nrt) {
    /* Trap old regime */
    pe_info(pe, "You have %s in the input file\n", key);
    pe_info(pe, "Please replace *_type by *_bc *_shape *_active etc\n");
    pe_fatal(pe, "Please check and try again\n");
  }
  else {
    /* New regime */
    /* Default */

    state->bc    = COLLOID_BC_BBL;
    state->shape = COLLOID_SHAPE_SPHERE;
    state->active = 0;
    state->magnetic = 0;

    /* Boundary conditions */
    {
      int nbc = 0;
      snprintf(key, BUFSIZ-1, "%s_%s", stub, "bc");
      nbc = rt_string_parameter(rt, key, value, BUFSIZ);
      if (nbc) {
	pe_info(pe, format_s1, stub, value);
	if (strcmp(value, "bbl") == 0) {
	  state->bc = COLLOID_BC_BBL;
	}
	else if (strcmp(value, "subgrid") == 0) {
	  state->bc = COLLOID_BC_SUBGRID;
	}
	else {
	  pe_fatal(pe, "colloid bc %s not recognised\n", value);
	}
      }
    }

    /* Shape */
    {
      int nshape = 0;
      snprintf(key, BUFSIZ-1, "%s_%s", stub, "shape");
      nshape = rt_string_parameter(rt, key, value, BUFSIZ);
      if (nshape) {
	pe_info(pe, format_s1, stub, value);
	if (strcmp(value, "disk") == 0) {
	  state->shape = COLLOID_SHAPE_DISK;
	}
	else if (strcmp(value, "sphere") == 0) {
	  state->shape = COLLOID_SHAPE_SPHERE;
	}
	else if (strcmp(value, "ellipsoid") == 0) {
	  state->shape = COLLOID_SHAPE_ELLIPSOID;
	}
	else {
	  pe_fatal(pe, "colloid shape %s not recognised\n", value);
	}
      }
    }

    /* Active */
    snprintf(key, BUFSIZ-1, "%s_%s", stub, "active");
    state->active = rt_switch(rt, key);
    if (state->active) pe_info(pe, format_s1, stub, "active");

    /* Magnetic */
    snprintf(key, BUFSIZ-1, "%s_%s", stub, "magnetic");
    state->magnetic = rt_switch(rt, key);
  }

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "rng");
  nrt = rt_int_parameter(rt, key, &state->rng);
  if (nrt) pe_info(pe, format_i1, key, state->rng);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "interact_type");
  nrt = rt_int_parameter(rt, key, &state->inter_type);
  if (nrt) pe_info(pe, format_i1, key, state->inter_type);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "a0");
  nrt = rt_double_parameter(rt, key, &state->a0);
  if (nrt) pe_info(pe, format_e1, key, state->a0);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "ah");
  nrt = rt_double_parameter(rt, key, &state->ah);
  if (nrt) pe_info(pe, format_e1, key, state->ah);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "al");
  nrt = rt_double_parameter(rt, key, &state->al);
  if (nrt) pe_info(pe, format_e1, key, state->al);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "r");
  nrt = rt_double_parameter_vector(rt, key, state->r);
  if (nrt) pe_info(pe, format_e3, key, state->r[X], state->r[Y], state->r[Z]);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "v");
  nrt = rt_double_parameter_vector(rt, key, state->v);
  if (nrt) pe_info(pe, format_e3, key, state->v[X], state->v[Y], state->v[Z]);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "w");
  nrt = rt_double_parameter_vector(rt, key, state->w);
  if (nrt) pe_info(pe, format_e3, key, state->w[X], state->w[Y], state->w[Z]);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "s");
  nrt = rt_double_parameter_vector(rt, key, state->s);
  if (nrt) pe_info(pe, format_e3, key, state->s[X], state->s[Y], state->s[Z]);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "m");
  nrt = rt_double_parameter_vector(rt, key, state->m);

  /* Initial direction for spherical squirmers */

  if (state->active && state->shape == COLLOID_SHAPE_SPHERE) {
    if (nrt == 0) {
      pe_info(pe, "An active sphere must have a direction\n");
      pe_info(pe, "Use `colloid_m` to specify an initial vector\n");
      pe_exit(pe, "Please check and try again\n");
    }
    else {
      /* Ensure m is non-zero, and force a unit vector */
      double rmod = modulus(state->m);
      if (rmod <= 0.0) {
	pe_info(pe, "Must specify colloid_m as non-zero\n");
	pe_exit(pe, "Please check and try again\n");
      }
      state->m[X] /= rmod;
      state->m[Y] /= rmod;
      state->m[Z] /= rmod;
    }
  }

  if (nrt) pe_info(pe, format_e3, key, state->m[X], state->m[Y], state->m[Z]);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "b1");
  nrt = rt_double_parameter(rt, key, &state->b1);
  if (nrt) pe_info(pe, format_e1, key, state->b1);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "b2");
  nrt = rt_double_parameter(rt, key, &state->b2);
  if (nrt) pe_info(pe, format_e1, key, state->b2);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "c");
  nrt = rt_double_parameter(rt, key, &state->c);
  if (nrt) pe_info(pe, format_e1, key, state->c);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "h");
  nrt = rt_double_parameter(rt, key, &state->h);
  if (nrt) pe_info(pe, format_e1, key, state->h);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "q0");
  nrt = rt_double_parameter(rt, key, &state->q0);
  if (nrt) pe_info(pe, format_e1, key, state->q0);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "q1");
  nrt = rt_double_parameter(rt, key, &state->q1);
  if (nrt) pe_info(pe, format_e1, key, state->q1);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "epsilon");
  nrt = rt_double_parameter(rt, key, &state->epsilon);
  if (nrt) pe_info(pe, format_e1, key, state->epsilon);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "elabc");
  nrt = rt_double_parameter_vector(rt, key, state->elabc);

  if (nrt) {
    /* An ellipsoid is defined by (a,b,c), and we expect a >= b >= c */
    double a = state->elabc[0];
    double b = state->elabc[1];
    double c = state->elabc[2];
    pe_info(pe, format_e3, key, a, b, c);

    if (a < b || b < c)  {
      pe_info(pe, "Error specifying principal semi-axes of ellipse\n");
      pe_info(pe, "Please specify a_b_c with a >= b >= c\n");
      pe_exit(pe, "Please check and try again\n");
    }

    if (state->shape != COLLOID_SHAPE_ELLIPSOID) {
      pe_info(pe, "Key elabc requires shape to be ellipsoidal\n");
      pe_exit(pe, "Please check colloid input and try again\n");
    }

    /* Active ellipsoids must have b == c for tangent computation */
    /* Also need b == c if surface anchoring required. */

    if (state->active && (fabs(b - c) > FLT_EPSILON)) {
      pe_info(pe, "Active ellipsoids must have b == c\n");
      pe_exit(pe, "Please check and try again\n");
    }
  }

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "euler");
  nrteuler = rt_double_parameter_vector(rt, key, euler);
  if (nrteuler) {
    pe_info(pe, format_e3, key, euler[X], euler[Y], euler[Z]);
    euler[X] = euler[X]/180.0*pi;
    euler[Y] = euler[Y]/180.0*pi;
    euler[Z] = euler[Z]/180.0*pi;
  }

  /* For vectors, both are necessary */

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "elev1");
  nrtv1 = rt_double_parameter_vector(rt, key, elev1);
  if (nrtv1) pe_info(pe, format_e3, key, elev1[X], elev1[Y], elev1[Z]);

  snprintf(key, BUFSIZ-1, "%s_%s", stub, "elev2");
  nrtv2 = rt_double_parameter_vector(rt, key, elev2);
  if (nrtv2) pe_info(pe, format_e3, key, elev2[X], elev2[Y], elev2[Z]);

  if (nrteuler && (nrtv1 || nrtv2)) {
    pe_info(pe, "Do not use both Euler angles and vectors for ellipsoids\n");
    pe_exit(pe, "Please remove one or other from the input and try again\n");
  }

  if (nrtv1 != nrtv2) {
    pe_info(pe, "You have specified only one ellipsoid orientation vector\n");
    pe_exit(pe, "Both are required. Please check the input and try again\n");
  }

  if (nrtv1 && nrtv2) {
    /* Just translate to Euler angles ... */
    int ifail = util_ellipsoid_euler_from_vectors(elev1, elev2, euler);
    if (ifail != 0) {
      pe_info(pe, "Vectors elev1 and elev2 must not be zero, and\n");
      pe_info(pe, "must not be parallel to specify ellipsoid orientation\n");
      pe_exit(pe, "Please check the input and try again\n");
    }
    pe_info(pe, format_e3, "Euler angles", euler[X], euler[Y], euler[Z]);
  }

  util_q4_from_euler_angles(euler[X], euler[Y], euler[Z], state->quat);
  util_vector_copy(4, state->quat, state->quatold);

  /* If active and ellipsoid, assign the squirmer orientation as along the
   * long axis. In fact, this should be available for all ellipsoids,
   * e.g., for anchoring */

  if (state->shape == COLLOID_SHAPE_ELLIPSOID) {
    double v1[3] = {1.0, 0.0, 0.0}; /* x-axis */
    util_q4_rotate_vector(state->quat, v1, state->m);
    if (state->active) {
      pe_info(pe,
	      "Squirmer swimming direction: %14.7e %14.7e %14.7e\n",
	      state->m[X], state->m[Y], state->m[Z]);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_rt_gravity
 *
 *  Sedimentation force and density.
 *  Look at the separate buoyancy at the same time. This is a later
 *  addition and is treated differently, as physics_t should be on
 *  the way out.
 *
 *****************************************************************************/

int colloids_rt_gravity(pe_t * pe, rt_t * rt, colloids_info_t * cinfo) {

  int nc;
  double rho0;
  double g[3] = {0.0, 0.0, 0.0};

  assert(cinfo);

  colloids_info_ntotal(cinfo, &nc);
  if (nc == 0) return 0;

  rt_double_parameter_vector(rt, "colloid_gravity", g);
  colloids_gravity_set(cinfo, g);

  if (cinfo->isgravity) {
    pe_info(pe, "\n");
    pe_info(pe, "Sedimentation force on:       yes\n");
    pe_info(pe, "Sedimentation force:         %14.7e %14.7e %14.7e\n",
	 g[X], g[Y], g[Z]);
  }

  {
    double b[3] = {0};
    int isb = rt_double_parameter_vector(rt, "colloid_buoyancy", b);

    colloids_buoyancy_set(cinfo, b);

    if (isb) {
      pe_info(pe, "\n");
      pe_info(pe, "Colloid buoyancy force:      %14.7e %14.7e %14.7e\n",
	      b[X], b[Y], b[Z]);
      if (cinfo->isgravity) {
	pe_exit(pe, "Buoyancy and gravity both set; use one only!\n");
      }
    }
  }

  nc = rt_double_parameter(rt, "colloid_rho0", &rho0);

  if (nc) {
    colloids_info_rho0_set(cinfo, rho0);
    pe_info(pe, "Colloid density:             %14.7e\n", rho0);
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
 *  The lb_model_t is included here to get the dimensionsality;
 *  in principle one could have an entirely separate procedure for
 *  two-dimensional systems of disks.
 *
 *  The cell width should be as small as possible to prevent
 *  unnecessary halo transfers.
 *
 *****************************************************************************/

int colloids_rt_cell_list_checks(pe_t * pe, cs_t * cs,
				 const lb_model_t * model,
				 colloids_info_t ** pinfo,
				 interact_t * interact) {
  int nc;
  int nlocal[3];
  int nbest[3];
  int nhalo;

  double a0max, ahmax;  /* maximum radii */
  double rcmax, hcmax;  /* Interaction ranges */
  double rmax;          /* Maximum interaction range */
  double wcell[3];      /* Final cell widths */

  assert(pe);
  assert(cs);
  assert(pinfo);
  assert(*pinfo);

  colloids_info_ntotal(*pinfo, &nc);

  if (nc == 0) return 0;

  cs_nlocal(cs, nlocal);
  cs_nhalo(cs, &nhalo);
  colloids_info_a0max(*pinfo, &a0max);

  /* For nhalo = 1, we require an additional + 0.5 to identify BBL;
   * for nhalo > 1, the constraint is on the colloid map in the
   * halo region, begin an additional nhalo - 0.5.
   * The absolute minimum is 2 units to accommodate subgrid partciles
   * an associated interpolations onto lattice. */

  a0max = dmax(1.0, a0max);

  nbest[X] = (int) floor(1.0*(nlocal[X]) / (dmax(a0max + nhalo - 0.5, 2.0)));
  nbest[Y] = (int) floor(1.0*(nlocal[Y]) / (dmax(a0max + nhalo - 0.5, 2.0)));
  nbest[Z] = (int) floor(1.0*(nlocal[Z]) / (dmax(a0max + nhalo - 0.5, 2.0)));


  pe_info(pe, "\n");
  pe_info(pe, "Colloid cell list information\n");
  pe_info(pe, "-----------------------------\n");
  pe_info(pe, "Input radius maximum:        %14.7e\n", a0max);

  if (nc > 1) {
    /* Interaction case */
    colloids_info_ahmax(*pinfo, &ahmax);
    interact_rcmax(interact, &rcmax);
    interact_hcmax(interact, &hcmax);
    rmax = dmax(2.0*ahmax + hcmax, rcmax);
    rmax = dmax(rmax, 1.5);                  /* subgrid particles again */
    rmax = dmax(rmax, a0max + nhalo - 0.5);  /* halo, as above */
    nbest[X] = (int) floor(1.0*nlocal[X] / rmax);
    nbest[Y] = (int) floor(1.0*nlocal[Y] / rmax);
    nbest[Z] = (int) floor(1.0*nlocal[Z] / rmax);

    pe_info(pe, "Hydrodynamic radius maximum: %14.7e\n", ahmax);
    pe_info(pe, "Surface-surface interaction: %14.7e\n", hcmax);
    pe_info(pe, "Centre-centre interaction:   %14.7e\n", rcmax);
  }

  /* If we have 2d disks, then prevent nbest[Z] going to zero... */
  if (model->ndim == 2) nbest[Z] = imax(1, nbest[Z]);

  /* Transfer colloids to new cell list if required */

  if (nbest[X] > 2 || nbest[Y] > 2 || nbest[Z] > 2) {
    colloids_info_recreate(nbest, pinfo);
  }

  colloids_info_lcell(*pinfo, wcell);
  pe_info(pe, "Final cell list:              %d %d %d\n",
       nbest[X], nbest[Y], nbest[Z]);
  pe_info(pe, "Final cell lengths:          %14.7e %14.7e %14.7e\n",
       wcell[X], wcell[Y], wcell[Z]);


  return 0;
}

/*****************************************************************************
 *
 *  colloids_init_ewald_rt
 *
 *****************************************************************************/

int colloids_init_ewald_rt(pe_t * pe, rt_t * rt, cs_t * cs,
			   colloids_info_t * cinfo,
			   ewald_t ** pewald) {

  int ncolloid;
  int iarg;
  int is_required = 0;
  double mu;               /* Dipole strength */
  double rc;               /* Real space cut off */

  assert(cinfo);

  colloids_info_ntotal(cinfo, &ncolloid);
  if (ncolloid < 2) return 0;

  rt_int_parameter(rt, "ewald_sum", &is_required);

  if (is_required) {

    iarg = rt_double_parameter(rt, "ewald_mu", &mu);
    if (iarg == 0) pe_fatal(pe, "Ewald sum requires dipole strength input\n");
    iarg = rt_double_parameter(rt, "ewald_rc", &rc);
    if (iarg == 0) pe_fatal(pe, "Ewald sum requires a real space cut off\n");

    ewald_create(pe, cs, mu, rc, cinfo, pewald);
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

int lubrication_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * inter) {

  int n, on = 0;
  double rcnorm = 0.0;
  double rctang = 0.0;
  lubr_t * lubr = NULL;

  assert(pe);
  assert(rt);

  rt_int_parameter(rt, "lubrication_on", &on);

  if (on) {
    pe_info(pe, "\nColloid-colloid lubrication corrections\n");
    pe_info(pe, "Lubrication corrections are switched on\n");
    n = rt_double_parameter(rt, "lubrication_normal_cutoff", &rcnorm);
    pe_info(pe, (n == 0) ? "[Default] " : "[User   ] ");
    pe_info(pe, "Normal force cutoff is %f\n", rcnorm);

    n = rt_double_parameter(rt, "lubrication_tangential_cutoff", &rctang);
    pe_info(pe, (n == 0) ? "[Default] " : "[User   ] ");
    pe_info(pe, "Tangential force cutoff is %f\n", rctang);

    lubrication_create(pe, cs, &lubr);
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

int pair_ss_cut_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * inter) {

  int n;
  int on = 0;
  double epsilon ;
  double sigma;
  int nu;
  double cutoff;

  pair_ss_cut_t * pair = NULL;

  assert(pe);
  assert(rt);

  rt_int_parameter(rt, "soft_sphere_on", &on);

  if (on) {

    n = rt_double_parameter(rt, "soft_sphere_epsilon", &epsilon);
    if (n == 0) pe_fatal(pe, "Please define soft_sphere_epsilon in input\n");

    n = rt_double_parameter(rt, "soft_sphere_sigma", &sigma);
    if (n == 0) pe_fatal(pe, "Please define soft_sphere_sigme in input\n");

    n = rt_int_parameter(rt, "soft_sphere_nu", &nu);
    if (n == 0) pe_fatal(pe, "Please check soft_sphere_nu appears in input\n");
    if (nu <= 0) pe_fatal(pe, "Please check soft_sphere_nu is positive\n");

    n = rt_double_parameter(rt, "soft_sphere_cutoff", &cutoff);
    if (n == 0) pe_fatal(pe, "Check soft_sphere_cutoff appears in input\n");

    pair_ss_cut_create(pe, cs, &pair);
    pair_ss_cut_param_set(pair, epsilon, sigma, nu, cutoff);
    pair_ss_cut_register(pair, inter);
    pair_ss_cut_info(pair);
  }

  return 0;
}

/*****************************************************************************
 *
 *  pair_ss_cut_ij_init
 *
 *****************************************************************************/

int pair_ss_cut_ij_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * intrct) {

  int ison = 0;

  assert(pe);
  assert(cs);
  assert(rt);
  assert(intrct);

  ison = rt_switch(rt, "pair_ss_cut_ij");

  if (ison) {
    int ntypes = 0;
    int nsymm  = 0;
    double epsilon[BUFSIZ] = {0};
    double sigma[BUFSIZ] = {0};
    double nu[BUFSIZ] = {0};
    double hc[BUFSIZ] = {0};
    pair_ss_cut_ij_t * pair = NULL;

    rt_key_required(rt, "pair_ss_cut_ij_ntypes",  RT_FATAL);
    rt_key_required(rt, "pair_ss_cut_ij_epsilon", RT_FATAL);
    rt_key_required(rt, "pair_ss_cut_ij_sigma",   RT_FATAL);
    rt_key_required(rt, "pair_ss_cut_ij_nu",      RT_FATAL);
    rt_key_required(rt, "pair_ss_cut_ij_hc",      RT_FATAL);

    rt_int_parameter(rt, "pair_ss_cut_ij_ntypes", &ntypes);
    if (ntypes < 1) pe_fatal(pe, "pair_ss_cut_ij_ntypes < 1 (%d)\n", ntypes);
    if (ntypes >= BUFSIZ) pe_fatal(pe, "pair_ss_cut_ij_ntypes INTERNAL\n");

    nsymm = ntypes*(ntypes + 1)/2;

    rt_double_nvector(rt, "pair_ss_cut_ij_epsilon", nsymm, epsilon, RT_FATAL);
    rt_double_nvector(rt, "pair_ss_cut_ij_sigma",   nsymm, sigma,   RT_FATAL);
    rt_double_nvector(rt, "pair_ss_cut_ij_nu",      nsymm, nu,      RT_FATAL);
    rt_double_nvector(rt, "pair_ss_cut_ij_hc",      nsymm, hc,      RT_FATAL);

    pair_ss_cut_ij_create(pe, cs, ntypes, epsilon, sigma, nu, hc, &pair);
    pair_ss_cut_ij_register(pair, intrct);
    pair_ss_cut_ij_info(pair);
  }

  return 0;
}

/*****************************************************************************
 *
 *  pair_yukawa_init
 *
 *****************************************************************************/

int pair_yukawa_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * interact) {

  int n;
  int on = 0;
  double epsilon;
  double kappa;
  double cutoff;

  pair_yukawa_t * yukawa = NULL;

  assert(pe);
  assert(rt);
  assert(interact);

  rt_int_parameter(rt, "yukawa_on", &on);

  if (on) {
    n = rt_double_parameter(rt, "yukawa_epsilon", &epsilon);
    if (n == 0) pe_fatal(pe, "Please check yukawa_epsilon appears in input\n");
    n = rt_double_parameter(rt, "yukawa_kappa", &kappa);
    if (n == 0) pe_fatal(pe, "Please check yukawa_kappa appears in input");
    n = rt_double_parameter(rt, "yukawa_cutoff", &cutoff);
    if (n == 0) pe_fatal(pe, "Please check yukawa_cutoff appears in input\n");

    pair_yukawa_create(pe, cs, &yukawa);
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

int pair_lj_cut_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * inter) {

  int n;
  int on = 0;
  double epsilon;
  double sigma;
  double cutoff;

  pair_lj_cut_t * lj = NULL;

  assert(pe);
  assert(rt);
  assert(inter);

  rt_int_parameter(rt, "lennard_jones_on", &on);

  if (on) {
    n = rt_double_parameter(rt, "lj_epsilon", &epsilon);
    if (n == 0) pe_fatal(pe, "Set lj_epsilon in input for LJ potential\n");
    n = rt_double_parameter(rt, "lj_sigma", &sigma);
    if (n == 0) pe_fatal(pe, "Must set lj_sigma in input for LJ potential\n");
    n = rt_double_parameter(rt, "lj_cutoff", &cutoff);
    if (n == 0) pe_fatal(pe, "Must set lj_cutoff in input for LJ potential\n");

    pair_lj_cut_create(pe, cs, &lj);
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

int bond_fene_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * interact) {

  int n;
  int on = 0;
  double kappa;
  double r0;

  bond_fene_t * fene = NULL;

  assert(pe);
  assert(rt);
  assert(interact);

  rt_int_parameter(rt, "bond_fene_on", &on);

  if (on) {
    n = rt_double_parameter(rt, "bond_fene_k", &kappa);
    if (n == 0) pe_fatal(pe, "Must set bond_fene_k int input for fene bond\n");
    n = rt_double_parameter(rt, "bond_fene_r0", &r0);
    if (n == 0) pe_fatal(pe, "Must set bond_fene_r0 in input for fene bond\n");

    bond_fene_create(pe, cs,&fene);
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

int angle_cosine_init(pe_t * pe, cs_t * cs, rt_t * rt, interact_t * interact) {

  int n;
  int on = 0;
  double kappa;

  angle_cosine_t * angle = NULL;

  assert(rt);
  assert(interact);

  rt_int_parameter(rt,"angle_cosine_on", &on);

  if (on) {
    n = rt_double_parameter(rt, "angle_cosine_k", &kappa);
    if (n == 0) pe_fatal(pe, "Must set anagle_cosine_k int input for angle\n");

    angle_cosine_create(pe, cs, &angle);
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
 *  message passing for BBL based on input radii a0. This is independent
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
 *  There are some edge cases where we can relax these constraints:
 *
 *  a) A direction which is non-periodic (eg., has walls)
 *     and is not decomposed (mpisz == 1) will have no message passing.
 *     In such a case,  (1) and (2) may be ignored, and (3) is relaxed to
 *     ncell >= 1. This may be useful for systems which are "narrow" (cf. a0).
 *
 *****************************************************************************/

int colloids_init_halo_range_check(pe_t * pe, cs_t * cs,
				   colloids_info_t * cinfo) {

  int ifail = 0;
  int ncolloid = 0;
  int ncell[3] = {0};
  int nlocal[3] = {0};
  int nhalo = 1;       /* Always, for purpose of BBL. */

  int nar[3] = {0};    /* See point (a) above */

  double a0max = 0.0;  /* Maximum colloid a0 present */
  double lcell[3];

  assert(cs);
  assert(cinfo);

  colloids_info_ntotal(cinfo, &ncolloid);
  if (ncolloid == 0) return 0;

  cs_nlocal(cs, nlocal);
  colloids_info_ncell(cinfo, ncell);
  colloids_info_lcell(cinfo, lcell);

  colloids_info_a0max(cinfo, &a0max);

  if (cs->param->periodic[X] == 0 && cs->param->mpi_cartsz[X] == 1) nar[X] = 1;
  if (cs->param->periodic[Y] == 0 && cs->param->mpi_cartsz[Y] == 1) nar[Y] = 1;
  if (cs->param->periodic[Z] == 0 && cs->param->mpi_cartsz[Z] == 1) nar[Z] = 1;

  if (nar[X] == 0 && (2.0*a0max >= 1.0*(nlocal[X] - nhalo))) ifail = 1;
  if (nar[Y] == 0 && (2.0*a0max >= 1.0*(nlocal[Y] - nhalo))) ifail = 1;
  if (nar[Z] == 0 && (2.0*a0max >= 1.0*(nlocal[Z] - nhalo))) ifail = 1;
  if (ifail == 1) {
    pe_fatal(pe, "Particle diameter larger than (nlocal - 1) domain size\n");
  }

  if (nar[X] == 0 && (lcell[X] <= a0max)) ifail = 1;
  if (nar[Y] == 0 && (lcell[Y] <= a0max)) ifail = 1;
  if (nar[Z] == 0 && (lcell[Z] <= a0max)) ifail = 1;
  if (ifail == 1) {
    pe_fatal(pe, "Particle a0 > cell width breaks BBL message passing\n");
  }

  if (ncell[X] < (2 - nar[X])) ifail = 1;
  if (ncell[Y] < (2 - nar[Y])) ifail = 1;
  if (ncell[Z] < (2 - nar[Z])) ifail = 1;

  if (ifail == 1) {
    pe_fatal(pe, "Must have two cells minimum\n");
  }

  /* Halo region colloid map constraint */

  cs_nhalo(cs, &nhalo);

  if (nar[X] == 0 && (lcell[X] < (a0max + nhalo - 0.5))) ifail = 1;
  if (nar[Y] == 0 && (lcell[Y] < (a0max + nhalo - 0.5))) ifail = 1;
  if (nar[Z] == 0 && (lcell[Z] < (a0max + nhalo - 0.5))) ifail = 1;

  if (ifail == 1) {
    pe_fatal(pe, "Must have cell width > a0_max + nhalo\n");
  }

  return ifail;
}

/*****************************************************************************
 *
 *  wall_ss_cut_init
 *
 *****************************************************************************/

int wall_ss_cut_init(pe_t * pe, cs_t * cs, rt_t * rt, wall_t * wall,
		     interact_t * interact) {

  int have_wall_ss_cut = 0;

  assert(pe);
  assert(cs);
  assert(rt);

  have_wall_ss_cut = rt_switch(rt, "wall_ss_cut_on");

  if (have_wall_ss_cut) {

    wall_ss_cut_t * wall_ss_cut = NULL;
    wall_ss_cut_options_t opts = {0};

    rt_key_required(rt, "wall_ss_cut_epsilon", RT_FATAL);
    rt_key_required(rt, "wall_ss_cut_sigma", RT_FATAL);
    rt_key_required(rt, "wall_ss_cut_nu", RT_FATAL);
    rt_key_required(rt, "wall_ss_cut_hc", RT_FATAL);

    rt_double_parameter(rt, "wall_ss_cut_epsilon", &opts.epsilon);
    rt_double_parameter(rt, "wall_ss_cut_sigma", &opts.sigma);
    rt_double_parameter(rt, "wall_ss_cut_nu", &opts.nu);
    rt_double_parameter(rt, "wall_ss_cut_hc", &opts.hc);

    if (opts.nu <= 0) pe_fatal(pe, "Please ensure wall_ss_cut_nu is +ve\n");
    if (opts.hc <= 0) pe_fatal(pe, "Please ensure wall_ss_cut_hc is +ve\n");

    wall_ss_cut_create(pe, cs, wall, &opts, &wall_ss_cut);
    wall_ss_cut_register(wall_ss_cut, interact);
    wall_ss_cut_info(wall_ss_cut);
  }

  return 0;
}
