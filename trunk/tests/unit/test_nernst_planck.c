/*****************************************************************************
 *
 *  test_nernst_planck.c
 *
 *  Unit test for electrokinetic quantities.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Oliver Henrich (o.henrich@ucl.ac.uk)
 *
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "control.h"
#include "map.h"
#include "psi.h"
#include "psi_s.h"
#include "psi_sor.h"
#include "psi_stats.h"
#include "fe_electro.h"
#include "nernst_planck.h"
#include "tests.h"

static int do_test_gouy_chapman(pe_t * pe);
static int test_io(cs_t * cs, psi_t * psi, int tstep);

/*****************************************************************************
 *
 *  test_nernst_planck_suite
 *
 *****************************************************************************/

int test_nernst_planck_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  do_test_gouy_chapman(pe);

  pe_info(pe, "PASS     ./unit/test_nernst_planck\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_gouy_chapman
 *
 *  A theory exists for symmetric electrolytes near a flat surface
 *  owing to Gouy and Chapman. (See Lyklema "Fundamentals of
 *  Interface and Colloid Science" Vol. II Section 3.5.)
 *
 *  Here we approximate this by a quasi-one dimensional system
 *  with walls at each end in the z-direction. An initial
 *  charge distribution is set up which must be net neutral,
 *  and has +ve charge at the wall and a mixture in the
 *  fluid. The resulting diffusion sets up a double layer in the
 *  fluid near the walls.
 *
 *  Set rho(z = 1)  = + 1 / (2 Nx Ny)
 *      rho(z = Lz) = + 1 / (2 Nx Ny)
 *      rho         = - 1 / (Nx Ny*(Nz - 2)) + electrolyte
 *
 *  The time to reach equilibrium is diffusional: L_z^2 / D_eff
 *  where D_eff ~= D_k e beta rho_k (from the Nernst Planck
 *  equation). The parameters make 20,000 steps reasonable.
 *
 *  The tolerances on the SOR could be investigated.
 *
 *  This is a test of the Gouy-Chapman theory.
 *
 *****************************************************************************/

static int do_test_gouy_chapman(pe_t * pe) {

  int nk = 2;          /* Number of species */

  int ic, jc, kc, index;
  int nlocal[3];
  int noffst[3];
  int test_output_required = 0;
  int tstep;
  double rho_w;               /* wall charge density */
  double rho_i;               /* Interior charge density */
  double rho_b, rho_b_local;  /* background ionic strength */

  int valency[2] = {+1, -1};
  double diffusivity[2] = {1.e-2, 1.e-2};

  double eunit = 1.;           /* Unit charge, ... */
  double epsilon = 3.3e3;      /* ... epsilon, and ... */
  double beta = 3.0e4;         /* ... the Boltzmann factor i.e., t ~ 10^5 */
  double lb;                   /* ... make up the Bjerrum length */
  double ldebye;               /* Debye length */
  double rho_el = 1.0e-3;      /* charge density */
  double yd;                   /* Dimensionless surface potential */

  int ntotal[3] = {64, 4, 4};  /* Quasi-one-dimensional system */
  int grid[3];                 /* Processor decomposition */

  int tmax = 200;

  cs_t * cs = NULL;
  map_t * map = NULL;
  psi_t * psi = NULL;
  physics_t * phys = NULL;
  fe_electro_t * fe = NULL;

  assert(pe);

  physics_create(pe, &phys);

  cs_create(pe, &cs);
  cs_nhalo_set(cs, 1);
  cs_ntotal_set(cs, ntotal);

  grid[X] = pe_mpi_size(pe);
  grid[Y] = 1;
  grid[Z] = 1;
  cs_decomposition_set(cs, grid);

  cs_init(cs);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffst);

  map_create(pe, cs, 0, &map);
  assert(map);

  psi_create(pe, cs, nk, &psi);
  assert(psi);

  psi_valency_set(psi, 0, valency[0]);
  psi_valency_set(psi, 1, valency[1]);
  psi_diffusivity_set(psi, 0, diffusivity[0]);
  psi_diffusivity_set(psi, 1, diffusivity[1]);
  psi_unit_charge_set(psi, eunit);
  psi_epsilon_set(psi, epsilon);
  psi_beta_set(psi, beta);

  fe_electro_create(psi, &fe);

  /* wall charge density */
  rho_w = 1.e+0 / (2.0*L(Y)*L(Z));

  /* counter charge density */
  rho_i = rho_w * (2.0*L(Y)*L(Z)) / ((L(X) - 2.0)*L(Y)*L(Z));


  /* apply counter charges & electrolyte */
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_psi_set(psi, index, 0.0);
	psi_rho_set(psi, index, 0, rho_el);
	psi_rho_set(psi, index, 1, rho_el + rho_i);

      }
    }
  }

  /* apply wall charges */
  if (cart_coords(X) == 0) {
    ic = 1;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);
	map_status_set(map, index, MAP_BOUNDARY); 

	psi_rho_set(psi, index, 0, rho_w);
	psi_rho_set(psi, index, 1, 0.0);

      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {
    ic = nlocal[X];
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);
	map_status_set(map, index, MAP_BOUNDARY);

	psi_rho_set(psi, index, 0, rho_w);
	psi_rho_set(psi, index, 1, 0.0);

      }
    }
  }

  map_halo(map);

  for (tstep = 1; tstep <= tmax; tstep++) {

    psi_halo_psi(psi);
    psi_sor_poisson(psi);
    psi_halo_rho(psi);
    /* The test is run with no hydrodynamics, hence NULL here. */
    nernst_planck_driver(psi, (fe_t *) fe, NULL, map);

    if (tstep % 1000 == 0) {

      info("%d\n", tstep);
      psi_stats_info(psi);
      if (test_output_required) test_io(cs, psi, tstep);
    }
  }

  /* We adopt a rather simple way to extract the answer from the
   * MPI task holding the centre of the system. The charge
   * density must be > 0 to compute the debye length and the
   * surface potential. */

  jc = 2;
  kc = 2;

  rho_b_local = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {

    index = coords_index(ic, jc, kc);
 
    if (noffst[X] + ic == ntotal[X] / 2) {
      psi_ionic_strength(psi, index, &rho_b_local);
    }
  }

  MPI_Allreduce(&rho_b_local, &rho_b, 1, MPI_DOUBLE, MPI_SUM, cart_comm());

  psi_bjerrum_length(psi, &lb);
  psi_debye_length(psi, rho_b, &ldebye);
  psi_surface_potential(psi, rho_w, rho_b, &yd);

  assert(tmax == 200);
  assert(ntotal[X] == 64);
  assert(fabs(lb     - 7.2343156e-01) < FLT_EPSILON);
  assert(fabs(ldebye - 6.0648554e+00) < FLT_EPSILON);
  assert(fabs(yd     - 5.1997576e-05) < FLT_EPSILON);

  map_free(map);
  fe_electro_free(fe);
  psi_free(psi);
  cs_free(cs);
  physics_free(phys);

  return 0;
}

/*****************************************************************************
 *
 *  test_io
 *
 *****************************************************************************/

static int test_io(cs_t * cs, psi_t * psi, int tstep) {

  int ntotalx;
  int nlocal[3];
  int ic, jc, kc, index;

  double * field;               /* 1-d field (local) */
  double * psifield;            /* 1-d psi field for output */
  double * rho0field;           /* 1-d rho0 field for output */
  double * rho1field;           /* 1-d rho0 field for output */

  char filename[BUFSIZ];
  FILE * out;

  coords_nlocal(nlocal);
  ntotalx = N_total(X);

  jc = 2;
  kc = 2;

  /* 1D output. calloc() is used to zero the arays, then
   * MPI_Gather to get complete picture. */

  field = (double *) calloc(nlocal[X], sizeof(double));
  psifield = (double *) calloc(ntotalx, sizeof(double));
  rho0field = (double *) calloc(ntotalx, sizeof(double));
  rho1field = (double *) calloc(ntotalx, sizeof(double));
  if (field == NULL) fatal("calloc(field) failed\n");
  if (psifield == NULL) fatal("calloc(psifield) failed\n");
  if (rho0field == NULL) fatal("calloc(rho0field) failed\n");
  if (rho1field == NULL) fatal("calloc(rho1field) failed\n");

  for (ic = 1; ic <= nlocal[X]; ic++) {

    index = coords_index(ic, jc, kc);
    psi_psi(psi, index, field + ic - 1);
  }

  MPI_Gather(field, nlocal[X], MPI_DOUBLE,
	     psifield, nlocal[X], MPI_DOUBLE, 0, cart_comm());

  for (ic = 1; ic <= nlocal[X]; ic++) {
    index = coords_index(ic, jc, kc);
    psi_rho(psi, index, 0, field + ic - 1);
  }

  MPI_Gather(field, nlocal[X], MPI_DOUBLE,
	     rho0field, nlocal[X], MPI_DOUBLE, 0, cart_comm());

  for (ic = 1; ic <= nlocal[X]; ic++) {
    index = coords_index(ic, jc, kc);
    psi_rho(psi, index, 1, field + ic - 1);
  }

  MPI_Gather(field, nlocal[X], MPI_DOUBLE,
	     rho1field, nlocal[X], MPI_DOUBLE, 0, cart_comm());

  if (cs_cart_rank(cs) == 0) {

    sprintf(filename, "np_test-%d.dat", tstep);
    out = fopen(filename, "w");
    if (out == NULL) fatal("Could not open %s\n", filename);

    for (ic = 1; ic <= ntotalx; ic++) {
      fprintf(out, "%d %14.7e %14.7e %14.7e\n", ic, psifield[ic-1],
	      rho0field[ic-1], rho1field[ic-1]);
    }
    fclose(out);
  }

  free(rho1field);
  free(rho0field);
  free(psifield);
  free(field);

  return 0;
}
