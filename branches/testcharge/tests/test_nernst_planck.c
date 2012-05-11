/*****************************************************************************
 *
 *  test_nernst_planck.c
 *
 *  Unit test for electrokinetic quantities.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Oliver Henrich (o.henrich@ucl.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "site_map.h"
#include "psi.h"
#include "psi_s.h"
#include "psi_sor.h"
#include "psi_stats.h"
#include "nernst_planck.h"

static int do_test_gouy_chapman(void);
static int test_io(psi_t * psi, int tstep);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);
  pe_init();

  do_test_gouy_chapman();

  pe_finalise();
  MPI_Finalize();

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

static int do_test_gouy_chapman(void) {

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

  int ntotal[3] = {4, 4, 64};  /* Quasi-one-dimensional system */
  int grid[3];                 /* Processor decomposition */

  int tmax = 20000;

  coords_nhalo_set(1);
  coords_ntotal_set(ntotal);

  grid[X] = 1;
  grid[Y] = 1;
  grid[Z] = pe_size();
  coords_decomposition_set(grid);

  coords_init();
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffst);

  site_map_init();

  psi_create(nk, &psi);

  psi_valency_set(psi, 0, valency[0]);
  psi_valency_set(psi, 1, valency[1]);
  psi_diffusivity_set(psi, 0, diffusivity[0]);
  psi_diffusivity_set(psi, 1, diffusivity[1]);
  psi_unit_charge_set(psi, eunit);
  psi_epsilon_set(psi, epsilon);
  psi_beta_set(psi, beta);

  /* wall charge density */
  rho_w = 1.e+0 / (2.0*L(X)*L(Y));

  /* counter charge density */
  rho_i = rho_w * (2.0*L(X)*L(Y)) / (L(X)*L(Y)*(L(Z) - 2.0));


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
  if (cart_coords(Z) == 0) {
    kc = 1;
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

	index = coords_index(ic, jc, kc);

	site_map_set_status(ic,jc,kc,BOUNDARY);

	psi_rho_set(psi, index, 0, rho_w);
	psi_rho_set(psi, index, 1, 0.0);

      }
    }
  }

  if (cart_coords(Z) == cart_size(Z) - 1) {
    kc = nlocal[Z];
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

	index = coords_index(ic, jc, kc);

	site_map_set_status(ic, jc, kc, BOUNDARY);

	psi_rho_set(psi, index, 0, rho_w);
	psi_rho_set(psi, index, 1, 0.0);

      }
    }
  }

  site_map_halo();

  for (tstep = 1; tstep <= tmax; tstep++) {

    psi_halo_psi(psi);
    psi_sor_poisson(psi);
    psi_halo_rho(psi);
    nernst_planck_driver(psi);

    if (tstep % 1000 == 0) {

      info("%d\n", tstep);
      psi_stats_info(psi);
      if (test_output_required) test_io(psi, tstep);
    }
  }

  /* We adopt a rather simple way to extract the answer from the
   * MPI task holding the centre of the system. The charge
   * density must be > 0 to compute the debye length and the
   * surface potential. */

  ic = 2;
  jc = 2;

  rho_b_local = 0.0;

  for (kc = 1; kc <= nlocal[Z]; kc++) {

    index = coords_index(ic, jc, kc);
 
    if (noffst[Z] + kc == ntotal[Z] / 2) {
      psi_ionic_strength(psi, index, &rho_b_local);
    }
  }

  MPI_Allreduce(&rho_b_local, &rho_b, 1, MPI_DOUBLE, MPI_SUM, cart_comm());

  psi_bjerrum_length(psi, &lb);
  psi_debye_length(psi, rho_b, &ldebye);
  psi_surface_potential(psi, rho_w, rho_b, &yd);
  info("Bjerrum length is %le\n", lb);
  info("Debye length is %le\n", ldebye);
  info("Surface potential is %le\n", yd);

  /* These are the reference answers. */

  assert(tmax == 20000);
  assert(ntotal[Z] == 64);
  /*assert(fabs((lb - 7.234316e-01)/0.723431) < FLT_EPSILON);
  assert(fabs((ldebye - 6.420075)/6.420075) < FLT_EPSILON);
  assert(fabs((yd - 5.451449e-05)/5.45e-05) < FLT_EPSILON);*/
  assert(fabs((lb - 7.234316e-01)/0.723431) < FLT_EPSILON);
  assert(fabs((ldebye - 6.420068)/6.420068) < FLT_EPSILON);
  assert(fabs((yd - 5.451444e-05)/5.45e-05) < FLT_EPSILON);

  site_map_finish();
  psi_free(psi);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  test_io
 *
 *****************************************************************************/

static int test_io(psi_t * psi, int tstep) {

  int ntotalz;
  int nlocal[3];
  int ic, jc, kc, index;

  double * field;               /* 1-d field (local) */
  double * psifield;            /* 1-d psi field for output */
  double * rho0field;           /* 1-d rho0 field for output */
  double * rho1field;           /* 1-d rho0 field for output */

  char filename[BUFSIZ];
  FILE * out;

  coords_nlocal(nlocal);
  ntotalz = N_total(Z);

  ic = 2;
  jc = 2;

  /* 1D output. calloc() is used to zero the arays, then
   * MPI_Gather to get complete picture. */

  field = calloc(nlocal[Z], sizeof(double));
  psifield = calloc(ntotalz, sizeof(double));
  rho0field = calloc(ntotalz, sizeof(double));
  rho1field = calloc(ntotalz, sizeof(double));
  if (field == NULL) fatal("calloc(field) failed\n");
  if (psifield == NULL) fatal("calloc(psifield) failed\n");
  if (rho0field == NULL) fatal("calloc(rho0field) failed\n");
  if (rho1field == NULL) fatal("calloc(rho1field) failed\n");

  for (kc = 1; kc <= nlocal[Z]; kc++) {

    index = coords_index(ic, jc, kc);
    psi_psi(psi, index, field + kc - 1);
  }

  MPI_Gather(field, nlocal[Z], MPI_DOUBLE,
	     psifield, nlocal[Z], MPI_DOUBLE, 0, cart_comm());

  for (kc = 1; kc <= nlocal[Z]; kc++) {
    index = coords_index(ic, jc, kc);
    psi_rho(psi, index, 0, field + kc - 1);
  }

  MPI_Gather(field, nlocal[Z], MPI_DOUBLE,
	     rho0field, nlocal[Z], MPI_DOUBLE, 0, cart_comm());

  for (kc = 1; kc <= nlocal[Z]; kc++) {
    index = coords_index(ic, jc, kc);
    psi_rho(psi, index, 1, field + kc - 1);
  }

  MPI_Gather(field, nlocal[Z], MPI_DOUBLE,
	     rho1field, nlocal[Z], MPI_DOUBLE, 0, cart_comm());

  if (cart_rank() == 0) {

    sprintf(filename, "np_test-%d.dat", tstep);
    out = fopen(filename, "w");
    if (out == NULL) fatal("Could not open %s\n", filename);

    for (kc = 1; kc <= ntotalz; kc++) {
      fprintf(out, "%d %le %le %le\n", kc, psifield[kc-1],
	      rho0field[kc-1], rho1field[kc-1]);
    }
    fclose(out);
  }

  free(rho1field);
  free(rho0field);
  free(psifield);
  free(field);

  return 0;
}
