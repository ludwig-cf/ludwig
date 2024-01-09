/*****************************************************************************
 *
 *  test_nernst_planck.c
 *
 *  Unit test for electrokinetic quantities.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Oliver Henrich (o.henrich@ucl.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "map.h"
#include "psi_sor.h"
#include "fe_electro.h"
#include "nernst_planck.h"

static int test_nernst_planck_driver(pe_t * pe);

/*****************************************************************************
 *
 *  test_nernst_planck_suite
 *
 *****************************************************************************/

int test_nernst_planck_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_nernst_planck_driver(pe);

  pe_info(pe, "PASS     ./unit/test_nernst_planck\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_nernst_planck_driver
 *
 *  This is the Gouy-Chapman problem.
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
 *
 *  This is a test of the Gouy-Chapman theory if one runs a significant
 *  number of time steps...
 *
 *****************************************************************************/

static int test_nernst_planck_driver(pe_t * pe) {

  int nhalo = 1;
  int ntotal[3] = {64, 4, 4};  /* Quasi-one-dimensional system */

  int nlocal[3];
  int noffst[3];
  int mpi_cartsz[3];
  int mpi_cartcoords[3];

  double rho_w;               /* wall charge density */
  double rho_i;               /* Interior charge density */
  double rho_b, rho_b_local;  /* background ionic strength */

  double rho_el = 1.0e-3;      /* charge density */
  double ltot[3];


  cs_t * cs = NULL;
  map_t * map = NULL;
  psi_t * psi = NULL;
  physics_t * phys = NULL;
  fe_electro_t * fe = NULL;

  double epsilon = 3.3e3;      /* ... epsilon, and ... */
  double beta = 3.0e4;         /* ... the Boltzmann factor i.e., t ~ 10^5 */

  psi_options_t opts = psi_options_default(nhalo);

  assert(pe);

  physics_create(pe, &phys);

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_ntotal_set(cs, ntotal);

  {
    /* If parallel, make sure the decomposition is in x-direction */
    int grid[3] = {pe_mpi_size(pe), 1, 1};
    cs_decomposition_set(cs, grid);
  }
  
  cs_init(cs);

  cs_ltot(cs, ltot);
  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffst);
  cs_cartsz(cs, mpi_cartsz);
  cs_cart_coords(cs, mpi_cartcoords);

  map_create(pe, cs, 0, &map);
  assert(map);

  opts.beta     = beta;
  opts.epsilon1 = epsilon;
  opts.epsilon2 = epsilon;
  psi_create(pe, cs, &opts, &psi);

  /* Care. the free energy gets the temperature from global physics_t. */
  fe_electro_create(pe, psi, &fe);

  /* wall charge density */
  rho_w = 1.e+0 / (2.0*ltot[Y]*ltot[Z]);

  /* counter charge density */
  rho_i = rho_w * (2.0*ltot[Y]*ltot[Z]) / ((ltot[X] - 2.0)*ltot[Y]*ltot[Z]);


  /* apply counter charges & electrolyte */
  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index = cs_index(cs, ic, jc, kc);

	psi_psi_set(psi, index, 0.0);
	psi_rho_set(psi, index, 0, rho_el);
	psi_rho_set(psi, index, 1, rho_el + rho_i);

      }
    }
  }

  /* apply wall charges */
  if (mpi_cartcoords[X] == 0) {
    int ic = 1;
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index = cs_index(cs, ic, jc, kc);
	map_status_set(map, index, MAP_BOUNDARY); 

	psi_rho_set(psi, index, 0, rho_w);
	psi_rho_set(psi, index, 1, 0.0);

      }
    }
  }

  if (mpi_cartcoords[X] == mpi_cartsz[X] - 1) {
    int ic = nlocal[X];
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index = cs_index(cs, ic, jc, kc);
	map_status_set(map, index, MAP_BOUNDARY);

	psi_rho_set(psi, index, 0, rho_w);
	psi_rho_set(psi, index, 1, 0.0);

      }
    }
  }

  /* Make a single update ... */

  map_halo(map);

  psi_halo_psi(psi);

  {
    psi_solver_sor_t * sor = NULL;
    psi_solver_sor_create(psi, &sor);
    psi_solver_sor_solve(sor, -1);
    psi_solver_sor_free(&sor);
  }

  psi_halo_rho(psi);

  nernst_planck_driver(psi, (fe_t *) fe, map);

  /* We adopt a rather simple way to extract the answer from the
   * MPI task holding the centre of the system. The charge
   * density must be > 0 to compute the debye length and the
   * surface potential. */

  rho_b_local = 0.0;

  for (int ic = 1; ic <= nlocal[X]; ic++) {

    int jc = 2;
    int kc = 2;
    int index = cs_index(cs, ic, jc, kc);
 
    if (noffst[X] + ic == ntotal[X] / 2) {
      psi_ionic_strength(psi, index, &rho_b_local);
    }
  }

  {
    MPI_Comm comm = MPI_COMM_NULL;
    cs_cart_comm(cs, &comm);
    MPI_Allreduce(&rho_b_local, &rho_b, 1, MPI_DOUBLE, MPI_SUM, comm);
  }

  {
    double lb = 0.0;                /* Bjerrum length */
    double ldebye = 0.0;            /* Debye length */
    double yd = 0.0;                /* Dimensionless surface potential */

    psi_bjerrum_length1(&opts, &lb);
    psi_debye_length1(&opts, rho_b, &ldebye);
    psi_surface_potential(psi, rho_w, rho_b, &yd);

    /* Only the surface potential has really changed compared with the
     * initial conditions ... */

    assert(fabs(lb     - 7.23431560e-01) < FLT_EPSILON);
    assert(fabs(ldebye - 6.04727364e+00) < FLT_EPSILON);
    assert(fabs(yd     - 5.18713579e-05) < FLT_EPSILON);
  }

  map_free(map);
  fe_electro_free(fe);
  psi_free(&psi);
  cs_free(cs);
  physics_free(phys);

  return 0;
}
