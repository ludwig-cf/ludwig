/*****************************************************************************
 *
 *  test_fft_sor.c
  *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Ruairi Short (Rshort@sms.ed.ac.uk)
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
#include "psi_s.h"
#include "psi_sor.h"
#include "timer.h"
#include "psi_stats.h"
#include "psi_colloid.h"

#include "psi_fft.h"

#include "util.h"
#include "psi_stats.h"

#include "runtime.h"
#include "coords_rt.h"

#define REF_PERMEATIVITY 1.0

/*This is copied from test_psi_sor.c*/
static int test_charge1_set(psi_t * psi);

int main(int argc, char ** argv) {

  int i,j,k;
  int index = 0;
  int nlocal[3] = {0, 0, 0};
  int global_coord[3] = {0, 0, 0};
  int global_coord_save[3] = {0, 0, 0};
  int iter = 1;
  double pi = 4.0*atan(1.0); 
  double max;

  MPI_Init(&argc, &argv);
  
  pe_init();

  info("Testing FFT solver\n");

  coords_init();
  decomp_init();

/*initialise two psi_t objects and these will be compared after being solved by the
 * two solvers*/
  psi_t *psi_sor = NULL;
  psi_t *psi_fft = NULL;
  
  psi_create(2, &psi_sor);
  psi_create(2, &psi_fft);
  assert(psi_sor);
  assert(psi_fft);

  psi_valency_set(psi_sor, 0, +1.0);
  psi_valency_set(psi_sor, 1, -1.0);
  psi_epsilon_set(psi_sor, REF_PERMEATIVITY);

  psi_valency_set(psi_fft, 0, +1.0);
  psi_valency_set(psi_fft, 1, -1.0);
  psi_epsilon_set(psi_fft, REF_PERMEATIVITY);

  psi_halo_psi(psi_sor);
  psi_halo_rho(psi_sor);

/*initialise the charges*/
  test_charge1_set(psi_sor);
  test_charge1_set(psi_fft);


 /*use psi_sor_poisson to solve*/
  info("Solving with SOR\n");
  psi_sor_poisson(psi_sor);


  /*use psi_fft_poisson to solve*/
  info("Solving with FFT\n");
  psi_fft_poisson(psi_fft);

  /*check results are acceptably similar
   * k^2 method requires an absolute tolerance of 1e-4 to pass
   * the 6 point stencil method requires 1e-11*/
  for(i=1; i<=nlocal[X]; i++) {
    for(j=1; j<=nlocal[Y]; j++) {
      for(k=1; k<=nlocal[Z]; k++) {
        index = coords_index(i,j,k);
        assert(fabs(psi_sor->psi[index] - psi_fft->psi[index]) < 1e-4);
      }
    }
  }

  psi_free(psi_sor);
  psi_free(psi_fft);

  decomp_finish();
  coords_finish();
  pe_finalise();
  MPI_Finalize();

  return 0;

}


/*****************************************************************************
 *
 *  test_charge1_set
 *
 *  Sets a uniform 'wall' charge at z = 1 and z = L_z and a uniform
 *  interior value elsewhere such that the system is overall charge
 *  neutral.
 *
 *  There is no sign, just a density. We expect valency[0] and valency[1]
 *  to be \pm 1.
 *
 *****************************************************************************/

static int test_charge1_set(psi_t * psi) {

  int nk;
  int ic, jc, kc, index;
  int nlocal[3];
  
  double rho0, rho1;

  double rho_min[4];  /* For psi_stats */
  double rho_max[4];  /* For psi_stats */
  double rho_tot[4];  /* For psi_stats */

  coords_nlocal(nlocal);

  rho0 = 1.0 / (2.0*L(X)*L(Y));           /* Edge values */
  rho1 = 1.0 / (L(X)*L(Y)*(L(Z) - 2.0));  /* Interior values */

  psi_nk(psi, &nk);
  assert(nk == 2);
  
  /* Throughout set to rho1 */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_psi_set(psi, index, 0.0);
	psi_rho_set(psi, index, 0, 0.0);
	psi_rho_set(psi, index, 1, rho1);
      }
    }
  }

  /* Now overwrite at the edges with rho0 */

  if (cart_coords(Z) == 0) {

    kc = 1;
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	index = coords_index(ic, jc, kc);

	psi_rho_set(psi, index, 0, rho0);
	psi_rho_set(psi, index, 1, 0.0);
      }
    }
  }

  if (cart_coords(Z) == cart_size(Z) - 1) {

    kc = nlocal[Z];
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	index = coords_index(ic, jc, kc);

	psi_rho_set(psi, index, 0, rho0);
	psi_rho_set(psi, index, 1, 0.0);
      }
    }
  }

  psi_stats_reduce(psi, rho_min, rho_max, rho_tot, 0, pe_comm());

  if (pe_rank() == 0) {
    /* psi all zero */
    assert(fabs(rho_min[0] - 0.0) < DBL_EPSILON);
    assert(fabs(rho_max[0] - 0.0) < DBL_EPSILON);
    assert(fabs(rho_tot[0] - 0.0) < DBL_EPSILON);
    /* First rho0 interior */
    assert(fabs(rho_min[1] - 0.0) < DBL_EPSILON);
    assert(fabs(rho_max[1] - rho0) < DBL_EPSILON);
    assert(fabs(rho_tot[1] - 1.0) < DBL_EPSILON);
    /* Next rho1 edge */
    assert(fabs(rho_min[2] - 0.0) < DBL_EPSILON);
    assert(fabs(rho_max[2] - rho1) < DBL_EPSILON);
    assert(fabs(rho_tot[2] - 1.0) < FLT_EPSILON);
    /* Total rho_elec */
    assert(fabs(rho_min[3] + rho1) < DBL_EPSILON); /* + because valency is - */
    assert(fabs(rho_max[3] - rho0) < DBL_EPSILON);
    assert(fabs(rho_tot[3] - 0.0) < FLT_EPSILON);
  }

  return 0;
}
