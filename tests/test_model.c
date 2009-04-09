/*****************************************************************************
 *
 *  test_model.c
 *
 *****************************************************************************/

#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "model.h"
#include "tests.h"

void test_halo_swap();
void test_reduced_halo_swap();
int on_corner(int x, int y, int z, int mx, int my, int mz);


int xfwd[NVEL];
int xbwd[NVEL];
int yfwd[NVEL];
int ybwd[NVEL];
int zfwd[NVEL];
int zbwd[NVEL];

int main(int argc, char ** argv) {

  int i, j, k, p;
  double sum, sumx, sumy, sumz;
  double dij;
  double rho;
  double u[ND];

  pe_init(argc, argv);
  init_control();

  for (p = 0; p < NVEL; p++) {
    xfwd[p] = 0;
    xbwd[p] = 0;
    yfwd[p] = 0;
    ybwd[p] = 0;
    zfwd[p] = 0;
    zbwd[p] = 0;
  }
    
  for (i = 0; i < CVXBLOCK; i++) {
    for (k = 0; k < xblocklen_cv[i]; k++) {
      p = xdisp_fwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][X] == +1);
      xfwd[p] = 1;
      p = xdisp_bwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][X] == -1);
      xbwd[p] = 1;
    }
  }

  for (i = 0; i < CVYBLOCK; i++) {
    for (k = 0; k < yblocklen_cv[i]; k++) {
      p = ydisp_fwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][Y] == +1);
      yfwd[p] = 1;
      p = ydisp_bwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][Y] == -1);
      ybwd[p] = 1;
    }
  }

  for (i = 0; i < CVZBLOCK; i++) {
    for (k = 0; k < zblocklen_cv[i]; k++) {
      p = zdisp_fwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][Z] == +1);
      zfwd[p] = 1;
      p = zdisp_bwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][Z] == -1);
      zbwd[p] = 1;
    }
  }


  coords_init();


  info("Checking model.c objects...\n\n");

  /* Check we compiled the right model. */

  info("The number of dimensions appears to be ND = %d\n", ND);
  info("The model appears to have NVEL = %d\n", NVEL);

  /* Speed of sound */

  info("The speed of sound is 1/3... ");
  test_assert(fabs(rcs2 - 3.0) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  /* Kronecker delta */

  info("Checking Kronecker delta d_ij...");

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      if (i == j) {
	test_assert(fabs(d_[i][j] - 1.0) < TEST_DOUBLE_TOLERANCE);
      }
      else {
	test_assert(fabs(d_[i][j] - 0.0) < TEST_DOUBLE_TOLERANCE);
      }
    }
  }
  info("ok\n");

  info("Checking cv[0][X] = 0 etc...");

  test_assert(cv[0][X] == 0);
  test_assert(cv[0][Y] == 0);
  test_assert(cv[0][Z] == 0);

  info("ok\n");


  info("Checking cv[p][X] = -cv[NVEL-p][X] (p != 0) etc...");

  for (p = 1; p < NVEL; p++) {
    test_assert(cv[p][X] == -cv[NVEL-p][X]);
    test_assert(cv[p][Y] == -cv[NVEL-p][Y]);
    test_assert(cv[p][Z] == -cv[NVEL-p][Z]);
  }

  info("ok\n");

  /* Sum of quadrature weights, velcoities */

  info("Checking sum of wv[p]... ");

  sum = 0.0; sumx = 0.0; sumy = 0.0; sumz = 0.0;

  for (p = 0; p < NVEL; p++) {
    sum += wv[p];
    sumx += wv[p]*cv[p][X];
    sumy += wv[p]*cv[p][Y];
    sumz += wv[p]*cv[p][Z];
  }

  test_assert(fabs(sum - 1.0) < TEST_DOUBLE_TOLERANCE);
  info("ok\n"); 
  info("Checking sum of wv[p]*cv[p][X]... ");
  test_assert(fabs(sumx) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");
  info("Checking sum of wv[p]*cv[p][Y]... ");
  test_assert(fabs(sumy) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");
  info("Checking sum of wv[p]*cv[p][Z]... ");
  test_assert(fabs(sumz) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  /* Quadratic terms = cs^2 d_ij */

  info("Checking wv[p]*cv[p][i]*cv[p][j]...");

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      sum = 0.0;
      for (p = 0; p < NVEL; p++) {
	sum += wv[p]*cv[p][i]*cv[p][j];
      }
      test_assert(fabs(sum - d_[i][j]/rcs2) < TEST_DOUBLE_TOLERANCE);
    }
  }
  info("ok\n");


  /* Check q_ */

  info("Checking q_[p][i][j] = cv[p][i]*cv[p][j] - c_s^2*d_[i][j]...");

  for (p = 0; p < NVEL; p++) {
    for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; j++) {
	sum = cv[p][i]*cv[p][j] - d_[i][j]/rcs2;
	test_assert(fabs(sum - q_[p][i][j]) < TEST_DOUBLE_TOLERANCE);
      }
    }
  }

  info("ok\n");


  info("Checking wv[p]*q_[p][i][j]...");

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      sum = 0.0;
      for (p = 0; p < NVEL; p++) {
	sum += wv[p]*q_[p][i][j];
      }
      test_assert(fabs(sum - 0.0) < TEST_DOUBLE_TOLERANCE);
    }
  }
  info("ok\n");

  info("Checking wv[p]*cv[p][i]*q_[p][j][k]...");

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      for (k = 0; k < 3; k++) {
	sum = 0.0;
	for (p = 0; p < NVEL; p++) {
	  sum += wv[p]*cv[p][i]*q_[p][j][k];
	}
	test_assert(fabs(sum - 0.0) < TEST_DOUBLE_TOLERANCE);
      }
    }
  }
  info("ok\n");

  /* No actual test here yet. Requires a theoretical answer. */
  info("Checking d_[i][j]*q_[p][i][j]...\n");

  for (p = 0; p < NVEL; p++) {
    sum = 0.0;
    for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; j++) {
	sum += d_[i][j]*q_[p][i][j];
      }
    }
    /* test_assert(fabs(sum - 0.0) < TEST_DOUBLE_TOLERANCE);*/
    /* info("p = %d sum = %f\n", p, sum);*/
  }
  info("ok\n");


  info("Check ma_ against rho, cv ... ");

  for (p = 0; p < NVEL; p++) {
    test_assert(fabs(ma_[0][p] - 1.0) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(ma_[1][p] - cv[p][X]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(ma_[2][p] - cv[p][Y]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(ma_[3][p] - cv[p][Z]) < TEST_DOUBLE_TOLERANCE);
  }

  info("ok\n");

  info("Check ma_ against q_ ...");

  for (p = 0; p < NVEL; p++) {
    test_assert(fabs(ma_[4][p] - q_[p][X][X]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(ma_[5][p] - q_[p][X][Y]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(ma_[6][p] - q_[p][X][Z]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(ma_[7][p] - q_[p][Y][Y]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(ma_[8][p] - q_[p][Y][Z]) < TEST_DOUBLE_TOLERANCE);
    test_assert(fabs(ma_[9][p] - q_[p][Z][Z]) < TEST_DOUBLE_TOLERANCE);
  }

  info("ok\n");

  info("Checking normalisers norm_[i]*wv[p]*ma_[i][p]*ma_[j][p] = dij... ");

  for (i = 0; i < NVEL; i++) {
    for (j = 0; j < NVEL; j++) {
      dij = (double) (i == j);
      sum = 0.0;
      for (p = 0; p < NVEL; p++) {
	sum += norm_[i]*wv[p]*ma_[i][p]*ma_[j][p];
      }
      test_assert(fabs(sum - dij) < TEST_DOUBLE_TOLERANCE);
    }
  }
  info("ok\n");

  info("Checking ma_[i][p]*mi_[p][j] = dij ... ");

  for (i = 0; i < NVEL; i++) {
    for (j = 0; j < NVEL; j++) {
      dij = (double) (i == j);
      sum = 0.0;
      for (p = 0; p < NVEL; p++) {
	sum += ma_[i][p]*mi_[p][j];
      }
      test_assert(fabs(sum - dij) < TEST_DOUBLE_TOLERANCE);
    }
  }
  info("ok\n");


  /* Tests of the basic distribution functions. */

  info("\n\n\nDistribution functions:\n");
  info("Allocate lattice sites...\n");

  init_site();

  info("Allocated 1 site\n");

  info("Set rho = 1 at site... ");
  set_rho(1.0, 0);
  info("ok\n");

  info("Check rho is correct... ");
  rho = get_rho_at_site(0);
  test_assert(fabs(rho - 1.0) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  info("Check individual distributions... ");
  for (p = 0; p < NVEL; p++) {
    rho = get_f_at_site(0, p);
    test_assert(fabs(rho - wv[p]) < TEST_DOUBLE_TOLERANCE);
  }
  info("ok\n");

  info("Check momentum... ");
  get_momentum_at_site(0, u);
  for (i = 0; i < ND; i++) {
    test_assert(fabs(u[i] - 0.0) < TEST_DOUBLE_TOLERANCE);
  }
  info("ok\n");

  info("Set rho and u... ");

  for (i = 0; i < ND; i++) {
    u[i] = 0.001;
  }

  set_rho_u_at_site(1.0, u, 0);

  rho = get_rho_at_site(0);
  test_assert(fabs(rho - 1.0) < TEST_DOUBLE_TOLERANCE);

  for (i = 0; i < ND; i++) {
    test_assert(fabs(u[i] - 0.001) < TEST_DOUBLE_TOLERANCE);
  }
  info("ok\n");

  info("Set individual f[p]... ");

  for (p = 0; p< NVEL; p++) {
    set_f_at_site(0, p, wv[p]);
    rho = get_f_at_site(0, p);
    test_assert(fabs(rho - wv[p]) < TEST_DOUBLE_TOLERANCE);
  }
  rho = get_rho_at_site(0);
  test_assert(fabs(rho - 1.0) < TEST_DOUBLE_TOLERANCE);

  finish_site();
  info("ok\n");

  test_halo_swap();
  test_reduced_halo_swap();

  info("\nModel tests passed ok.\n\n");

  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_halo_swap
 *
 *  Test full halo swap.
 *
 *****************************************************************************/

void test_halo_swap() {
  int i, j, k, p;
  int index, N[ND];
  double u[ND];
  double rho;
  /* Test the periodic/processor halo swap:
   * (1) load f[0] at each site with the index,
   * (2) swap
   * (3) check halos. */

  info("\nHalo swap (full distributions)...\n\n");
  distribution_halo_set_complete();

  init_site();
  get_N_local(N);

  for (i = 1; i <= N[X]; i++) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {
	index = get_site_index(i, j, k);

	set_f_at_site(index, X, (double) (i));
	set_f_at_site(index, Y, (double) (j));
	set_f_at_site(index, Z, (double) (k));

	for (p = 3; p < NVEL; p++) {
	  set_f_at_site(index, p, (double) p);
	}
      }
    }
  }

  halo_site();

  for (i = 0; i <= N[X] + 1; i++) {
    for (j = 0; j <= N[Y] + 1; j++) {
      for (k = 0; k <= N[Z] + 1; k++) {
	index = get_site_index(i, j, k);

	u[X] = get_f_at_site(index, X);
	u[Y] = get_f_at_site(index, Y);
	u[Z] = get_f_at_site(index, Z);
	
	if (i == 0) {
	  test_assert(fabs(u[X] - (double) N[X]) < TEST_DOUBLE_TOLERANCE);
	}
	if (i == N[X] + 1) {
	  test_assert(fabs(u[X] - 1.0) < TEST_DOUBLE_TOLERANCE);
	}

	if (j == 0) {
	  test_assert(fabs(u[Y] - (double) N[Y]) < TEST_DOUBLE_TOLERANCE);
	}
	if (j == N[Y] + 1) {
	  test_assert(fabs(u[Y] - 1.0) < TEST_DOUBLE_TOLERANCE);
	}

	if (k == 0) {
	  test_assert(fabs(u[Z] - (double) N[Z]) < TEST_DOUBLE_TOLERANCE);
	}
	if (k == N[Z] + 1) {
	  test_assert(fabs(u[Z] - 1.0) < TEST_DOUBLE_TOLERANCE);
	}

	for (p = 3; p < NVEL; p++) {
	  rho = get_f_at_site(index, p);
	  test_assert(fabs(rho - (double) p) < TEST_DOUBLE_TOLERANCE);
	}
      }
    }
  }

  info("Halo swap ok\n");
  finish_site();
}

/*****************************************************************************
 *
 *  test_reduced_halo_swap
 *
 *****************************************************************************/

void test_reduced_halo_swap() {  
  int index, N[ND];
  double f;  
  int i, j, k, p;

  info("\nHalo swap (reduced)...\n\n");
  distribution_halo_set_reduced();

  coords_init();
  init_site();
  get_N_local(N);
  
  /* Set everything which is NOT in a halo */
  for (i = 1; i <= N[X]; i++) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {
	index = get_site_index(i, j, k);
	for (p = 0; p < NVEL; p++) {
	  set_f_at_site(index, p, (double) p);
	}
      }
    }
  }

  /* do swap */
  halo_site();

  /* Now check that the sites are right still
   * Also check the halos sites now
   */
  for (i = 0; i <= N[X]+1; i++) {
    for (j = 0; j <= N[Y]+1; j++) {
      for (k = 0; k <= N[Z]+1; k++) {
	index = get_site_index(i, j, k);

	for(p = 0; p < NVEL; p++) {
	  f = get_f_at_site(index, p);
	  if(i == 0 && !on_corner(i, j, k, N[X]+1, N[Y]+1, N[Z]+1)) {
	    if(xfwd[p] > 0) {
	      test_assert(fabs(f - p) < TEST_DOUBLE_TOLERANCE);
	    }
	  }
	  if(i == N[X]+1 && !on_corner(i, j, k, N[X]+1, N[Y]+1, N[Z]+1)) {
	    if(xbwd[p] > 0) {
	      test_assert(fabs(f - p) < TEST_DOUBLE_TOLERANCE);  
	    }
	  }
	  if(j == 0 && !on_corner(i, j, k, N[X]+1, N[Y]+1, N[Z]+1)) {
	    if(yfwd[p] > 0) {
	      test_assert(fabs(f - p) < TEST_DOUBLE_TOLERANCE);
	    }
	  }
	  if(j == N[Y]+1 && !on_corner(i, j, k, N[X]+1, N[Y]+1, N[Z]+1)) {
	    if(ybwd[p] > 0) {
	      test_assert(fabs(f - p) < TEST_DOUBLE_TOLERANCE);  
	    }
	  }
	  if(k == 0 && !on_corner(i, j, k, N[X]+1, N[Y]+1, N[Z]+1)) {
	    if(zfwd[p] > 0) {
	      test_assert(fabs(f - p) < TEST_DOUBLE_TOLERANCE);
	    }
	  }
	  if(k == N[Z]+1 && !on_corner(i, j, k, N[X]+1, N[Y]+1, N[Z]+1)) {
	    if(zbwd[p] > 0) {
	      test_assert(fabs(f - p) < TEST_DOUBLE_TOLERANCE);  
	    }
	  }
	  
	  /* Check the interior is still the same. */
	  if(i > 0 && j > 0 && k > 0 &&
	     i < N[X]+1 && j < N[Y]+1 && k < N[Z]+1) {
	    test_assert(fabs(f - p) < TEST_DOUBLE_TOLERANCE);
	  }
	}

      }
    }
  }

  info("Reduced halo swapping... ok");

}


/*************************************
 *
 * Returns 0(false) if on a corner,
 *         1(true)  otherwise.
 *
 *************************************/

int on_corner(int x, int y, int z, int mx, int my, int mz) {

  int iscorner = 0;

  /* on the axes */

  if (fabs(x) + fabs(y) == 0 || fabs(x) + fabs(z) == 0 ||
      fabs(y) + fabs(z) == 0 ) {
    iscorner = 1;
  }

  /* opposite corners from axes */

  if ((x == mx && y == my) || (x == mx && z == mz) || (y == my && z == mz)) {
    iscorner = 1;
  }
  
  if ((x == 0 && y == my) || (x == 0 && z == mz) || (y == 0 && x == mx) ||
      (y == 0 && z == mz) || (z == 0 && x == mx) || (z == 0 && y == my)) {
    iscorner = 1;
  }

  return iscorner;
}
