/*****************************************************************************
 *
 *  test_model.c
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "model.h"
#include "tests.h"

int main(int argc, char ** argv) {

  int i, j, k, p;
  int index, N[ND];
  double sum, sumx, sumy, sumz;
  double dij;
  double rho;
  double u[ND];

  pe_init(argc, argv);
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


  /* Test the periodic/processor halo swap:
   * (1) load f[0] at each site with the index,
   * (2) swap
   * (3) check halos. */

  info("\nHalo swap...\n\n");

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

  info("\nModel tests passed ok.\n\n");

  pe_finalise();

  return 0;
}
