/*****************************************************************************
 *
 *  test.c
 *
 *  Statistics on fluid/particle conservation laws.
 *  Single fluid and binary fluid.
 *
 *  $Id: test.c,v 1.13 2008-08-24 16:36:26 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "site_map.h"
#include "model.h"
#include "physics.h"
#include "bbl.h"
#include "phi.h"
#include "free_energy.h"
#include "test.h"


extern Site * site;

/*****************************************************************************
 *
 *  TEST_statistics
 *
 *  Report the minimum, maximum, total, mean and variance of rho, phi
 *  over the whole system.
 *
 *  In the presence of solid particles, the current deficit in order
 *  parameter is taken into account in working out the total phi.
 *
 *****************************************************************************/

void TEST_statistics() {

  double rhosum, phisum;
  double phibar, rhobar;
  double phivar, rhovar;
  double rho, phi;
  double rfluid;
  double partsum[3], partmin[2], partmax[2];

  int     i, j, k, p, index;
  int     N[3];

#ifdef _MPI_
  double g_sum[3], g_min[2], g_max[2];
#endif

  get_N_local(N);

  partsum[0] =  0.0;
  partsum[1] =  bbl_order_parameter_deficit();
  partsum[2] =  0.0;     /* volume of fluid */

  partmin[0] =  2.0;     /* rho_min */
  partmax[0] =  0.0;     /* rho_max */
  rhovar     =  0.0;

  partmin[1] = +1.0;     /* phi_min */
  partmax[1] = -1.0;     /* phi_max */
  phivar     =  0.0;

  phi_compute_phi_site();

  /* Accumulate the sums, minima, and maxima */

  for (i = 1; i <= N[X]; i++) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {

	if (site_map_get_status(i, j, k) != FLUID) continue;
	index = get_site_index(i, j, k);

	rho = site[index].f[0];

	for (p = 1; p < NVEL; p++) {
	  rho += site[index].f[p];
	}

	if (rho < partmin[0]) partmin[0] = rho;
	if (rho > partmax[0]) partmax[0] = rho;
	partsum[0] += rho;

	phi = phi_get_phi_site(index);
	if (phi < partmin[1]) partmin[1] = phi;
	if (phi > partmax[1]) partmax[1] = phi;
	partsum[1] += phi;

	partsum[2] += 1.0;
      }
    }
  }

#ifdef _MPI_
  MPI_Allreduce(partsum, g_sum, 3, MPI_DOUBLE, MPI_SUM, cart_comm());

  partsum[0] = g_sum[0];
  partsum[1] = g_sum[1];
  partsum[2] = g_sum[2];
#endif

  rhosum = partsum[0];
  phisum = partsum[1];
  rfluid = 1.0/partsum[2];

  rhobar = rhosum*rfluid;
  phibar = phisum*rfluid;


  /* Have to go round again to get the variances... */

  partsum[0] = 0.0;
  partsum[1] = 0.0;

  for (i = 1; i <= N[X]; i++) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {

	if (site_map_get_status(i, j, k) != FLUID) continue;
	index = get_site_index(i, j, k);

	rho = site[index].f[0];

	for (p = 1; p < NVEL; p++) {
	  rho += site[index].f[p];
	}

	phi = phi_get_phi_site(index);
	partsum[0] += (rho - rhobar)*(rho - rhobar);
	partsum[1] += (phi - phibar)*(phi - phibar);

      }
    }
  }

#ifdef _MPI_

  MPI_Reduce(partsum, g_sum, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(partmin, g_min, 2, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(partmax, g_max, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  partsum[0] = g_sum[0];
  partsum[1] = g_sum[1];

  partmin[0] = g_min[0];
  partmin[1] = g_min[1];
  partmax[0] = g_max[0];
  partmax[1] = g_max[1];
#endif

  rhovar = partsum[0]*rfluid;
  phivar = partsum[1]*rfluid;

  info("\nTEST_statistics [total, mean, variance, min, max]\n");
  info("[rho][%.8g, %.8g, %.8g, %.8g, %.8g]\n", rhosum, rhobar, rhovar,
       partmin[0], partmax[0]);
  info("[phi][%.8g, %.8g, %.8g, %.8g, %.8g]\n", phisum, phibar, phivar,
       partmin[1], partmax[1]);

  return;
}


/*****************************************************************************
 *
 *  TEST_momentum
 *
 *  Compute the total system momentum (fluid+colloids).
 *
 *****************************************************************************/

void TEST_momentum() {

  int       ic, jc, kc, index;
  int       N[3];
  int       p;

  double     gx, gy, gz, cx, cy, cz;
  double     mass;
  double   * f;

  Colloid * p_colloid;


  /* Work out the fluid momentum (gx, gy, gz) */
  gx = gy = gz = 0.0;

  get_N_local(N);

  for (ic = 1; ic <= N[X]; ic++) {
    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {

	if (site_map_get_status(ic, jc, kc) != FLUID) continue;
	index = get_site_index(ic, jc, kc);

	f = site[index].f;

	for (p = 1; p < NVEL; p++) {
	  gx += cv[p][0]*f[p];
	  gy += cv[p][1]*f[p];
	  gz += cv[p][2]*f[p];
	}
      }
    }
  }

  /* Work out the net colloid momemtum (cx, cy, cz) */
  cx = cy = cz = 0.0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {

	  mass = 4.0*PI*pow(p_colloid->a0, 3)/3.0;

	  cx += mass*p_colloid->v.x;
	  cy += mass*p_colloid->v.y;
	  cz += mass*p_colloid->v.z;

	  /* Next colloid */
	  p_colloid = p_colloid->next;
	}

	/* Next cell */
      }
    }
  }

#ifdef _MPI_
  {
    double   partsum[6], g_sum[6];
    partsum[0] = gx;
    partsum[1] = gy;
    partsum[2] = gz;
    partsum[3] = cx;
    partsum[4] = cy;
    partsum[5] = cz;

    MPI_Reduce(partsum, g_sum, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    gx = g_sum[0];
    gy = g_sum[1];
    gz = g_sum[2];
    cx = g_sum[3];
    cy = g_sum[4];
    cz = g_sum[5];
  }
#endif

  info("\nTEST_momentum [x, y, z]\n");
  info("[total][%g, %g, %g]\n", gx + cx,  gy + cy,  gz + cz);
  info("[fluid][%g, %g, %g]\n", gx,       gy,       gz);

  return;
}


/*****************************************************************************
 *
 *  TEST_fluid_temperature
 *
 *  This computes and reports on the statistics of the fluid
 *  temperature when fluctauations are present.
 *
 *  For the temperature, a contribution to a long-term mean is
 *  accumulated each time the fuction is called, and the time
 *  mean reported (for what it's worth).
 *
 *****************************************************************************/

void TEST_fluid_temperature() {

  double   uvar, uxvar, uyvar, uzvar;
  double   rhovar, chi2var;
  double   rfluid;
  int      i, j, k, index, p;
  int      N[3];
  double   rho, ux, uy, uz, chi2;
  double   *f;

  get_N_local(N);

  uvar    = 0.0;   /* Total u variance */
  uxvar   = 0.0;   /* u_x variance */
  uyvar   = 0.0;   /* u_y variance */
  uzvar   = 0.0;   /* u_z variance */
  rhovar  = 0.0;   /* (1 - rho)^2  */
  chi2var = 0.0;   /* chi2 ghost mode variance */
  rfluid  = 0.0;   /* Fluid volume */

  /* Single loop: variances are computed assuming the appropriate
   * means are well-behaved (i.e., mean of u is zero, mean of rho
   * is 1) */ 

  for (i = 1; i <= N[X]; i++) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {

	if (site_map_get_status(i, j, k) == FLUID) {
	  index = get_site_index(i, j, k);
	  f = site[index].f;

	  rho  = f[0];
	  ux   = 0.0;
	  uy   = 0.0;
	  uz   = 0.0;
	  chi2 = 0.0;        /* Ghost mode temperature */

	  for (p = 1; p < NVEL; p++) {
	    rho  += f[p];
	    ux   += f[p]*cv[p][0];
	    uy   += f[p]*cv[p][1];
	    uz   += f[p]*cv[p][2];
	    chi2 += f[p]*cv[p][0]*cv[p][1]*cv[p][2];
	  }

	  ux = ux/rho;
	  uy = uy/rho;
	  uz = uz/rho;

	  uvar    += ux*ux + uy*uy + uz*uz;
	  uxvar   += ux*ux;
	  uyvar   += uy*uy;
	  uzvar   += uz*uz;
	  rhovar  += (1.0 - rho)*(1.0 - rho);
	  chi2var += 9.0*chi2*chi2;
	  rfluid  += 1.0;
	}
      }
    }
  }

#ifdef _MPI_
  {
    double   partsum[7], g_sum[7];
    partsum[0] = uvar;
    partsum[1] = uxvar;
    partsum[2] = uyvar;
    partsum[3] = uzvar;
    partsum[4] = rhovar;
    partsum[5] = chi2var;
    partsum[6] = rfluid;

    MPI_Reduce(partsum, g_sum, 7, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    uvar    = g_sum[0];
    uxvar   = g_sum[1];
    uyvar   = g_sum[2];
    uzvar   = g_sum[3];
    rhovar  = g_sum[4];
    chi2var = g_sum[5];
    rfluid  = g_sum[6];
  }
#endif

  rfluid  = 1.0/rfluid;

  uvar    = uvar*rfluid;
  uxvar   = uxvar*rfluid;
  uyvar   = uyvar*rfluid;
  uzvar   = uzvar*rfluid;
  rhovar  = rhovar*rfluid;
  chi2var = chi2var*rfluid;

  info("TEST_fluid_temperature:\n");
  info("  <v_x^2> = %g\n", uxvar);
  info("  <v_y^2> = %g\n", uyvar);
  info("  <v_z^2> = %g\n", uzvar);
  info("   <mv^2> = %g (target: %g)\n", uvar, get_kT()*ND);
  info(" <drho^2> = %g\n", rhovar);
  info("  <ghost> = %g\n", chi2var);

  return;
}

/*****************************************************************************
 *
 *  test_rheology
 *
 *****************************************************************************/

void test_rheology() {

  int get_step(void);

  double stress[3][3];
  double sneq[3][3];
  double rhouu[3][3];
  double pchem[3][3], plocal[3][3];
  double s[3][3];
  double u[3];
  double rho, rrho, rv;
  int nlocal[3];
  int ic, jc, kc, index, p, ia, ib;

  get_N_local(nlocal);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      stress[ia][ib] = 0.0;
      plocal[ia][ib] = 0.0;
      pchem[ia][ib] = 0.0;
      rhouu[ia][ib] = 0.0;
      sneq[ia][ib] = 0.0;
    }
  }

  /* Accumulate contributions to the stress */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

        rho = 0.0;
        for (ia = 0; ia < 3; ia++) {
          u[ia] = 0.0;
          for (ib = 0; ib < 3; ib++) {
            s[ia][ib] = 0.0;
          }
        }
        
        for (p = 0; p < NVEL; p++) {
          rho     += site[index].f[p]*ma_[MRHO][p];
          u[X]    += site[index].f[p]*ma_[MRUX][p];
          u[Y]    += site[index].f[p]*ma_[MRUY][p];
          u[Z]    += site[index].f[p]*ma_[MRUZ][p];
          s[X][X] += site[index].f[p]*ma_[MSXX][p];
          s[X][Y] += site[index].f[p]*ma_[MSXY][p];
          s[X][Z] += site[index].f[p]*ma_[MSXZ][p];
          s[Y][Y] += site[index].f[p]*ma_[MSYY][p];
          s[Y][Z] += site[index].f[p]*ma_[MSYZ][p];
          s[Z][Z] += site[index].f[p]*ma_[MSZZ][p];
        }

#ifdef _SINGLE_FLUID_
#else
	free_energy_get_chemical_stress(index, plocal);
#endif

        pchem[X][X] += plocal[X][X];
        pchem[X][Y] += plocal[X][Y];
        pchem[X][Z] += plocal[X][Z];
        pchem[Y][Y] += plocal[Y][Y];
        pchem[Y][Z] += plocal[Y][Z];
        pchem[Z][Z] += plocal[Z][Z];

        rrho = 1.0/rho;
        for (ia = 0; ia < 3; ia++) {
          for (ib = 0; ib < 3; ib++) {
            rhouu[ia][ib] += rrho*u[ia]*u[ib];
            stress[ia][ib] += s[ia][ib];
            sneq[ia][ib] += (s[ia][ib] - rrho*u[ia]*u[ib]);
          }
        }

      }
    }
  }

#ifdef _MPI_
  {
    double send[24];
    double recv[24];

    kc = 0;
    for (ic = 0; ic < 3; ic++) {
      for (jc = 0; jc < 3; jc++) {
        if (ic <= jc) {
          send[kc++] = stress[ic][jc];
          send[kc++] = pchem[ic][jc];
          send[kc++] = rhouu[ic][jc];
          send[kc++] = sneq[ic][jc];
        }
      }
    }

    MPI_Reduce(send, recv, 24, MPI_DOUBLE, MPI_SUM, 0, cart_comm());

    kc = 0;
    for (ic = 0; ic < 3; ic++) {
      for (jc = 0; jc < 3; jc++) {
        if (ic <= jc) {
          stress[ic][jc] = recv[kc++];
          pchem[ic][jc] = recv[kc++];
          rhouu[ic][jc] = recv[kc++];
          sneq[ic][jc] = recv[kc++];
        }
      }
    }

  }
#endif

  rv = 1.0/(L(X)*L(Y)*L(Z));
  info("stress_hy x %d %12.6g %12.6g %12.6g\n", get_step(),
       rv*stress[X][X], rv*stress[X][Y], rv*stress[X][Z]);
  info("stress_hy y %d %12.6g %12.6g %12.6g\n", get_step(),
       rv*stress[X][Y], rv*stress[Y][Y], rv*stress[Y][Z]);
  info("stress_hy z %d %12.6g %12.6g %12.6g\n", get_step(),
       rv*stress[X][Z], rv*stress[Y][Z], rv*stress[Z][Z]);

  info("stress_th x %d %12.6g %12.6g %12.6g\n", get_step(),
       rv*pchem[X][X], rv*pchem[X][Y], rv*pchem[X][Z]);
  info("stress_th y %d %12.6g %12.6g %12.6g\n", get_step(),
       rv*pchem[X][Y], rv*pchem[Y][Y], rv*pchem[Y][Z]);
  info("stress_th z %d %12.6g %12.6g %12.6g\n", get_step(),
       rv*pchem[X][Z], rv*pchem[Y][Z], rv*pchem[Z][Z]);

  info("stress_uu x %d %12.6g %12.6g %12.6g\n", get_step(),
       rv*rhouu[X][X], rv*rhouu[X][Y], rv*rhouu[X][Z]);
  info("stress_uu y %d %12.6g %12.6g %12.6g\n", get_step(),
       rv*rhouu[X][Y], rv*rhouu[Y][Y], rv*rhouu[Y][Z]);
  info("stress_uu z %d %12.6g %12.6g %12.6g\n", get_step(),
       rv*rhouu[X][Z], rv*rhouu[Y][Z], rv*rhouu[Z][Z]);

  info("stress_ne x %d %12.6g %12.6g %12.6g\n", get_step(),
       rv*sneq[X][X], rv*sneq[X][Y], rv*sneq[X][Z]);
  info("stress_ne y %d %12.6g %12.6g %12.6g\n", get_step(),
       rv*sneq[X][Y], rv*sneq[Y][Y], rv*sneq[Y][Z]);
  info("stress_ne z %d %12.6g %12.6g %12.6g\n", get_step(),
       rv*sneq[X][Z], rv*sneq[Y][Z], rv*sneq[Z][Z]);

  return;
}
