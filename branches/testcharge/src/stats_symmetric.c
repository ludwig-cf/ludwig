/*****************************************************************************
 *
 *  stats_symmetric.c
 *
 *  Statistics related to the symetric free energy Model H, binary fluid.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "site_map.h"
#include "symmetric.h"
#include "field_grad.h"

/*****************************************************************************
 *
 *  stats_symmetric_length
 *
 *  Computes estimates of domain length scales in real space based on
 *  a gradient statistic.
 *
 *  The reduction is in pe_comm() for output.
 *
 *****************************************************************************/

int stats_symmetric_length(field_grad_t * phi_grad, int timestep) {

#define NSTAT 7

  int ic, jc, kc, index, ia;
  int nlocal[3];

  double a, b;                          /* Free energy parameters */
  double xi0;                           /* interfacial width */

  double dphi[3];                       /* grad phi */
  double dphi_local[NSTAT];             /* Local sum grad phi (plus volume) */
  double dphi_total[NSTAT];             /* Global sum (plus volume) */
  double dphiab[3][3];                  /* Gradient tensor d_a phi d_b phi */

  double lcoordinate[3];                /* Lengths in coordinate directions */
  double lnatural[3];                   /* Lengths in 'natural' directions */
  double alpha, beta;                   /* Angles in 'natural' picture */

  double eigenvals[3], eigenvecs[3][3];
  double rvolume;

  MPI_Comm comm;

  assert(phi_grad);

  coords_nlocal(nlocal);
  comm = pe_comm();

  a = symmetric_a();
  b = symmetric_b();
  xi0 = symmetric_interfacial_width();

  for (ia = 0; ia < NSTAT; ia++) {
    dphi_local[ia] = 0.0;
  }

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	if (site_map_get_status_index(index) != FLUID) continue;

	field_grad_scalar_grad(phi_grad, index, dphi);

	dphi_local[0] += dphi[X]*dphi[X];
        dphi_local[1] += dphi[X]*dphi[Y];
        dphi_local[2] += dphi[X]*dphi[Z];
        dphi_local[3] += dphi[Y]*dphi[Y];
        dphi_local[4] += dphi[Y]*dphi[Z];
        dphi_local[5] += dphi[Z]*dphi[Z];
	dphi_local[6] += 1.0;
      }
    }
  }

  MPI_Reduce(dphi_local, dphi_total, NSTAT, MPI_DOUBLE, MPI_SUM, 0, comm);

  /* Set up the normalised gradient tensor. */

  assert(dphi_total[6] > 0.0);

  rvolume = 1.0/dphi_total[6];
  dphiab[X][X] = rvolume*dphi_total[0];
  dphiab[X][Y] = rvolume*dphi_total[1];
  dphiab[X][Z] = rvolume*dphi_total[2];
  dphiab[Y][X] = dphiab[X][Y];
  dphiab[Y][Y] = rvolume*dphi_total[3];
  dphiab[Y][Z] = rvolume*dphi_total[4];
  dphiab[Z][X] = dphiab[X][Z];
  dphiab[Z][Y] = dphiab[X][Y];
  dphiab[Z][Z] = rvolume*dphi_total[5];

  /* Length scales in coordinate directions, and natural directions */

  for (ia = 0; ia < 3; ia++) {
    lcoordinate[ia] = -4.0*a/(3.0*b*xi0*dphiab[ia][ia]);
  }

  util_jacobi_sort(dphiab, eigenvals, eigenvecs);

  for (ia = 0; ia < 3; ia++) {
    lnatural[ia] = -4.0*a/(3.0*b*xi0*eigenvals[ia]);
  }

  /* Angles are defined... */

  alpha = atan2(eigenvecs[0][0], eigenvecs[1][0]);
  beta  = atan2(eigenvecs[2][0], eigenvecs[1][0]);

  info("\n");
  info("[length xyz] %8d %14.7e %14.7e %14.7e\n", timestep,
       lcoordinate[X], lcoordinate[Y], lcoordinate[Z]);
  info("[length abc] %8d %14.7e %14.7e %14.7e\n", timestep,
       lnatural[0], lnatural[1], lnatural[2]);
  info("[angles abc] %8d %14.7e %14.7e\n", timestep, alpha, beta);

#undef NSTAT

  return 0;
}

