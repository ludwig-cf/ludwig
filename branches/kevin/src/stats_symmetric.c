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
 *  (c) 2011-2013 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ignacio Pagonabarraga (ipagonabarraga@ub.edu)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>

#include "util.h"
#include "symmetric.h"
#include "stats_symmetric.h"

/*****************************************************************************
 *
 *  stats_symmetric_length
 *
 *  Computes estimates of domain length scales in real space based on
 *  a gradient statistic.
 *
 *****************************************************************************/

int stats_symmetric_length(coords_t * cs, field_grad_t * phi_grad, map_t * map,
			   int timestep) {

#define NSTAT 7

  int ic, jc, kc, index, ia;
  int nlocal[3];
  int status;

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
  assert(map);

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
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

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

/*****************************************************************************
 *
 *  stats_symmetric_moment_inertia
 *
 *  This computes the moment of inertia (e.g, for a droplet test).
 *  First, we have to locate the centre of the droplet r0. Then we
 *  can work out contributions to the moment of interia tensor
 *
 *      M_ab = (r_a - r0)*(r_b - r0)*phi
 *
 *  The eigenvalues and eigenvectors of this matrix are used to
 *  provide a measure of the shape of the drop.
 *
 *  We assume the centre of the droplet is identified by phi < 0.
 *  
 *****************************************************************************/

int stats_symmetric_moment_inertia(coords_t * cs, field_t * phi, map_t * map,
				   int timestep) {

  int ic, jc, kc, index;
  int nlocal[3], noffset[3];
  int status;

  double phi0;
  double rr[4], rr_global[4];           /* Centre of droplet calculation */
  double mom[6];                        /* Flattened tensor contribution */
  double mom_global[6];                 /* Flattened tensor */
  double inertia[3][3];                 /* Final tensor */

  double alpha, beta;                   /* Angles in 'natural' picture */
  double eigenvals[3], eigenvecs[3][3]; /* Eigenvalues / eigenvectors */

  MPI_Comm comm;

  assert(cs);
  assert(phi);
  assert(map);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
  coords_cart_comm(cs, &comm);

  rr[X] = rr[Y] = rr[Z] = rr[3] = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	field_scalar(phi, index, &phi0);

	if (phi0 < 0.0) {
	  rr[X] += ic + noffset[X];
	  rr[Y] += jc + noffset[Y];
	  rr[Z] += kc + noffset[Z];
	  rr[3] -= phi0;
	}
      }
    }
  }

  MPI_Reduce(rr, rr_global, 4, MPI_DOUBLE, MPI_SUM, 0, comm);
        
  rr_global[X] /= (1.0*rr_global[3]);
  rr_global[Y] /= (1.0*rr_global[3]);
  rr_global[Z] /= (1.0*rr_global[3]);

  mom[XX] = mom[XY] = mom[XZ] = mom[YY] = mom[YZ] = mom[ZZ] = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	field_scalar(phi, index, &phi0);
                                
	if (phi0 < 0.0) {
	  rr[X] = ic + noffset[X];
          rr[Y] = jc + noffset[Y];
	  rr[Z] = kc + noffset[Z];
	  mom[XX] -= (rr[X] - rr_global[X])*(rr[X] - rr_global[X])*phi0;
	  mom[XY] -= (rr[X] - rr_global[X])*(rr[Y] - rr_global[Y])*phi0;
	  mom[XZ] -= (rr[X] - rr_global[X])*(rr[Z] - rr_global[Z])*phi0;
	  mom[YY] -= (rr[Y] - rr_global[Y])*(rr[Y] - rr_global[Y])*phi0;
	  mom[YZ] -= (rr[Y] - rr_global[Y])*(rr[Z] - rr_global[Z])*phi0;
	  mom[ZZ] -= (rr[Z] - rr_global[Z])*(rr[Z] - rr_global[Z])*phi0;
	}
      }
    }
  }

  MPI_Reduce(mom, mom_global, 6, MPI_DOUBLE, MPI_SUM, 0, comm);

  inertia[X][X] = mom_global[XX];
  inertia[X][Y] = mom_global[XY];
  inertia[X][Z] = mom_global[XZ];
  inertia[Y][X] = inertia[X][Y];
  inertia[Y][Y] = mom_global[YY];
  inertia[Y][Z] = mom_global[YZ];
  inertia[Z][X] = inertia[X][Z];
  inertia[Z][Y] = inertia[Y][Z];
  inertia[Z][Z] = mom_global[ZZ];

  util_jacobi_sort(inertia, eigenvals, eigenvecs);

  /* Angles are defined... */

  alpha = atan2(eigenvecs[0][0], eigenvecs[1][0]);
  beta  = atan2(eigenvecs[2][0], eigenvecs[1][0]);

  info("\n");
  info("Droplet shape at time - %8d\n", timestep);
  info("[Droplet eigenvalues]   %8d %14.7e %14.7e %14.7e\n",
       timestep, eigenvals[X], eigenvals[Y], eigenvals[Z]);
  info("[Droplet angles]        %8d %14.7e %14.7e\n",
       timestep, alpha, beta);

  return 0;
}
