/*****************************************************************************
 *
 *  fe_lc_blue_phase.h
 *
 *  Landau-de Gennes free energy with tensor order appropriate for
 *  liquid crystals including blue phases ("one elastic constant").
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2015 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *    Juho Lintuvuori
 *    Davide Marenduzzo
 *
 *****************************************************************************/

#ifndef FE_LC_BLUEPHASE_H
#define FE_LC_BLUEPHASE_H

#include "pe.h"
#include "fe.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_lcbp_param_s fe_lcbp_param_t;
typedef struct fe_lcbp_s fe_lcbp_t;

struct fe_lcbp_param_s {
  double a0;            /* Bulk free energy parameter A_0 */
  double gamma;         /* Controls magnitude of order */
  double kappa0;        /* Elastic constant \kappa_0 */
  double kappa1;        /* Elastic constant \kappa_1 */
  double q0;            /* Pitch = 2pi / q0 */

  double xi;            /* effective molecular aspect ration (<= 1.0) */
  double zeta;          /* Apolar activity constant */

  double redshift;      /* Redshift parameter */
  double rredshift;     /* Reciprocal of current redshift parameter */

  double epsilon;       /* Dielectric anisotropy (e/12pi) */
  double electric[3];   /* External electric field */
};

/* Host only */

__host__ int fe_lcbp_create(field_t * f, field_grad_t * grd,
			    fe_lcbp_t ** p);
__host__ int fe_lcbp_free(fe_lcbp_t * fe);
__host__ int fe_lcbp_param_set(fe_lcbp_t * fe, fe_lcbp_param_t values);


/* Host / device functions */

__host__ __device__ int fe_lcbp_param(fe_lcbp_t * fe, fe_lcbp_param_t * vals);
__host__ __device__ int fe_lcbp_fed(fe_lcbp_t * fe, int index, double * fed);
__host__ __device__ int fe_lcbp_mol_field(fe_lcbp_t * fe, int index,
					  double h[3][3]);
__host__ __device__ int fe_lcbp_stress(fe_lcbp_t * fe, int index,
				       double s[3][3]);

__host__ __device__ int fe_lcbp_compute_fed(fe_lcbp_t * fe, double q[3][3],
					    double dq[3][3][3], double * fed);
__host__ __device__ int fe_lcbp_compute_h(fe_lcbp_t * fe, double q[3][3],
					  double dq[3][3][3],
					  double dsq[3][3], double h[3][3]);
__host__ __device__ int fe_lcbp_compute_stress(fe_lcbp_t * fe, double q[3][3],
					       double dq[3][3][3],
					       double h[3][3],
					       double sth[3][3]);

__host__ __device__ int fe_lcbp_chirality(fe_lcbp_t * fe, double * chirality);
__host__ __device__ int fe_lcbp_reduced_temperature(fe_lcbp_t * fe,
						    double * tau);
__host__ __device__ int fe_lcbp_amplitude_compute(fe_lcbp_t * fe, double * a);
__host__ __device__ int fe_lcbp_dimensionless_field_strength(fe_lcbp_t * fe,
							     double * edm);

#endif
