/*****************************************************************************
 *
 *  blue_phase.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *    Juho Lintuvuori
 *    Davide Marenduzzo
 *
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef BLUEPHASE_H
#define BLUEPHASE_H

#include "free_energy.h"
#include "field.h"
#include "field_grad.h"
#include "io_harness.h"

typedef struct fe_lc_s fe_lc_t;
typedef struct fe_lc_param_s fe_lc_param_t;

/* Free energy structure */

struct fe_lc_s {
  fe_t super;
  fe_lc_param_t * param;      /* Parameters */
  field_t * q;                /* Q_ab (compresse rank 1 field) */
  field_grad_t * dq;          /* Gradients thereof */
  fe_lc_t * target;           /* Device structure */
};

/* Liquid crystal free energy parameters */

struct fe_lc_param_s {
  double a0;
  double q0;
  double gamma;
  double kappa0;
  double kappa1;

  double xi;
  double zeta;
  double redshift;                        /* Redshift */
  double rredshift;                       /* Reciprocal redshift */
  double epsilon;                         /* Dielectric anistropy */
  double amplitude0;                      /* Initial amplitude from input */
  double e0coswt[3];                      /* Electric field */

  double w1_coll;                         /* Anchoring strength parameter */
  double w2_coll;                         /* Second anchoring parameter */
  double w1_wall;
  double w2_wall;

  int anchoring_coll;                     /* Colloids anchoring type */
  int anchoring_wall;                     /* Wall anchoring type */
  int is_redshift_updated;                /* Switch */
};

/* Surface anchoring types */
enum lc_anchoring_enum {LC_ANCHORING_PLANAR = 0,
			LC_ANCHORING_NORMAL, 
			LC_ANCHORING_FIXED,
			LC_ANCHORING_TYPES /* Last entry */
};


__host__ int fe_lc_create(field_t * q, field_grad_t * dq, fe_lc_t ** fe);
__host__ int fe_lc_free(fe_lc_t * fe);
__host__ int fe_lc_param_set(fe_lc_t * fe, fe_lc_param_t values);
__host__ int fe_lc_param_commit(fe_lc_t * fe);
__host__ int fe_lc_redshift_set(fe_lc_t * fe,  double redshift);
__host__ int fe_lc_redshift_compute(fe_lc_t * fe);
__host__ int fe_lc_target(fe_lc_t * fe, fe_t ** target);

/* Host / device functions */

__host__ __device__
int fe_lc_param(fe_lc_t * fe, fe_lc_param_t * vals);

__host__ __device__
int fe_lc_fed(fe_lc_t * fe, int index, double * fed);

__host__ __device__
int fe_lc_mol_field(fe_lc_t * fe, int index, double h[3][3]);

__host__ __device__
int fe_lc_stress(fe_lc_t * fe, int index, double s[3][3]);

__host__ __device__
int fe_lc_compute_fed(fe_lc_t * fe, double gamma,  double q[3][3],
		      double dq[3][3][3], double * fed);

__host__ __device__
int fe_lc_compute_h(fe_lc_t * fe, double gaama,	double q[3][3],
		    double dq[3][3][3],	double dsq[3][3], double h[3][3]);

__host__ __device__
int fe_lc_compute_stress(fe_lc_t * fe, double q[3][3], double dq[3][3][3],
			 double h[3][3], double sth[3][3]);

__host__ __device__
int fe_lc_chirality(fe_lc_t * fe, double * chirality);

__host__ __device__
int fe_lc_reduced_temperature(fe_lc_t * fe,  double * tau);

__host__ __device__
int fe_lc_dimensionless_field_strength(fe_lc_t * fe, double * edm);


__host__ __device__
void fe_lc_stress_v(fe_lc_t * fe, int index, double s[3][3][NSIMDVL]);

__host__ __device__
void fe_lc_compute_h_v(fe_lc_t * fe,
		       double q[3][3][NSIMDVL], 
		       double dq[3][3][3][NSIMDVL],
		       double dsq[3][3][NSIMDVL], 
		       double h[3][3][NSIMDVL]);
__host__ __device__
void fe_lc_compute_stress_v(fe_lc_t * fe,
			    double q[3][3][NSIMDVL],
			    double dq[3][3][3][NSIMDVL],
			    double h[3][3][NSIMDVL],
			    double s[3][3][NSIMDVL]);


/* Function of the parameters only */

__host__ __device__
int fe_lc_amplitude_compute(fe_lc_param_t * param, double * a);

__host__ __device__
int fe_lc_q_uniaxial(fe_lc_param_t * param, const double n[3], double q[3][3]);
__host__ int fe_lc_scalar_ops(double q[3][3], double qs[NQAB]);


#endif
 
