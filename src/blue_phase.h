/*****************************************************************************
 *
 *  fe_lc.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2018 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *    Juho Lintuvuori
 *    Davide Marenduzzo
 *
 *****************************************************************************/

#ifndef LUDWIG_FE_LC_H
#define LUDWIG_FE_LC_H

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"
#include "io_harness.h"

typedef struct fe_lc_s fe_lc_t;
typedef struct fe_lc_param_s fe_lc_param_t;

/* Free energy structure */

struct fe_lc_s {
  fe_t super;
  pe_t * pe;                  /* Parallel environment */
  cs_t * cs;                  /* Coordinate system */
  fe_lc_param_t * param;      /* Parameters */
  field_t * q;                /* Q_ab (compresse rank 1 field) */
  field_grad_t * dq;          /* Gradients thereof */
  field_t * p;                /* Active term P_a = Q_ak d_m Q_mk */
  field_grad_t * dp;          /* Active term gradient d_a P_b */
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
  double zeta0;                           /* active stress term delta_ab */
  double zeta1;                           /* active stress term Q_ab */
  double zeta2;                           /* active stress d_a P_b + d_b P_a */
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
  int is_active;                          /* Switch for active fluid */
};

/* Surface anchoring types */
enum lc_anchoring_enum {LC_ANCHORING_PLANAR = 0,
			LC_ANCHORING_NORMAL, 
			LC_ANCHORING_FIXED,
			LC_ANCHORING_TYPES /* Last entry */
};


__host__ int fe_lc_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			  field_t * q, field_grad_t * dq, fe_lc_t ** fe);
__host__ int fe_lc_free(fe_lc_t * fe);
__host__ int fe_lc_param_set(fe_lc_t * fe, fe_lc_param_t values);
__host__ int fe_lc_param_commit(fe_lc_t * fe);
__host__ int fe_lc_redshift_set(fe_lc_t * fe,  double redshift);
__host__ int fe_lc_redshift_compute(cs_t * cs, fe_lc_t * fe);
__host__ int fe_lc_target(fe_lc_t * fe, fe_t ** target);
__host__ int fe_lc_active_stress(fe_lc_t * fe);

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
int fe_lc_str_symm(fe_lc_t * fe, int index, double s[3][3]);

__host__ __device__
int fe_lc_str_anti(fe_lc_t * fe, int index, double s[3][3]);

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
int fe_lc_compute_stress_active(fe_lc_t * fe, double q[3][3], double dp[3][3],
				double sa[3][3]);

__host__ __device__
int fe_lc_chirality(fe_lc_t * fe, double * chirality);

__host__ __device__
int fe_lc_reduced_temperature(fe_lc_t * fe,  double * tau);

__host__ __device__
int fe_lc_dimensionless_field_strength(fe_lc_t * fe, double * edm);

__host__ __device__
void fe_lc_mol_field_v(fe_lc_t * fe, int index, double h[3][3][NSIMDVL]);

__host__ __device__
void fe_lc_stress_v(fe_lc_t * fe, int index, double s[3][3][NSIMDVL]);

__host__ __device__
void fe_lc_str_symm_v(fe_lc_t * fe, int index, double s[3][3][NSIMDVL]);

__host__ __device__
void fe_lc_str_anti_v(fe_lc_t * fe, int index, double s[3][3][NSIMDVL]);

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
 
