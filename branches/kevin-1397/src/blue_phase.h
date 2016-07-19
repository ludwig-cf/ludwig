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
  io_info_t * io;             /* IO stuff */
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
  double electric[3];                     /* Electric field */
  double amplitude0;                      /* Initial amplitude from input */

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
			LC_ANCHROING_TYPES /* Last entry */
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
void fe_lc_compute_h_v(fe_lc_t * fe, double q[3][3][NSIMDVL], 
		       double dq[3][3][3][NSIMDVL],
		       double dsq[3][3][NSIMDVL], 
		       double h[3][3][NSIMDVL]);


/* Function of the parameters only */

__host__ __device__
int fe_lc_amplitude_compute(fe_lc_param_t * param, double * a);

__host__ __device__
int fe_lc_q_uniaxial(fe_lc_param_t * param, const double n[3], double q[3][3]);
__host__ int fe_lc_scalar_ops(double q[3][3], double qs[NQAB]);



#ifdef OLD_SHIT
struct bluePhaseKernelConstants {

  int nSites;
  int Nall[3];
  int nhalo;
  int nextra;
  double e0[3];
  double q0;
  double kappa0;
  double kappa1;
  double a0_;
  double gamma_;
  double r3_;
  double epsilon_;
  double xi_;
  double zeta_;
  double d_[3][3];
  double e_[3][3][3];
  double w1_coll;                         /* Anchoring strength parameter */
  double w2_coll;                         /* Second anchoring parameter */
  double w1_wall;
  double w2_wall;
  double amplitude;
  char dc_[3][3];
  char ec_[3][3][3];
};

typedef struct bluePhaseKernelConstants bluePhaseKernelConstants_t;

__targetHost__ void blue_phase_set_kernel_constants();
__targetHost__ void blue_phase_target_constant_ptr(void** tmp);
__targetHost__ void blue_phase_host_constant_ptr(void** tmp);
__targetHost__ int blue_phase_q_set(field_t * q, field_grad_t * dq);

__targetHost__ void   blue_phase_set_free_energy_parameters(double, double, double, double);
__targetHost__ void   blue_phase_set_xi(double);
__targetHost__ void   blue_phase_set_zeta(double);
__targetHost__ void   blue_phase_set_gamma(double);

__targetHost__ double blue_phase_free_energy_density(const int);
__targetHost__ __target__ double blue_phase_compute_fed(double q[3][3], double dq[3][3][3],bluePhaseKernelConstants_t* pbpc);
__targetHost__ double blue_phase_compute_bulk_fed(double q[3][3]);
__targetHost__ double blue_phase_compute_gradient_fed(double q[3][3], double dq[3][3][3]);

__targetHost__ void   blue_phase_molecular_field(const int, double h[3][3]);
__targetHost__ __target__ void blue_phase_compute_h(double q[3][3], double dq[3][3][3],
						     double dsq[3][3], double h[3][3],
						     bluePhaseKernelConstants_t* pbpc);
__targetHost__ __target__ void blue_phase_compute_stress(double q[3][3], double dq[3][3][3],
					  double h[3][3], double sth[3][3],
					  bluePhaseKernelConstants_t* pbpc);
__targetHost__ __target__ void blue_phase_compute_stress_vec(double q[3][3][VVL], double dq[3][3][3][VVL],
					  double h[3][3][VVL], double* sth, 
							     bluePhaseKernelConstants_t* pbpc, int baseIndex);
__targetHost__ void blue_phase_chemical_stress(int index,  double sth[3][3]);
 __target__ void blue_phase_chemical_stress_dev(int index, field_t* t_q, field_grad_t* t_q_grad, double* t_pth, void* pcon,int calledFromPhiForceStress);
 __target__ void blue_phase_chemical_stress_dev_vec(int index, field_t* t_q, field_grad_t* t_q_grad, double* t_pth, void* pcon,int calledFromPhiForceStress);
__targetHost__ void   blue_phase_redshift_set(const double redshift);
__targetHost__ void   blue_phase_electric_field_set(const double e[3]);
__targetHost__ void   blue_phase_dielectric_anisotropy_set(double e);
__targetHost__ double   blue_phase_dielectric_anisotropy(void);
__targetHost__ double blue_phase_amplitude_compute(void);
__targetHost__ double blue_phase_get_xi(void);
__targetHost__ double blue_phase_get_zeta(void);
__targetHost__ double blue_phase_get_gamma(void);
__targetHost__ double blue_phase_chirality(void);
__targetHost__ double blue_phase_reduced_temperature(void);
__targetHost__ double blue_phase_redshift(void);
__targetHost__ double blue_phase_q0(void);
__targetHost__ double blue_phase_kappa0(void);
__targetHost__ double blue_phase_kappa1(void);
__targetHost__ double blue_phase_redshift(void);
__targetHost__ double blue_phase_rredshift(void);
__targetHost__ double blue_phase_a0(void);
__targetHost__ double blue_phase_gamma(void);
__targetHost__ double blue_phase_dimensionless_field_strength(void);

__targetHost__ void blue_phase_redshift_update_set(int onoff);

__targetHost__ void blue_phase_q_uniaxial(double amplitude, const double n[3], double q[3][3]);

__targetHost__ void blue_phase_set_active_region_gamma_zeta(const int index);

__targetHost__ int fed_io_info_set(int grid[3], int form_out);
__targetHost__ int fed_io_info(io_info_t ** info);
__targetHost__ int blue_phase_scalar_ops(double q[3][3], double qs[5]);
#endif
#endif
 
