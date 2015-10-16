/*****************************************************************************
 *
 *  blue_phase.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2014 The University of Edinburgh
 *
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *    Juho Lintuvuori
 *    Davide Marenduzzo
 *
 *****************************************************************************/

#ifndef BLUEPHASE_H
#define BLUEPHASE_H

/* 'Extension' of free energy (pending free_energy_tensor.h) */

#include "free_energy.h"
#include "field.h"
#include "field_grad.h"
#include "io_harness.h"


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

//__targetEntry__ void   blue_phase_chemical_stress(const int, double sth[3][3], ...);
__targetHost__ void blue_phase_chemical_stress(int index,  double sth[3][3]);
__targetHost__ __target__ void blue_phase_chemical_stress_dev(int index, field_t* t_q, field_grad_t* t_q_grad, double* t_pth, void* pcon,int calledFromPhiForceStress);
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
__targetHost__ void blue_phase_redshift_compute(void);

__targetHost__ void blue_phase_q_uniaxial(double amplitude, const double n[3], double q[3][3]);

__targetHost__ void blue_phase_set_active_region_gamma_zeta(const int index);

__targetHost__ int fed_io_info_set(io_info_t * info);
__targetHost__ int  fed_io_info(io_info_t ** info);
__targetHost__ int blue_phase_scalar_ops(double q[3][3], double qs[5]);

#endif
 
