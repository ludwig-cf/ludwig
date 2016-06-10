/****************************************************************************
 *
 *  symmetric.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef SYMMETRIC_H
#define SYMMETRIC_H

#include "memory.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"

#ifndef OLD_SHIT
typedef struct fe_symmetric_param_s fe_symm_param_t;
typedef struct fe_symm_s fe_symm_t;

struct fe_symmetric_param_s {
  double a;
  double b;
  double kappa;
};

struct fe_symm_s {
  fe_symm_param_t param;       /* Parameters */
  field_t * phi;               /* Scalar order parameter or composition */
  field_grad_t * dphi;         /* Gradients thereof */
  fe_symm_t * target;          /* Target copy */
};

__host__ int fe_symm_create(field_t * f, field_grad_t * grd, fe_symm_t ** p);
__host__ int fe_symm_free(fe_symm_t * fe);
__host__ int fe_symm_param_set(fe_symm_t * fe, fe_symm_param_t values);

__host__ __device__ int fe_symm_param(fe_symm_t * fe, fe_symm_param_t * values);
__host__ __device__ int fe_symm_interfacial_tension(fe_symm_t * fe, double * s);
__host__ __device__ int fe_symm_interfacial_width(fe_symm_t * fe, double * xi);
__host__ __device__ int fe_symm_fed(fe_symm_t * fe, int index, double * fed);
__host__ __device__ int fe_symm_mu(fe_symm_t * fe, int index, double * mu);
__host__ __device__ int fe_symm_str(fe_symm_t * fe, int index, double s[3][3]);

__target__
void fe_symm_chemical_potential_target(fe_symm_t * fe, int index, double * mu);
__target__
void fe_symm_chemical_stress_target(fe_symm_t * fe, int index,
						 double s[3][3*NSIMDVL]);

#else
__target__
double symmetric_chemical_potential_target(const int index, const int nop,
					   const double * t_phi,
					   const double * t_delsqphi);
__target__ void symmetric_chemical_stress_target(const int index,
						 double s[3][3*NSIMDVL],
						 const double* t_phi, 
						 const double* t_gradphi,
						 const double* t_delsqphi);

__targetHost__ int symmetric_phi_set(field_t * phi, field_grad_t * dphi);

__targetHost__ void   symmetric_free_energy_parameters_set(double a, double b, double kappa);
__targetHost__ double symmetric_a(void);
__targetHost__ double symmetric_b(void);
__targetHost__ double symmetric_interfacial_tension(void);
__targetHost__ double symmetric_interfacial_width(void);
__targetHost__ double symmetric_free_energy_density(const int index);
__targetHost__ double symmetric_chemical_potential(const int index, const int nop);
__targetHost__ double symmetric_isotropic_pressure(const int index);
__targetHost__ void   symmetric_chemical_stress(const int index, double s[3][3]);
#endif

#endif

