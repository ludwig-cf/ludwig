/*****************************************************************************
 *
 *  phi_force_stress.h
 *  
 *  Wrapper functions for stress computation.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHI_FORCE_STRESS_H
#define PHI_FORCE_STRESS_H

enum {PTH_METHOD_NO_FORCE, PTH_METHOD_DIVERGENCE, PTH_METHOD_GRADMU};

typedef struct pth_s pth_t;

__host__ int pth_create(int method, pth_t ** pth);
__host__ int pth_free(pth_t * pth);
__host__ int pth_memcpy(pth_t * pth, int flag);
__host__ int phi_force_stress_compute(pth_t * pth, field_t* q, field_grad_t* q_grad);

__host__ __device__ void phi_force_stress(pth_t * pth,  int index, double p[3][3]);
__host__ __device__ void phi_force_stress_set(pth_t * pth, int index, double p[3][3]);

#endif
