/*****************************************************************************
 *
 *  phi_force_stress.h
 *  
 *  Wrapper functions for stress computation.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PHI_FORCE_STRESS_H
#define LUDWIG_PHI_FORCE_STRESS_H

#include "pe.h"
#include "coords.h"
#include "free_energy.h"

enum {PTH_METHOD_NO_FORCE, PTH_METHOD_DIVERGENCE, PTH_METHOD_GRADMU, 
      PTH_METHOD_STRESS_ONLY};

typedef struct pth_s pth_t;

__host__ int pth_create(pe_t * pe, cs_t * cs, int method, pth_t ** pth);
__host__ int pth_free(pth_t * pth);
__host__ int pth_memcpy(pth_t * pth, tdpMemcpyKind flag);
__host__ int pth_stress_compute(pth_t * pth, fe_t * fe);

__host__ __device__ void pth_stress(pth_t * pth,  int index, double p[3][3]);
__host__ __device__ void pth_stress_set(pth_t * pth, int index, double p[3][3]);

#endif
