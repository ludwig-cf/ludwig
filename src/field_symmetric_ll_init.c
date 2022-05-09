/****************************************************************************
 *
 *  field_symmetric_ll_init.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Shan Chen (shan.chen@epfl.ch)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "noise.h"
#include "util.h"
#include "field_s.h"
#include "field_symmetric_ll_init.h"

#define FE_PHI 0
#define FE_PSI 1

/*****************************************************************************
 *
 *  field_symmetric_ll_phi_uniform
 *
 *****************************************************************************/

int field_symmetric_ll_phi_uniform(field_t * phi, double phi0) {
    
  int nlocal[3];
  int ic, jc, kc, index;

  assert(phi);
    
  cs_nlocal(phi->cs, nlocal);
    
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = cs_index(phi->cs, ic, jc, kc);
	phi->data[addr_rank1(phi->nsites, phi->nf, index, FE_PHI)] = phi0;
      }
    }
  }
    
  return 0;
}


/*****************************************************************************
 *
 *  field_symmetric_ll_psi_uniform
 *
 *****************************************************************************/

int field_symmetric_ll_psi_uniform(field_t * phi, double psi0) {
    
  int nlocal[3];
  int ic, jc, kc, index;

  assert(phi);
    
  cs_nlocal(phi->cs, nlocal);
    
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = cs_index(phi->cs, ic, jc, kc);
	phi->data[addr_rank1(phi->nsites, phi->nf, index, FE_PSI)] = psi0;
      }
    }
  }
    
  return 0;
}

