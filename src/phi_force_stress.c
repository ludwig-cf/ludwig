/*****************************************************************************
 *
 *  phi_force_stress.c
 *  
 *  Wrapper functions for stress computation.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "free_energy.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "phi_force_stress.h"
#include "util.h"
#include "math.h"
#include "blue_phase.h"
#include "timer.h"

double * pth_;
double * t_pth_;

/*****************************************************************************
 *
 *  phi_force_stress_compute
 *
 *  Compute the stress everywhere and store.
 *
 *****************************************************************************/

__targetEntry__ void chemical_stress_lattice(double pth_local[3][3], field_t* t_q, field_grad_t* t_q_grad, double* t_pth, void* pcon, void (* chemical_stress)(const int index, double s[3][3]),int isBPCS){ 

  int index;

  __targetTLPNoStride__(index,tc_nSites){
    
    int coords[3];
    targetCoords3D(coords,tc_Nall,index);
    
    /*  if not a halo site:*/
    if (coords[0] >= (tc_nhalo-tc_nextra) && 
	coords[1] >= (tc_nhalo-tc_nextra) && 
	coords[2] >= (tc_nhalo-tc_nextra) &&
	coords[0] < tc_Nall[X]-(tc_nhalo-tc_nextra) &&  
	coords[1] < tc_Nall[Y]-(tc_nhalo-tc_nextra)  &&  
	coords[2] < tc_Nall[Z]-(tc_nhalo-tc_nextra) ){ 
      

      if (isBPCS){
	/* we are using blue_phase_chemical_stress which is ported to targetDP
	 * for the time being we are explicitly calling
	 * blue_phase_chemical_stress
	 * ultimitely this will be generic when the other options are
	 * ported to targetDP */
	 int calledFromPhiForceStress=1;
	 blue_phase_chemical_stress_dev(index, t_q, t_q_grad, t_pth, pcon, 
					calledFromPhiForceStress);
      }
      else{

#ifndef __NVCC__
	/* only blue_phase_chemical_stress support for CUDA. */
	chemical_stress(index, pth_local);
	phi_force_stress_set(index, pth_local); 
#endif

      }
      
    }
  }

return;
}

/*****************************************************************************
 *
 *  phi_force_stress_compute
 *
 *****************************************************************************/

__targetHost__
void phi_force_stress_compute(field_t * q, field_grad_t* q_grad) {

  int nlocal[3];
  int nextra = 1;
  coords_nlocal(nlocal);
  assert(coords_nhalo() >= 2);

  int nhalo = coords_nhalo();
  int Nall[3];
  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;
  int nSites=Nall[X]*Nall[Y]*Nall[Z];

  double pth_local[3][3];
  void (* chemical_stress)(const int index, double s[3][3]);


  chemical_stress = fe_chemical_stress_function();

  
  /* initialise kernel constants on both host and target */
  blue_phase_set_kernel_constants();

  void* pcon=NULL;
  blue_phase_target_constant_ptr(&pcon);

  field_t* t_q = NULL;
  field_grad_t* t_q_grad = NULL;


  /* isBPCS is 1 if we are using  blue_phase_chemical_stress 
   * (which is ported to targetDP), 0 otherwise*/

  int isBPCS=((void*)chemical_stress)==((void*) blue_phase_chemical_stress);

#ifdef __NVCC__
    if (!isBPCS) fatal("only Blue Phase chemical stress is currently supported for CUDA");
#endif

  if (isBPCS){ 

    t_q = q->tcopy; 
    t_q_grad = q_grad->tcopy;

    double* tmpptr;

#ifndef KEEPFIELDONTARGET    

    copyFromTarget(&tmpptr,&(t_q->data),sizeof(double*)); 
    copyToTarget(tmpptr,q->data,q->nf*nSites*sizeof(double));
    
    copyFromTarget(&tmpptr,&(t_q_grad->grad),sizeof(double*)); 
    copyToTarget(tmpptr,q_grad->grad,q_grad->nf*NVECTOR*nSites*sizeof(double));
    
    copyFromTarget(&tmpptr,&(t_q_grad->delsq),sizeof(double*)); 
    copyToTarget(tmpptr,q_grad->delsq,q_grad->nf*nSites*sizeof(double));
#endif

  }


  /* copy lattice shape constants to target ahead of execution*/
  copyConstToTarget(&tc_nSites,&nSites, sizeof(int));
  copyConstToTarget(&tc_nextra,&nextra, sizeof(int));
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int));
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int));
  
  TIMER_start(TIMER_CHEMICAL_STRESS_KERNEL);

  /* execute lattice-based operation on target*/
  chemical_stress_lattice __targetLaunch__(nSites) (pth_local, t_q, t_q_grad, t_pth_, pcon, chemical_stress, isBPCS);
  targetSynchronize();

  TIMER_stop(TIMER_CHEMICAL_STRESS_KERNEL);
  

  if (isBPCS){
    /* we are using blue_phase_chemical_stress which is ported to targetDP
       copy result from target back to host */

#ifndef KEEPFIELDONTARGET    
    copyFromTarget(pth_,t_pth_,3*3*nSites*sizeof(double));      
#endif
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress_set
 *
 *****************************************************************************/

__targetHost__  void phi_force_stress_set(const int index, double p[3][3]) {

  int ia, ib;

  assert(pth_);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      pth_[addr_rank2(tc_nSites,3,3,index,ia,ib)] = p[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress
 *
 *****************************************************************************/

__targetHost__  void phi_force_stress(const int index, double p[3][3]) {

  int ia, ib;

  assert(pth_);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      p[ia][ib] = pth_[addr_rank2(tc_nSites,3,3,index,ia,ib)];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  phi_force_stress_allocate
 *
 *****************************************************************************/

__targetHost__  void phi_force_stress_allocate() {

  int n;

  assert(coords_nhalo() >= 2);

  n = coords_nsites();

  pth_ = (double *) malloc(9*n*sizeof(double));
  if (pth_ == NULL) fatal("malloc(pth_) failed\n");

  targetMalloc((void**) &t_pth_,9*n*sizeof(double));


  return;
}

/*****************************************************************************
 *
 *  phi_force_stress_free
 *
 *****************************************************************************/

__targetHost__  void phi_force_stress_free() {

  free(pth_);
  targetFree(t_pth_);

  return;
}
